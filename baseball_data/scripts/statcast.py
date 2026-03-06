#!/usr/bin/env python3
"""
Statcast loader: fetches pitch-level data for the 2025 season and appends to the database.

Pulls data in 6-day chunks, keeps minimal columns for HR prediction and batter outcome
modeling. Run statcast.sql once before first use.

Required for in-play labels (pitch_result, immediate_event, in_play_outcome): events,
description, type. Statcast provides these; _normalize_statcast_columns maps alternate
names (e.g. des, Type, Events) to our schema so they are preserved.

PostgreSQL: set DATABASE_URL or PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE.
"""

from __future__ import annotations

import logging
import os
import random
import sys

# Disable tqdm progress bars from pybaseball statcast to keep logs readable
os.environ["TQDM_DISABLE"] = "1"

import time
import warnings

warnings.filterwarnings("ignore", message=".*errors='ignore'.*", module="pybaseball.datahelpers.postprocessing")
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

from dotenv import load_dotenv
from pybaseball import statcast
from pybaseball.statcast import StatcastException
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

_BASE = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BASE.parent

_SEASON_START = "2025-03-27"
_SEASON_END = "2025-09-29"
_CHUNK_DAYS = 6

_COLS = [
    "game_date", "game_year", "game_pk", "game_type", "player_name",
    "batter", "pitcher", "events", "description",
    "release_speed", "launch_speed", "launch_angle", "pitch_type", "type",
    "hit_distance_sc", "stand", "p_throws",
    "home_team", "away_team", "inning", "inning_topbot",
    "balls", "strikes", "outs_when_up", "zone", "bb_type",
    "at_bat_number", "pitch_number",
    "plate_x", "plate_z",
    "release_spin_rate",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_env() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(_BASE / ".env")


def _db_url() -> str:
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    dbname = os.getenv("PGDATABASE")
    pw = quote_plus(password) if password else ""
    return f"postgresql://{user}:{pw}@{host}:{port}/{dbname}"


def _date_range_chunks() -> list[tuple[str, str]]:
    start = datetime.strptime(_SEASON_START, "%Y-%m-%d").date()
    end = datetime.strptime(_SEASON_END, "%Y-%m-%d").date()
    chunks = []
    while start <= end:
        chunk_end = min(start + timedelta(days=_CHUNK_DAYS - 1), end)
        chunks.append((start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        start = chunk_end + timedelta(days=1)
    return chunks


def _normalize_statcast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Statcast/pybaseball column names to our schema so events, description, type
    are preserved for pitch-result and in-play labels. Statcast provides these;
    the API or pybaseball may return different names (e.g. des, Type, Events).
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    # Normalize case: some sources return Type, Events, Description
    for alt, canonical in [("Type", "type"), ("Events", "events"), ("Description", "description")]:
        if alt in df.columns and canonical not in df.columns:
            df[canonical] = df[alt]
        elif alt in df.columns and canonical in df.columns:
            df[canonical] = df[canonical].fillna(df[alt])
    # Baseball Savant / pybaseball: "des" = plate appearance description; "description" = pitch result.
    # Use "des" as fallback for "description" so we don't lose pitch-level outcome text.
    if "des" in df.columns and "description" not in df.columns:
        df["description"] = df["des"]
    elif "des" in df.columns and "description" in df.columns:
        df["description"] = df["description"].fillna(df["des"])
    # Some Statcast exports use spin_rate; we store as release_spin_rate.
    if "spin_rate" in df.columns and "release_spin_rate" not in df.columns:
        df["release_spin_rate"] = df["spin_rate"]
    elif "spin_rate" in df.columns and "release_spin_rate" in df.columns:
        df["release_spin_rate"] = df["release_spin_rate"].fillna(df["spin_rate"])
    return df


def _filter_and_trim(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize_statcast_columns(df)

    if "game_type" in df.columns:
        df = df.loc[df["game_type"] == "R"].copy()
    else:
        df = df.copy()

    if "hit_distance" in df.columns and "hit_distance_sc" not in df.columns:
        df["hit_distance_sc"] = df["hit_distance"]
    elif "hit_distance" in df.columns and "hit_distance_sc" in df.columns:
        df["hit_distance_sc"] = df["hit_distance_sc"].fillna(df["hit_distance"])

    keep = [c for c in _COLS if c in df.columns]
    df = df[keep].copy()

    # Truncate to schema lengths to avoid StringDataRightTruncation
    def _trunc(val, w: int):
        if pd.isna(val):
            return pd.NA
        t = str(val).strip()
        return t[:w] if len(t) > w else t

    for col, width in ({"stand": 1, "p_throws": 1, "type": 1}).items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x, w=width: _trunc(x, w))

    for c in _COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[_COLS]
    return df


def main() -> None:
    _load_env()
    url = _db_url()

    try:
        engine = create_engine(url)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
    except SQLAlchemyError as e:
        log.error("Database connection failed: %s", e)
        raise SystemExit(1) from e

    chunks = _date_range_chunks()
    log.info("Season %s to %s, %d chunks (6-day windows)", _SEASON_START, _SEASON_END, len(chunks))

    total_appended = 0
    for i, (start_dt, end_dt) in enumerate(chunks):
        log.info("Chunk %d/%d: %s to %s", i + 1, len(chunks), start_dt, end_dt)
        try:
            raw = statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
        except StatcastException as e:
            log.warning("Statcast request failed for %s–%s: %s; skipping chunk", start_dt, end_dt, e)
            time.sleep(random.uniform(2, 5))
            continue
        except Exception as e:
            log.warning("Request error for %s–%s: %s; skipping chunk", start_dt, end_dt, e)
            time.sleep(random.uniform(2, 5))
            continue

        n_fetched = 0 if raw is None or raw.empty else len(raw)
        df = _filter_and_trim(raw)
        if df.empty:
            log.info("  Fetched %d rows, 0 regular-season rows to append", n_fetched)
            time.sleep(random.uniform(2, 5))
            continue

        try:
            df.to_sql(
                "statcast_pitches_2025",
                engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=5000,
            )
        except SQLAlchemyError as e:
            log.error("Upload failed for %s–%s: %s", start_dt, end_dt, e)
            raise SystemExit(1) from e

        n_append = len(df)
        total_appended += n_append
        # Warn if events/description/type are mostly null (needed for batter in-play labels)
        for col in ("events", "description", "type"):
            if col in df.columns:
                pct = df[col].notna().mean() * 100
                if pct < 10 and n_append > 100:
                    log.warning("  %s is %.1f%% non-null; batter pitch-result/in-play labels may be missing", col, pct)
        log.info("  Fetched %d rows, appended %d (cumulative %d)", n_fetched, n_append, total_appended)
        time.sleep(random.uniform(2, 5))

    log.info("Done. Total rows appended: %d", total_appended)


if __name__ == "__main__":
    main()
