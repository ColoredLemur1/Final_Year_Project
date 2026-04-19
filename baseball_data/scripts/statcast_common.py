#!/usr/bin/env python3
"""
Shared Statcast fetch / normalize / DB helpers for statcast.py, statcast_seasons.py,
and statcast_plate_backfill.py.
"""

from __future__ import annotations

import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from pybaseball import statcast
from pybaseball.statcast import StatcastException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

_BASE = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BASE.parent

CHUNK_DAYS = 6

COLS = [
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


def load_env() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(_BASE / ".env")


def db_url() -> str:
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    dbname = os.getenv("PGDATABASE")
    pw = quote_plus(password) if password else ""
    return f"postgresql://{user}:{pw}@{host}:{port}/{dbname}"


def date_range_chunks(season_start: str, season_end: str, chunk_days: int = CHUNK_DAYS) -> list[tuple[str, str]]:
    start = datetime.strptime(season_start, "%Y-%m-%d").date()
    end = datetime.strptime(season_end, "%Y-%m-%d").date()
    chunks = []
    while start <= end:
        chunk_end = min(start + timedelta(days=chunk_days - 1), end)
        chunks.append((start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        start = chunk_end + timedelta(days=1)
    return chunks


def normalize_statcast_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for alt, canonical in [("Type", "type"), ("Events", "events"), ("Description", "description")]:
        if alt in df.columns and canonical not in df.columns:
            df[canonical] = df[alt]
        elif alt in df.columns and canonical in df.columns:
            df[canonical] = df[canonical].fillna(df[alt])
    if "des" in df.columns and "description" not in df.columns:
        df["description"] = df["des"]
    elif "des" in df.columns and "description" in df.columns:
        df["description"] = df["description"].fillna(df["des"])
    if "spin_rate" in df.columns and "release_spin_rate" not in df.columns:
        df["release_spin_rate"] = df["spin_rate"]
    elif "spin_rate" in df.columns and "release_spin_rate" in df.columns:
        df["release_spin_rate"] = df["release_spin_rate"].fillna(df["spin_rate"])
    return df


def filter_and_trim(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    df = normalize_statcast_columns(df)

    if "game_type" in df.columns:
        df = df.loc[df["game_type"] == "R"].copy()
    else:
        df = df.copy()

    if "hit_distance" in df.columns and "hit_distance_sc" not in df.columns:
        df["hit_distance_sc"] = df["hit_distance"]
    elif "hit_distance" in df.columns and "hit_distance_sc" in df.columns:
        df["hit_distance_sc"] = df["hit_distance_sc"].fillna(df["hit_distance"])

    keep = [c for c in COLS if c in df.columns]
    df = df[keep].copy()

    def _trunc(val, w: int):
        if pd.isna(val):
            return pd.NA
        t = str(val).strip()
        return t[:w] if len(t) > w else t

    for col, width in ({"stand": 1, "p_throws": 1, "type": 1}).items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x, w=width: _trunc(x, w))

    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[COLS]
    return df


def append_to_table(engine, df: pd.DataFrame, table_name: str) -> None:
    df.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )


def check_db(engine) -> None:
    with engine.connect() as c:
        c.execute(text("SELECT 1"))


def fetch_statcast_chunk(start_dt: str, end_dt: str, verbose: bool = False) -> pd.DataFrame | None:
    try:
        return statcast(start_dt=start_dt, end_dt=end_dt, verbose=verbose)
    except StatcastException:
        time.sleep(random.uniform(2, 5))
        return None
    except Exception:
        time.sleep(random.uniform(2, 5))
        return None


def load_season_to_table(
    engine,
    table_name: str,
    season_start: str,
    season_end: str,
    log,
) -> int:
    """Fetch Statcast in chunks and append to ``table_name``. Returns total rows appended."""
    chunks = date_range_chunks(season_start, season_end)
    log.info("Table %s: %s to %s, %d chunks", table_name, season_start, season_end, len(chunks))
    total = 0
    for i, (start_dt, end_dt) in enumerate(chunks):
        log.info("Chunk %d/%d: %s to %s", i + 1, len(chunks), start_dt, end_dt)
        raw = fetch_statcast_chunk(start_dt, end_dt, verbose=False)
        n_fetched = 0 if raw is None or raw.empty else len(raw)
        df = filter_and_trim(raw) if raw is not None else pd.DataFrame()
        if df.empty:
            log.info("  Fetched %d rows, 0 regular-season rows to append", n_fetched)
            time.sleep(random.uniform(2, 5))
            continue
        try:
            append_to_table(engine, df, table_name)
        except SQLAlchemyError as e:
            log.error("Upload failed for %s–%s: %s", start_dt, end_dt, e)
            raise
        n_append = len(df)
        total += n_append
        for col in ("events", "description", "type"):
            if col in df.columns:
                pct = df[col].notna().mean() * 100
                if pct < 10 and n_append > 100:
                    log.warning(
                        "  %s is %.1f%% non-null; batter pitch-result labels may be missing",
                        col,
                        pct,
                    )
        log.info("  Fetched %d rows, appended %d (cumulative %d)", n_fetched, n_append, total)
        time.sleep(random.uniform(2, 5))
    return total
