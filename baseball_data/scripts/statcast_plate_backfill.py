#!/usr/bin/env python3
"""
Backfill plate_x and plate_z for existing statcast_pitches_2025 rows.

Fetches Statcast data in 6-day chunks for the 2025 season (same as statcast.py),
filters to regular-season, and UPDATEs existing rows by matching on
(game_pk, at_bat_number, pitch_number).

Ensures plate_x/plate_z columns exist (ALTER TABLE if needed).
Uses root .env for DB connection.

Run from baseball_data/: python -m scripts.statcast_plate_backfill
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

try:
    from dotenv import load_dotenv
    from pybaseball import statcast
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
    try:
        from pybaseball.statcast import StatcastException
    except ImportError:
        StatcastException = Exception
except ImportError as e:
    print(f"Missing dependency: {e}. Install with: pip install pybaseball pandas sqlalchemy python-dotenv", file=sys.stderr)
    raise SystemExit(1) from e

_BASE = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BASE.parent

_SEASON_START = "2025-03-27"
_SEASON_END = "2025-09-29"
_CHUNK_DAYS = 6

# Required columns for matching and update
_KEY_COLS = ["game_pk", "at_bat_number", "pitch_number"]
_UPDATE_COLS = ["plate_x", "plate_z"]

os.environ["TQDM_DISABLE"] = "1"
import warnings
warnings.filterwarnings("ignore", message=".*errors='ignore'.*", module="pybaseball.datahelpers.postprocessing")

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
    host = os.getenv("DB_HOST") or os.getenv("PGHOST", "localhost")
    port = os.getenv("DB_PORT") or os.getenv("PGPORT", "5432")
    user = os.getenv("DB_USER") or os.getenv("PGUSER", "postgres")
    password = os.getenv("DB_PASS") or os.getenv("PGPASSWORD", "")
    dbname = os.getenv("DB_NAME") or os.getenv("PGDATABASE", "baseball")
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


def _ensure_columns(engine) -> None:
    """Add plate_x, plate_z if they don't exist."""
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE statcast_pitches_2025 ADD COLUMN IF NOT EXISTS plate_x REAL"))
        conn.execute(text("ALTER TABLE statcast_pitches_2025 ADD COLUMN IF NOT EXISTS plate_z REAL"))


def _filter_and_extract(df: pd.DataFrame) -> pd.DataFrame:
    """Filter game_type R and keep only key + update columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    if "game_type" in df.columns:
        df = df.loc[df["game_type"] == "R"].copy()
    else:
        df = df.copy()
    cols = _KEY_COLS + _UPDATE_COLS
    available = [c for c in cols if c in df.columns]
    df = df[available].copy()
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[_KEY_COLS + _UPDATE_COLS].dropna(subset=_KEY_COLS, how="all")


def _update_chunk(engine, df: pd.DataFrame) -> int:
    """Update statcast_pitches_2025 with plate_x, plate_z from df via temp table."""
    if df.empty:
        return 0
    # Drop rows where both plate_x and plate_z are null (nothing to update)
    df = df.dropna(subset=_UPDATE_COLS, how="all")
    if df.empty:
        return 0
    with engine.begin() as conn:
        df.to_sql("_statcast_plate_temp", conn, if_exists="replace", index=False, method="multi", chunksize=5000)
        result = conn.execute(text("""
            UPDATE statcast_pitches_2025 s
            SET plate_x = COALESCE(t.plate_x, s.plate_x),
                plate_z = COALESCE(t.plate_z, s.plate_z)
            FROM _statcast_plate_temp t
            WHERE s.game_pk = t.game_pk
              AND s.at_bat_number = t.at_bat_number
              AND s.pitch_number = t.pitch_number
        """))
        conn.execute(text("DROP TABLE _statcast_plate_temp"))
    return result.rowcount if hasattr(result, "rowcount") else 0


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

    _ensure_columns(engine)
    log.info("Ensured plate_x, plate_z columns exist")

    chunks = _date_range_chunks()
    log.info("Season %s to %s, %d chunks (6-day windows)", _SEASON_START, _SEASON_END, len(chunks))

    total_updated = 0
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
        df = _filter_and_extract(raw)
        if df.empty:
            log.info("  Fetched %d rows, 0 regular-season rows with plate data to update", n_fetched)
            time.sleep(random.uniform(2, 5))
            continue

        try:
            n_updated = _update_chunk(engine, df)
        except SQLAlchemyError as e:
            log.error("Update failed for %s–%s: %s", start_dt, end_dt, e)
            raise SystemExit(1) from e

        total_updated += n_updated
        log.info("  Fetched %d rows, updated %d with plate_x/plate_z (cumulative %d)", n_fetched, n_updated, total_updated)
        time.sleep(random.uniform(2, 5))

    log.info("Done. Total rows updated with plate_x/plate_z: %d", total_updated)


if __name__ == "__main__":
    main()
