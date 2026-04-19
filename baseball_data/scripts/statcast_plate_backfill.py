#!/usr/bin/env python3
"""
Statcast plate backfill: fills plate_x and plate_z for existing pitch rows.

Fetches Statcast in 6-day chunks and UPDATEs rows by game_pk, at_bat_number, pitch_number.
Adds columns if missing. Default table statcast_pitches_2025; use --year for other
statcast_pitches_* tables after loading that season with statcast_seasons.py.

PostgreSQL: set DATABASE_URL or PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
import warnings

os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore", message=".*errors='ignore'.*", module="pybaseball.datahelpers.postprocessing")

import pandas as pd
from pybaseball import statcast
from pybaseball.statcast import StatcastException
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from statcast_common import (
    CHUNK_DAYS,
    check_db,
    date_range_chunks,
    db_url,
    load_env,
    normalize_statcast_columns,
)

# Default regular-season windows for --year when --start/--end omitted
_SEASON_BOUNDS: dict[int, tuple[str, str]] = {
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-29"),
}

_KEY_COLS = ["game_pk", "at_bat_number", "pitch_number"]
_UPDATE_COLS = ["plate_x", "plate_z"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _ensure_columns(engine, table_name: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f'ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS plate_x REAL'))
        conn.execute(text(f'ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS plate_z REAL'))


def _filter_and_extract(df: pd.DataFrame) -> pd.DataFrame:
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


def _update_chunk(engine, df: pd.DataFrame, table_name: str) -> int:
    if df.empty:
        return 0
    df = df.dropna(subset=_UPDATE_COLS, how="all")
    if df.empty:
        return 0
    temp = "_statcast_plate_temp"
    with engine.begin() as conn:
        df.to_sql(temp, conn, if_exists="replace", index=False, method="multi", chunksize=5000)
        result = conn.execute(
            text(
                f"""
            UPDATE {table_name} s
            SET plate_x = COALESCE(t.plate_x, s.plate_x),
                plate_z = COALESCE(t.plate_z, s.plate_z)
            FROM {temp} t
            WHERE s.game_pk = t.game_pk
              AND s.at_bat_number = t.at_bat_number
              AND s.pitch_number = t.pitch_number
        """
            )
        )
        conn.execute(text(f"DROP TABLE {temp}"))
    return result.rowcount if hasattr(result, "rowcount") and result.rowcount else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill plate_x / plate_z from Statcast.")
    p.add_argument("--year", type=int, default=2025, help="Season / table suffix (default: 2025).")
    p.add_argument("--start", default=None, help="Season start YYYY-MM-DD (default: built-in range for --year).")
    p.add_argument("--end", default=None, help="Season end YYYY-MM-DD (default: built-in range for --year).")
    args = p.parse_args()

    year = args.year
    if year < 2000 or year > 2100:
        log.error("Invalid --year")
        raise SystemExit(1)
    table_name = f"statcast_pitches_{year}"
    if args.start and args.end:
        season_start, season_end = args.start, args.end
    elif year in _SEASON_BOUNDS:
        season_start, season_end = _SEASON_BOUNDS[year]
    else:
        log.error("No default date range for year %s; pass --start and --end.", year)
        raise SystemExit(1)

    load_env()
    url = db_url()

    try:
        engine = create_engine(url)
        check_db(engine)
    except SQLAlchemyError as e:
        log.error("Database connection failed: %s", e)
        raise SystemExit(1) from e

    _ensure_columns(engine, table_name)
    log.info("Ensured plate_x, plate_z on %s", table_name)

    chunks = date_range_chunks(season_start, season_end, CHUNK_DAYS)
    log.info("Table %s: %s to %s, %d chunks", table_name, season_start, season_end, len(chunks))

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
        if raw is not None and not raw.empty:
            raw = normalize_statcast_columns(raw)
        df = _filter_and_extract(raw) if raw is not None else pd.DataFrame()
        if df.empty:
            log.info("  Fetched %d rows, 0 regular-season rows with plate data to update", n_fetched)
            time.sleep(random.uniform(2, 5))
            continue

        try:
            n_updated = _update_chunk(engine, df, table_name)
        except SQLAlchemyError as e:
            log.error("Update failed for %s–%s: %s", start_dt, end_dt, e)
            raise SystemExit(1) from e

        total_updated += n_updated
        log.info("  Fetched %d rows, updated %d with plate_x/plate_z (cumulative %d)", n_fetched, n_updated, total_updated)
        time.sleep(random.uniform(2, 5))

    log.info("Done. Total rows updated on %s: %d", table_name, total_updated)


if __name__ == "__main__":
    main()
