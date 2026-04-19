#!/usr/bin/env python3
"""
Load Statcast regular-season pitches for extra years (default 2023 and 2024).

Run baseball_data/sql/z_statcast_multiseason.sql once for season tables, clean views, and
clean_statcast_with_batter. Same column set as statcast.py; use statcast_plate_backfill.py
--year YYYY if plate_x/plate_z need filling.

PostgreSQL: PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE (or .env at project root).
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings

warnings.filterwarnings("ignore", message=".*errors='ignore'.*", module="pybaseball.datahelpers.postprocessing")

os.environ["TQDM_DISABLE"] = "1"

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from statcast_common import check_db, db_url, load_env, load_season_to_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Approximate MLB regular-season windows (regular season only; game_type R filtered in Python).
DEFAULT_SEASON_BOUNDS: dict[int, tuple[str, str]] = {
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
}


def main() -> None:
    p = argparse.ArgumentParser(description="Load Statcast for extra seasons (e.g. 2023–2024).")
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2023, 2024],
        help="Seasons to load (default: 2023 2024).",
    )
    p.add_argument(
        "--start",
        default=None,
        help="Override season start YYYY-MM-DD for every season in --years.",
    )
    p.add_argument(
        "--end",
        default=None,
        help="Override season end YYYY-MM-DD for every season in --years.",
    )
    args = p.parse_args()
    years = args.years

    if (args.start is None) ^ (args.end is None):
        log.error("Pass both --start and --end, or neither.")
        raise SystemExit(1)

    load_env()
    url = db_url()
    try:
        engine = create_engine(url)
        check_db(engine)
    except SQLAlchemyError as e:
        log.error("Database connection failed: %s", e)
        raise SystemExit(1) from e

    for year in years:
        table = f"statcast_pitches_{year}"
        if args.start and args.end:
            start, end = args.start, args.end
        elif year in DEFAULT_SEASON_BOUNDS:
            start, end = DEFAULT_SEASON_BOUNDS[year]
        else:
            log.error("No default date range for year %s; pass --start and --end.", year)
            raise SystemExit(1)
        log.info("Loading year %s into %s (%s .. %s)", year, table, start, end)
        n = load_season_to_table(engine, table, start, end, log)
        log.info("Year %s: appended %d rows to %s", year, n, table)

    log.info("All requested seasons finished.")


if __name__ == "__main__":
    main()
