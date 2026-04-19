#!/usr/bin/env python3
"""
Statcast loader: fetches pitch-level data for the 2025 season and appends to the database.

Pulls data in 6-day chunks, keeps minimal columns for HR prediction and batter outcome
modeling. Run statcast.sql once before first use.

PostgreSQL: set DATABASE_URL or PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE.
"""

from __future__ import annotations

import logging
import os
import warnings

os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore", message=".*errors='ignore'.*", module="pybaseball.datahelpers.postprocessing")

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from statcast_common import check_db, db_url, load_env, load_season_to_table

_SEASON_START = "2025-03-27"
_SEASON_END = "2025-09-29"
_TABLE = "statcast_pitches_2025"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    load_env()
    url = db_url()

    try:
        engine = create_engine(url)
        check_db(engine)
    except SQLAlchemyError as e:
        log.error("Database connection failed: %s", e)
        raise SystemExit(1) from e

    total = load_season_to_table(engine, _TABLE, _SEASON_START, _SEASON_END, log)
    log.info("Done. Total rows appended to %s: %d", _TABLE, total)


if __name__ == "__main__":
    main()
