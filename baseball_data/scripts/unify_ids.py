#!/usr/bin/env python3
"""
Unify IDs: builds a player map from the Chadwick Register and uploads it to PostgreSQL.

Maps Statcast (MLBAM), Lahman (bbref), and Retrosheet IDs. Creates v_master_player_list.

PostgreSQL: set DATABASE_URL or PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE.
"""

from __future__ import annotations

import logging
import os
import sys
import unicodedata
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

try:
    from dotenv import load_dotenv
    from pybaseball import chadwick_register
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
except ImportError as e:
    print(f"Missing dependency: {e}. Install with: pip install pybaseball pandas sqlalchemy python-dotenv", file=sys.stderr)
    raise SystemExit(1) from e


_BASE = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BASE.parent
_BACKUP_PATH = _BASE / "data" / "player_map_backup.csv"

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


def _normalize_name(s: str) -> str:
    if pd.isna(s) or not isinstance(s, str):
        return ""
    s = str(s).strip()
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def fetch_and_transform() -> pd.DataFrame:
    log.info("Fetching Chadwick register via pybaseball.chadwick_register()")
    df = chadwick_register()

    log.info("Filtering players active from 2015 onwards (mlb_played_last >= 2015)")
    df = df.loc[df["mlb_played_last"] >= 2015].copy()

    cols = ["key_mlbam", "key_bbref", "key_retro", "name_first", "name_last"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Chadwick register missing columns: {missing}")

    df = df[cols].copy()

    log.info("Cleaning name_first and name_last (remove accents/special chars)")
    df["name_first"] = df["name_first"].apply(_normalize_name)
    df["name_last"] = df["name_last"].apply(_normalize_name)

    # Chadwick uses -1 for missing key_mlbam; use NaN for DB null
    df.loc[df["key_mlbam"] == -1, "key_mlbam"] = pd.NA
    df = df.dropna(subset=["key_bbref"])  # need bbref for join with people

    log.info("Players mapped: %d", len(df))
    return df


def save_backup(df: pd.DataFrame) -> None:
    _BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_BACKUP_PATH, index=False)
    log.info("Backup saved to %s", _BACKUP_PATH)


def upload_and_create_view(df: pd.DataFrame, engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS player_map CASCADE"))
        conn.execute(text("""
            CREATE TABLE player_map (
                key_mlbam   INTEGER,
                key_bbref   VARCHAR(10) NOT NULL,
                key_retro   VARCHAR(10),
                name_first  VARCHAR(255),
                name_last   VARCHAR(255)
            )
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_player_map_key_bbref ON player_map (key_bbref)"))

    df.to_sql(
        "player_map",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )
    log.info("Uploaded player_map to Postgres")

    with engine.begin() as conn:
        conn.execute(text("DROP VIEW IF EXISTS v_master_player_list"))
        conn.execute(text("""
            CREATE VIEW v_master_player_list AS
            SELECT
                pm.key_mlbam,
                pm.key_bbref,
                pm.key_retro,
                pm.name_first  AS name_first_map,
                pm.name_last   AS name_last_map,
                p.playerID,
                p.nameFirst,
                p.nameLast,
                p.birthYear,
                p.debut,
                p.finalGame,
                p.retroID,
                p.bbrefID
            FROM player_map pm
            LEFT JOIN people p ON p.playerID = pm.key_bbref
        """))
    log.info("Created view v_master_player_list")


def main() -> None:
    _load_env()

    df = fetch_and_transform()
    if df.empty:
        log.warning("No players to map (filtered empty). Exiting.")
        return

    save_backup(df)

    url = _db_url()
    try:
        engine = create_engine(url)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
    except SQLAlchemyError as e:
        log.error("Database connection failed: %s", e)
        raise SystemExit(1) from e

    try:
        upload_and_create_view(df, engine)
    except SQLAlchemyError as e:
        log.error("Upload or view creation failed: %s", e)
        raise SystemExit(1) from e

    log.info("Done. Total players mapped: %d", len(df))


if __name__ == "__main__":
    main()
