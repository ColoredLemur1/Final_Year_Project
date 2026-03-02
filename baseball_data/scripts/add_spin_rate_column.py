#!/usr/bin/env python3
"""
Add release_spin_rate column to statcast_pitches_2025 table and update views.

This script:
1. Adds the release_spin_rate column to statcast_pitches_2025 table
2. Updates the clean_statcast_pitches_2025 view to include the column
3. Updates the clean_statcast_with_batter view to include the column

PostgreSQL: set DATABASE_URL or PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

_BASE = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BASE.parent

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

    log.info("Adding release_spin_rate column to statcast_pitches_2025...")
    
    try:
        with engine.begin() as conn:
            # Add column to table
            conn.execute(text("""
                ALTER TABLE statcast_pitches_2025 
                ADD COLUMN IF NOT EXISTS release_spin_rate REAL
            """))
            log.info("✓ Added release_spin_rate column to statcast_pitches_2025")
            
            # Drop dependent view first (clean_statcast_with_batter depends on clean_statcast_pitches_2025)
            log.info("Dropping views to recreate with new column...")
            conn.execute(text("DROP VIEW IF EXISTS clean_statcast_with_batter CASCADE"))
            conn.execute(text("DROP VIEW IF EXISTS clean_statcast_pitches_2025 CASCADE"))
            
            # Recreate clean_statcast_pitches_2025 view
            log.info("Recreating clean_statcast_pitches_2025 view...")
            conn.execute(text("""
                CREATE VIEW clean_statcast_pitches_2025 AS
                SELECT DISTINCT ON (game_pk, at_bat_number, pitch_number)
                    game_date,
                    game_year,
                    game_pk,
                    game_type,
                    player_name,
                    batter,
                    pitcher,
                    events,
                    description,
                    release_speed,
                    launch_speed,
                    launch_angle,
                    pitch_type,
                    "type",
                    hit_distance_sc,
                    stand,
                    p_throws,
                    home_team,
                    away_team,
                    inning,
                    inning_topbot,
                    balls,
                    strikes,
                    outs_when_up,
                    zone,
                    bb_type,
                    at_bat_number,
                    pitch_number,
                    plate_x,
                    plate_z,
                    release_spin_rate
                FROM statcast_pitches_2025
                ORDER BY game_pk, at_bat_number, pitch_number, game_date
            """))
            log.info("✓ Recreated clean_statcast_pitches_2025 view")
            
            # Recreate clean_statcast_with_batter view
            log.info("Recreating clean_statcast_with_batter view...")
            conn.execute(text("""
                CREATE VIEW clean_statcast_with_batter AS
                SELECT
                    -- Statcast pitch data
                    s.game_date,
                    s.game_year,
                    s.game_pk,
                    s.game_type,
                    s.player_name,
                    s.batter,
                    s.pitcher,
                    s.events,
                    s.description,
                    s.release_speed,
                    s.launch_speed,
                    s.launch_angle,
                    s.pitch_type,
                    s."type",
                    s.hit_distance_sc,
                    s.stand,
                    s.p_throws,
                    s.home_team,
                    s.away_team,
                    s.inning,
                    s.inning_topbot,
                    s.balls,
                    s.strikes,
                    s.outs_when_up,
                    s.zone,
                    s.bb_type,
                    s.at_bat_number,
                    s.pitch_number,
                    s.plate_x,
                    s.plate_z,
                    s.release_spin_rate,
                    -- Batter bio (from people via player_map)
                    p.nameFirst   AS batter_name_first,
                    p.nameLast    AS batter_name_last,
                    p.birthYear   AS batter_birth_year,
                    p.debut       AS batter_debut,
                    p.finalGame   AS batter_final_game,
                    -- Batter career stats (from vw_career_batting)
                    cb.AB         AS career_AB,
                    cb.H          AS career_H,
                    cb.HR         AS career_HR,
                    cb.BA         AS career_BA,
                    cb.OBP        AS career_OBP,
                    cb.SLG        AS career_SLG
                FROM clean_statcast_pitches_2025 s
                LEFT JOIN player_map pm ON s.batter = pm.key_mlbam
                LEFT JOIN people p ON pm.key_bbref = p.playerID
                LEFT JOIN vw_career_batting cb ON pm.key_bbref = cb.playerID
            """))
            log.info("✓ Recreated clean_statcast_with_batter view")
            
    except SQLAlchemyError as e:
        log.error("Migration failed: %s", e)
        raise SystemExit(1) from e

    log.info("Migration complete! The release_spin_rate column is now available.")
    log.info("Note: Existing rows will have NULL for release_spin_rate until you reload data with statcast.py")


if __name__ == "__main__":
    main()

