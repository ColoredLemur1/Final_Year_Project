-- Statcast pitch-level data (minimal columns for HR prediction and game/player context).
-- Populated by scripts/statcast.py. Run this before first statcast load.

DROP TABLE IF EXISTS statcast_pitches_2025;

CREATE TABLE statcast_pitches_2025 (
    game_date       DATE         NOT NULL,
    game_year       SMALLINT,
    game_pk         BIGINT       NOT NULL,
    game_type       VARCHAR(2),
    player_name     VARCHAR(255),
    batter          BIGINT,
    pitcher         BIGINT,
    events          VARCHAR(255),
    description     VARCHAR(255),
    release_speed   REAL,
    launch_speed    REAL,
    launch_angle    REAL,
    pitch_type      VARCHAR(10),
    "type"          CHAR(1),
    hit_distance_sc REAL,
    stand           CHAR(1),
    p_throws        CHAR(1),
    home_team       VARCHAR(3),
    away_team       VARCHAR(3),
    inning          SMALLINT,
    inning_topbot   VARCHAR(10),
    balls           SMALLINT,
    strikes         SMALLINT,
    outs_when_up    SMALLINT,
    zone            SMALLINT,
    bb_type         VARCHAR(50),
    at_bat_number   SMALLINT,
    pitch_number    SMALLINT,
    plate_x         REAL,
    plate_z         REAL,
    release_spin_rate REAL
);

CREATE INDEX IF NOT EXISTS idx_statcast_2025_game ON statcast_pitches_2025 (game_date, game_pk);
CREATE INDEX IF NOT EXISTS idx_statcast_2025_batter ON statcast_pitches_2025 (batter);
CREATE INDEX IF NOT EXISTS idx_statcast_2025_pitcher ON statcast_pitches_2025 (pitcher);

-- For existing tables: add plate_x, plate_z, release_spin_rate without dropping data
-- Run this if the table already exists and lacks these columns:
-- ALTER TABLE statcast_pitches_2025 ADD COLUMN IF NOT EXISTS plate_x REAL;
-- ALTER TABLE statcast_pitches_2025 ADD COLUMN IF NOT EXISTS plate_z REAL;
-- ALTER TABLE statcast_pitches_2025 ADD COLUMN IF NOT EXISTS release_spin_rate REAL;
