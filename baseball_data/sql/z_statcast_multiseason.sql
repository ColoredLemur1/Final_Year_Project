-- Multi-season Statcast tables (2023–2024) + unified clean_statcast_with_batter.
-- Run AFTER: schema.sql, statcast.sql (2025 table), clean_views.sql so clean_statcast_pitches_2025 exists.
-- Does NOT drop statcast_pitches_2025 or existing 2025 data.

-- ---------------------------------------------------------------------------
-- Raw pitch tables (same column layout as statcast_pitches_2025)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS statcast_pitches_2023 (
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

CREATE TABLE IF NOT EXISTS statcast_pitches_2024 (
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

CREATE INDEX IF NOT EXISTS idx_statcast_2023_game ON statcast_pitches_2023 (game_date, game_pk);
CREATE INDEX IF NOT EXISTS idx_statcast_2023_batter ON statcast_pitches_2023 (batter);
CREATE INDEX IF NOT EXISTS idx_statcast_2023_pitcher ON statcast_pitches_2023 (pitcher);

CREATE INDEX IF NOT EXISTS idx_statcast_2024_game ON statcast_pitches_2024 (game_date, game_pk);
CREATE INDEX IF NOT EXISTS idx_statcast_2024_batter ON statcast_pitches_2024 (batter);
CREATE INDEX IF NOT EXISTS idx_statcast_2024_pitcher ON statcast_pitches_2024 (pitcher);

-- Ensure extra columns exist if tables were created from an older template
ALTER TABLE statcast_pitches_2023 ADD COLUMN IF NOT EXISTS plate_x REAL;
ALTER TABLE statcast_pitches_2023 ADD COLUMN IF NOT EXISTS plate_z REAL;
ALTER TABLE statcast_pitches_2023 ADD COLUMN IF NOT EXISTS release_spin_rate REAL;
ALTER TABLE statcast_pitches_2024 ADD COLUMN IF NOT EXISTS plate_x REAL;
ALTER TABLE statcast_pitches_2024 ADD COLUMN IF NOT EXISTS plate_z REAL;
ALTER TABLE statcast_pitches_2024 ADD COLUMN IF NOT EXISTS release_spin_rate REAL;

-- ---------------------------------------------------------------------------
-- Dedup clean pitch views (mirror clean_statcast_pitches_2025)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW clean_statcast_pitches_2023 AS
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
FROM statcast_pitches_2023
ORDER BY game_pk, at_bat_number, pitch_number, game_date;

CREATE OR REPLACE VIEW clean_statcast_pitches_2024 AS
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
FROM statcast_pitches_2024
ORDER BY game_pk, at_bat_number, pitch_number, game_date;

-- ---------------------------------------------------------------------------
-- Unified view: all regular-season pitches (2023–2025) + batter context
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW clean_statcast_with_batter AS
SELECT
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
    p.nameFirst   AS batter_name_first,
    p.nameLast    AS batter_name_last,
    p.birthYear   AS batter_birth_year,
    p.debut       AS batter_debut,
    p.finalGame   AS batter_final_game,
    cb.AB         AS career_AB,
    cb.H          AS career_H,
    cb.HR         AS career_HR,
    cb.BA         AS career_BA,
    cb.OBP        AS career_OBP,
    cb.SLG        AS career_SLG
FROM (
    SELECT * FROM clean_statcast_pitches_2023
    UNION ALL
    SELECT * FROM clean_statcast_pitches_2024
    UNION ALL
    SELECT * FROM clean_statcast_pitches_2025
) s
LEFT JOIN player_map pm ON s.batter = pm.key_mlbam
LEFT JOIN people p ON pm.key_bbref = p.playerID
LEFT JOIN vw_career_batting cb ON pm.key_bbref = cb.playerID;
