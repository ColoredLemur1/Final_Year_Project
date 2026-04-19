-- Clean views for Lahman, Statcast, and cross-source data.
-- Run after: schema.sql, views.sql, player_map.sql, statcast.sql
-- These views apply cleaning rules: implausible value filters, deduplication, and ID bridging.


--------------------------------------------------------------------------------
-- 1. LAHMAN-ONLY CLEAN VIEWS
--------------------------------------------------------------------------------

-- clean_vw_player_batting: batting + people with implausible value filters
CREATE OR REPLACE VIEW clean_vw_player_batting AS
SELECT *
FROM vw_player_batting
WHERE (AB >= 0 OR AB IS NULL)
  AND (H >= 0 OR H IS NULL)
  AND (H <= AB OR AB IS NULL OR H IS NULL)
  AND (HR >= 0 OR HR IS NULL)
  AND (BA <= 1 OR BA IS NULL)
  AND (OBP <= 1 OR OBP IS NULL);


-- clean_vw_player_pitching: pitching + people with implausible value filters
CREATE OR REPLACE VIEW clean_vw_player_pitching AS
SELECT *
FROM vw_player_pitching
WHERE (IPouts >= 0 OR IPouts IS NULL)
  AND (W >= 0 OR W IS NULL)
  AND (L >= 0 OR L IS NULL)
  AND (ER >= 0 OR ER IS NULL)
  AND (ERA <= 100 OR ERA IS NULL);


-- clean_vw_career_batting: career batting totals with implausible value filters
CREATE OR REPLACE VIEW clean_vw_career_batting AS
SELECT *
FROM vw_career_batting
WHERE (AB >= 0 OR AB IS NULL)
  AND (H >= 0 OR H IS NULL)
  AND (H <= AB OR AB IS NULL OR H IS NULL)
  AND (HR >= 0 OR HR IS NULL)
  AND (BA <= 1 OR BA IS NULL)
  AND (OBP <= 1 OR OBP IS NULL);


-- clean_vw_career_pitching: career pitching totals with implausible value filters
CREATE OR REPLACE VIEW clean_vw_career_pitching AS
SELECT *
FROM vw_career_pitching
WHERE (IPouts >= 0 OR IPouts IS NULL)
  AND (W >= 0 OR W IS NULL)
  AND (L >= 0 OR L IS NULL)
  AND (ER >= 0 OR ER IS NULL)
  AND (ERA <= 100 OR ERA IS NULL);


--------------------------------------------------------------------------------
-- 2. STATCAST-ONLY CLEAN VIEW
--------------------------------------------------------------------------------

-- clean_statcast_pitches_2025: deduplicated pitches (one row per game_pk, at_bat_number, pitch_number)
CREATE OR REPLACE VIEW clean_statcast_pitches_2025 AS
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
ORDER BY game_pk, at_bat_number, pitch_number, game_date;


--------------------------------------------------------------------------------
-- 3. CROSS-SOURCE VIEW
--------------------------------------------------------------------------------

-- clean_statcast_with_batter: Statcast pitches joined to batter bio + career batting stats
-- Uses LEFT JOINs so pitches with unmapped batters are retained (with NULL Lahman fields).
-- Required for batter pitch-result / in-play labels: events (PA outcome on last pitch),
-- description (pitch-level result text), type (B/S/X). Populated by statcast.py loader.
-- When loading 2023–2024, run z_statcast_multiseason.sql after this file to replace this view
-- with a UNION across clean_statcast_pitches_2023/2024/2025.
CREATE OR REPLACE VIEW clean_statcast_with_batter AS
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
LEFT JOIN vw_career_batting cb ON pm.key_bbref = cb.playerID;
