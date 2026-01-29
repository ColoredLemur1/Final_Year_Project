-- player_map and v_master_player_list
-- Created by scripts/unify_ids.py from Chadwick Register (pybaseball).
-- Run unify_ids.py to populate; this file is for reference only.

DROP TABLE IF EXISTS player_map CASCADE;
CREATE TABLE player_map (
    key_mlbam   INTEGER,
    key_bbref   VARCHAR(10) NOT NULL,
    key_retro   VARCHAR(10),
    name_first  VARCHAR(255),
    name_last   VARCHAR(255)
);
CREATE INDEX IF NOT EXISTS idx_player_map_key_bbref ON player_map (key_bbref);

DROP VIEW IF EXISTS v_master_player_list;
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
LEFT JOIN people p ON p.playerID = pm.key_bbref;
