-- Lahman views 

-- vw_player_batting: batting + people, with derived stats (BA, OBP, SLG, OPS)
CREATE OR REPLACE VIEW vw_player_batting AS
SELECT
    p.playerID,
    p.nameFirst,
    p.nameLast,
    (p.nameFirst || ' ' || p.nameLast) AS nameFull,
    b.yearID,
    b.stint,
    b.teamID,
    b.lgID,
    b.G,
    b.AB,
    b.R,
    b.H,
    b."X2B" AS "2B",
    b."X3B" AS "3B",
    b.HR,
    b.RBI,
    b.SB,
    b.CS,
    b.BB,
    b.SO,
    b.IBB,
    b.HBP,
    b.SH,
    b.SF,
    b.GIDP,
    CASE WHEN b.AB > 0 THEN ROUND((b.H::NUMERIC / b.AB), 3) END AS BA,
    CASE WHEN (b.AB + b.BB + COALESCE(b.HBP, 0) + COALESCE(b.SF, 0)) > 0
         THEN ROUND(((b.H + b.BB + COALESCE(b.HBP, 0))::NUMERIC
             / (b.AB + b.BB + COALESCE(b.HBP, 0) + COALESCE(b.SF, 0))), 3)
         END AS OBP,
    CASE WHEN b.AB > 0
         THEN ROUND(((b.H + COALESCE(b."X2B", 0) + 2 * COALESCE(b."X3B", 0) + 3 * b.HR)::NUMERIC / b.AB), 3)
         END AS SLG,
    CASE WHEN b.AB > 0 AND (b.AB + b.BB + COALESCE(b.HBP, 0) + COALESCE(b.SF, 0)) > 0
         THEN ROUND(((b.H + COALESCE(b."X2B", 0) + 2 * COALESCE(b."X3B", 0) + 3 * b.HR)::NUMERIC / b.AB)
             + ((b.H + b.BB + COALESCE(b.HBP, 0))::NUMERIC
                 / (b.AB + b.BB + COALESCE(b.HBP, 0) + COALESCE(b.SF, 0))), 3)
         END AS OPS
FROM people p
JOIN batting b ON b.playerID = p.playerID;

-- vw_player_pitching: pitching + people, IP = IPouts / 3
CREATE OR REPLACE VIEW vw_player_pitching AS
SELECT
    p.playerID,
    p.nameFirst,
    p.nameLast,
    (p.nameFirst || ' ' || p.nameLast) AS nameFull,
    pit.yearID,
    pit.stint,
    pit.teamID,
    pit.lgID,
    pit.W,
    pit.L,
    pit.G,
    pit.GS,
    pit.CG,
    pit.SHO,
    pit.SV,
    pit.IPouts,
    ROUND((pit.IPouts::NUMERIC / 3), 1) AS IP,
    pit.H,
    pit.ER,
    pit.HR,
    pit.BB,
    pit.SO,
    pit.BAOpp,
    pit.ERA,
    pit.IBB,
    pit.WP,
    pit.HBP,
    pit.BK,
    pit.BFP,
    pit.GF,
    pit.R,
    pit.SH,
    pit.SF,
    pit.GIDP
FROM people p
JOIN pitching pit ON pit.playerID = p.playerID;

-- vw_player_fielding: fielding + people
CREATE OR REPLACE VIEW vw_player_fielding AS
SELECT
    p.playerID,
    p.nameFirst,
    p.nameLast,
    (p.nameFirst || ' ' || p.nameLast) AS nameFull,
    f.yearID,
    f.stint,
    f.teamID,
    f.lgID,
    f.POS,
    f.G,
    f.GS,
    f.PO,
    f.A,
    f.E,
    f.DP,
    f.PB,
    f.WP,
    f.SB,
    f.CS,
    f.ZR
FROM people p
JOIN fielding f ON f.playerID = p.playerID;

-- vw_team_seasons: teams with WinPct, R/G, RA/G, run differential
CREATE OR REPLACE VIEW vw_team_seasons AS
SELECT
    yearID,
    lgID,
    teamID,
    franchID,
    divID,
    Rank,
    G,
    Ghome,
    W,
    L,
    DivWin,
    WCWin,
    LgWin,
    WSWin,
    R,
    AB,
    H,
    "X2B" AS "2B",
    "X3B" AS "3B",
    HR,
    BB,
    SO,
    SB,
    CS,
    HBP,
    SF,
    RA,
    ER,
    ERA,
    CG,
    SHO,
    SV,
    IPouts,
    HA,
    HRA,
    BBA,
    SOA,
    E,
    DP,
    FP,
    name,
    park,
    attendance,
    BPF,
    PPF,
    teamIDBR,
    teamIDlahman45,
    teamIDretro,
    CASE WHEN G > 0 THEN ROUND((W::NUMERIC / G), 3) END AS WinPct,
    CASE WHEN G > 0 THEN ROUND((R::NUMERIC / G), 2) END AS Rpg,
    CASE WHEN G > 0 THEN ROUND((RA::NUMERIC / G), 2) END AS RApg,
    (R - RA) AS run_diff
FROM teams;

-- vw_player_season_batting: one row per player-year, summed across stints/teams
CREATE OR REPLACE VIEW vw_player_season_batting AS
SELECT
    b.playerID,
    b.yearID,
    MIN(b.teamID) AS teamID_first,
    MIN(b.lgID)   AS lgID,
    SUM(b.G)      AS G,
    SUM(b.AB)     AS AB,
    SUM(b.R)      AS R,
    SUM(b.H)      AS H,
    SUM(b."X2B")  AS "2B",
    SUM(b."X3B")  AS "3B",
    SUM(b.HR)     AS HR,
    SUM(b.RBI)    AS RBI,
    SUM(b.SB)     AS SB,
    SUM(b.CS)     AS CS,
    SUM(b.BB)     AS BB,
    SUM(b.SO)     AS SO,
    SUM(COALESCE(b.IBB, 0))   AS IBB,
    SUM(COALESCE(b.HBP, 0))   AS HBP,
    SUM(COALESCE(b.SH, 0))    AS SH,
    SUM(COALESCE(b.SF, 0))    AS SF,
    SUM(COALESCE(b.GIDP, 0))  AS GIDP
FROM batting b
GROUP BY b.playerID, b.yearID;

-- vw_player_season_pitching: one row per player-year, summed across stints/teams
CREATE OR REPLACE VIEW vw_player_season_pitching AS
SELECT
    pit.playerID,
    pit.yearID,
    MIN(pit.teamID) AS teamID_first,
    MIN(pit.lgID)   AS lgID,
    SUM(pit.W)      AS W,
    SUM(pit.L)      AS L,
    SUM(pit.G)      AS G,
    SUM(pit.GS)     AS GS,
    SUM(COALESCE(pit.CG, 0))   AS CG,
    SUM(COALESCE(pit.SHO, 0))  AS SHO,
    SUM(COALESCE(pit.SV, 0))   AS SV,
    SUM(pit.IPouts) AS IPouts,
    ROUND(SUM(pit.IPouts)::NUMERIC / 3, 1) AS IP,
    SUM(pit.H)      AS H,
    SUM(pit.ER)     AS ER,
    SUM(pit.HR)     AS HR,
    SUM(pit.BB)     AS BB,
    SUM(pit.SO)     AS SO,
    SUM(COALESCE(pit.IBB, 0))  AS IBB,
    SUM(COALESCE(pit.WP, 0))   AS WP,
    SUM(COALESCE(pit.HBP, 0))  AS HBP,
    SUM(COALESCE(pit.BK, 0))   AS BK,
    SUM(COALESCE(pit.BFP, 0))  AS BFP,
    SUM(COALESCE(pit.GF, 0))   AS GF,
    SUM(COALESCE(pit.R, 0))    AS R
FROM pitching pit
GROUP BY pit.playerID, pit.yearID;

-- vw_player_season_summary: player-year with batting + pitching (FULL OUTER join)
CREATE OR REPLACE VIEW vw_player_season_summary AS
SELECT
    COALESCE(sb.playerID, sp.playerID) AS playerID,
    COALESCE(sb.yearID, sp.yearID)     AS yearID,
    COALESCE(sb.teamID_first, sp.teamID_first) AS teamID_first,
    COALESCE(sb.lgID, sp.lgID)         AS lgID,
    sb.G     AS G_bat,
    sb.AB,
    sb.R     AS R_bat,
    sb.H,
    sb."2B",
    sb."3B",
    sb.HR    AS HR_bat,
    sb.RBI,
    sb.SB,
    sb.BB    AS BB_bat,
    sb.SO    AS SO_bat,
    sp.G     AS G_pit,
    sp.W,
    sp.L,
    sp.GS,
    sp.IP,
    sp.IPouts,
    sp.ER,
    sp.BB    AS BB_pit,
    sp.SO    AS SO_pit,
    sp.SV
FROM vw_player_season_batting sb
FULL OUTER JOIN vw_player_season_pitching sp
    ON sb.playerID = sp.playerID AND sb.yearID = sp.yearID;

-- vw_career_batting: career totals by player (people + aggregated batting)
CREATE OR REPLACE VIEW vw_career_batting AS
SELECT
    p.playerID,
    p.nameFirst,
    p.nameLast,
    (p.nameFirst || ' ' || p.nameLast) AS nameFull,
    p.debut,
    p.finalGame,
    SUM(b.G)    AS G,
    SUM(b.AB)   AS AB,
    SUM(b.R)    AS R,
    SUM(b.H)    AS H,
    SUM(b."X2B") AS "2B",
    SUM(b."X3B") AS "3B",
    SUM(b.HR)   AS HR,
    SUM(b.RBI)  AS RBI,
    SUM(b.SB)   AS SB,
    SUM(b.CS)   AS CS,
    SUM(b.BB)   AS BB,
    SUM(b.SO)   AS SO,
    SUM(COALESCE(b.IBB, 0))   AS IBB,
    SUM(COALESCE(b.HBP, 0))   AS HBP,
    SUM(COALESCE(b.SH, 0))    AS SH,
    SUM(COALESCE(b.SF, 0))    AS SF,
    SUM(COALESCE(b.GIDP, 0))  AS GIDP,
    CASE WHEN SUM(b.AB) > 0 THEN ROUND(SUM(b.H)::NUMERIC / SUM(b.AB), 3) END AS BA,
    CASE WHEN (SUM(b.AB) + SUM(b.BB) + SUM(COALESCE(b.HBP, 0)) + SUM(COALESCE(b.SF, 0))) > 0
         THEN ROUND((SUM(b.H) + SUM(b.BB) + SUM(COALESCE(b.HBP, 0)))::NUMERIC
             / (SUM(b.AB) + SUM(b.BB) + SUM(COALESCE(b.HBP, 0)) + SUM(COALESCE(b.SF, 0))), 3)
         END AS OBP,
    CASE WHEN SUM(b.AB) > 0
         THEN ROUND((SUM(b.H) + SUM(COALESCE(b."X2B", 0)) + 2 * SUM(COALESCE(b."X3B", 0)) + 3 * SUM(b.HR))::NUMERIC
             / SUM(b.AB), 3)
         END AS SLG
FROM people p
JOIN batting b ON b.playerID = p.playerID
GROUP BY p.playerID, p.nameFirst, p.nameLast, p.debut, p.finalGame;

-- vw_career_pitching: career totals by player (people + aggregated pitching)
CREATE OR REPLACE VIEW vw_career_pitching AS
SELECT
    p.playerID,
    p.nameFirst,
    p.nameLast,
    (p.nameFirst || ' ' || p.nameLast) AS nameFull,
    p.debut,
    p.finalGame,
    SUM(pit.W)      AS W,
    SUM(pit.L)      AS L,
    SUM(pit.G)      AS G,
    SUM(pit.GS)     AS GS,
    SUM(COALESCE(pit.CG, 0))   AS CG,
    SUM(COALESCE(pit.SHO, 0))  AS SHO,
    SUM(COALESCE(pit.SV, 0))   AS SV,
    SUM(pit.IPouts) AS IPouts,
    ROUND(SUM(pit.IPouts)::NUMERIC / 3, 1) AS IP,
    SUM(pit.H)      AS H,
    SUM(pit.ER)     AS ER,
    SUM(pit.HR)     AS HR,
    SUM(pit.BB)     AS BB,
    SUM(pit.SO)     AS SO,
    SUM(COALESCE(pit.IBB, 0))  AS IBB,
    SUM(COALESCE(pit.WP, 0))   AS WP,
    SUM(COALESCE(pit.HBP, 0))  AS HBP,
    SUM(COALESCE(pit.BK, 0))   AS BK,
    SUM(COALESCE(pit.BFP, 0))  AS BFP,
    SUM(COALESCE(pit.GF, 0))   AS GF,
    SUM(COALESCE(pit.R, 0))    AS R,
    CASE WHEN SUM(pit.IPouts) > 0
         THEN ROUND(9 * SUM(pit.ER)::NUMERIC / (SUM(pit.IPouts) / 3.0), 2)
         END AS ERA
FROM people p
JOIN pitching pit ON pit.playerID = p.playerID
GROUP BY p.playerID, p.nameFirst, p.nameLast, p.debut, p.finalGame;
