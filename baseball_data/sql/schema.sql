-- Lahman core schema (Step 1)
-- Five base tables: people, teams, batting, pitching, fielding
-- PostgreSQL-friendly; adjust types if using MySQL/SQL Server.


-- people (player identity, bio, and cross-source IDs)

CREATE TABLE IF NOT EXISTS people (
    playerID    VARCHAR(10)  PRIMARY KEY,
    birthYear   SMALLINT,
    birthMonth  SMALLINT,
    birthDay    SMALLINT,
    birthCountry VARCHAR(255),
    birthState  VARCHAR(255),
    birthCity   VARCHAR(255),
    deathYear   SMALLINT,
    deathMonth  SMALLINT,
    deathDay    SMALLINT,
    deathCountry VARCHAR(255),
    deathState  VARCHAR(255),
    deathCity   VARCHAR(255),
    nameFirst   VARCHAR(255),
    nameLast    VARCHAR(255),
    nameGiven   VARCHAR(255),
    weight      SMALLINT,
    height      SMALLINT,
    bats        CHAR(1),
    throws      CHAR(1),
    debut       DATE,
    finalGame   DATE,
    retroID     VARCHAR(9),
    bbrefID     VARCHAR(9)
);


-- teams (year-level team stats, standings, park; teamIDretro for Retrosheet)

CREATE TABLE IF NOT EXISTS teams (
    yearID      SMALLINT     NOT NULL,
    lgID        CHAR(2)      NOT NULL,
    teamID      VARCHAR(3)   NOT NULL,
    franchID    VARCHAR(3),
    divID       CHAR(1),
    Rank        SMALLINT,
    G           SMALLINT,
    Ghome       SMALLINT,
    W           SMALLINT,
    L           SMALLINT,
    DivWin      CHAR(1),
    WCWin       CHAR(1),
    LgWin       CHAR(1),
    WSWin       CHAR(1),
    R           SMALLINT,
    AB          SMALLINT,
    H           SMALLINT,
    "X2B"       SMALLINT,
    "X3B"       SMALLINT,
    HR          SMALLINT,
    BB          SMALLINT,
    SO          SMALLINT,
    SB          SMALLINT,
    CS          SMALLINT,
    HBP         SMALLINT,
    SF          SMALLINT,
    RA          SMALLINT,
    ER          SMALLINT,
    ERA         DECIMAL(5, 2),
    CG          SMALLINT,
    SHO         SMALLINT,
    SV          SMALLINT,
    IPouts      SMALLINT,
    HA          SMALLINT,
    HRA         SMALLINT,
    BBA         SMALLINT,
    SOA         SMALLINT,
    E           SMALLINT,
    DP          SMALLINT,
    FP          DECIMAL(4, 3),
    name        VARCHAR(255),
    park        VARCHAR(255),
    attendance  INT,
    BPF         SMALLINT,
    PPF         SMALLINT,
    teamIDBR    VARCHAR(3),
    teamIDlahman45 VARCHAR(3),
    teamIDretro VARCHAR(5),
    PRIMARY KEY (yearID, lgID, teamID)
);


-- batting (player-year-stint batting stats)

CREATE TABLE IF NOT EXISTS batting (
    playerID    VARCHAR(10)  NOT NULL REFERENCES people (playerID),
    yearID      SMALLINT     NOT NULL,
    stint       SMALLINT     NOT NULL,
    teamID      VARCHAR(3)   NOT NULL,
    lgID        CHAR(2)      NOT NULL,
    G           SMALLINT,
    AB          SMALLINT,
    R           SMALLINT,
    H           SMALLINT,
    "X2B"       SMALLINT,
    "X3B"       SMALLINT,
    HR          SMALLINT,
    RBI         SMALLINT,
    SB          SMALLINT,
    CS          SMALLINT,
    BB          SMALLINT,
    SO          SMALLINT,
    IBB         SMALLINT,
    HBP         SMALLINT,
    SH          SMALLINT,
    SF          SMALLINT,
    GIDP        SMALLINT,
    PRIMARY KEY (playerID, yearID, stint),
    FOREIGN KEY (yearID, lgID, teamID) REFERENCES teams (yearID, lgID, teamID)
);


-- pitching (player-year-stint pitching stats)

CREATE TABLE IF NOT EXISTS pitching (
    playerID    VARCHAR(10)  NOT NULL REFERENCES people (playerID),
    yearID      SMALLINT     NOT NULL,
    stint       SMALLINT     NOT NULL,
    teamID      VARCHAR(3)   NOT NULL,
    lgID        CHAR(2)      NOT NULL,
    W           SMALLINT,
    L           SMALLINT,
    G           SMALLINT,
    GS          SMALLINT,
    CG          SMALLINT,
    SHO         SMALLINT,
    SV          SMALLINT,
    IPouts      SMALLINT,
    H           SMALLINT,
    ER          SMALLINT,
    HR          SMALLINT,
    BB          SMALLINT,
    SO          SMALLINT,
    BAOpp       DECIMAL(4, 3),
    ERA         DECIMAL(5, 2),
    IBB         SMALLINT,
    WP          SMALLINT,
    HBP         SMALLINT,
    BK          SMALLINT,
    BFP         SMALLINT,
    GF          SMALLINT,
    R           SMALLINT,
    SH          SMALLINT,
    SF          SMALLINT,
    GIDP        SMALLINT,
    PRIMARY KEY (playerID, yearID, stint),
    FOREIGN KEY (yearID, lgID, teamID) REFERENCES teams (yearID, lgID, teamID)
);

-- fielding (player-year-stint-position fielding stats)

CREATE TABLE IF NOT EXISTS fielding (
    playerID    VARCHAR(10)  NOT NULL REFERENCES people (playerID),
    yearID      SMALLINT     NOT NULL,
    stint       SMALLINT     NOT NULL,
    teamID      VARCHAR(3)   NOT NULL,
    lgID        CHAR(2)      NOT NULL,
    POS         VARCHAR(2)   NOT NULL,
    G           SMALLINT,
    GS          SMALLINT,
    PO          SMALLINT,
    A           SMALLINT,
    E           SMALLINT,
    DP          SMALLINT,
    PB          SMALLINT,
    WP          SMALLINT,
    SB          SMALLINT,
    CS          SMALLINT,
    ZR          DECIMAL(4, 1),
    PRIMARY KEY (playerID, yearID, stint, POS),
    FOREIGN KEY (yearID, lgID, teamID) REFERENCES teams (yearID, lgID, teamID)
);

-- Indexes for common filters and joins
CREATE INDEX IF NOT EXISTS idx_batting_player_year ON batting (playerID, yearID);
CREATE INDEX IF NOT EXISTS idx_batting_team_year ON batting (teamID, yearID);
CREATE INDEX IF NOT EXISTS idx_pitching_player_year ON pitching (playerID, yearID);
CREATE INDEX IF NOT EXISTS idx_pitching_team_year ON pitching (teamID, yearID);
CREATE INDEX IF NOT EXISTS idx_fielding_player_year ON fielding (playerID, yearID);
CREATE INDEX IF NOT EXISTS idx_fielding_team_year ON fielding (teamID, yearID);
CREATE INDEX IF NOT EXISTS idx_teams_year_lg ON teams (yearID, lgID);
CREATE INDEX IF NOT EXISTS idx_people_retro ON people (retroID);
CREATE INDEX IF NOT EXISTS idx_people_bbref ON people (bbrefID);
