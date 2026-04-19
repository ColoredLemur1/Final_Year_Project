#!/usr/bin/env python3
"""
Data prep: pulls pitch-level data from PostgreSQL and encodes it for XGBoost.

Pitcher vs batter, pitch-by-pitch: adds pitcher workload (pitches in game/inning),
joins pitcher career stats, converts all text/categorical columns to numbers.
Target is configurable (e.g. is_batted_ball or outcome_class).

PostgreSQL: set DATABASE_URL or PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder

_BASE = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASE.parent.parent

_SEASON_START = "2025-03-27"

_DROP_FEATURE_COLS = [
    "player_name",
    "description",
    "batter_name_first",
    "batter_name_last",
    "events",
]

_CAT_COLS = [
    "pitch_type",
    "previous_pitch_type",
    "type",
    "stand",
    "p_throws",
    "inning_topbot",
    "home_team",
    "away_team",
    "game_type",
    "bb_type",
]


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


def load_pitches(engine) -> pd.DataFrame:
    """Load pitch-level data from clean_statcast_with_batter (regular season only)."""
    q = """
    SELECT
        game_date, game_year, game_pk, game_type, player_name,
        batter, pitcher, events, description,
        release_speed, launch_speed, launch_angle, pitch_type, type,
        hit_distance_sc, stand, p_throws,
        home_team, away_team, inning, inning_topbot,
        balls, strikes, outs_when_up, zone, bb_type,
        at_bat_number, pitch_number, plate_x, plate_z, release_spin_rate,
        batter_name_first, batter_name_last,
        career_AB, career_H, career_HR, career_BA, career_OBP, career_SLG
    FROM clean_statcast_with_batter
    WHERE game_type = 'R'
    """
    return pd.read_sql(q, engine)


def load_pitcher_career(engine) -> pd.DataFrame:
    """Load pitcher MLBAM -> career pitching stats (player_map + clean_vw_career_pitching)."""
    q = """
    SELECT pm.key_mlbam,
           cp.playerid AS pitcher_bbref,
           cp.IP AS pitcher_career_IP,
           cp.ERA AS pitcher_career_ERA,
           cp.SO AS pitcher_career_SO,
           cp.BB AS pitcher_career_BB,
           cp.H AS pitcher_career_H,
           cp.ER AS pitcher_career_ER,
           cp.HR AS pitcher_career_HR,
           cp.BFP AS pitcher_career_BFP,
           cp.IPouts AS pitcher_career_IPouts
    FROM player_map pm
    JOIN clean_vw_career_pitching cp ON cp.playerid = pm.key_bbref
    """
    return pd.read_sql(q, engine)


def add_pitcher_workload(df: pd.DataFrame) -> pd.DataFrame:
    """Add pitcher_pitches_this_game and pitcher_pitches_this_inning (cumulative per game)."""
    df = df.sort_values(["game_pk", "pitcher", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    # Pitches thrown so far in this game by this pitcher
    df["pitcher_pitches_this_game"] = df.groupby(["game_pk", "pitcher"]).cumcount()
    # Pitches thrown so far in this inning by this pitcher
    df["pitcher_pitches_this_inning"] = df.groupby(["game_pk", "pitcher", "inning"]).cumcount()
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features for better prediction.
    - velocity_differential: current minus previous pitch speed (within at-bat). Deception.
    - previous_pitch_type: pitch type of previous pitch in same at-bat (sequencing).
    - distance_from_heart: sqrt(plate_x^2 + (plate_z - 2.5)^2); middle-middle = smaller.
    - is_pitcher_count: 1 if 0-2 or 1-2 (pitcher ahead). is_batter_count: 1 if 3-0, 3-1, 2-0 (batter ahead).
    - batter_rolling_hit_rate_10g: fraction of last 10 games in which batter had a hit.
    - batter_ba_last_10g: rolling BA (hits / PA) over last 10 games; hot streaks.
    """
    df = df.copy()
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    # Velocity differential: within each at-bat, difference from previous pitch speed
    if "release_speed" in df.columns:
        prev_speed = df.groupby(["game_pk", "at_bat_number"])["release_speed"].shift(1)
        df["velocity_differential"] = (df["release_speed"] - prev_speed).fillna(0.0)

    # Previous pitch type (within at-bat) for sequencing/deception
    if "pitch_type" in df.columns:
        df["previous_pitch_type"] = (
            df.groupby(["game_pk", "at_bat_number"])["pitch_type"].shift(1).fillna("__NA__")
        )

    # Distance from heart of zone (plate_z 2.5 ft = typical middle); smaller = more hittable
    if "plate_x" in df.columns and "plate_z" in df.columns:
        df["distance_from_heart"] = (
            (df["plate_x"].fillna(0) ** 2 + (df["plate_z"].fillna(2.5) - 2.5) ** 2) ** 0.5
        )

    # Count context: pitcher ahead (0-2, 1-2) vs batter ahead (3-0, 3-1, 2-0)
    if "balls" in df.columns and "strikes" in df.columns:
        b, s = df["balls"].fillna(0), df["strikes"].fillna(0)
        df["is_pitcher_count"] = ((b == 0) & (s >= 2)) | ((b == 1) & (s == 2)).astype(int)
        df["is_batter_count"] = ((b >= 3) & (s <= 1)) | ((b == 2) & (s == 0)).astype(int)

    # Batter rolling hit rate (last 10 games): fraction of games with a hit
    hit_events = {"single", "double", "triple", "home_run"}
    if "batter" in df.columns and "game_pk" in df.columns and "events" in df.columns and "game_date" in df.columns:
        batter_game = df.groupby(["batter", "game_pk"]).agg(
            game_date=("game_date", "min"),
            hit_in_game=(
                "events",
                lambda s: 1 if s.dropna().astype(str).str.strip().str.lower().isin(hit_events).any() else 0,
            ),
        ).reset_index()
        batter_game = batter_game.sort_values(["batter", "game_date"]).reset_index(drop=True)
        batter_game["batter_rolling_hit_rate_10g"] = (
            batter_game.groupby("batter")["hit_in_game"]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        )
        # Rolling BA last 10 games: hits / PA over last 10 games (shift 1 = exclude current game)
        pa_events = df.dropna(subset=["events"]).groupby(["batter", "game_pk"]).agg(
            hits_in_game=("events", lambda s: s.astype(str).str.strip().str.lower().isin(hit_events).sum()),
            pa_in_game=("events", "count"),
        ).reset_index()
        batter_game = batter_game.merge(pa_events, on=["batter", "game_pk"], how="left")
        batter_game["hits_in_game"] = batter_game["hits_in_game"].fillna(0)
        batter_game["pa_in_game"] = batter_game["pa_in_game"].fillna(0)
        roll_hits = batter_game.groupby("batter")["hits_in_game"].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).sum()
        )
        roll_pa = batter_game.groupby("batter")["pa_in_game"].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).sum()
        )
        batter_game["batter_ba_last_10g"] = (roll_hits / roll_pa.replace(0, 1)).fillna(0.0)
        df = df.merge(
            batter_game[["batter", "game_pk", "batter_rolling_hit_rate_10g", "batter_ba_last_10g"]],
            on=["batter", "game_pk"],
            how="left",
        )
        df["batter_rolling_hit_rate_10g"] = df["batter_rolling_hit_rate_10g"].fillna(0.0)
        df["batter_ba_last_10g"] = df["batter_ba_last_10g"].fillna(0.0)

    return df


def add_previous_release_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous_release_speed (speed of last pitch in at-bat). Known at inference when simulating pitch-by-pitch."""
    df = df.copy()
    if "release_speed" in df.columns:
        df["previous_release_speed"] = (
            df.groupby(["game_pk", "at_bat_number"])["release_speed"].shift(1).fillna(0.0)
        )
    return df


_PITCHER_SIM_FEATURE_COLS = [
    "pitcher", "p_throws", "stand",
    "balls", "strikes", "is_pitcher_count", "is_batter_count",
    "inning", "inning_topbot", "outs_when_up", "at_bat_number", "pitch_number",
    "previous_pitch_type", "previous_release_speed",
    "pitcher_pitches_this_game", "pitcher_pitches_this_inning",
    "game_date", "home_team", "away_team", "game_type",
]
_PITCHER_SIM_CAREER_COLS = [
    "pitcher_career_ip",
    "pitcher_career_era",
    "pitcher_career_so",
    "pitcher_career_bb",
    "pitcher_career_h",
    "pitcher_career_er",
    "pitcher_career_hr",
    "pitcher_career_bfp",
    "pitcher_career_ipouts",
]
_PITCHER_SIM_TARGETS = ["pitch_type", "plate_x", "plate_z", "release_speed", "release_spin_rate"]

# Deciding counts (3-2, 0-2, 1-2, 2-2, 3-1) for count-aware weighting in pitcher training.
# Kept in sync with data_prep_batters.is_deciding_count / DECIDING_COUNTS.
DECIDING_COUNTS = {(0, 2), (1, 2), (2, 2), (3, 1), (3, 2)}


def is_deciding_count(balls: int, strikes: int) -> bool:
    """True if (balls, strikes) is a deciding count (pitcher ahead or full count)."""
    return (int(balls), int(strikes)) in DECIDING_COUNTS


def load_pitcher_simulator_data(engine) -> pd.DataFrame:
    """
    Load pitch-level data for pitcher simulator: context + targets.
    Adds pitcher workload, derived features, previous_release_speed.
    Drops rows with null targets; filters out KN (knuckleball).
    """
    pitches = load_pitches(engine)
    pitcher_career = load_pitcher_career(engine)
    pitches = pitches.merge(
        pitcher_career,
        left_on="pitcher",
        right_on="key_mlbam",
        how="left",
    )
    if "key_mlbam" in pitches.columns:
        pitches = pitches.drop(columns=["key_mlbam"], errors="ignore")
    if "pitcher_bbref" in pitches.columns:
        pitches = pitches.drop(columns=["pitcher_bbref"], errors="ignore")

    pitches = add_pitcher_workload(pitches)
    pitches = add_derived_features(pitches)
    pitches = add_previous_release_speed(pitches)

    for col in _PITCHER_SIM_TARGETS:
        if col not in pitches.columns:
            raise ValueError(f"Missing target column: {col}")
    pitches = pitches.dropna(subset=_PITCHER_SIM_TARGETS)
    pitches = pitches.loc[pitches["pitch_type"] != "KN"].copy()
    return pitches


def prepare_pitcher_features(
    df: pd.DataFrame,
    *,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict | None]:
    """
    Build X, y_dict, and optional encoder dict for the pitcher simulator (context-only features).

    encoders=None on train: fits categoricals on df; return the third value and pass it into
    val/test calls. Pass that dict as encoders on val/test; third return value is then None.
    """
    feature_cols = [c for c in _PITCHER_SIM_FEATURE_COLS if c in df.columns]
    career_cols = [c for c in _PITCHER_SIM_CAREER_COLS if c in df.columns]
    X_cols = feature_cols + career_cols
    X = df[X_cols].copy()

    X, enc_out = _encode_pitcher_categoricals(X, encoders=encoders)

    y_dict = {t: df[t].copy() for t in _PITCHER_SIM_TARGETS}
    return X, y_dict, enc_out


def _encode_pitcher_categoricals(df: pd.DataFrame, encoders: dict | None = None) -> tuple[pd.DataFrame, dict | None]:
    """Encode categorical columns; fit new encoders or apply a bundle from the training split."""
    df = df.copy()
    fit_mode = encoders is None
    cat_classes: dict[str, list[str]] = {} if fit_mode else encoders.get("cat_classes", {})

    if "game_date" in df.columns:
        start = pd.Timestamp(_SEASON_START)
        df["game_date"] = (pd.to_datetime(df["game_date"]) - start).dt.days

    cat_cols = ["previous_pitch_type", "stand", "p_throws", "inning_topbot", "home_team", "away_team", "game_type"]
    for col in cat_cols:
        if col not in df.columns:
            continue
        s = df[col].fillna("__NA__").astype(str).str.strip()
        if s.str.len().eq(0).any():
            s = s.replace("", "__NA__")
        if fit_mode:
            le = LabelEncoder()
            df[col] = le.fit_transform(s)
            cat_classes[col] = le.classes_.tolist()
        else:
            classes = cat_classes.get(col)
            if not classes:
                le = LabelEncoder()
                df[col] = le.fit_transform(s)
                continue
            class_to_i = {c: i for i, c in enumerate(classes)}
            unk = class_to_i.get("__NA__", 0)
            df[col] = s.map(lambda x, m=class_to_i, u=unk: m.get(x, u)).astype(np.int64)

    for col in ("pitcher",):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int64")

    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        df[col] = df[col].fillna(-1.0)
    for col in df.select_dtypes(include=["Int64", "Int32"]).columns:
        df[col] = df[col].fillna(-1).astype("int64")

    if fit_mode:
        return df, {"cat_classes": cat_classes}
    return df, None


def fit_and_save_pitcher_encoders(engine, out_path: str | Path) -> None:
    """
    Load pitcher simulator data, fit encoders on full dataset, save to JSON.
    Call after training so the web app can use identical encodings.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_pitcher_simulator_data(engine)
    type_to_idx = {t: i for i, t in enumerate(sorted(df["pitch_type"].unique()))}

    feature_cols = [c for c in _PITCHER_SIM_FEATURE_COLS if c in df.columns]
    career_cols = [c for c in _PITCHER_SIM_CAREER_COLS if c in df.columns]
    feats_s1 = feature_cols + career_cols

    X = df[feats_s1].copy()

    # game_date
    if "game_date" in X.columns:
        start = pd.Timestamp(_SEASON_START)
        X["game_date"] = (pd.to_datetime(X["game_date"]) - start).dt.days

    cat_cols = ["previous_pitch_type", "stand", "p_throws", "inning_topbot", "home_team", "away_team", "game_type"]
    cat_encodings: dict[str, list] = {}
    for col in cat_cols:
        if col not in X.columns:
            continue
        s = X[col].fillna("__NA__").astype(str).str.strip()
        if s.str.len().eq(0).any():
            s = s.replace("", "__NA__")
        le = LabelEncoder()
        X[col] = le.fit_transform(s)
        cat_encodings[col] = le.classes_.tolist()

    for col in ("pitcher",):
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-1).astype("int64")

    for col in X.select_dtypes(include=["float64", "float32"]).columns:
        X[col] = X[col].fillna(-1.0)
    for col in X.select_dtypes(include=["Int64", "Int32"]).columns:
        X[col] = X[col].fillna(-1).astype("int64")

    payload = {
        "type_to_idx": type_to_idx,
        "cat_encodings": cat_encodings,
        "feats_s1": feats_s1,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return None


#fallback category to number mappings when the encoder file is missing so the app still runs
_DEFAULT_CAT_ENCODINGS = {
    "p_throws": ["L", "R", "__NA__"],
    "stand": ["L", "R", "__NA__"],
    "inning_topbot": ["Bottom", "Top", "__NA__"],
    "game_type": ["R", "__NA__"],
    "previous_pitch_type": ["__NA__"],
    "home_team": ["__NA__"],
    "away_team": ["__NA__"],
}


def load_pitcher_encoders(path: str | Path) -> dict | None:
    """Load encoder config from JSON. Returns {type_to_idx, cat_encodings, feats_s1} or None if file missing."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_residual_stds(path: str | Path) -> dict:
    """Load residual stds per pitch type per target from JSON. Returns {target: {pitch_type: std}}."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_pitcher_repertoire(path: str | Path) -> dict:
    """Load pitcher -> list of pitch_type strings from JSON. For inference-time repertoire masking. Missing file -> {}."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {int(k): list(v) for k, v in raw.items()}


def load_pitcher_plate_means(path: str | Path) -> dict:
    """Load {pitcher_id: {pitch_type: {plate_x, plate_z}}} from JSON. Missing file -> {}."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_pitcher_pitch_type_rates(path: str | Path) -> dict:
    """Load {pitcher_id: {pitch_type: proportion}} from JSON. For inference-time proba blend. Missing file -> {}."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def encode_context_row(row: dict, encoders: dict, pitch_type_code: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode a context dict into model-ready DataFrames.
    row: dict with pitcher, p_throws, stand, balls, strikes, inning, etc.
    encoders: from load_pitcher_encoders.
    pitch_type_code: if provided, used for Stage 2 (feats_s2).
    Returns (X_s1, X_s2) - X_s2 is None if pitch_type_code is None.
    """
    feats_s1 = encoders["feats_s1"]
    cat_encodings = encoders.get("cat_encodings") or {}

    def _encode_cat(col: str, val: str | None) -> int:
        classes = cat_encodings.get(col) or _DEFAULT_CAT_ENCODINGS.get(col, ["__NA__"])
        v = str(val).strip() if val is not None and pd.notna(val) else "__NA__"
        if not v:
            v = "__NA__"
        return classes.index(v) if v in classes else (classes.index("__NA__") if "__NA__" in classes else 0)

    out = {}
    for col in feats_s1:
        if col == "game_date":
            val = row.get(col, 0)
            if isinstance(val, str):
                start = pd.Timestamp(_SEASON_START)
                val = (pd.to_datetime(val) - start).days
            elif hasattr(val, "year") or isinstance(val, pd.Timestamp):
                # datetime.date, datetime.datetime, or pd.Timestamp from SQL
                start = pd.Timestamp(_SEASON_START)
                val = (pd.to_datetime(val) - start).days
            out[col] = int(val) if pd.notna(val) else 0
        elif col in cat_encodings or col in _DEFAULT_CAT_ENCODINGS:
            out[col] = _encode_cat(col, row.get(col))
        elif col == "pitcher":
            out[col] = int(pd.to_numeric(row.get(col, -1), errors="coerce") or -1)
        elif col in ("balls", "strikes", "inning", "outs_when_up", "at_bat_number", "pitch_number",
                     "is_pitcher_count", "is_batter_count",
                     "pitcher_pitches_this_game", "pitcher_pitches_this_inning"):
            out[col] = int(pd.to_numeric(row.get(col, 0), errors="coerce") or 0)
        elif col == "previous_release_speed":
            out[col] = float(pd.to_numeric(row.get(col, 0), errors="coerce") or 0)
        elif col.startswith("pitcher_career_"):
            out[col] = float(pd.to_numeric(row.get(col, -1), errors="coerce") or -1)
        else:
            out[col] = row.get(col, -1)

    X_s1 = pd.DataFrame([{c: out.get(c, -1) for c in feats_s1}])
    if pitch_type_code is not None:
        X_s2 = X_s1.copy()
        X_s2["pitch_type_code"] = pitch_type_code
        return X_s1, X_s2
    return X_s1, None


def prepare_features_and_target(
    df: pd.DataFrame,
    target_type: str = "is_batted_ball",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature DataFrame and target Series.
    target_type: 'is_batted_ball' (binary: type == 'X'), 'is_hit' (1 if single/double/triple/HR), 'is_home_run', or 'outcome_class' (multiclass from events).
    For is_hit and is_home_run: filter to batted balls only (type == 'X') so we don't mix "missed hittable pitch" with "out."
    Drops rows where target is NaN.
    """
    if target_type == "is_batted_ball":
        target = (df["type"] == "X").astype(int)
    elif target_type == "is_hit":
        # Only batted balls: hit vs out (exclude swinging strike, foul, etc.)
        if "type" in df.columns:
            df = df.loc[df["type"] == "X"].copy()
        hit_events = {"single", "double", "triple", "home_run"}
        def _is_hit(ev):
            if pd.isna(ev):
                return None
            return 1 if str(ev).strip().lower() in hit_events else 0
        target = df["events"].apply(_is_hit)
        target = target.astype("Int64")
    elif target_type == "is_home_run":
        # Only batted balls
        if "type" in df.columns:
            df = df.loc[df["type"] == "X"].copy()
        def _is_hr(ev):
            if pd.isna(ev):
                return None
            return 1 if str(ev).strip().lower() == "home_run" else 0
        target = df["events"].apply(_is_hr)
        target = target.astype("Int64")
    elif target_type == "outcome_class":
        # Map events to a small set of labels
        event_map = {
            "strikeout": 0,
            "walk": 1,
            "single": 2,
            "double": 3,
            "triple": 4,
            "home_run": 5,
            "field_out": 6,
            "ground_out": 6,
            "fly_out": 6,
            "line_out": 6,
            "pop_out": 6,
            "force_out": 6,
            "double_play": 6,
            "sac_fly": 7,
            "sac_bunt": 7,
            "hit_by_pitch": 8,
            "caught_stealing_2b": 9,
            "other": 10,
        }
        def _event_to_class(ev):
            if pd.isna(ev) or ev is None:
                return None
            s = str(ev).strip().lower()
            return event_map.get(s, 10)  # 10 = other

        target = df["events"].apply(_event_to_class)
        target = target.astype("Int64")  # nullable int; NaN where events missing
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # Drop rows with missing target
    valid = target.notna()
    df = df.loc[valid].copy()
    target = target.loc[valid]

    # Drop target-related columns from features
    drop = [c for c in _DROP_FEATURE_COLS if c in df.columns]
    df = df.drop(columns=drop, errors="ignore")
    return df, target


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all text/categorical columns to numeric codes. Fill NaN with -1 for categoricals."""
    df = df.copy()
    # game_date -> days since season start
    if "game_date" in df.columns:
        start = pd.Timestamp(_SEASON_START)
        df["game_date"] = (pd.to_datetime(df["game_date"]) - start).dt.days

    # Label-encode categoricals
    for col in _CAT_COLS:
        if col not in df.columns:
            continue
        s = df[col].fillna("__NA__").astype(str).str.strip()
        if s.str.len().eq(0).any():
            s = s.replace("", "__NA__")
        le = LabelEncoder()
        df[col] = le.fit_transform(s)

    # batter / pitcher: keep as int (MLBAM IDs can be large but valid); fill NaN with -1
    for col in ("batter", "pitcher"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int64")

    # Numeric columns: fill NaN with -1 so XGBoost gets a constant
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        df[col] = df[col].fillna(-1.0)
    for col in df.select_dtypes(include=["Int64", "Int32"]).columns:
        df[col] = df[col].fillna(-1).astype("int64")

    return df


def load_and_prepare(
    engine,
    target_type: str = "is_batted_ball",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Pull data from Postgres, add pitcher workload and career stats, encode all to numbers, return (X, y).
    """
    pitches = load_pitches(engine)
    pitcher_career = load_pitcher_career(engine)
    # Left join pitcher career onto pitches
    pitches = pitches.merge(
        pitcher_career,
        left_on="pitcher",
        right_on="key_mlbam",
        how="left",
        suffixes=("", "_pitcher"),
    )
    if "key_mlbam" in pitches.columns:
        pitches = pitches.drop(columns=["key_mlbam"], errors="ignore")
    if "pitcher_bbref" in pitches.columns:
        pitches = pitches.drop(columns=["pitcher_bbref"], errors="ignore")

    pitches = add_pitcher_workload(pitches)
    pitches = add_derived_features(pitches)
    X, y = prepare_features_and_target(pitches, target_type=target_type)
    X = encode_categoricals(X)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pitch-level numeric data for XGBoost")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Save (X, y) as parquet: OUT_features.parquet, OUT_target.parquet (e.g. data/pitches_numeric)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="is_batted_ball",
        choices=["is_batted_ball", "is_hit", "is_home_run", "outcome_class"],
        help="Target variable type",
    )
    parser.add_argument(
        "--save-pitcher-encoders",
        action="store_true",
        help="Fit and save pitcher simulator encoders to saved_models/pitcher_encoders.json",
    )
    args = parser.parse_args()

    _load_env()
    engine = create_engine(_db_url())

    if args.save_pitcher_encoders:
        out_path = _BASE / "saved_models" / "pitcher_encoders.json"
        fit_and_save_pitcher_encoders(engine, out_path)
        print("Saved pitcher encoders to", out_path)
        return

    X, y = load_and_prepare(engine, target_type=args.target)
    print("Shape:", X.shape[0], "rows", X.shape[1], "features")
    print("Target:", args.target, "->", y.shape[0], "labels")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        base = out_path.parent / out_path.stem
        X.to_parquet(f"{base}_features.parquet", index=False)
        y.to_frame("target").to_parquet(f"{base}_target.parquet", index=False)
        print("Saved:", f"{base}_features.parquet", f"{base}_target.parquet")


if __name__ == "__main__":
    main()
