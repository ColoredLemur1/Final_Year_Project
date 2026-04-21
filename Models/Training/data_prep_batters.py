#!/usr/bin/env python3
"""
Batter simulator data prep: pitch-level data with pitch_result target.

Loads from clean_statcast_with_batter; maps events/description/type to pitch_result;
builds batter feature matrix (pitch, count, matchup, batter career, park);
train/val/test split (temporal forward-chaining by default, or random).

PostgreSQL: set PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE (or .env).
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder

from temporal_split import temporal_train_val_test

_BASE = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASE.parent.parent

# pitch_result labels (full Statcast mapping + reduced targets below)
PITCH_RESULT_CLASSES = [
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
    "foul_tip",
    "hit_by_pitch",
    "in_play_out",
    "in_play_1b",
    "in_play_2b",
    "in_play_3b",
    "in_play_hr",
]
RESULT_TO_IDX = {r: i for i, r in enumerate(PITCH_RESULT_CLASSES)}
IDX_TO_RESULT = {i: r for i, r in enumerate(PITCH_RESULT_CLASSES)}

# 4-class immediate target (no in_play / hbp; foul_tip -> foul)
IMMEDIATE_PITCH_RESULT_CLASSES = [
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
]
IMMEDIATE_RESULT_TO_IDX = {r: i for i, r in enumerate(IMMEDIATE_PITCH_RESULT_CLASSES)}

# 5-class immediate + hit (in_play hits -> hit; drops out/hbp)
IMMEDIATE_PLUS_HIT_CLASSES = [
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
    "hit",
]
IMMEDIATE_PLUS_HIT_RESULT_TO_IDX = {r: i for i, r in enumerate(IMMEDIATE_PLUS_HIT_CLASSES)}
PITCH_RESULT_TO_IMMEDIATE_PLUS_HIT = {
    "ball": "ball",
    "called_strike": "called_strike",
    "swinging_strike": "swinging_strike",
    "foul": "foul",
    "foul_tip": "foul",
    "in_play_1b": "hit",
    "in_play_2b": "hit",
    "in_play_3b": "hit",
    "in_play_hr": "hit",
}

# at-bat events -> 5 buckets (names in BUCKET_CLASS_NAMES)
OUTCOME_MAPPING = {
    "field_out": 0,
    "ground_out": 0,
    "fly_out": 0,
    "line_out": 0,
    "pop_out": 0,
    "grounded_into_dp": 0,
    "force_out": 0,
    "double_play": 0,
    "triple_play": 0,
    "field_error": 0,
    "sac_fly": 0,
    "sac_bunt": 0,
    "fielders_choice": 0,
    "fielders_choice_out": 0,
    "strikeout": 1,
    "strikeout_double_play": 1,
    "walk": 2,
    "hit_by_pitch": 2,
    "single": 3,
    "double": 4,
    "triple": 4,
    "home_run": 4,
}
BUCKET_CLASS_NAMES = ["in_play_out", "strikeout", "free_pass", "single", "extra_base"]
NUM_BUCKET_CLASSES = len(BUCKET_CLASS_NAMES)

# coarse labels for sim / viz (immediate vs in_play branch)
IMMEDIATE_EVENT_NAMES = ["ball", "strike", "foul", "hit_by_pitch", "in_play"]
IN_PLAY_OUTCOME_NAMES = ["out", "1b", "2b", "3b", "hr"]
PITCH_RESULT_TO_IMMEDIATE = {
    "ball": "ball",
    "called_strike": "strike",
    "swinging_strike": "strike",
    "foul": "foul",
    "foul_tip": "foul",
    "hit_by_pitch": "hit_by_pitch",
    "in_play_out": "in_play",
    "in_play_1b": "in_play",
    "in_play_2b": "in_play",
    "in_play_3b": "in_play",
    "in_play_hr": "in_play",
}
PITCH_RESULT_TO_IN_PLAY_OUTCOME = {
    "in_play_out": "out",
    "in_play_1b": "1b",
    "in_play_2b": "2b",
    "in_play_3b": "3b",
    "in_play_hr": "hr",
}

# (balls, strikes) set shared with pitcher sim for count-aware weighting
DECIDING_COUNTS = {(0, 2), (1, 2), (2, 2), (3, 1), (3, 2)}


def is_deciding_count(balls: int, strikes: int) -> bool:
    return (int(balls), int(strikes)) in DECIDING_COUNTS


# core + optional + derived feature column names
BATTER_FEATURE_COLS = [
    "pitch_type",
    "plate_x",
    "plate_z",
    "release_speed",
    "release_spin_rate",
    # Statcast movement (pfx_* inches) + spin; spin_axis_x_pthrows = spin_axis * sign(R=+1,L=-1).
    "hb_in",
    "ivb_in",
    "spin_axis",
    "spin_axis_x_pthrows",
    "balls",
    "strikes",
    "stand",
    "p_throws",
    "batter",
    "career_AB",
    "career_H",
    "career_HR",
    "career_BA",
    "career_OBP",
    "career_SLG",
    "home_team",
]
BATTER_OPTIONAL_FEATURE_COLS = [
    "inning",
    "outs_when_up",
    "is_pitcher_count",
    "is_batter_count",
]
BATTER_EXTRA_FEATURE_COLS = [
    "pitcher",
    "previous_pitch_type",
    "previous_was_strike",
    "in_zone",
    "is_fastball",
    "game_year",
]
BATTER_SEQUENCE_FEATURE_COLS = [
    "previous_2_pitch_type",
    "previous_2_was_strike",
    "previous_3_pitch_type",
    "previous_3_was_strike",
]
FASTBALL_TYPES = {"FF", "FT", "FC", "SI", "FS"}


def _load_env() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(_BASE / ".env")


def _db_url() -> str:
    host = os.getenv("PGHOST") or os.getenv("DB_HOST", "localhost")
    port = os.getenv("PGPORT") or os.getenv("DB_PORT", "5432")
    user = os.getenv("PGUSER") or os.getenv("DB_USER", "postgres")
    password = os.getenv("PGPASSWORD") or os.getenv("DB_PASS", "")
    dbname = os.getenv("PGDATABASE") or os.getenv("DB_NAME", "baseball")
    pw = quote_plus(password) if password else ""
    return f"postgresql://{user}:{pw}@{host}:{port}/{dbname}"


# Statcast description / type / PA events -> pitch_result
DESCRIPTION_TO_RESULT = {
    "ball": "ball",
    "blocked_ball": "ball",
    "intent_ball": "ball",
    "pitchout": "ball",
    "called_strike": "called_strike",
    "swinging_strike": "swinging_strike",
    "swinging_strike_blocked": "swinging_strike",
    "foul": "foul",
    "foul_tip": "foul_tip",
    "foul_bunt": "foul",
    "missed_bunt": "swinging_strike",
    "hit_by_pitch": "hit_by_pitch",
    "hit_into_play": None,
    "hit_into_play_no_out": None,
    "hit_into_play_score": None,
}
EVENTS_TO_IN_PLAY_RESULT = {
    "single": "in_play_1b",
    "double": "in_play_2b",
    "triple": "in_play_3b",
    "home_run": "in_play_hr",
    "field_out": "in_play_out",
    "ground_out": "in_play_out",
    "fly_out": "in_play_out",
    "line_out": "in_play_out",
    "pop_out": "in_play_out",
    "force_out": "in_play_out",
    "double_play": "in_play_out",
    "triple_play": "in_play_out",
    "sac_fly": "in_play_out",
    "sac_bunt": "in_play_out",
    "fielders_choice": "in_play_out",
    "fielders_choice_out": "in_play_out",
}


def _safe_str(x) -> str:
    """String for DB fields; empty if null so type X still parses."""
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def events_description_to_pitch_result(
    events: str | None,
    description: str | None,
    type_val: str | None,
) -> str | None:
    """Map Statcast row to a PITCH_RESULT_CLASSES label or None."""
    desc = _safe_str(description).lower()
    typ = _safe_str(type_val).upper()
    ev = _safe_str(events).lower()

    if typ == "X" or "hit_into_play" in desc or "hit_into_play_no_out" in desc or "hit_into_play_score" in desc:
        if ev and ev in EVENTS_TO_IN_PLAY_RESULT:
            return EVENTS_TO_IN_PLAY_RESULT[ev]
        if typ == "X":
            return "in_play_out"
        return None

    if desc in DESCRIPTION_TO_RESULT:
        res = DESCRIPTION_TO_RESULT[desc]
        if res is not None:
            return res
    if typ == "B":
        return "ball"
    if typ == "S":
        return "called_strike"
    return None


def event_to_pitch_result(row: pd.Series) -> str | None:
    return events_description_to_pitch_result(
        row.get("events"),
        row.get("description"),
        row.get("type"),
    )


def _p_throws_sign(s: pd.Series) -> np.ndarray:
    u = s.astype(str).str.upper().str.strip().str[0]
    return np.where(u == "R", 1.0, np.where(u == "L", -1.0, 0.0))


def add_batter_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    HB / IVB proxies from Statcast pfx_x / pfx_z (inches); spin_axis (deg) and handedness interaction.
    Missing values -> -1.0 for model sentinel consistency.
    """
    df = df.copy()
    px = pd.to_numeric(df.get("pfx_x"), errors="coerce")
    pz = pd.to_numeric(df.get("pfx_z"), errors="coerce")
    df["hb_in"] = px
    df["ivb_in"] = pz
    sa = pd.to_numeric(df.get("spin_axis"), errors="coerce")
    df["spin_axis"] = sa
    sig = _p_throws_sign(df["p_throws"])
    cross = sa * sig
    invalid = sa.isna() | (sig == 0)
    cross = cross.mask(invalid, np.nan)
    df["spin_axis_x_pthrows"] = cross
    for c in ("hb_in", "ivb_in", "spin_axis", "spin_axis_x_pthrows"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1.0)
    return df


def load_batter_simulator_data(engine) -> pd.DataFrame:
    """Load regular-season pitches from clean_statcast_with_batter; add lags, pitch_result, target_class."""
    q = """
    SELECT
        game_date, game_year, game_pk, game_type,
        batter, pitcher, events, description, "type",
        pitch_type, plate_x, plate_z, release_speed, release_spin_rate,
        balls, strikes, stand, p_throws,
        home_team, away_team, inning, outs_when_up,
        career_AB, career_H, career_HR, career_BA, career_OBP, career_SLG
    FROM clean_statcast_with_batter
    WHERE game_type = 'R'
    """
    q_extra = """
        SELECT
            game_date, game_year, game_pk, game_type,
            batter, pitcher, events, description, "type",
            pitch_type, plate_x, plate_z, release_speed, release_spin_rate,
            balls, strikes, stand, p_throws,
            home_team, away_team, inning, outs_when_up,
            zone, at_bat_number, pitch_number,
            career_AB, career_H, career_HR, career_BA, career_OBP, career_SLG
        FROM clean_statcast_with_batter
        WHERE game_type = 'R'
        """
    q_movement = """
        SELECT
            game_date, game_year, game_pk, game_type,
            batter, pitcher, events, description, "type",
            pitch_type, plate_x, plate_z, release_speed, release_spin_rate,
            pfx_x, pfx_z, spin_axis,
            balls, strikes, stand, p_throws,
            home_team, away_team, inning, outs_when_up,
            zone, at_bat_number, pitch_number,
            career_AB, career_H, career_HR, career_BA, career_OBP, career_SLG
        FROM clean_statcast_with_batter
        WHERE game_type = 'R'
        """
    df = None
    for q_try in (q_movement, q_extra, q):
        try:
            df = pd.read_sql(q_try, engine)
            break
        except Exception:
            continue
    if df is None or len(df.columns) == 0:
        df = pd.read_sql(q, engine)
    if "pfx_x" not in df.columns:
        df["pfx_x"] = np.nan
        df["pfx_z"] = np.nan
        df["spin_axis"] = np.nan
    if "zone" not in df.columns:
        df["zone"] = np.nan
        df["at_bat_number"] = 0
        df["pitch_number"] = 0

    for alt, canonical in [("Type", "type"), ("Events", "events"), ("Description", "description")]:
        if alt in df.columns and canonical not in df.columns:
            df[canonical] = df[alt]
        elif alt in df.columns and canonical in df.columns:
            df[canonical] = df[canonical].fillna(df[alt])

    career_cols = ["career_AB", "career_H", "career_HR", "career_BA", "career_OBP", "career_SLG"]
    for c in career_cols:
        if c in df.columns:
            continue
        low = c.lower()
        if low in df.columns:
            df[c] = df[low]
        else:
            df[c] = -1.0
    for c in career_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1.0)

    b = df["balls"].fillna(0).astype(int)
    s = df["strikes"].fillna(0).astype(int)
    df["is_pitcher_count"] = ((b == 0) & (s >= 2)) | ((b == 1) & (s == 2)).astype(int)
    df["is_batter_count"] = ((b >= 3) & (s <= 1)) | ((b == 2) & (s == 0)).astype(int)

    if "pitcher" not in df.columns:
        df["pitcher"] = -1

    if all(k in df.columns for k in ["game_pk", "at_bat_number", "pitch_number"]):
        df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
        g = df.groupby(["game_pk", "at_bat_number"])
        df["previous_pitch_type"] = g["pitch_type"].shift(1).fillna("__NA__").astype(str)
        df["previous_was_strike"] = (g["type"].shift(1) == "S").astype(int).fillna(0)
        df["previous_2_pitch_type"] = g["pitch_type"].shift(2).fillna("__NA__").astype(str)
        df["previous_2_was_strike"] = (g["type"].shift(2) == "S").astype(int).fillna(0)
        df["previous_3_pitch_type"] = g["pitch_type"].shift(3).fillna("__NA__").astype(str)
        df["previous_3_was_strike"] = (g["type"].shift(3) == "S").astype(int).fillna(0)
    else:
        df["previous_pitch_type"] = "__NA__"
        df["previous_was_strike"] = 0
        df["previous_2_pitch_type"] = "__NA__"
        df["previous_2_was_strike"] = 0
        df["previous_3_pitch_type"] = "__NA__"
        df["previous_3_was_strike"] = 0

    if "zone" in df.columns:
        z = pd.to_numeric(df["zone"], errors="coerce")
        df["in_zone"] = np.where(z.notna() & (z >= 1) & (z <= 9), 1, np.where(z.notna(), 0, -1))
    else:
        df["in_zone"] = -1

    df["is_fastball"] = df["pitch_type"].astype(str).str.upper().isin(FASTBALL_TYPES).astype(int)

    df = add_batter_movement_features(df)

    if "game_year" not in df.columns:
        df["game_year"] = -1
    else:
        df["game_year"] = pd.to_numeric(df["game_year"], errors="coerce").fillna(-1).astype(int)

    required = [
        "pitch_type", "plate_x", "plate_z", "release_speed", "release_spin_rate",
        "balls", "strikes", "stand", "p_throws", "batter",
        "career_AB", "career_H", "career_HR", "career_BA", "career_OBP", "career_SLG",
        "home_team",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df = df.dropna(subset=required)
    df = df.loc[df["pitch_type"] != "KN"].copy()

    df["pitch_result"] = df.apply(event_to_pitch_result, axis=1)
    if "type" in df.columns:
        n_type_x = (df["type"].apply(lambda x: _safe_str(x).upper() == "X")).sum()
    else:
        n_type_x = 0
    if "events" in df.columns:
        n_events_ok = int(df["events"].notna().sum())
    else:
        n_events_ok = 0
    pitch_result_counts = df["pitch_result"].value_counts()
    in_play_labels = [c for c in PITCH_RESULT_CLASSES if c.startswith("in_play")]
    n_in_play = pitch_result_counts.reindex(in_play_labels, fill_value=0).sum()
    if n_in_play == 0 and n_type_x > 0:
        import warnings
        warnings.warn(
            f"load_batter_simulator_data: type=='X' rows={n_type_x}, events non-null={n_events_ok}, but 0 in_play pitch_result. "
            "Check that 'events' (and description/type) are populated in DB for in-play pitches (re-run statcast.py and ensure views include them)."
        )
    df = df.loc[df["pitch_result"].notna()].copy()

    if "game_pk" in df.columns and "at_bat_number" in df.columns and "events" in df.columns:
        ab_events = (
            df.loc[df["events"].notna()]
            .groupby(["game_pk", "at_bat_number"], as_index=False)["events"]
            .last()
        )
        ab_events["events_clean"] = ab_events["events"].astype(str).str.strip().str.lower()
        ab_events["target_class"] = ab_events["events_clean"].map(OUTCOME_MAPPING)
        ab_events = ab_events[["game_pk", "at_bat_number", "target_class"]].dropna(subset=["target_class"])
        ab_events["target_class"] = ab_events["target_class"].astype(int)
        df = df.merge(
            ab_events,
            on=["game_pk", "at_bat_number"],
            how="left",
            suffixes=("", "_ab"),
        )
        df = df.drop(columns=[c for c in df.columns if c.endswith("_ab")], errors="ignore")
    else:
        df["target_class"] = np.nan

    df["immediate_event"] = df["pitch_result"].map(PITCH_RESULT_TO_IMMEDIATE)
    df["in_play_outcome"] = df["pitch_result"].map(PITCH_RESULT_TO_IN_PLAY_OUTCOME)

    return df


def _encode_batter_categoricals(df: pd.DataFrame, cat_encodings: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Label-encode categoricals; fit encodings if cat_encodings is None."""
    df = df.copy()
    feats = [c for c in BATTER_FEATURE_COLS if c in df.columns]
    opt = [c for c in BATTER_OPTIONAL_FEATURE_COLS if c in df.columns]
    extra = [c for c in BATTER_EXTRA_FEATURE_COLS + BATTER_SEQUENCE_FEATURE_COLS if c in df.columns]
    all_cols = feats + opt + extra
    cat_cols = ["pitch_type", "stand", "p_throws", "home_team", "previous_pitch_type", "previous_2_pitch_type", "previous_3_pitch_type"]
    if cat_encodings is None:
        cat_encodings = {}

    for col in all_cols:
        if col not in df.columns:
            continue
        if col in cat_cols:
            s = df[col].fillna("__NA__").astype(str).str.strip()
            s = s.replace("", "__NA__")
            if col not in cat_encodings:
                uniq = sorted(s.unique().tolist())
                cat_encodings[col] = uniq
            classes = cat_encodings[col]
            df[col] = s.map(lambda x: classes.index(x) if x in classes else (classes.index("__NA__") if "__NA__" in classes else 0))
        elif col in ("batter", "pitcher"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int64")
        elif col in ("previous_was_strike", "previous_2_was_strike", "previous_3_was_strike", "in_zone", "is_fastball", "game_year"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int64")
        elif col in ("career_AB", "career_H", "career_HR", "career_BA", "career_OBP", "career_SLG"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1.0)
        elif col in ("balls", "strikes", "inning", "outs_when_up", "is_pitcher_count", "is_batter_count"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        elif col in (
            "plate_x",
            "plate_z",
            "release_speed",
            "release_spin_rate",
            "hb_in",
            "ivb_in",
            "spin_axis",
            "spin_axis_x_pthrows",
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1.0)

    return df, cat_encodings


def prepare_batter_features(
    df: pd.DataFrame,
    cat_encodings: dict | None = None,
    feature_cols: list[str] | None = None,
    use_pitch_result_target: bool = False,
    use_immediate_outcomes_only: bool = False,
    use_immediate_plus_hit: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Build X, y, cat_encodings. y is pitch_result or target_class; optional immediate-only / +hit filters."""
    df = df.copy()
    if use_immediate_plus_hit and use_pitch_result_target:
        allowed = list(PITCH_RESULT_TO_IMMEDIATE_PLUS_HIT.keys())
        df = df.loc[df["pitch_result"].isin(allowed)].copy()
        df["pitch_result"] = df["pitch_result"].map(PITCH_RESULT_TO_IMMEDIATE_PLUS_HIT)
    elif use_immediate_outcomes_only and use_pitch_result_target:
        allowed = list(IMMEDIATE_PITCH_RESULT_CLASSES) + ["foul_tip"]
        df = df.loc[df["pitch_result"].isin(allowed)].copy()
        df["pitch_result"] = df["pitch_result"].replace({"foul_tip": "foul"})
    if not use_pitch_result_target and "target_class" in df.columns and df["target_class"].isna().any():
        df = df.loc[df["target_class"].notna()].copy()
    if feature_cols is None:
        feats = [c for c in BATTER_FEATURE_COLS if c in df.columns]
        opt = [c for c in BATTER_OPTIONAL_FEATURE_COLS if c in df.columns]
        extra = [c for c in BATTER_EXTRA_FEATURE_COLS + BATTER_SEQUENCE_FEATURE_COLS if c in df.columns]
        feature_cols = feats + opt + extra
    int_fill_cols = ("batter", "pitcher", "balls", "strikes", "inning", "outs_when_up", "is_pitcher_count", "is_batter_count", "previous_was_strike", "previous_2_was_strike", "previous_3_was_strike", "in_zone", "is_fastball", "game_year")
    na_pitch_type_cols = ("previous_pitch_type", "previous_2_pitch_type", "previous_3_pitch_type")
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c in df.columns:
            X[c] = df[c].values
        else:
            X[c] = "__NA__" if c in na_pitch_type_cols else (-1 if c in int_fill_cols else -1.0)
    X, cat_encodings = _encode_batter_categoricals(X, cat_encodings)
    if use_pitch_result_target:
        y = df["pitch_result"].copy()
    elif "target_class" in df.columns:
        y = df["target_class"].astype(int)
    else:
        y = df["pitch_result"].copy()
    return X, y, cat_encodings


def train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    random_state: int = 42,
    time_based: bool = True,
    date_col: str = "game_date",
    year_col: str = "game_year",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/val/test: temporal_train_val_test if time_based else stratified random split."""
    n = len(df)
    stratify_col = df["target_class"] if "target_class" in df.columns and df["target_class"].notna().all() else df.get("pitch_result")
    if time_based and (date_col in df.columns or year_col in df.columns):
        train_df, val_df, test_df, _meta = temporal_train_val_test(
            df, date_col=date_col, year_col=year_col, train_frac=train_frac, val_frac=val_frac
        )
        return train_df, val_df, test_df
    from sklearn.model_selection import train_test_split
    try:
        train_df, rest = train_test_split(
            df, train_size=train_frac, random_state=random_state, stratify=stratify_col
        )
    except ValueError:
        train_df, rest = train_test_split(df, train_size=train_frac, random_state=random_state)
    val_ratio = val_frac / (1 - train_frac)
    stratify_rest = rest["target_class"] if "target_class" in rest.columns and rest["target_class"].notna().all() else rest.get("pitch_result")
    try:
        val_df, test_df = train_test_split(
            rest, train_size=val_ratio, random_state=random_state, stratify=stratify_rest
        )
    except ValueError:
        val_df, test_df = train_test_split(rest, train_size=val_ratio, random_state=random_state)
    return train_df, val_df, test_df


def load_and_prepare_batter(
    engine,
    time_based: bool = True,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    random_state: int = 42,
    use_pitch_result_target: bool = False,
    use_immediate_outcomes_only: bool = False,
    use_immediate_plus_hit: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict, list]:
    """Load, split, prepare; returns X/y splits, encodings from train, feature_cols."""
    df = load_batter_simulator_data(engine)
    train_df, val_df, test_df = train_val_test_split(
        df, train_frac=train_frac, val_frac=val_frac, random_state=random_state, time_based=time_based
    )
    X_train, y_train, cat_encodings = prepare_batter_features(
        train_df, cat_encodings=None, use_pitch_result_target=use_pitch_result_target,
        use_immediate_outcomes_only=use_immediate_outcomes_only,
        use_immediate_plus_hit=use_immediate_plus_hit,
    )
    feature_cols = list(X_train.columns)
    X_val, y_val, _ = prepare_batter_features(
        val_df, cat_encodings=cat_encodings, feature_cols=feature_cols,
        use_pitch_result_target=use_pitch_result_target, use_immediate_outcomes_only=use_immediate_outcomes_only,
        use_immediate_plus_hit=use_immediate_plus_hit,
    )
    X_test, y_test, _ = prepare_batter_features(
        test_df, cat_encodings=cat_encodings, feature_cols=feature_cols,
        use_pitch_result_target=use_pitch_result_target, use_immediate_outcomes_only=use_immediate_outcomes_only,
        use_immediate_plus_hit=use_immediate_plus_hit,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, cat_encodings, feature_cols


def encode_batter_row(pitch: dict, context: dict, encoders: dict) -> pd.DataFrame:
    """One-row feature frame for inference (feats_batter + cat_encodings from JSON)."""
    feats = encoders.get("feats_batter", BATTER_FEATURE_COLS)
    cat_encodings = encoders.get("cat_encodings", {})
    out = {}
    for col in feats:
        if col == "pitch_type":
            v = str(pitch.get(col, "__NA__")).strip() or "__NA__"
            classes = cat_encodings.get(col, [])
            out[col] = classes.index(v) if v in classes else (classes.index("__NA__") if "__NA__" in classes else 0)
        elif col in ("stand", "p_throws", "home_team"):
            v = str(context.get(col, "__NA__")).strip() or "__NA__"
            classes = cat_encodings.get(col, [])
            out[col] = classes.index(v) if v in classes else (classes.index("__NA__") if "__NA__" in classes else 0)
        elif col in ("batter", "pitcher"):
            out[col] = int(context.get(col, -1) or -1)
        elif col in ("previous_pitch_type", "previous_2_pitch_type", "previous_3_pitch_type"):
            v = str(context.get(col, "__NA__")).strip() or "__NA__"
            classes = cat_encodings.get(col, [])
            out[col] = classes.index(v) if v in classes else (classes.index("__NA__") if "__NA__" in classes else 0)
        elif col in ("previous_was_strike", "previous_2_was_strike", "previous_3_was_strike", "in_zone", "is_fastball", "game_year"):
            out[col] = int(context.get(col, -1) if context.get(col) is not None else -1)
        elif col in ("plate_x", "plate_z", "release_speed", "release_spin_rate"):
            out[col] = float(pitch.get(col, -1) or -1)
        elif col in ("hb_in", "ivb_in"):
            raw = pitch.get(col)
            if raw is None or raw == "":
                alt = "pfx_x" if col == "hb_in" else "pfx_z"
                raw = pitch.get(alt, -1)
            try:
                out[col] = float(raw)
            except (TypeError, ValueError):
                out[col] = -1.0
            if out[col] != out[col]:  # NaN
                out[col] = -1.0
        elif col == "spin_axis":
            raw = pitch.get("spin_axis", pitch.get("spin_axis_deg", -1))
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = -1.0
            out[col] = v if v == v else -1.0
        elif col == "spin_axis_x_pthrows":
            if pitch.get("spin_axis_x_pthrows") is not None:
                try:
                    out[col] = float(pitch["spin_axis_x_pthrows"])
                except (TypeError, ValueError):
                    out[col] = -1.0
            else:
                try:
                    sa = float(pitch.get("spin_axis", -1))
                except (TypeError, ValueError):
                    sa = -1.0
                if sa < 0 or sa > 360 or sa != sa:
                    out[col] = -1.0
                else:
                    pt = str(context.get("p_throws", "R")).upper().strip()[:1]
                    sig = 1.0 if pt == "R" else (-1.0 if pt == "L" else 0.0)
                    out[col] = sa * sig if sig else -1.0
            if out[col] != out[col]:
                out[col] = -1.0
        elif col in ("balls", "strikes"):
            out[col] = int(context.get(col, 0) or 0)
        elif col in ("career_AB", "career_H", "career_HR", "career_BA", "career_OBP", "career_SLG"):
            out[col] = float(context.get(col, -1) or -1)
        else:
            out[col] = context.get(col, -1) if col in context else -1
    return pd.DataFrame([out])[feats]


def load_batter_encoders(path: str | Path) -> dict | None:
    """Load batter_encoders.json or None."""
    path = Path(path)
    if not path.exists():
        return None
    import json
    with open(path) as f:
        return json.load(f)
