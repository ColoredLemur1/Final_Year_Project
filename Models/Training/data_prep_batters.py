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

# ---------------------------------------------------------------------------
# Phase 1.1: pitch_result classes (single source of truth)
# ---------------------------------------------------------------------------
# Game state update:
#   ball           -> balls += 1; if balls == 4 -> walk
#   called_strike  -> strikes += 1; if strikes == 3 -> strikeout
#   swinging_strike-> strikes += 1; if strikes == 3 -> strikeout
#   foul           -> strikes = min(2, strikes + 1)
#   hit_by_pitch   -> batter to first, advance forced runners
#   in_play_out    -> outs += 1; advance runners; if outs == 3 -> end half-inning
#   in_play_1b/2b/3b/hr -> update bases and runs; next batter
# ---------------------------------------------------------------------------
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

# Immediate outcomes only (4 classes): ball, strikes, foul (foul_tip merged into foul) — no in_play_*, no hit_by_pitch
IMMEDIATE_PITCH_RESULT_CLASSES = [
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
]
IMMEDIATE_RESULT_TO_IDX = {r: i for i, r in enumerate(IMMEDIATE_PITCH_RESULT_CLASSES)}

# 5-class immediate + hit: ball, called_strike, swinging_strike, foul, hit (successful hits only: 1b, 2b, 3b, hr)
# in_play_out and hit_by_pitch are excluded (rows dropped) when using this target
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
    # in_play_out and hit_by_pitch not included — exclude those rows when using this target
}

# ---------------------------------------------------------------------------
# Phase 1.2: 5-bucket outcome mapping (at-bat level, from events column)
# Tactical buckets for higher accuracy; map raw events to 5 classes.
# ---------------------------------------------------------------------------
OUTCOME_MAPPING = {
    # 0: In-Play Outs (Poor contact) — include all Statcast out types so at-bats are retained
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
    # 1: Strikeouts (Pitcher Dominance)
    "strikeout": 1,
    "strikeout_double_play": 1,
    # 2: Free Passes (Batter Patience)
    "walk": 2,
    "hit_by_pitch": 2,
    # 3: Singles (Base Hits)
    "single": 3,
    # 4: Extra Base Hits (Damage)
    "double": 4,
    "triple": 4,
    "home_run": 4,
}
BUCKET_CLASS_NAMES = ["in_play_out", "strikeout", "free_pass", "single", "extra_base"]
NUM_BUCKET_CLASSES = len(BUCKET_CLASS_NAMES)

# Two-stage evaluation: Stage 1 = immediate event (simulator); Stage 2 = in-play outcome (deeper viz)
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

# Deciding counts: 3-2, 0-2, 1-2, 2-2, 3-1 (used for count-aware weighting / zone nudge in pitcher sim)
DECIDING_COUNTS = {(0, 2), (1, 2), (2, 2), (3, 1), (3, 2)}


def is_deciding_count(balls: int, strikes: int) -> bool:
    """True if (balls, strikes) is a deciding count (pitcher ahead or full count)."""
    return (int(balls), int(strikes)) in DECIDING_COUNTS


# ---------------------------------------------------------------------------
# Phase 1.4: Batter feature columns (documented)
# ---------------------------------------------------------------------------
# Pitch: pitch_type, plate_x, plate_z, release_speed, release_spin_rate
# Count: balls, strikes; optional: is_pitcher_count, is_batter_count
# Matchup: stand, p_throws
# Batter: batter (id), batter career: career_AB, career_H, career_HR, career_BA, career_OBP, career_SLG
# Park: home_team (proxy when venue/park_id not in DB)
# ---------------------------------------------------------------------------
BATTER_FEATURE_COLS = [
    "pitch_type",
    "plate_x",
    "plate_z",
    "release_speed",
    "release_spin_rate",
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
# Optional: inning, outs_when_up, is_pitcher_count, is_batter_count
BATTER_OPTIONAL_FEATURE_COLS = [
    "inning",
    "outs_when_up",
    "is_pitcher_count",
    "is_batter_count",
]
# Derived / context: pitcher, previous pitch, zone, fastball, year (added in load when available)
BATTER_EXTRA_FEATURE_COLS = [
    "pitcher",
    "previous_pitch_type",
    "previous_was_strike",
    "in_zone",
    "is_fastball",
    "game_year",
]
# Sequence: multiple lags of (pitch_type, was_strike) for "previous balls" context (last 2–3 pitches)
BATTER_SEQUENCE_FEATURE_COLS = [
    "previous_2_pitch_type",
    "previous_2_was_strike",
    "previous_3_pitch_type",
    "previous_3_was_strike",
]
# Fastball pitch types for is_fastball flag
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


# ---------------------------------------------------------------------------
# Phase 1.3: Mapping raw event/description/type -> pitch_result
# ---------------------------------------------------------------------------
# Statcast: description = pitch-level; events = PA-level (on last pitch of AB);
# type = 'B' | 'S' | 'X' (ball, strike, in play).
# ---------------------------------------------------------------------------
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
    "hit_into_play": None,  # use events
    "hit_into_play_no_out": None,  # use events
    "hit_into_play_score": None,  # use events
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
    """Coerce value to string; treat None/NaN as empty so we don't lose in-play (type X)."""
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
    """
    Map raw Statcast event/description/type to pitch_result class.
    Returns one of PITCH_RESULT_CLASSES or None (drop row).
    Uses _safe_str so NaN from DB (NULL) is treated as empty and type 'X' is still recognized.
    """
    desc = _safe_str(description).lower()
    typ = _safe_str(type_val).upper()
    ev = _safe_str(events).lower()

    # In play: use events (populated on the pitch that ends the AB)
    if typ == "X" or "hit_into_play" in desc or "hit_into_play_no_out" in desc or "hit_into_play_score" in desc:
        if ev and ev in EVENTS_TO_IN_PLAY_RESULT:
            return EVENTS_TO_IN_PLAY_RESULT[ev]
        # fallback: generic out
        if typ == "X":
            return "in_play_out"
        return None

    # Non–in-play: use description
    if desc in DESCRIPTION_TO_RESULT:
        res = DESCRIPTION_TO_RESULT[desc]
        if res is not None:
            return res
    # Fallback by type
    if typ == "B":
        return "ball"
    if typ == "S":
        return "called_strike"  # generic strike if description missing
    return None


def event_to_pitch_result(row: pd.Series) -> str | None:
    """Convenience: map a row (events, description, type) to pitch_result."""
    return events_description_to_pitch_result(
        row.get("events"),
        row.get("description"),
        row.get("type"),
    )


# ---------------------------------------------------------------------------
# Phase 2: Loaders and prep
# ---------------------------------------------------------------------------
def load_batter_simulator_data(engine) -> pd.DataFrame:
    """
    Load pitch-level rows: pitch + context + events for mapping.
    Uses clean_statcast_with_batter; adds is_pitcher_count, is_batter_count.
    Drops rows with null required cols; excludes KN; applies pitch_result mapping.
    """
    # Base columns; zone, at_bat_number, pitch_number may be missing in older views.
    # Quote "type" (PostgreSQL reserved word) so the column is selected correctly.
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
    try:
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
        df = pd.read_sql(q_extra, engine)
    except Exception:
        df = pd.read_sql(q, engine)
        df["zone"] = np.nan
        df["at_bat_number"] = 0
        df["pitch_number"] = 0

    # Normalize events/description/type column names (some DB drivers return Type, Events, Description)
    for alt, canonical in [("Type", "type"), ("Events", "events"), ("Description", "description")]:
        if alt in df.columns and canonical not in df.columns:
            df[canonical] = df[alt]
        elif alt in df.columns and canonical in df.columns:
            df[canonical] = df[canonical].fillna(df[alt])

    # Normalize career column names (PostgreSQL often returns lowercase) and add if missing.
    # If your view has no career stats, run baseball_data/sql/views.sql and clean_views.sql.
    career_cols = ["career_AB", "career_H", "career_HR", "career_BA", "career_OBP", "career_SLG"]
    for c in career_cols:
        if c in df.columns:
            continue
        low = c.lower()
        if low in df.columns:
            df[c] = df[low]
        else:
            df[c] = -1.0
    # Fill NULL career stats (e.g. batters with no Lahman mapping) so we don't drop in-play rows in dropna(required)
    for c in career_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1.0)

    # Derived count flags
    b = df["balls"].fillna(0).astype(int)
    s = df["strikes"].fillna(0).astype(int)
    df["is_pitcher_count"] = ((b == 0) & (s >= 2)) | ((b == 1) & (s == 2)).astype(int)
    df["is_batter_count"] = ((b >= 3) & (s <= 1)) | ((b == 2) & (s == 0)).astype(int)

    # Pitcher: already in SELECT; ensure column exists for older views
    if "pitcher" not in df.columns:
        df["pitcher"] = -1

    # Previous pitch(es) in PA: lag 1 and optional lags 2–3 for sequence context
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

    # In zone (Statcast zone 1-9 is strike zone)
    if "zone" in df.columns:
        z = pd.to_numeric(df["zone"], errors="coerce")
        df["in_zone"] = np.where(z.notna() & (z >= 1) & (z <= 9), 1, np.where(z.notna(), 0, -1))
    else:
        df["in_zone"] = -1

    # Is fastball (FF, FT, FC, SI, FS)
    df["is_fastball"] = df["pitch_type"].astype(str).str.upper().isin(FASTBALL_TYPES).astype(int)

    # Game year (run environment); normalize if missing
    if "game_year" not in df.columns:
        df["game_year"] = -1
    else:
        df["game_year"] = pd.to_numeric(df["game_year"], errors="coerce").fillna(-1).astype(int)

    # Required pitch/context columns (career_* already ensured above)
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

    # Map to pitch_result (Phase 2.4)
    df["pitch_result"] = df.apply(event_to_pitch_result, axis=1)
    # Diagnostic: if no in-play outcomes, type/events/description may be missing or wrong in DB
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

    # 5-bucket target: at-bat outcome from events (populated on last pitch of AB)
    if "game_pk" in df.columns and "at_bat_number" in df.columns and "events" in df.columns:
        # Per at-bat: get events from the row where events is not null (last pitch of AB)
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
    """
    Encode categorical columns for batter features. If cat_encodings provided, use it; else fit.
    Returns (encoded_df, cat_encodings).
    """
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
        elif col in ("plate_x", "plate_z", "release_speed", "release_spin_rate"):
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
    """
    Build (X, y, cat_encodings) for batter simulator.
    X = batter feature matrix; y = pitch_result (or target_class); cat_encodings for inference.
    If feature_cols is provided (e.g. from train), val/test use same columns (missing filled -1).
    When use_pitch_result_target=True, y is always pitch_result (per-pitch outcome); otherwise
    y is 5-bucket target_class (only rows with mapped at-bat outcome are kept).
    When use_immediate_outcomes_only=True (and use_pitch_result_target=True), only rows with
    pitch_result in {ball, called_strike, swinging_strike, foul, foul_tip} are kept; foul_tip
    is mapped to foul so the target has 4 classes — no in_play_*, no hit_by_pitch.
    When use_immediate_plus_hit=True (and use_pitch_result_target=True), keep ball/called_strike/
    swinging_strike/foul/foul_tip and all in_play_*; map to 5 classes: ball, called_strike,
    swinging_strike, foul, hit. hit_by_pitch rows are excluded.
    """
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
    # Ensure all columns exist for consistent matrix
    int_fill_cols = ("batter", "pitcher", "balls", "strikes", "inning", "outs_when_up", "is_pitcher_count", "is_batter_count", "previous_was_strike", "previous_2_was_strike", "previous_3_was_strike", "in_zone", "is_fastball", "game_year")
    na_pitch_type_cols = ("previous_pitch_type", "previous_2_pitch_type", "previous_3_pitch_type")
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c in df.columns:
            X[c] = df[c].values
        else:
            X[c] = "__NA__" if c in na_pitch_type_cols else (-1 if c in int_fill_cols else -1.0)
    X, cat_encodings = _encode_batter_categoricals(X, cat_encodings)
    # Per-pitch target (pitch_result or immediate-only 5 classes) vs at-bat target (5-bucket target_class)
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
    """
    Split into train/val/test.

    time_based=True (default): chronological split via temporal_train_val_test (by season when
    multiple years exist, else by game_date order).

    time_based=False: random stratified split on target_class if present, else pitch_result.
    Returns (train_df, val_df, test_df).
    """
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
    """
    Load batter data, build pitch_result, split, prepare features.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test, cat_encodings, feature_cols).
    Default time_based=True uses chronological splitting (see train_val_test_split).
    cat_encodings and feature_cols come from train so val/test use the same codes.
    When use_pitch_result_target=True, y is per-pitch outcome (pitch_result); otherwise
    y is 5-bucket at-bat outcome when available.
    When use_immediate_outcomes_only=True, only ball/called_strike/swinging_strike/foul/foul_tip
    are kept (no in_play_*, no hit_by_pitch).
    When use_immediate_plus_hit=True, 5 classes: ball, called_strike, swinging_strike, foul, hit
    (all in_play_* mapped to hit; hit_by_pitch excluded).
    """
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
    """
    Encode one pitch + context for batter model inference.
    pitch: {pitch_type, plate_x, plate_z, release_speed, release_spin_rate}
    context: {balls, strikes, stand, p_throws, batter, career_AB, career_H, ...}, home_team
    encoders: from load_batter_encoders (feats_batter, cat_encodings).
    Returns one-row DataFrame with feats_batter columns.
    """
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
        elif col in ("balls", "strikes"):
            out[col] = int(context.get(col, 0) or 0)
        elif col in ("career_AB", "career_H", "career_HR", "career_BA", "career_OBP", "career_SLG"):
            out[col] = float(context.get(col, -1) or -1)
        else:
            out[col] = context.get(col, -1) if col in context else -1
    return pd.DataFrame([out])[feats]


def load_batter_encoders(path: str | Path) -> dict | None:
    """Load batter encoders from JSON (feats_batter, result_to_idx, cat_encodings). Returns None if missing."""
    path = Path(path)
    if not path.exists():
        return None
    import json
    with open(path) as f:
        return json.load(f)
