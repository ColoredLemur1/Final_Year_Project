#!/usr/bin/env python3
"""
Evaluation utilities for pitcher simulator: load models, generate pitches, load real data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import xgboost as xgb

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent.parent
_MODELS_DIR = _PROJECT_ROOT / "Models" / "saved_models"
_ENCODERS_PATH = _PROJECT_ROOT / "Models" / "Training" / "saved_models" / "pitcher_encoders.json"

sys.path.insert(0, str(_PROJECT_ROOT / "Models" / "Training"))


def _db_url() -> str:
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    dbname = os.getenv("PGDATABASE")
    pw = quote_plus(password) if password else ""
    return f"postgresql://{user}:{pw}@{host}:{port}/{dbname}"


def load_models_and_encoders() -> dict:
    """Load classifier, regressors, encoders. Same logic as web_app/app.py _load_models_and_encoders."""
    from data_prep import (
        load_pitcher_encoders,
        load_residual_stds,
        load_pitcher_repertoire,
        load_pitcher_plate_means,
        load_pitcher_pitch_type_rates,
        _DEFAULT_CAT_ENCODINGS,
        encode_context_row,
    )

    _saved = _PROJECT_ROOT / "Models" / "Training" / "saved_models"
    residual_stds = load_residual_stds(_saved / "residual_stds.json")
    pitcher_repertoire = load_pitcher_repertoire(_saved / "pitcher_repertoire.json")
    pitcher_plate_means = load_pitcher_plate_means(_saved / "pitcher_plate_means.json")
    pitcher_pitch_type_rates = load_pitcher_pitch_type_rates(_saved / "pitcher_pitch_type_rates.json")

    clf = xgb.XGBClassifier()
    clf.load_model(str(_MODELS_DIR / "pitcher_pitch_type.json"))
    booster = clf.get_booster()
    model_feature_names = booster.feature_names
    if model_feature_names is None:
        model_feature_names = [f"f{i}" for i in range(booster.num_features())]
    elif callable(model_feature_names):
        model_feature_names = model_feature_names()
    n_feats = booster.num_features()

    encoders = load_pitcher_encoders(_ENCODERS_PATH)
    if encoders is None:
        all_pts = set()
        for pts in pitcher_repertoire.values():
            all_pts.update(pts)
        for rates in pitcher_pitch_type_rates.values():
            all_pts.update(rates.keys() if isinstance(rates, dict) else [])
        type_to_idx = {t: i for i, t in enumerate(sorted(all_pts))} if all_pts else {"FF": 0}
        encoders = {
            "feats_s1": list(model_feature_names),
            "type_to_idx": type_to_idx,
            "cat_encodings": _DEFAULT_CAT_ENCODINGS,
        }
    feats_s1 = encoders["feats_s1"]
    feats_for_model = feats_s1[:n_feats] if len(feats_s1) >= n_feats else list(model_feature_names)
    feats_s2 = feats_for_model + ["pitch_type_code"]
    type_to_idx = encoders["type_to_idx"]
    idx_to_type = {v: k for k, v in type_to_idx.items()}

    reg_models = {}
    for t in ["plate_x", "plate_z", "release_speed", "release_spin_rate"]:
        m = xgb.XGBRegressor()
        m.load_model(str(_MODELS_DIR / f"pitcher_{t}.json"))
        reg_models[t] = m

    return {
        "clf": clf,
        "reg_models": reg_models,
        "encoders": encoders,
        "residual_stds": residual_stds,
        "pitcher_repertoire": pitcher_repertoire,
        "pitcher_plate_means": pitcher_plate_means,
        "pitcher_pitch_type_rates": pitcher_pitch_type_rates,
        "feats_s1": feats_s1,
        "feats_for_model": feats_for_model,
        "feats_s2": feats_s2,
        "type_to_idx": type_to_idx,
        "idx_to_type": idx_to_type,
        "encode_context_row": encode_context_row,
    }


def predict_pitch(models: dict, context: dict) -> dict:
    """Predict a single pitch from context. Returns dict with pitch_type, plate_x, plate_z, release_speed, release_spin_rate."""
    enc = models["encoders"]
    encode_fn = models["encode_context_row"]
    clf = models["clf"]
    reg = models["reg_models"]
    type_to_idx = models["type_to_idx"]
    idx_to_type = models["idx_to_type"]
    feats_s2 = models["feats_s2"]

    X_s1, _ = encode_fn(context, enc, pitch_type_code=None)
    feats_clf = models["feats_for_model"]
    proba = clf.predict_proba(X_s1[feats_clf])
    p = np.asarray(proba).ravel()
    # Restrict to pitcher's repertoire so we never generate types they don't throw
    pitcher_id = context.get("pitcher")
    repertoire = models.get("pitcher_repertoire", {}).get(int(pitcher_id) if pitcher_id is not None else None)
    if repertoire:
        for idx in range(len(p)):
            if idx_to_type.get(idx, "FF") not in repertoire:
                p[idx] = 0.0
        p_sum = p.sum()
        if p_sum > 0:
            p = p / p_sum
    # Blend with pitcher's empirical pitch-type rates
    pitcher_rates = models.get("pitcher_pitch_type_rates") or {}
    rates_dict = pitcher_rates.get(int(pitcher_id) if pitcher_id is not None else -1) if pitcher_id is not None else None
    if rates_dict:
        p_rates = np.array([rates_dict.get(idx_to_type.get(i, "FF"), 0.0) for i in range(len(p))], dtype=float)
        if p_rates.sum() > 0:
            p_rates = p_rates / p_rates.sum()
            p = 0.5 * p + 0.5 * p_rates
            p = p / p.sum()
    # Temperature > 1 flattens probabilities so we get more pitch-type diversity (not just FF/SL)
    temperature = 1.6
    p = np.power(np.clip(p, 1e-8, 1.0), 1.0 / temperature)
    p = p / p.sum()
    pred_idx = int(np.random.choice(len(p), p=p))
    pitch_type = idx_to_type.get(pred_idx, "FF")
    pt_code = type_to_idx.get(pitch_type, 0)
    _, X_s2 = encode_fn(context, enc, pitch_type_code=pt_code)
    feats_reg = models["feats_s2"]
    residual_stds = models.get("residual_stds") or {}
    default_stds = {"plate_x": 0.5, "plate_z": 0.5, "release_speed": 1.5, "release_spin_rate": 200}

    # Full residual variance so locations don't over-cluster; light blend toward pitcher mean if available
    pitcher_plate_means = models.get("pitcher_plate_means") or {}
    means = pitcher_plate_means.get(int(pitcher_id) if pitcher_id is not None else -1, {}).get(pitch_type) if pitcher_id is not None else None
    plate_x = float(reg["plate_x"].predict(X_s2[feats_reg])[0])
    plate_x += residual_stds.get("plate_x", {}).get(pitch_type, default_stds["plate_x"]) * np.random.randn()
    if means:
        plate_x = 0.85 * plate_x + 0.15 * means["plate_x"]

    plate_z = float(reg["plate_z"].predict(X_s2[feats_reg])[0])
    plate_z += residual_stds.get("plate_z", {}).get(pitch_type, default_stds["plate_z"]) * np.random.randn()
    if means:
        plate_z = 0.85 * plate_z + 0.15 * means["plate_z"]

    release_speed = float(reg["release_speed"].predict(X_s2[feats_reg])[0])
    release_speed += residual_stds.get("release_speed", {}).get(pitch_type, default_stds["release_speed"]) * np.random.randn()

    release_spin_rate = float(reg["release_spin_rate"].predict(X_s2[feats_reg])[0])
    release_spin_rate += residual_stds.get("release_spin_rate", {}).get(pitch_type, default_stds["release_spin_rate"]) * np.random.randn()

    return {
        "pitch_type": pitch_type,
        "plate_x": plate_x,
        "plate_z": plate_z,
        "release_speed": release_speed,
        "release_spin_rate": release_spin_rate,
    }


def generate_pitches(
    models: dict,
    pitcher_id: int,
    p_throws: str,
    career: dict,
    n_pitches: int = 300,
) -> list[dict]:
    """Generate n_pitches using the simulator. Returns list of pitch dicts."""
    pitches: list[dict] = []
    prev_pt = "__NA__"
    prev_speed = 0.0

    for pitch_num in range(1, n_pitches + 1):
        context = {
            "pitcher": pitcher_id,
            "p_throws": p_throws,
            "stand": "R",
            "balls": 0,
            "strikes": 0,
            "is_pitcher_count": 0,
            "is_batter_count": 0,
            "inning": 1,
            "inning_topbot": "Top",
            "outs_when_up": 0,
            "at_bat_number": 1,
            "pitch_number": pitch_num,
            "previous_pitch_type": prev_pt,
            "previous_release_speed": prev_speed,
            "pitcher_pitches_this_game": pitch_num - 1,
            "pitcher_pitches_this_inning": pitch_num - 1,
            "game_date": 0,
            "home_team": "NYY",
            "away_team": "BOS",
            "game_type": "R",
        }
        for k, v in career.items():
            context[k] = v if v is not None else -1

        pred = predict_pitch(models, context)
        pitches.append(pred)
        prev_pt = pred["pitch_type"]
        prev_speed = pred["release_speed"]

    return pitches


def load_real_pitcher_pitches_with_context(engine, pitcher_id: int) -> pd.DataFrame:
    """Load real pitches with full context (balls, strikes, inning, etc.) for evaluation.
    Adds previous_pitch_type, previous_release_speed, pitcher_pitches_this_game/inning.
    Use with generate_pitches_from_real_contexts so the model sees the same situations as real data."""
    from sqlalchemy import text

    q = text("""
    SELECT
        game_pk, pitcher, p_throws, stand, balls, strikes, inning, inning_topbot, outs_when_up,
        at_bat_number, pitch_number, game_date, home_team, away_team, game_type,
        pitch_type, plate_x, plate_z, release_speed, release_spin_rate
    FROM clean_statcast_with_batter
    WHERE pitcher = :pid AND game_type = 'R'
    ORDER BY game_pk, at_bat_number, pitch_number
    """)
    df = pd.read_sql(q, engine, params={"pid": pitcher_id})
    if len(df) == 0:
        return df
    # Previous pitch within same at-bat
    df = df.copy()
    df["previous_pitch_type"] = df.groupby(["game_pk", "at_bat_number"])["pitch_type"].shift(1).fillna("__NA__")
    df["previous_release_speed"] = df.groupby(["game_pk", "at_bat_number"])["release_speed"].shift(1).fillna(0.0)
    # Pitcher workload (pitches so far in game / in inning)
    df["pitcher_pitches_this_game"] = df.groupby("game_pk").cumcount()
    df["pitcher_pitches_this_inning"] = df.groupby(["game_pk", "inning"]).cumcount()
    # Encoder expects these (optional)
    df["is_pitcher_count"] = 0
    df["is_batter_count"] = 0
    return df


def generate_pitches_from_real_contexts(
    models: dict,
    df_real: pd.DataFrame,
    career: dict,
) -> list[dict]:
    """Generate one pitch per row of df_real using that row's context (balls, strikes, inning, etc.).
    This matches the variety of situations the model sees in the app and gives diverse pitch types."""
    if "balls" not in df_real.columns or "strikes" not in df_real.columns:
        # Fallback to fixed-context generation if no context columns
        return generate_pitches(
            models,
            int(df_real["pitcher"].iloc[0]) if "pitcher" in df_real.columns else 0,
            df_real["p_throws"].iloc[0] if "p_throws" in df_real.columns else "R",
            career,
            n_pitches=len(df_real),
        )
    pitches: list[dict] = []
    for _, row in df_real.iterrows():
        context = {
            "pitcher": int(row.get("pitcher", -1)),
            "p_throws": str(row.get("p_throws", "R")),
            "stand": str(row.get("stand", "R")),
            "balls": int(row.get("balls", 0)),
            "strikes": int(row.get("strikes", 0)),
            "is_pitcher_count": int(row.get("is_pitcher_count", 0)),
            "is_batter_count": int(row.get("is_batter_count", 0)),
            "inning": int(row.get("inning", 1)),
            "inning_topbot": str(row.get("inning_topbot", "Top")),
            "outs_when_up": int(row.get("outs_when_up", 0)),
            "at_bat_number": int(row.get("at_bat_number", 1)),
            "pitch_number": int(row.get("pitch_number", 1)),
            "previous_pitch_type": str(row.get("previous_pitch_type", "__NA__")),
            "previous_release_speed": float(row.get("previous_release_speed", 0.0)),
            "pitcher_pitches_this_game": int(row.get("pitcher_pitches_this_game", 0)),
            "pitcher_pitches_this_inning": int(row.get("pitcher_pitches_this_inning", 0)),
            "game_date": row.get("game_date", 0),
            "home_team": str(row.get("home_team", "NYY")),
            "away_team": str(row.get("away_team", "BOS")),
            "game_type": str(row.get("game_type", "R")),
        }
        for k, v in career.items():
            context[k] = v if v is not None else -1
        pred = predict_pitch(models, context)
        pitches.append(pred)
    return pitches


def load_pitcher_career(engine, pitcher_id: int) -> dict | None:
    """Load pitcher career stats for context. Same logic as web_app/app.py _load_pitcher_career."""
    from sqlalchemy import text

    q = text("""
    SELECT pm.key_mlbam AS pitcher,
           cp.IP AS pitcher_career_IP, cp.ERA AS pitcher_career_ERA,
           cp.SO AS pitcher_career_SO, cp.BB AS pitcher_career_BB,
           cp.H AS pitcher_career_H, cp.ER AS pitcher_career_ER,
           cp.HR AS pitcher_career_HR, cp.BFP AS pitcher_career_BFP,
           cp.IPouts AS pitcher_career_IPouts
    FROM player_map pm
    JOIN clean_vw_career_pitching cp ON cp.playerid = pm.key_bbref
    WHERE pm.key_mlbam = :pid
    """)
    df = pd.read_sql(q, engine, params={"pid": pitcher_id})
    if len(df) == 0:
        return None
    row = df.iloc[0]
    return {
        "pitcher_career_ip": row.get("pitcher_career_ip"),
        "pitcher_career_era": row.get("pitcher_career_era"),
        "pitcher_career_so": row.get("pitcher_career_so"),
        "pitcher_career_bb": row.get("pitcher_career_bb"),
        "pitcher_career_h": row.get("pitcher_career_h"),
        "pitcher_career_er": row.get("pitcher_career_er"),
        "pitcher_career_hr": row.get("pitcher_career_hr"),
        "pitcher_career_bfp": row.get("pitcher_career_bfp"),
        "pitcher_career_ipouts": row.get("pitcher_career_ipouts"),
    }

def get_engine():
    """Create SQLAlchemy engine for the database."""
    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(_EVAL_DIR / ".env")
    return create_engine(_db_url())