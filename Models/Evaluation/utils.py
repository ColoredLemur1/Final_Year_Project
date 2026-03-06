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

# Deciding-count location nudge: in 3-2, 0-2, 1-2, 2-2, 3-1, blend predicted location slightly toward zone center.
# Small blend preserves spread (no collapse to middle). Set DECIDING_COUNT_ZONE_BLEND to 0 to disable.
DECIDING_COUNT_ZONE_BLEND = 0.15
ZONE_CENTER_X, ZONE_CENTER_Z = 0.0, 2.5


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
    from data_prep_batters import is_deciding_count

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

    # Deciding counts: nudge location toward zone center (preserve spread; blend is small so we don't collapse to middle)
    if DECIDING_COUNT_ZONE_BLEND > 0:
        balls_ctx = int(context.get("balls", 0))
        strikes_ctx = int(context.get("strikes", 0))
        if is_deciding_count(balls_ctx, strikes_ctx):
            plate_x = (1 - DECIDING_COUNT_ZONE_BLEND) * plate_x + DECIDING_COUNT_ZONE_BLEND * ZONE_CENTER_X
            plate_z = (1 - DECIDING_COUNT_ZONE_BLEND) * plate_z + DECIDING_COUNT_ZONE_BLEND * ZONE_CENTER_Z

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


# ---------------------------------------------------------------------------
# Batter model and game sim (Phase 5)
# ---------------------------------------------------------------------------
def load_batter_models() -> dict:
    """Load batter pitch_result classifier and encoders. Prefer calibrated model (same as evaluation)."""
    sys.path.insert(0, str(_PROJECT_ROOT / "Models" / "Training"))
    from data_prep_batters import load_batter_encoders, encode_batter_row

    import joblib as _joblib

    _saved = _PROJECT_ROOT / "Models" / "Training" / "saved_models"
    encoders = load_batter_encoders(_saved / "batter_encoders.json")
    if encoders is None:
        encoders = load_batter_encoders(_MODELS_DIR / "batter_encoders.json")
    if encoders is None:
        raise FileNotFoundError("batter_encoders.json not found")

    # Prefer calibrated model (same artifact as batter_evaluation) for accurate real-world depiction
    cal_path = _saved / "batter_pitch_result_calibrated.joblib"
    if not cal_path.exists():
        cal_path = _MODELS_DIR / "batter_pitch_result_calibrated.joblib"
    if cal_path.exists():
        clf = _joblib.load(cal_path)
    else:
        clf = xgb.XGBClassifier()
        p = _saved / "batter_pitch_result.json"
        if not p.exists():
            p = _MODELS_DIR / "batter_pitch_result.json"
        clf.load_model(str(p))

    out = {
        "clf": clf,
        "encoders": encoders,
        "feats_batter": encoders["feats_batter"],
        "result_to_idx": encoders["result_to_idx"],
        "idx_to_result": {int(k): v for k, v in encoders["idx_to_result"].items()},
        "encode_batter_row": encode_batter_row,
    }
    # Optional: BIP launch regressor + final outcome model (when batter model predicts "hit")
    import json
    import joblib
    _try = [_saved, _MODELS_DIR]
    for _d in _try:
        bip_path = _d / "bip_launch_regressor.joblib"
        bip_meta_path = _d / "bip_launch_regressor_metadata.json"
        fo_path = _d / "final_outcome_model.joblib"
        fo_meta_path = _d / "final_outcome_metadata.json"
        if bip_path.exists() and bip_meta_path.exists():
            out["bip_regressor"] = joblib.load(bip_path)
            with open(bip_meta_path) as f:
                out["bip_metadata"] = json.load(f)
            break
    for _d in _try:
        fo_meta_path = _d / "final_outcome_metadata.json"
        if not fo_meta_path.exists():
            continue
        with open(fo_meta_path) as f:
            out["final_outcome_metadata"] = json.load(f)
        fo_meta = out["final_outcome_metadata"]
        if fo_meta.get("two_stage"):
            s1_path = _d / "final_outcome_stage1_model.joblib"
            s2_path = _d / "final_outcome_stage2_model.joblib"
            if s1_path.exists() and s2_path.exists():
                out["final_outcome_stage1_model"] = joblib.load(s1_path)
                out["final_outcome_stage2_model"] = joblib.load(s2_path)
                out["final_outcome_model"] = None
                break
        else:
            fo_path = _d / "final_outcome_model.joblib"
            if fo_path.exists():
                out["final_outcome_model"] = joblib.load(fo_path)
                break
    return out


def _resolve_hit_to_in_play(batter_models: dict, pitch: dict, context: dict) -> str:
    """
    When batter model predicted 'hit', resolve to concrete in-play result via BIP -> physics -> final outcome model.
    Returns in_play_out (immediate out), in_play_1b, in_play_2b, in_play_3b, or in_play_hr. HR rule applied only when model predicted a hit.
    """
    bip = batter_models.get("bip_regressor")
    bip_meta = batter_models.get("bip_metadata")
    fo_model = batter_models.get("final_outcome_model")
    fo_meta = batter_models.get("final_outcome_metadata")
    stage1 = batter_models.get("final_outcome_stage1_model")
    stage2 = batter_models.get("final_outcome_stage2_model")
    two_stage = fo_meta and fo_meta.get("two_stage") and stage1 is not None and stage2 is not None
    has_single = fo_meta and fo_model is not None
    if not bip or not bip_meta or not fo_meta or (not two_stage and not has_single):
        # Fallback: random in_play outcome if models not loaded
        in_play = ["in_play_out", "in_play_1b", "in_play_2b", "in_play_3b", "in_play_hr"]
        return str(np.random.choice(in_play))
    pitch_mph = float(pitch.get("release_speed") or 0)
    plate_x = float(pitch.get("plate_x") or 0)
    plate_z = float(pitch.get("plate_z") or 0)
    slg = context.get("career_SLG") or context.get("batter_power")
    batter_power = 0.4 if (slg is None or float(slg) < 0) else float(slg)
    bip_feats = bip_meta["feature_cols"]
    X_bip = pd.DataFrame([{
        "pitch_mph": pitch_mph,
        "plate_x": plate_x,
        "plate_z": plate_z,
        "batter_power": batter_power,
    }])
    X_bip = X_bip.reindex(columns=bip_feats, fill_value=0)
    pred_launch = bip.predict(X_bip[bip_feats])[0]
    launch_speed = float(pred_launch[0])
    launch_angle = float(pred_launch[1])
    from physics_engine import compute_trajectory
    traj = compute_trajectory(launch_speed, launch_angle, 0.0)
    distance_ft = traj["distance_ft"]
    hr_dist = fo_meta.get("hr_rule_distance_ft", 380)
    hr_lo = fo_meta.get("hr_rule_launch_angle_min", 20)
    hr_hi = fo_meta.get("hr_rule_launch_angle_max", 40)
    class_names = fo_meta["class_names"]
    fo_feats = fo_meta["feature_cols"]
    # Build feature row: launch + distance + career_iso + zone × handedness (if model expects them)
    ba = context.get("career_BA")
    slg = context.get("career_SLG") or context.get("batter_power")
    career_iso = 0.12
    if ba is not None and slg is not None:
        try:
            career_iso = max(0.0, float(slg) - float(ba))
        except (TypeError, ValueError):
            pass
    stand = (str(context.get("stand") or "R").upper().strip() == "L")
    stand_L = 1 if stand else 0
    plate_x_stand = plate_x * (1 if stand else -1)
    plate_z_stand = plate_z * (1 if stand else -1)
    row = {
        "launch_speed": launch_speed, "launch_angle": launch_angle, "distance_ft": distance_ft,
        "career_iso": career_iso, "plate_x": plate_x, "plate_z": plate_z,
        "stand_L": stand_L, "plate_x_stand": plate_x_stand, "plate_z_stand": plate_z_stand,
    }
    X_fo = pd.DataFrame([row]).reindex(columns=fo_feats, fill_value=0)
    if two_stage:
        pred_bin = int(stage1.predict(X_fo[fo_feats])[0])
        if pred_bin == 0:
            return "in_play_out"
        # Stage 2: use predict_proba + HR tax (0.8x) to suppress over-estimation, then argmax
        stage2_probs = np.asarray(stage2.predict_proba(X_fo[fo_feats])).ravel()
        hr_tax = fo_meta.get("hr_suppress_multiplier", 0.8)
        if len(stage2_probs) >= 3 and hr_tax != 1.0:
            stage2_probs[2] *= hr_tax  # HR is index 2 (single=0, xbh=1, hr=2)
            stage2_probs = stage2_probs / stage2_probs.sum()
        pred_hit = int(np.argmax(stage2_probs))
        result = class_names[1 + pred_hit]  # in_play_1b, extra_base_hit, in_play_hr
    else:
        pred_idx = int(fo_model.predict(X_fo[fo_feats])[0])
        result = class_names[pred_idx] if pred_idx < len(class_names) else "in_play_out"
    # HR rule: only override when model predicted a hit (do not turn a predicted out into HR)
    if result != "in_play_out" and "in_play_hr" in class_names:
        if distance_ft >= hr_dist and hr_lo <= launch_angle <= hr_hi:
            result = "in_play_hr"
    # Map 4-class extra_base_hit to game-state label (in_play_2b)
    if result == "extra_base_hit":
        result = "in_play_2b"
    return result


def predict_pitch_result(batter_models: dict, pitch: dict, context: dict, sample: bool = True) -> str | tuple[str, np.ndarray]:
    """
    Predict pitch_result from one pitch + context.
    pitch: {pitch_type, plate_x, plate_z, release_speed, release_spin_rate}
    context: {balls, strikes, stand, p_throws, batter, career_*, home_team, ...}
    If sample=True, returns sampled class (str). If sample=False, returns (class, proba array).
    When balls >= 3, P(ball) is reduced (count-aware) so the pitcher does not give away walks as often.
    """
    enc = batter_models["encoders"]
    X = batter_models["encode_batter_row"](pitch, context, enc)
    feats = batter_models["feats_batter"]
    proba = np.asarray(batter_models["clf"].predict_proba(X[feats])).ravel()
    idx_to_result = batter_models["idx_to_result"]
    n = len(proba)
    if n < len(idx_to_result):
        proba = np.resize(proba, len(idx_to_result))
        proba[n:] = 0
        proba = proba / proba.sum()
    # Post-hoc calibration (match batter_evaluation): suppress foul 0.70, boost hit 1.3 for 5-class
    n_classes = len(proba)
    foul_idx = next((int(i) for i, name in idx_to_result.items() if name == "foul"), None)
    hit_idx = next((int(i) for i, name in idx_to_result.items() if name == "hit"), None)
    if foul_idx is not None and foul_idx < n_classes:
        proba = proba.copy()
        proba[foul_idx] *= 0.70
        if n_classes == 5 and hit_idx is not None and hit_idx < n_classes:
            proba[hit_idx] *= 1.3
        proba = proba / proba.sum()
    # Count-aware: when pitcher has 3+ balls, reduce P(ball) and renormalize
    if context.get("balls", 0) >= 3:
        ball_idx = None
        for idx, name in idx_to_result.items():
            if name == "ball":
                ball_idx = int(idx) if isinstance(idx, str) else idx
                break
        if ball_idx is not None and ball_idx < len(proba):
            proba = proba.copy()
            proba[ball_idx] *= 0.6
            proba = proba / proba.sum()
    if sample:
        pred_idx = int(np.random.choice(len(proba), p=proba))
        return idx_to_result.get(pred_idx, "ball")
    pred_idx = int(np.argmax(proba))
    return idx_to_result.get(pred_idx, "ball"), proba


def update_game_state(state: dict, pitch_result: str) -> dict:
    """
    Update game state after one pitch. state: balls, strikes, outs, bases (list 0-2 for 1st,2nd,3rd), runs, inning.
    Returns new state. Caller should check for walk (balls==4), strikeout (strikes==3), HBP, or in_play_* to advance runners.
    """
    s = state.copy()
    s["balls"] = state["balls"]
    s["strikes"] = state["strikes"]
    s["outs"] = state["outs"]
    s["bases"] = list(state.get("bases", [0, 0, 0]))
    s["runs"] = state.get("runs", 0)

    if pitch_result == "ball":
        s["balls"] = state["balls"] + 1
        if s["balls"] >= 4:
            s["balls"], s["strikes"] = 0, 0
            # Walk: batter to first, advance forced
            b = s["bases"]
            if b[0]: b[1], b[0] = b[1] or b[0], 1
            else: b[0] = 1
            s["bases"] = b
    elif pitch_result in ("called_strike", "swinging_strike", "foul_tip"):
        s["strikes"] = state["strikes"] + 1
        if s["strikes"] >= 3:
            s["balls"], s["strikes"] = 0, 0
            s["outs"] = state["outs"] + 1
    elif pitch_result == "foul":
        s["strikes"] = min(2, state["strikes"] + 1)
    elif pitch_result == "hit_by_pitch":
        s["balls"], s["strikes"] = 0, 0
        b = s["bases"]
        if b[0]: b[1], b[0] = b[1] or b[0], 1
        else: b[0] = 1
        s["bases"] = b
    elif pitch_result == "in_play_out":
        s["balls"], s["strikes"] = 0, 0
        s["outs"] = state["outs"] + 1
    elif pitch_result == "in_play_1b":
        s["balls"], s["strikes"] = 0, 0
        b = s["bases"]
        s["runs"] += b[2]
        s["bases"] = [1, b[0], b[1]]
    elif pitch_result == "in_play_2b":
        s["balls"], s["strikes"] = 0, 0
        b = s["bases"]
        s["runs"] += b[1] + b[2]
        s["bases"] = [0, 1, b[0]]
    elif pitch_result == "in_play_3b":
        s["balls"], s["strikes"] = 0, 0
        b = s["bases"]
        s["runs"] += b[0] + b[1] + b[2]
        s["bases"] = [0, 0, 1]
    elif pitch_result == "in_play_hr":
        s["balls"], s["strikes"] = 0, 0
        b = s["bases"]
        s["runs"] += 1 + b[0] + b[1] + b[2]
        s["bases"] = [0, 0, 0]
    return s


def simulate_at_bat_step(
    pitcher_models: dict,
    batter_models: dict,
    context: dict,
    batter_id: int,
    batter_career: dict,
    home_team: str,
    state: dict,
    prev_pitch: dict | None = None,
    prev_result: str | None = None,
    pitch_history: list[tuple[str, int]] | None = None,
) -> tuple[dict, str, dict]:
    """
    One at-bat step: full pipeline mounted for the app.

    Flow:
    1. Pitcher models generate the pitch (ball): predict_pitch -> pitch dict (plate_x, plate_z, release_speed, etc.).
    2. Batter models (calibrated, same as evaluation) consume that pitch: predict_pitch_result -> immediate outcome
       (ball, called_strike, swinging_strike, foul, foul_tip, hit_by_pitch, or hit).
    3. Foul/strike/ball: result is that immediate event; game state updated.
    4. Hit: goes through BIP regressor -> physics engine -> final outcome model -> in_play_out / in_play_1b /
       in_play_2b / in_play_3b / in_play_hr. A hit can end in an immediate out (in_play_out) from the final outcome model.
    5. Game state is updated from the final result (all immediate events + in_play_*).

    context: pitcher context (pitcher, p_throws, stand, balls, strikes, ...) for predict_pitch.
    prev_pitch / prev_result / pitch_history: for batter context.
    Returns (pitch_dict, pitch_result, new_state).
    """
    pitch = predict_pitch(pitcher_models, context)
    previous_pitch_type = (prev_pitch.get("pitch_type", "__NA__") if prev_pitch else "__NA__")
    previous_was_strike = 1 if (prev_result and prev_result in ("called_strike", "swinging_strike", "foul", "foul_tip")) else 0

    batter_ctx = {
        "balls": context.get("balls", 0),
        "strikes": context.get("strikes", 0),
        "stand": context.get("stand", "R"),
        "p_throws": context.get("p_throws", "R"),
        "batter": batter_id,
        "home_team": home_team,
        "previous_pitch_type": previous_pitch_type,
        "previous_was_strike": previous_was_strike,
        **batter_career,
    }
    if context.get("pitcher") is not None:
        batter_ctx["pitcher"] = context["pitcher"]
    if pitch_history:
        _apply_pitch_history_to_batter_ctx(batter_ctx, pitch_history)
    result = predict_pitch_result(batter_models, pitch, batter_ctx, sample=True)
    fo_meta = batter_models.get("final_outcome_metadata")
    has_fo = (
        batter_models.get("final_outcome_model") is not None
        or (fo_meta and fo_meta.get("two_stage") and batter_models.get("final_outcome_stage1_model") is not None)
    )
    if result == "hit" and has_fo:
        result = _resolve_hit_to_in_play(batter_models, pitch, batter_ctx)  # BIP -> physics -> final outcome (can be in_play_out)
    new_state = update_game_state(state, result)
    return pitch, result, new_state


def simulate_half_inning(
    pitcher_models: dict,
    batter_models: dict,
    pitcher_context: dict,
    batting_order: list[tuple[int, dict]],
    home_team: str,
    max_pitches_per_batter: int = 20,
) -> tuple[int, list[dict]]:
    """
    Simulate half-inning until 3 outs. batting_order: list of (batter_id, batter_career).
    Returns (runs_scored, list of pitch/outcome logs for 2D viz).
    """
    state = {"balls": 0, "strikes": 0, "outs": 0, "bases": [0, 0, 0], "runs": 0}
    logs = []
    order_index = 0
    while state["outs"] < 3:
        batter_id, batter_career = batting_order[order_index % len(batting_order)]
        context = {**pitcher_context, "balls": state["balls"], "strikes": state["strikes"]}
        last_pitch: dict | None = None
        last_result: str | None = None
        pitch_history: list[tuple[str, int]] = []  # (pitch_type, was_strike) oldest to newest, max 3
        pitches_this_ab = 0
        while state["outs"] < 3 and pitches_this_ab < max_pitches_per_batter:
            pitch, result, state = simulate_at_bat_step(
                pitcher_models,
                batter_models,
                context,
                batter_id,
                batter_career,
                home_team,
                state,
                prev_pitch=last_pitch,
                prev_result=last_result,
                pitch_history=pitch_history if pitch_history else None,
            )
            logs.append({"pitch": pitch, "result": result, "state_after": state.copy()})
            context["balls"], context["strikes"] = state["balls"], state["strikes"]
            context["previous_pitch_type"] = pitch["pitch_type"]
            context["previous_release_speed"] = pitch["release_speed"]
            _was_strike = 1 if result in ("called_strike", "swinging_strike", "foul", "foul_tip") else 0
            pitch_history.append((pitch["pitch_type"], _was_strike))
            if len(pitch_history) > 3:
                pitch_history.pop(0)
            last_pitch = pitch
            last_result = result
            pitches_this_ab += 1
            # PA ends on walk (4 balls), strikeout (3 strikes), HBP, or in_play (state resets balls/strikes to 0)
            if state["balls"] == 0 and state["strikes"] == 0:
                break
        order_index += 1
    return state["runs"], logs


def _apply_pitch_history_to_batter_ctx(batter_ctx: dict, pitch_history: list[tuple[str, int]]) -> None:
    """Apply last 2–3 (pitch_type, was_strike) lags into batter_ctx for sequence features. In-place."""
    # pitch_history is ordered oldest to newest: [ (pt2, ws2), (pt1, ws1) ] for lag-2 and lag-1
    n = len(pitch_history)
    if n >= 2:
        batter_ctx["previous_2_pitch_type"] = pitch_history[-2][0]
        batter_ctx["previous_2_was_strike"] = pitch_history[-2][1]
    if n >= 3:
        batter_ctx["previous_3_pitch_type"] = pitch_history[-3][0]
        batter_ctx["previous_3_was_strike"] = pitch_history[-3][1]