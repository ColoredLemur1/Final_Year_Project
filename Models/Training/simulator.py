"""
Simulator pipeline: Outcome model → (for in-play only) regressor → physics → summary.

Given pitch context (speed, location, batter power), runs the 5-bucket outcome model.
For in-play outcomes (Out, Single, Extra Base Hit), runs the launch-parameter
regressor and physics engine and returns trajectory + summary with field/distance.
For strikeout and walk, skips regressor and physics and returns "No batted ball."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parent
_SAVED = _BASE / "saved_models"

# Human-readable outcome labels for summary (bucket index -> short label)
OUTCOME_LABELS = {
    0: "Out",
    1: "Strikeout",
    2: "Walk/HBP",
    3: "Single",
    4: "Extra Base Hit",
}

# In-play buckets get regressor + physics; non-in-play (strikeout, walk) do not.
IN_PLAY_BUCKETS = (0, 3, 4)  # in_play_out, single, extra_base
NON_IN_PLAY_BUCKETS = (1, 2)  # strikeout, free_pass

# Spray angle (degrees) -> field direction for summary (spray_angle 0 = center)
def _field_from_spray(spray_angle_deg: float) -> str:
    if spray_angle_deg < -15:
        return "Left Field"
    if spray_angle_deg > 15:
        return "Right Field"
    return "Center Field"


def _load_models() -> tuple[Any, dict, Any, dict]:
    """Load outcome model, batter encoders, regressor, regressor metadata. Raises if missing."""
    import joblib
    from data_prep_batters import encode_batter_row, load_batter_encoders

    encoders = load_batter_encoders(_SAVED / "batter_encoders.json")
    if encoders is None:
        raise FileNotFoundError(f"batter_encoders.json not found in {_SAVED}")
    outcome_path = _SAVED / "batter_pitch_result_calibrated.joblib"
    if not outcome_path.exists():
        raise FileNotFoundError(f"batter_pitch_result_calibrated.joblib not found in {_SAVED}")
    reg_path = _SAVED / "outcome_regressor.joblib"
    meta_path = _SAVED / "outcome_regressor_metadata.json"
    if not reg_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"outcome_regressor.joblib and outcome_regressor_metadata.json required in {_SAVED}")

    outcome_model = joblib.load(outcome_path)
    regressor = joblib.load(reg_path)
    with open(meta_path) as f:
        reg_metadata = json.load(f)

    return outcome_model, encoders, regressor, reg_metadata


def run(
    pitch_context: dict[str, Any],
    *,
    use_drag: bool = False,
    encode_batter_row_fn: Any = None,
) -> dict[str, Any]:
    """
    Run the full simulator: outcome → (for in-play only) regressor → physics → summary.

    pitch_context must include at least:
      - release_speed or pitch_mph (mph)
      - plate_x, plate_z (ft)
      - batter_power or career_SLG (for regressor; also pass career_SLG for outcome model if needed)

    Optional: balls, strikes, stand, p_throws, pitch_type, release_spin_rate,
    career_AB, career_H, career_HR, career_BA, career_OBP, home_team, batter.
    Defaults are used for any missing keys so the outcome model can run.

    For in-play outcomes (Out, Single, Extra Base Hit): runs regressor and physics;
    returns trajectory fields and a summary with field and distance.
    For non-in-play (Strikeout, Walk/HBP): skips regressor and physics; trajectory
    fields are None and summary is e.g. "Strikeout. No batted ball."

    Returns dict with: outcome, outcome_bucket, outcome_name, launch_speed_mph,
    launch_angle_deg, spray_angle_deg, distance_ft, max_height_ft, time_of_flight_s, summary.
    """
    from data_prep_batters import encode_batter_row as _encode_batter_row
    from physics_engine import compute_trajectory

    encode_fn = encode_batter_row_fn or _encode_batter_row
    outcome_model, encoders, regressor, reg_metadata = _load_models()
    feature_cols = reg_metadata["feature_cols"]
    outcome_names = reg_metadata.get("outcome_names", [])

    # Build pitch and context for batter outcome model
    pitch_mph = float(pitch_context.get("pitch_mph") or pitch_context.get("release_speed") or 0)
    plate_x = float(pitch_context.get("plate_x", 0))
    plate_z = float(pitch_context.get("plate_z", 0))
    batter_power = float(pitch_context.get("batter_power") or pitch_context.get("career_SLG") or 0.4)

    pitch = {
        "pitch_type": pitch_context.get("pitch_type", "FF"),
        "plate_x": plate_x,
        "plate_z": plate_z,
        "release_speed": pitch_mph,
        "release_spin_rate": pitch_context.get("release_spin_rate", 2000),
    }
    context = {
        "balls": pitch_context.get("balls", 0),
        "strikes": pitch_context.get("strikes", 0),
        "stand": pitch_context.get("stand", "R"),
        "p_throws": pitch_context.get("p_throws", "R"),
        "batter": pitch_context.get("batter", -1),
        "career_AB": pitch_context.get("career_AB", 100),
        "career_H": pitch_context.get("career_H", 25),
        "career_HR": pitch_context.get("career_HR", 5),
        "career_BA": pitch_context.get("career_BA", 0.25),
        "career_OBP": pitch_context.get("career_OBP", 0.32),
        "career_SLG": batter_power,
        "home_team": pitch_context.get("home_team", "__NA__"),
    }

    # 1) Outcome model → predicted bucket 0–4
    X_batter = encode_fn(pitch, context, encoders)
    feats = encoders.get("feats_batter", list(X_batter.columns))
    if not all(c in X_batter.columns for c in feats):
        feats = [c for c in feats if c in X_batter.columns]
    X_batter = X_batter.reindex(columns=feats, fill_value=-1)
    proba = np.asarray(outcome_model.predict_proba(X_batter)).ravel()
    pred_bucket = int(np.argmax(proba))
    outcome_name = outcome_names[pred_bucket] if pred_bucket < len(outcome_names) else f"bucket_{pred_bucket}"

    label = OUTCOME_LABELS.get(pred_bucket, outcome_name)

    # Non-in-play (strikeout, walk): skip regressor and physics; no trajectory.
    if pred_bucket in NON_IN_PLAY_BUCKETS:
        if pred_bucket == 1:
            summary = "Strikeout. No batted ball."
        else:
            summary = "Walk. No batted ball."
        return {
            "outcome": label,
            "outcome_bucket": pred_bucket,
            "outcome_name": outcome_name,
            "launch_speed_mph": None,
            "launch_angle_deg": None,
            "spray_angle_deg": None,
            "distance_ft": None,
            "max_height_ft": None,
            "time_of_flight_s": None,
            "summary": summary,
        }

    # In-play (0, 3, 4): regressor → physics → full summary with field and distance.
    one_hot = [1 if i == pred_bucket else 0 for i in range(reg_metadata.get("n_outcomes", 5))]
    row = {
        "pitch_mph": pitch_mph,
        "plate_x": plate_x,
        "plate_z": plate_z,
        "batter_power": batter_power,
    }
    for i, v in enumerate(one_hot):
        row[f"outcome_{i}"] = v
    X_reg = pd.DataFrame([row])
    X_reg = X_reg[feature_cols]
    pred_launch = regressor.predict(X_reg)[0]
    launch_speed = float(pred_launch[0])
    launch_angle = float(pred_launch[1])
    spray_angle = float(pred_launch[2]) if len(pred_launch) > 2 else 0.0

    traj = compute_trajectory(launch_speed, launch_angle, spray_angle, use_drag=use_drag)
    distance_ft = traj["distance_ft"]
    max_height_ft = traj["max_height_ft"]
    time_of_flight_s = traj["time_of_flight_s"]

    field = _field_from_spray(spray_angle)
    summary = f"{label} to {field}! Distance: {distance_ft:.0f} ft, Max Height: {max_height_ft:.0f} ft."

    return {
        "outcome": label,
        "outcome_bucket": pred_bucket,
        "outcome_name": outcome_name,
        "launch_speed_mph": round(launch_speed, 1),
        "launch_angle_deg": round(launch_angle, 1),
        "spray_angle_deg": round(spray_angle, 1),
        "distance_ft": round(distance_ft, 1),
        "max_height_ft": round(max_height_ft, 1),
        "time_of_flight_s": round(time_of_flight_s, 2),
        "summary": summary,
    }
