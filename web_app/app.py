#!/usr/bin/env python3
"""
Streamlit app for the vizualisation tool 
"""

import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "Models" / "Training"))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(_PROJECT_ROOT / ".env")

_MODELS_DIR = _PROJECT_ROOT / "Models" / "saved_models"
_TRAINING_SAVED = _PROJECT_ROOT / "Models" / "Training" / "saved_models"
_ENCODERS_PATH = _TRAINING_SAVED / "pitcher_encoders.json"

PITCH_TYPE_NAMES = {
    "FF": "4-Seam Fastball", "FT": "2-Seam Fastball", "SI": "Sinker",
    "FC": "Cutter", "SL": "Slider", "CH": "Changeup", "CU": "Curveball",
    "FS": "Splitter", "KC": "Knuckle Curve", "ST": "Sweeper",
    "SV": "Slurve", "SC": "Screwball", "CS": "Slow Curve",
    "FA": "Fastball", "FO": "Forkball", "EP": "Eephus", "PO": "Pitch Out",
    "KN": "Knuckleball", "UN": "Unknown",
}


def _db_url() -> str:
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    dbname = os.getenv("PGDATABASE")
    pw = quote_plus(password) if password else ""
    return f"postgresql://{user}:{pw}@{host}:{port}/{dbname}"


#streamlit reruns the script on every click so we cache the engine and models so we do not reload every time
@st.cache_resource
def _load_engine():
    return create_engine(_db_url())


@st.cache_data
def _load_pitchers(_engine):
    q = """
    SELECT DISTINCT s.pitcher, s.p_throws,
           p.nameFirst || ' ' || p.nameLast AS pitcher_name
    FROM clean_statcast_with_batter s
    JOIN player_map pm ON s.pitcher = pm.key_mlbam
    JOIN people p ON pm.key_bbref = p.playerID
    WHERE s.game_type = 'R' AND s.pitcher IS NOT NULL
    ORDER BY pitcher_name
    """
    return pd.read_sql(q, _engine)


@st.cache_data
def _load_pitcher_career(_engine, pitcher_id: int):
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
    df = pd.read_sql(q, _engine, params={"pid": pitcher_id})
    return df.iloc[0].to_dict() if len(df) > 0 else None


@st.cache_resource
def _load_models_and_encoders():
    from data_prep import (
        load_pitcher_encoders,
        load_residual_stds,
        load_pitcher_repertoire,
        load_pitcher_plate_means,
        load_pitcher_pitch_type_rates,
        _DEFAULT_CAT_ENCODINGS,
        encode_context_row,
    )

    residual_stds = load_residual_stds(_TRAINING_SAVED / "residual_stds.json")
    pitcher_repertoire = load_pitcher_repertoire(_TRAINING_SAVED / "pitcher_repertoire.json")
    pitcher_plate_means = load_pitcher_plate_means(_TRAINING_SAVED / "pitcher_plate_means.json")
    pitcher_pitch_type_rates = load_pitcher_pitch_type_rates(_TRAINING_SAVED / "pitcher_pitch_type_rates.json")

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


def _predict_pitch(models, context: dict) -> dict:
    enc = models["encoders"]
    encode_fn = models["encode_context_row"]
    clf = models["clf"]
    reg = models["reg_models"]
    type_to_idx = models["type_to_idx"]
    idx_to_type = models["idx_to_type"]
    feats_s1 = models["feats_s1"]
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
    # Blend with pitcher's empirical pitch-type rates so e.g. FF > ST for Ohtani
    pitcher_rates = models.get("pitcher_pitch_type_rates") or {}
    rates_dict = pitcher_rates.get(int(pitcher_id) if pitcher_id is not None else -1) if pitcher_id is not None else None
    if rates_dict:
        p_rates = np.array([rates_dict.get(idx_to_type.get(i, "FF"), 0.0) for i in range(len(p))], dtype=float)
        if p_rates.sum() > 0:
            p_rates = p_rates / p_rates.sum()
            p = 0.5 * p + 0.5 * p_rates
            p = p / p.sum()
    # Temperature > 1 flattens probs so we get more pitch-type diversity (not just FF/SL)
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


def main():
    st.set_page_config(page_title="Pitcher Simulator", layout="wide")

    engine = _load_engine()

    try:
        models = _load_models_and_encoders()
    except Exception as e:
        st.error(f"Failed to load models or encoders: {e}")
        st.info("Ensure pitcher models and pitcher_encoders.json exist. Run: python data_prep.py --save-pitcher-encoders")
        return

    pitchers_df = _load_pitchers(engine)
    name_count = {}
    options = []
    pitcher_id_map = {}
    for _, row in pitchers_df.iterrows():
        name = row["pitcher_name"]
        name_count[name] = name_count.get(name, 0) + 1
        label = name if name_count[name] == 1 else f"{name} (ID: {row['pitcher']})"
        options.append(label)
        pitcher_id_map[label] = (int(row["pitcher"]), str(row["p_throws"]))

    selected = st.selectbox("Select pitcher", options, placeholder="Type to search...")
    if not selected:
        st.stop()

    pitcher_id, p_throws = pitcher_id_map[selected]

    if "last_pitcher" not in st.session_state or st.session_state["last_pitcher"] != pitcher_id:
        st.session_state["last_pitcher"] = pitcher_id
        st.session_state["pitch_number"] = 1
        st.session_state["previous_pitch_type"] = "__NA__"
        st.session_state["previous_release_speed"] = 0.0
        st.session_state["pitches"] = []

    career = _load_pitcher_career(engine, pitcher_id)
    if career:
        st.subheader("Career stats")
        cols = st.columns(6)
        v = career.get("pitcher_career_ip")
        cols[0].metric("Innings Pitched", f"{v:.1f}" if v is not None else "-")
        v = career.get("pitcher_career_era")
        cols[1].metric("Earned Run Average", f"{v:.2f}" if v is not None else "-")
        v = career.get("pitcher_career_so")
        cols[2].metric("Strikeouts", int(v) if v is not None else "-")
        v = career.get("pitcher_career_bb")
        cols[3].metric("Walks (Base on Balls)", int(v) if v is not None else "-")
        v = career.get("pitcher_career_hr")
        cols[4].metric("Home Runs Allowed", int(v) if v is not None else "-")
        v = career.get("pitcher_career_bfp")
        cols[5].metric("Batters Faced", int(v) if v is not None else "-")
    else:
        career = {"pitcher_career_ip": -1, "pitcher_career_era": -1, "pitcher_career_so": -1,
                  "pitcher_career_bb": -1, "pitcher_career_h": -1, "pitcher_career_er": -1,
                  "pitcher_career_hr": -1, "pitcher_career_bfp": -1, "pitcher_career_ipouts": -1}

    if st.button("Next pitch"):
        pitch_num = st.session_state["pitch_number"]
        prev_pt = st.session_state["previous_pitch_type"]
        prev_speed = st.session_state["previous_release_speed"]

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

        pred = _predict_pitch(models, context)
        st.session_state["pitches"].append(pred)
        st.session_state["pitch_number"] = pitch_num + 1
        st.session_state["previous_pitch_type"] = pred["pitch_type"]
        st.session_state["previous_release_speed"] = pred["release_speed"]

        st.rerun()

    pitches = st.session_state.get("pitches", [])
    if pitches:
        SZ_LEFT, SZ_RIGHT = -0.71, 0.71
        SZ_BOT, SZ_TOP = 1.5, 3.5

        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        df_plot = pd.DataFrame(pitches)

        ax.plot([SZ_LEFT, SZ_RIGHT, SZ_RIGHT, SZ_LEFT, SZ_LEFT],
                [SZ_BOT, SZ_BOT, SZ_TOP, SZ_TOP, SZ_BOT],
                color="gray", linewidth=2, linestyle="-", label="Strike zone")

        # Draw pitch locations by type
        for pt in df_plot["pitch_type"].unique():
            sub = df_plot[df_plot["pitch_type"] == pt]
            ax.scatter(sub["plate_x"], sub["plate_z"], label=pt, alpha=0.7, s=50, zorder=5)

        latest = pitches[-1]
        ax.scatter(latest["plate_x"], latest["plate_z"], s=120, facecolors="none",
                   edgecolors="red", linewidths=2, zorder=6, label="Latest pitch")

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0.5, 5)
        ax.grid(True, alpha=0.4, linestyle="--")
        ax.set_xlabel("Horizontal position (feet from center)", fontsize=8)
        ax.set_ylabel("Vertical position (feet from ground)", fontsize=8)
        ax.set_title("Pitch locations – strike zone view", fontsize=9)
        ax.legend(loc="upper right", fontsize=6)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

        col_graph, col_metrics = st.columns([3, 2])
        with col_graph:
            st.pyplot(fig)
        plt.close(fig)

        with col_metrics:
            pt_name = PITCH_TYPE_NAMES.get(latest["pitch_type"], latest["pitch_type"])
            st.subheader(f"Pitch #{len(pitches)}")
            st.metric("Pitch type", pt_name)
            st.metric("Release speed", f"{latest['release_speed']:.1f} mph")
            st.metric("Spin rate", f"{latest['release_spin_rate']:.0f} rpm")
    else:
        st.info("Click **Next pitch** to simulate the first pitch.")


if __name__ == "__main__":
    main()
