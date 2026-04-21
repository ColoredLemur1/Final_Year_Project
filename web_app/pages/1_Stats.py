#!/usr/bin/env python3
"""Stats page for viewing batter and pitcher performance summaries."""

from __future__ import annotations

import sys
from pathlib import Path

_WEB_APP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _WEB_APP_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "Models" / "Training"))
sys.path.insert(0, str(_PROJECT_ROOT / "Models" / "Evaluation"))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_WEB_APP_DIR / ".env")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

import utils as eval_utils


PITCH_FULL = {
    "FF": "Four-seam fastball", "FT": "Two-seam fastball", "FC": "Cutter",
    "SI": "Sinker", "FS": "Splitter", "SL": "Slider", "CH": "Changeup",
    "CU": "Curveball", "KC": "Knuckle curve", "ST": "Sweeper",
    "SV": "Slurve", "GY": "Gyro", "UN": "Unknown",
}

def _pitch_full(abbrev: str) -> str:
    """Return a readable pitch name from its abbreviation."""
    return PITCH_FULL.get(abbrev, abbrev)



@st.cache_resource
def _get_engine():
    """Create and cache the database engine for this page."""
    return eval_utils.get_engine()




@st.cache_data(ttl=300)
def _load_all_batters(_engine) -> pd.DataFrame:
    """
    Returns one row per batter with career stats needed for the radar and detail cards.
    Columns (all lowercase after pd.read_sql): batter, display_name, stand, debut_year,
             career_ab, career_h, career_hr, career_ba, career_obp, career_slg,
             career_rbi, career_bb, career_so
    """
    # Main stats already in the cross-source view
    q_main = text("""
        SELECT
            batter,
            MAX(batter_name_last || ', ' || batter_name_first) AS display_name,
            MAX(stand)       AS stand,
            MAX(EXTRACT(YEAR FROM batter_debut))  AS debut_year,
            MAX(career_AB)   AS career_AB,
            MAX(career_H)    AS career_H,
            MAX(career_HR)   AS career_HR,
            MAX(career_BA)   AS career_BA,
            MAX(career_OBP)  AS career_OBP,
            MAX(career_SLG)  AS career_SLG
        FROM clean_statcast_with_batter
        WHERE game_type = 'R' AND batter IS NOT NULL
          AND batter_name_last IS NOT NULL
        GROUP BY batter
        ORDER BY MAX(batter_name_last), MAX(batter_name_first)
    """)
    df_main = pd.read_sql(q_main, _engine)

    # Supplementary: RBI, BB, SO not exposed in clean_statcast_with_batter
    q_supp = text("""
        SELECT pm.key_mlbam AS batter,
               cb.RBI AS career_RBI,
               cb.BB  AS career_BB,
               cb.SO  AS career_SO
        FROM clean_vw_career_batting cb
        JOIN player_map pm ON pm.key_bbref = cb.playerID
    """)
    df_supp = pd.read_sql(q_supp, _engine)

    df = df_main.merge(df_supp, on="batter", how="left")
    return df


@st.cache_data(ttl=300)
def _load_best_hit_pitch(_engine, batter_id: int) -> str | None:
    """Returns the full name of the pitch type the batter hits most often."""
    q = text("""
        SELECT pitch_type, COUNT(*) AS hits
        FROM clean_statcast_with_batter
        WHERE batter = :bid
          AND events IN ('single', 'double', 'triple', 'home_run')
          AND pitch_type IS NOT NULL
        GROUP BY pitch_type
        ORDER BY hits DESC
        LIMIT 1
    """)
    row = pd.read_sql(q, _engine, params={"bid": batter_id})
    if row.empty:
        return None
    return _pitch_full(str(row.iloc[0]["pitch_type"]))



@st.cache_data(ttl=300)
def _load_all_pitchers(_engine) -> pd.DataFrame:
    """
    Returns one row per pitcher with career stats for the radar and detail cards.
    Columns (all lowercase after pd.read_sql): pitcher, display_name, p_throws, debut_year,
             career_ip, w, l, career_so, career_bb, career_h, career_er,
             era, k9, bb9, whip, win_pct
    """
    # Pitcher list + throwing hand from Statcast
    q_sc = text("""
        SELECT
            s.pitcher,
            MAX(pm.name_last || ', ' || pm.name_first) AS display_name,
            MAX(s.p_throws) AS p_throws
        FROM clean_statcast_with_batter s
        LEFT JOIN player_map pm ON s.pitcher = pm.key_mlbam
        WHERE s.game_type = 'R' AND s.pitcher IS NOT NULL
          AND pm.name_last IS NOT NULL
        GROUP BY s.pitcher
        ORDER BY MAX(pm.name_last), MAX(pm.name_first)
    """)
    df_sc = pd.read_sql(q_sc, _engine)

    # Career pitching stats from Lahman
    q_lah = text("""
        SELECT pm.key_mlbam AS pitcher,
               cp.W, cp.L,
               ROUND(cp.IPouts::NUMERIC / 3, 1) AS career_IP,
               cp.SO AS career_SO,
               cp.BB AS career_BB,
               cp.H  AS career_H,
               cp.ER AS career_ER,
               EXTRACT(YEAR FROM p.debut) AS debut_year
        FROM clean_vw_career_pitching cp
        JOIN player_map pm ON pm.key_bbref = cp.playerID
        JOIN people p ON p.playerID = cp.playerID
    """)
    df_lah = pd.read_sql(q_lah, _engine)

    df = df_sc.merge(df_lah, on="pitcher", how="left")

    # Compute derived stats
    ip = df["career_ip"].replace(0, np.nan)
    df["era"]     = (9 * df["career_er"] / ip).round(2)
    df["k9"]      = (9 * df["career_so"] / ip).round(2)
    df["bb9"]     = (9 * df["career_bb"] / ip).round(2)
    df["whip"]    = ((df["career_bb"] + df["career_h"]) / ip).round(3)
    denom = (df["w"] + df["l"]).replace(0, np.nan)
    df["win_pct"] = (df["w"] / denom * 100).round(1)

    return df


@st.cache_data(ttl=300)
def _load_pitcher_pitch_insights(_engine, pitcher_id: int) -> dict:
    """Returns {'favourite': full_name, 'best_k': full_name} for a pitcher."""
    q_fav = text("""
        SELECT pitch_type, COUNT(*) AS cnt
        FROM clean_statcast_with_batter
        WHERE pitcher = :pid AND pitch_type IS NOT NULL
        GROUP BY pitch_type
        ORDER BY cnt DESC
        LIMIT 1
    """)
    q_k = text("""
        SELECT pitch_type,
               SUM(CASE WHEN events = 'strikeout'
                          OR description IN ('swinging_strike', 'called_strike',
                                             'swinging_strike_blocked')
                        THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS k_rate
        FROM clean_statcast_with_batter
        WHERE pitcher = :pid AND pitch_type IS NOT NULL
        GROUP BY pitch_type
        HAVING COUNT(*) >= 50
        ORDER BY k_rate DESC
        LIMIT 1
    """)
    fav_df = pd.read_sql(q_fav, _engine, params={"pid": pitcher_id})
    k_df   = pd.read_sql(q_k,   _engine, params={"pid": pitcher_id})

    return {
        "favourite": _pitch_full(str(fav_df.iloc[0]["pitch_type"])) if not fav_df.empty else "N/A",
        "best_k":    _pitch_full(str(k_df.iloc[0]["pitch_type"]))   if not k_df.empty  else "N/A",
    }


# Normalization helpers

def _minmax(series: pd.Series, val: float, invert: bool = False) -> float:
    """Normalize val against series min/max; optionally invert (lower=better stats)."""
    lo, hi = series.min(), series.max()
    if pd.isna(val) or pd.isna(lo) or pd.isna(hi) or lo == hi:
        return 0.0
    norm = float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))
    return round(1.0 - norm if invert else norm, 4)


def _fmt_int(v) -> str:
    """Format a numeric value as a comma-separated integer, or 'N/A'."""
    if pd.isna(v):
        return "N/A"
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_stat(v, decimals: int = 3) -> str:
    """Format a numeric value as a fixed-point float, or 'N/A'."""
    if pd.isna(v):
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"




def _render_batter_tab(engine) -> None:
    """Render batter selector, radar chart, details, and stat breakdown."""
    df = _load_all_batters(engine)
    if df.empty:
        st.warning("No batter data found.")
        return

    options = df["display_name"].tolist()
    batter_ids = df["batter"].tolist()

    selected_idx = st.selectbox(
        "Select Batter",
        range(len(options)),
        format_func=lambda i: options[i],
        key="stats_batter_select",
    )
    row = df.iloc[selected_idx]
    batter_id = int(batter_ids[selected_idx])

    # Radar values across all batters
    def _safe(col):
        """Return a valid float stat value or NaN for missing/sentinel values."""
        v = row.get(col)
        return float(v) if pd.notna(v) and v != -1 else np.nan

    ba_val      = _safe("career_ba")
    obp_val     = _safe("career_obp")
    slg_val     = _safe("career_slg")
    ab_val      = _safe("career_ab")
    hr_val      = _safe("career_hr")
    bb_val      = _safe("career_bb")
    so_val      = _safe("career_so")

    hr_rate_val = (hr_val / ab_val) if (not np.isnan(hr_val) and not np.isnan(ab_val) and ab_val > 0) else np.nan
    bb_k_val    = (bb_val / so_val) if (not np.isnan(bb_val) and not np.isnan(so_val) and so_val > 0) else np.nan

    # Compute HR rate and BB/K for the full population for normalization
    ab_s  = df["career_ab"].replace(-1, np.nan).astype(float)
    hr_s  = df["career_hr"].replace(-1, np.nan).astype(float)
    bb_s  = df["career_bb"].replace(-1, np.nan).astype(float)
    so_s  = df["career_so"].replace(-1, np.nan).astype(float)
    hr_rate_series = (hr_s / ab_s.replace(0, np.nan))
    bb_k_series    = (bb_s / so_s.replace(0, np.nan))

    r_vals = [
        _minmax(df["career_ba"].replace(-1, np.nan).astype(float),  ba_val),
        _minmax(df["career_obp"].replace(-1, np.nan).astype(float), obp_val),
        _minmax(df["career_slg"].replace(-1, np.nan).astype(float), slg_val),
        _minmax(hr_rate_series, hr_rate_val),
        _minmax(bb_k_series,    bb_k_val),
    ]
    axes = ["BA", "OBP", "SLG", "HR rate", "BB/K"]

    # Radar graph and player details
    col_radar, col_details = st.columns(2)

    with col_radar:
        st.markdown("**Career Stat Radar**")
        fig = go.Figure(go.Scatterpolar(
            r=r_vals + [r_vals[0]],
            theta=axes + [axes[0]],
            fill="toself",
            fillcolor="rgba(79,142,247,0.25)",
            line=dict(color="#4f8ef7", width=2),
            name=options[selected_idx],
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_details:
        st.markdown("**Player Details**")

        stand_map = {"R": "Right", "L": "Left", "S": "Switch"}
        stand_raw = str(row.get("stand", "")) if pd.notna(row.get("stand")) else ""
        stand_label = stand_map.get(stand_raw, stand_raw or "N/A")

        debut_year = int(row["debut_year"]) if pd.notna(row.get("debut_year")) else "N/A"

        d1, d2 = st.columns(2)
        d1.metric("Dominant Hand",         stand_label)
        d2.metric("Career At-Bats",        _fmt_int(ab_val))
        d3, d4 = st.columns(2)
        d3.metric("Career Home Runs",       _fmt_int(hr_val))
        d4.metric("Career Runs Batted In",  _fmt_int(_safe("career_rbi")))

        st.markdown("---")
        st.markdown("**Statcast Insights**")
        best_hit = _load_best_hit_pitch(engine, batter_id)
        st.markdown(f"**Best Hit Ball Type:** {best_hit or 'N/A'}")
        st.markdown(f"**Debut Year:** {debut_year}")

    # Stat breakdown
    st.markdown("### Stat Breakdown")

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("BA",      _fmt_stat(ba_val))
    s1.caption("Batting Average — hits per at-bat. Higher = better contact hitter.")
    s2.metric("OBP",     _fmt_stat(obp_val))
    s2.caption("On-Base % — how often the batter reaches base (hits, walks, hit-by-pitch). Higher = harder to get out.")
    s3.metric("SLG",     _fmt_stat(slg_val))
    s3.caption("Slugging % — measures power; weights extra-base hits more. Higher = more dangerous hitter.")
    s4.metric("HR rate", _fmt_stat(hr_rate_val))
    s4.caption("Home runs per at-bat. Higher = more likely to hit it out of the park.")
    s5.metric("BB/K",    _fmt_stat(bb_k_val, 2))
    s5.caption("Walk-to-strikeout ratio — plate discipline. Higher = better eye, takes walks, avoids strikeouts.")




def _render_pitcher_tab(engine) -> None:
    """Render pitcher selector, radar chart, details, and stat breakdown."""
    df = _load_all_pitchers(engine)
    if df.empty:
        st.warning("No pitcher data found.")
        return

    options     = df["display_name"].tolist()
    pitcher_ids = df["pitcher"].tolist()

    selected_idx = st.selectbox(
        "Select Pitcher",
        range(len(options)),
        format_func=lambda i: options[i],
        key="stats_pitcher_select",
    )
    row        = df.iloc[selected_idx]
    pitcher_id = int(pitcher_ids[selected_idx])

    # Radar values
    def _safe(col):
        """Return a valid float stat value or NaN for missing values."""
        v = row.get(col)
        return float(v) if pd.notna(v) else np.nan

    era_val     = _safe("era")
    k9_val      = _safe("k9")
    bb9_val     = _safe("bb9")
    whip_val    = _safe("whip")
    win_pct_val = _safe("win_pct")

    r_vals = [
        _minmax(df["era"].astype(float),     era_val,     invert=True),
        _minmax(df["k9"].astype(float),      k9_val),
        _minmax(df["bb9"].astype(float),     bb9_val,     invert=True),
        _minmax(df["whip"].astype(float),    whip_val,    invert=True),
        _minmax(df["win_pct"].astype(float), win_pct_val),
    ]
    axes = ["ERA", "K/9", "BB/9", "WHIP", "Win %"]

    # Radar graph and player details
    col_radar, col_details = st.columns(2)

    with col_radar:
        st.markdown("**Career Stat Radar**")
        fig = go.Figure(go.Scatterpolar(
            r=r_vals + [r_vals[0]],
            theta=axes + [axes[0]],
            fill="toself",
            fillcolor="rgba(239,68,68,0.2)",
            line=dict(color="#ef4444", width=2),
            name=options[selected_idx],
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_details:
        st.markdown("**Player Details**")

        throws_map   = {"R": "Right", "L": "Left"}
        throws_raw   = str(row.get("p_throws", "")) if pd.notna(row.get("p_throws")) else ""
        throws_label = throws_map.get(throws_raw, throws_raw or "N/A")

        debut_year = int(row["debut_year"]) if pd.notna(row.get("debut_year")) else "N/A"
        career_ip  = _fmt_stat(_safe("career_ip"), 1)
        career_w   = _fmt_int(_safe("w"))
        career_l   = _fmt_int(_safe("l"))

        d1, d2 = st.columns(2)
        d1.metric("Throwing Hand",          throws_label)
        d2.metric("Career Innings Pitched",  career_ip)
        d3, d4 = st.columns(2)
        d3.metric("Career Wins",            career_w)
        d4.metric("Career Losses",          career_l)

        st.markdown("---")
        st.markdown("**Statcast Insights**")
        insights = _load_pitcher_pitch_insights(engine, pitcher_id)
        st.markdown(f"**Favourite Ball Pitched:** {insights['favourite']}")
        st.markdown(f"**Best Strikeout Pitch:** {insights['best_k']}")
        st.markdown(f"**Debut Year:** {debut_year}")

    # Stat breakdown
    st.markdown("### Stat Breakdown")

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("ERA",   _fmt_stat(era_val, 2))
    s1.caption("Earned Run Average — runs given up per 9 innings. Lower = better pitcher.")
    s2.metric("K/9",   _fmt_stat(k9_val))
    s2.caption("Strikeouts per 9 innings. Higher = more dominant at getting batters out.")
    s3.metric("BB/9",  _fmt_stat(bb9_val))
    s3.caption("Walks per 9 innings — control. Lower = fewer free passes given to batters.")
    s4.metric("WHIP",  _fmt_stat(whip_val, 3))
    s4.caption("Walks + Hits per inning. Lower = fewer batters reaching base each inning.")
    s5.metric("Win %", (_fmt_stat(win_pct_val, 1) + "%") if not pd.isna(win_pct_val) else "N/A")
    s5.caption("Win percentage — wins out of total decisions. Higher = more games won.")


def main() -> None:
    """Render the Stats page with separate batter and pitcher tabs."""
    st.title("Player Stats")
    engine = _get_engine()
    tab_bat, tab_pit = st.tabs(["Batters", "Pitchers"])
    with tab_bat:
        _render_batter_tab(engine)
    with tab_pit:
        _render_pitcher_tab(engine)


main()
