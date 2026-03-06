#!/usr/bin/env python3
"""
Streamlit Pitcher–Batter Simulator: select pitcher and batter, throw pitch-by-pitch,
see pitch/outcome data, strike zone plot (X = strikes, O = hits/fouls), balls/strikes tally,
and per-pitch messages. Simulation ends on terminal outcome until player change.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WEB_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _WEB_APP_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "Models" / "Training"))
sys.path.insert(0, str(_PROJECT_ROOT / "Models" / "Evaluation"))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_WEB_APP_DIR / ".env")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sqlalchemy import text

# Import after path setup
import utils as eval_utils


def _result_to_message(result: str) -> str:
    """Map model result to short human-readable label (per-pitch and legacy 5-bucket)."""
    messages = {
        "ball": "Ball",
        "called_strike": "Called strike",
        "swinging_strike": "Swinging strike",
        "foul": "Foul",
        "foul_tip": "Foul tip",
        "hit_by_pitch": "Hit by pitch",
        "in_play_out": "Out",
        "in_play_1b": "Single",
        "in_play_2b": "Double",
        "in_play_3b": "Triple",
        "in_play_hr": "Home run!",
        "strikeout": "Strikeout",
        "free_pass": "Walk",
        "single": "Single",
        "extra_base": "Extra base",
    }
    return messages.get(result, result.replace("_", " ").title())


def _terminal_game_message(result: str, new_state: dict, entry: dict) -> tuple[str, str]:
    """For terminal outcomes, return (short_label, game_message) so the UI clearly states what happened in the game."""
    # Walk (4 balls) — state resets to 0-0 after walk
    if result == "ball" and new_state.get("balls", 0) == 0 and new_state.get("strikes", 0) == 0:
        return "Walk", "Player walks to first base!"
    # Strikeout
    if result in ("called_strike", "swinging_strike", "foul_tip"):
        return "Strikeout", "Batter is out. Three strikes."
    # Hit by pitch
    if result == "hit_by_pitch":
        return "Hit by pitch", "Batter takes first base."
    # In-play outcomes
    if result == "in_play_out":
        return "Out", _detailed_outcome_message(entry)
    if result == "in_play_1b":
        return "Single", "Player reaches first base!"
    if result == "in_play_2b":
        return "Double", "Player reaches second base!"
    if result == "in_play_3b":
        return "Triple", "Player reaches third base!"
    if result == "in_play_hr":
        return "Home run!", _detailed_outcome_message(entry)
    return _result_to_message(result), _detailed_outcome_message(entry)


def _outcome_style(result: str, pa_ended: bool, new_state: dict) -> str:
    """Return 'out' | 'on_base' | 'hr' | 'immediate' for color: red, green, yellow, grey."""
    if result == "in_play_out":
        return "out"
    if result == "in_play_hr":
        return "hr"
    if result in ("in_play_1b", "in_play_2b", "in_play_3b", "hit_by_pitch"):
        return "on_base"
    if pa_ended and result in ("called_strike", "swinging_strike", "foul_tip"):
        return "out"  # strikeout
    if pa_ended and result == "ball" and new_state.get("balls", 0) == 0 and new_state.get("strikes", 0) == 0:
        return "on_base"  # walk
    return "immediate"  # ball, strike, foul (non-terminal)


def _detailed_outcome_message(entry: dict) -> str:
    """Build a clear, descriptive message for what happened (e.g. HIT! ball to left field, distance, elevation, caught)."""
    result = entry["result"]
    pitch = entry["pitch"]
    pt = pitch.get("pitch_type", "?")
    speed = pitch.get("release_speed", 90)
    plate_x = pitch.get("plate_x", 0)

    if result == "ball":
        return f"Ball. {pt} at {speed:.0f} mph, outside the zone."
    if result == "called_strike":
        return f"Called strike. {pt} at {speed:.0f} mph, caught the corner."
    if result == "swinging_strike":
        return f"Swinging strike. {pt} at {speed:.0f} mph, batter missed."
    if result == "foul":
        return f"Foul. {pt} at {speed:.0f} mph, fouled off."
    if result == "foul_tip":
        return f"Foul tip. {pt} at {speed:.0f} mph, caught by catcher (strike)."
    if result == "hit_by_pitch":
        return "Hit by pitch. Batter takes first base."

    # In-play: direction from plate_x; distance/height from pitch data (deterministic, model doesn't return these)
    directions = ["left field", "left-center", "center field", "right-center", "right field"]
    idx = min(4, max(0, int((plate_x + 2) / 4 * 5)))
    direction = directions[idx]
    seed = abs(int(plate_x * 100) + int(speed) + sum(ord(c) for c in str(pt)))
    if result == "in_play_hr":
        dist = 380 + (seed % 70)
        height = 80 + (seed % 40)
        return f"HIT! Home run to {direction}! Distance: {dist} ft, apex: {height} ft."
    if result == "in_play_3b":
        dist = 320 + (seed % 60)
        height = 40 + (seed % 30)
        return f"HIT! Ball was hit to {direction}. Distance: {dist} ft, max height: {height} ft. Triple."
    if result == "in_play_2b":
        dist = 280 + (seed % 80)
        height = 25 + (seed % 30)
        return f"HIT! Ball was hit to {direction}. Distance: {dist} ft, max height: {height} ft. Double."
    if result == "in_play_1b":
        dist = 180 + (seed % 100)
        height = 15 + (seed % 30)
        return f"HIT! Ball was hit to {direction}. Distance: {dist} ft, max height: {height} ft. Single."
    if result == "in_play_out":
        dist = 180 + (seed % 140)
        height = 10 + (seed % 35)
        return f"Out. Ball was hit to {direction}. Distance: {dist} ft, max height: {height} ft. Caught."
    return _result_to_message(result)


# Full names for pitch type abbreviations (for table display)
PITCH_TYPE_FULL_NAMES = {
    "FF": "Four-seam fastball",
    "FT": "Two-seam fastball",
    "FC": "Cutter",
    "SI": "Sinker",
    "FS": "Splitter",
    "SL": "Slider",
    "CH": "Changeup",
    "CU": "Curveball",
    "KC": "Knuckle curve",
    "ST": "Sweeper",
    "SV": "Slurve",
    "GY": "Gyro",
    "UN": "Unknown",
}


def _pitch_type_full_name(abbrev: str) -> str:
    return PITCH_TYPE_FULL_NAMES.get(abbrev, abbrev.replace("_", " ").title())


def _get_hit_or_foul_stats(entry: dict) -> dict | None:
    """If this pitch was a hit (in_play_*) or foul, return dict with direction, distance_ft, max_height_ft for display. Else None."""
    result = entry["result"]
    pitch = entry["pitch"]
    plate_x = pitch.get("plate_x", 0)
    speed = pitch.get("release_speed", 90)
    pt = pitch.get("pitch_type", "?")
    directions = ["left field", "left-center", "center field", "right-center", "right field"]
    idx = min(4, max(0, int((plate_x + 2) / 4 * 5)))
    direction = directions[idx]
    seed = abs(int(plate_x * 100) + int(speed) + sum(ord(c) for c in str(pt)))

    if result == "foul":
        return {"label": "Foul", "direction": direction, "distance_ft": None, "max_height_ft": None}
    if result == "in_play_hr":
        return {"label": "Home run", "direction": direction, "distance_ft": 380 + (seed % 70), "max_height_ft": 80 + (seed % 40)}
    if result == "in_play_3b":
        return {"label": "Triple", "direction": direction, "distance_ft": 320 + (seed % 60), "max_height_ft": 40 + (seed % 30)}
    if result == "in_play_2b":
        return {"label": "Double", "direction": direction, "distance_ft": 280 + (seed % 80), "max_height_ft": 25 + (seed % 30)}
    if result == "in_play_1b":
        return {"label": "Single", "direction": direction, "distance_ft": 180 + (seed % 100), "max_height_ft": 15 + (seed % 30)}
    if result == "in_play_out":
        return {"label": "Out", "direction": direction, "distance_ft": 180 + (seed % 140), "max_height_ft": 10 + (seed % 35)}
    if result == "hit_by_pitch":
        return None
    return None


# Distinct colors per pitch type for the strike zone plot
PITCH_TYPE_COLORS = {
    "FF": "#e41a1c",
    "SI": "#377eb8",
    "FC": "#4daf4a",
    "SL": "#984ea3",
    "CH": "#ff7f00",
    "CU": "#ffff33",
    "ST": "#a65628",
    "FS": "#f781bf",
    "KC": "#999999",
    "SV": "#66c2a5",
    "GY": "#fc8d62",
    "UN": "#8da0cb",
}


def _pitch_type_color(pt: str) -> str:
    return PITCH_TYPE_COLORS.get(pt, "#333333")


def _is_terminal_outcome(result: str, new_state: dict, old_outs: int) -> bool:
    """True if this outcome ends the plate appearance."""
    if result in ("hit_by_pitch", "in_play_out", "in_play_1b", "in_play_2b", "in_play_3b", "in_play_hr"):
        return True
    if result == "ball" and new_state.get("balls", 0) == 0 and new_state.get("strikes", 0) == 0:
        return True  # walk
    if result in ("called_strike", "swinging_strike", "foul_tip") and new_state.get("strikes", 0) == 0 and new_state.get("outs", 0) > old_outs:
        return True  # strikeout
    return False


@st.cache_resource
def _get_engine():
    return eval_utils.get_engine()


@st.cache_resource
def _load_models():
    """Load pitcher and batter models. Pitcher generates pitch; batter (calibrated + BIP + final outcome) consumes it."""
    return eval_utils.load_models_and_encoders(), eval_utils.load_batter_models()


@st.cache_data(ttl=300)
def _load_pitcher_list(_engine):
    """Return (list of (pitcher_id, display_name)), dict pitcher_id -> p_throws."""
    q = text("""
        SELECT DISTINCT s.pitcher, s.p_throws,
               COALESCE(pm.name_last || ', ' || pm.name_first, 'Pitcher ' || s.pitcher::text) AS display_name
        FROM clean_statcast_with_batter s
        LEFT JOIN player_map pm ON s.pitcher = pm.key_mlbam
        WHERE s.game_type = 'R' AND s.pitcher IS NOT NULL
        ORDER BY display_name
    """)
    df = pd.read_sql(q, _engine)
    if len(df) == 0:
        return [], {}
    options = [(int(row["pitcher"]), str(row["display_name"])) for _, row in df.iterrows()]
    p_throws = {int(row["pitcher"]): str(row["p_throws"]) if pd.notna(row["p_throws"]) else "R" for _, row in df.iterrows()}
    return options, p_throws


@st.cache_data(ttl=300)
def _load_batter_list(_engine):
    """Return list of (batter_id, display_name) and dict batter_id -> {stand, career_AB, ...}."""
    q = text("""
        SELECT batter,
               MAX(batter_name_first) AS name_first,
               MAX(batter_name_last)  AS name_last,
               MAX(stand)             AS stand,
               MAX(career_AB)         AS career_AB,
               MAX(career_H)          AS career_H,
               MAX(career_HR)         AS career_HR,
               MAX(career_BA)         AS career_BA,
               MAX(career_OBP)        AS career_OBP,
               MAX(career_SLG)        AS career_SLG
        FROM clean_statcast_with_batter
        WHERE game_type = 'R' AND batter IS NOT NULL
        GROUP BY batter
        ORDER BY MAX(batter_name_last), MAX(batter_name_first)
    """)
    df = pd.read_sql(q, _engine)
    if len(df) == 0:
        return [], {}
    options = []
    batter_info = {}
    # PostgreSQL returns unquoted column names in lowercase
    for _, row in df.iterrows():
        bid = int(row.get("batter", 0))
        name_first = row.get("name_first", "") or ""
        name_last = row.get("name_last", "") or ""
        name = f"{name_last}, {name_first}".strip() or f"Batter {bid}"
        options.append((bid, name))
        career_ab = row.get("career_ab", row.get("career_AB", -1))
        career_h = row.get("career_h", row.get("career_H", -1))
        career_hr = row.get("career_hr", row.get("career_HR", -1))
        career_ba = row.get("career_ba", row.get("career_BA", -1))
        career_obp = row.get("career_obp", row.get("career_OBP", -1))
        career_slg = row.get("career_slg", row.get("career_SLG", -1))
        stand_val = row.get("stand", "R")
        batter_info[bid] = {
            "stand": str(stand_val) if pd.notna(stand_val) else "R",
            "career_AB": float(career_ab) if pd.notna(career_ab) else -1,
            "career_H": float(career_h) if pd.notna(career_h) else -1,
            "career_HR": float(career_hr) if pd.notna(career_hr) else -1,
            "career_BA": float(career_ba) if pd.notna(career_ba) else -1,
            "career_OBP": float(career_obp) if pd.notna(career_obp) else -1,
            "career_SLG": float(career_slg) if pd.notna(career_slg) else -1,
        }
    return options, batter_info


def _initial_game_state():
    return {"balls": 0, "strikes": 0, "outs": 0, "bases": [0, 0, 0], "runs": 0}


def _reset_session_for_ab():
    st.session_state.game_state = _initial_game_state()
    st.session_state.pitch_log = []
    st.session_state.pitch_history = []
    st.session_state.pa_ended = False
    st.session_state.last_pitch = None
    st.session_state.last_result = None


def main():
    st.set_page_config(page_title="Pitcher–Batter Simulator", layout="wide")
    st.title("Pitcher–Batter Simulator")

    engine = _get_engine()
    pitcher_models, batter_models = _load_models()
    pitcher_options, pitcher_p_throws = _load_pitcher_list(engine)
    batter_options, batter_info = _load_batter_list(engine)

    if not pitcher_options or not batter_options:
        st.warning("No pitchers or batters found in the database. Load Statcast data and run views.")
        return

    # Session state
    if "game_state" not in st.session_state:
        st.session_state.game_state = _initial_game_state()
    if "pitch_log" not in st.session_state:
        st.session_state.pitch_log = []
    if "pitch_history" not in st.session_state:
        st.session_state.pitch_history = []
    if "pa_ended" not in st.session_state:
        st.session_state.pa_ended = False
    if "last_pitch" not in st.session_state:
        st.session_state.last_pitch = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "selected_pitcher_id" not in st.session_state:
        st.session_state.selected_pitcher_id = None
    if "selected_batter_id" not in st.session_state:
        st.session_state.selected_batter_id = None

    # Dropdowns: display names, store ids
    pitcher_display = [name for _, name in pitcher_options]
    batter_display = [name for _, name in batter_options]
    pitcher_id_by_idx = {i: pid for i, (pid, _) in enumerate(pitcher_options)}
    batter_id_by_idx = {i: bid for i, (bid, _) in enumerate(batter_options)}

    col1, col2 = st.columns(2)
    with col1:
        pitcher_ix = st.selectbox(
            "Pitcher",
            range(len(pitcher_display)),
            format_func=lambda i: pitcher_display[i],
            key="pitcher_select",
        )
    with col2:
        batter_ix = st.selectbox(
            "Batter",
            range(len(batter_display)),
            format_func=lambda i: batter_display[i],
            key="batter_select",
        )

    pitcher_id = pitcher_id_by_idx[pitcher_ix]
    batter_id = batter_id_by_idx[batter_ix]

    # Reset at-bat when selection changes
    if st.session_state.selected_pitcher_id != pitcher_id or st.session_state.selected_batter_id != batter_id:
        st.session_state.selected_pitcher_id = pitcher_id
        st.session_state.selected_batter_id = batter_id
        _reset_session_for_ab()

    game_state = st.session_state.game_state
    pitch_log = st.session_state.pitch_log
    pitch_history = st.session_state.pitch_history
    pa_ended = st.session_state.pa_ended
    last_pitch = st.session_state.last_pitch
    last_result = st.session_state.last_result

    # Pitch button
    if st.button("Pitch", type="primary", disabled=pa_ended):
        pitcher_career = eval_utils.load_pitcher_career(engine, pitcher_id) or {}
        batter_career = batter_info.get(batter_id, {})
        stand = batter_career.get("stand", "R")
        p_throws = pitcher_p_throws.get(pitcher_id, "R")

        context = {
            "pitcher": pitcher_id,
            "p_throws": p_throws,
            "stand": stand,
            "balls": game_state["balls"],
            "strikes": game_state["strikes"],
            "inning": 1,
            "inning_topbot": "Top",
            "outs_when_up": 0,
            "at_bat_number": 1,
            "pitch_number": len(pitch_log) + 1,
            "previous_pitch_type": last_pitch.get("pitch_type", "__NA__") if last_pitch else "__NA__",
            "previous_release_speed": last_pitch.get("release_speed", 0.0) if last_pitch else 0.0,
            "pitcher_pitches_this_game": len(pitch_log),
            "pitcher_pitches_this_inning": len(pitch_log),
            "game_date": 0,
            "home_team": "NYY",
            "away_team": "BOS",
            "game_type": "R",
            "is_pitcher_count": 0,
            "is_batter_count": 0,
        }
        for k, v in (pitcher_career or {}).items():
            context[k] = v if v is not None else -1

        # Full pipeline: pitcher -> pitch; batter model -> immediate outcome (ball/strike/foul/hit);
        # on hit -> BIP -> physics -> final outcome model -> in_play_out/1b/2b/3b/hr; then game state updated.
        old_outs = game_state["outs"]
        pitch, result, new_state = eval_utils.simulate_at_bat_step(
            pitcher_models,
            batter_models,
            context,
            batter_id,
            batter_career,
            "NYY",
            game_state,
            prev_pitch=last_pitch,
            prev_result=last_result,
            pitch_history=pitch_history if pitch_history else None,
        )

        st.session_state.pitch_log = pitch_log + [{"pitch": pitch, "result": result, "state_after": new_state.copy()}]
        st.session_state.game_state = new_state
        st.session_state.last_pitch = pitch
        st.session_state.last_result = result
        _was_strike = 1 if result in ("called_strike", "swinging_strike", "foul", "foul_tip") else 0
        new_history = pitch_history + [(pitch["pitch_type"], _was_strike)]
        if len(new_history) > 3:
            new_history = new_history[-3:]
        st.session_state.pitch_history = new_history

        if _is_terminal_outcome(result, new_state, old_outs):
            st.session_state.pa_ended = True

        st.rerun()

    # Last outcome: clear, game-meaning message with color (red=out, green=on base, yellow=HR, grey=immediate)
    if pitch_log:
        last = pitch_log[-1]
        result = last["result"]
        pa_ended = st.session_state.pa_ended
        new_state = last.get("state_after") or st.session_state.game_state
        if pa_ended:
            short, detail = _terminal_game_message(result, new_state, last)
        else:
            short = _result_to_message(result)
            detail = _detailed_outcome_message(last)
        style = _outcome_style(result, pa_ended, new_state)
        if pa_ended:
            suffix = " At-bat over. Select a new pitcher or batter to throw again."
        else:
            suffix = ""
        msg = f"**{short}** — {detail}{suffix}"
        if style == "out":
            st.error(msg)
        elif style == "on_base":
            st.success(msg)
        elif style == "hr":
            st.warning(msg)
        else:
            st.markdown(f'<p style="color:#6b7280; margin:0;">**Last outcome:** {short} — {detail}</p>', unsafe_allow_html=True)

    # Balls/Strikes tally (above where the graph will be)
    st.subheader(f"Balls: {game_state['balls']}   Strikes: {game_state['strikes']}")

    # Pitch table: Type of ball (full name) and full outcome only
    if pitch_log:
        rows = []
        for i, entry in enumerate(pitch_log[-10:], start=max(1, len(pitch_log) - 9)):
            p = entry["pitch"]
            pt_abbrev = p.get("pitch_type", "UN")
            rows.append({
                "Type of ball": _pitch_type_full_name(pt_abbrev),
                "Outcome": _result_to_message(entry["result"]),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # Below table: last pitch ball stats (speed, spin rate)
        last = pitch_log[-1]
        p = last["pitch"]
        speed = p.get("release_speed")
        spin = p.get("release_spin_rate")
        if speed is not None or spin is not None:
            st.caption("**Ball stats (last pitch):**")
            parts = []
            if speed is not None:
                parts.append(f"Speed: {speed:.1f} mph")
            if spin is not None:
                parts.append(f"Spin rate: {spin:.0f} rpm")
            st.text("  ".join(parts))

        # Below ball stats: hit/foul stats when applicable
        hit_stats = _get_hit_or_foul_stats(last)
        if hit_stats:
            st.caption(f"**{hit_stats['label']} stats:**")
            line = f"Direction: {hit_stats['direction']}"
            if hit_stats.get("distance_ft") is not None:
                line += f"  ·  Distance: {hit_stats['distance_ft']} ft"
            if hit_stats.get("max_height_ft") is not None:
                line += f"  ·  Max height: {hit_stats['max_height_ft']} ft"
            st.text(line)

    # Strike zone plot at the bottom: only visible after at least one pitch, smaller and narrower
    if pitch_log:
        strike_results = {"called_strike", "swinging_strike", "foul_tip"}
        hit_foul_results = {"foul", "hit_by_pitch", "in_play_out", "in_play_1b", "in_play_2b", "in_play_3b", "in_play_hr"}

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        zone_width = 0.95
        zone_z_lo, zone_z_hi = 1.5, 3.5
        ax.add_patch(mpatches.Rectangle(
            (-zone_width, zone_z_lo), 2 * zone_width, zone_z_hi - zone_z_lo,
            fill=False, edgecolor="gray", linewidth=1.5,
        ))
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0.5, 4.5)
        ax.set_xlabel("plate_x (ft)")
        ax.set_ylabel("plate_z (ft)")
        ax.set_title("Pitch locations")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

        legend_handles = []
        for i, entry in enumerate(pitch_log):
            px = entry["pitch"]["plate_x"]
            pz = entry["pitch"]["plate_z"]
            res = entry["result"]
            pt = entry["pitch"].get("pitch_type", "UN")
            color = _pitch_type_color(pt)
            full_name = _pitch_type_full_name(pt)
            outcome = _result_to_message(entry["result"])
            label = f"Pitch {i + 1}: {full_name} ({outcome})"
            if res in strike_results:
                ax.scatter(px, pz, marker="x", s=60, c=color, linewidths=2)
            elif res in hit_foul_results:
                ax.scatter(px, pz, marker="o", s=45, facecolors="none", edgecolors=color, linewidths=2)
            else:
                ax.scatter(px, pz, marker="o", s=35, c=color, alpha=0.7)
            legend_handles.append(mpatches.Patch(facecolor=color, label=label))

        ax.legend(handles=legend_handles, loc="upper right", fontsize=6)

        # Place graph at bottom in a narrow column so it doesn't dominate
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
