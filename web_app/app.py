#!/usr/bin/env python3
"""Home page — navigation hub for the Baseball Simulator app."""

import streamlit as st

st.set_page_config(
    page_title="Baseball Simulator",
    layout="wide",
)

st.title("Baseball Simulator")
st.caption("ML-powered pitcher vs batter matchup analysis")

st.write("")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div style="background:#1f2937;border:1px solid #374151;border-radius:10px;
                    padding:1.6rem;text-align:center;">
            <div style="color:#fff;font-size:1.1rem;font-weight:600;">Player Stats</div>
            <div style="color:#9ca3af;font-size:0.85rem;margin-top:0.4rem;">
                Radar profiles for every batter and pitcher
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div style="background:#1f2937;border:1px solid #374151;border-radius:10px;
                    padding:1.6rem;text-align:center;">
            <div style="color:#fff;font-size:1.1rem;font-weight:600;">Simulator</div>
            <div style="color:#9ca3af;font-size:0.85rem;margin-top:0.4rem;">
                Pitch-by-pitch at-bat simulation
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.caption("Navigate using the sidebar.")
