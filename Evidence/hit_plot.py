#!/usr/bin/env python3
"""
Hit plot: launch_speed vs launch_angle, colored by Home Run vs other batted balls.

Evidence plot for HR prediction. Run from project root: python Evidence/hit_plot.py

PostgreSQL: set DATABASE_URL or PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dotenv import load_dotenv
from sqlalchemy import create_engine

_EVIDENCE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVIDENCE_DIR.parent


def _load_env():
    load_dotenv(_PROJECT_ROOT / ".env")


def _db_url():
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    dbname = os.getenv("PGDATABASE")
    pw = quote_plus(password) if password else ""
    return f"postgresql://{user}:{pw}@{host}:{port}/{dbname}"

def main():
    _load_env()
    engine = create_engine(_db_url())

    df = pd.read_sql(
        """
        SELECT launch_speed, launch_angle, events
        FROM clean_statcast_with_batter
        WHERE launch_speed IS NOT NULL AND launch_angle IS NOT NULL
        """,
        engine,
    )
    df["is_home_run"] = df["events"] == "home_run"

    jp = sns.jointplot(
        data=df,
        x="launch_speed",
        y="launch_angle",
        hue="is_home_run",
        kind="scatter",
        alpha=0.4,
    )
    jp.set_axis_labels("Launch Speed (mph)", "Launch Angle (deg)")
    jp.fig.suptitle("Launch Speed vs Launch Angle: Home Runs vs Other Batted Balls", y=1.02)

    out_path = _EVIDENCE_DIR / "hit_plot_launch_ev.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
