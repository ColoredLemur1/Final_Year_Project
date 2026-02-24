#!/usr/bin/env python3
"""
Type ball plot: hit frequency by pitch type (FF, SL, CH, etc.).

Grouped bar chart: Home Run, Hit, Out per pitch type. Run from project root: python Evidence/type_ball.py

PostgreSQL: set DATABASE_URL or PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import pandas as pd

try:
    from dotenv import load_dotenv
    from sqlalchemy import create_engine
except ImportError as e:
    print(f"Missing dependency: {e}. Install with: pip install pandas matplotlib sqlalchemy python-dotenv", file=sys.stderr)
    raise SystemExit(1) from e

_EVIDENCE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVIDENCE_DIR.parent

# Map Statcast pitch_type codes to readable labels
PITCH_LABELS = {
    "FF": "Four-Seam",
    "SI": "Sinker",
    "SL": "Slider",
    "CH": "Changeup",
    "CU": "Curveball",
    "FC": "Cutter",
    "ST": "Sweeper",
    "FS": "Splitter",
    "SV": "Slurve",
    "KC": "Knuckle Curve",
}


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


def _outcome_category(events):
    """Map events to outcome category for grouping."""
    if pd.isna(events):
        return "Other"
    e = str(events).lower()
    if e == "home_run":
        return "Home Run"
    if e in ("single", "double", "triple"):
        return "Hit"
    if "out" in e or e in ("grounded_into_double_play", "field_error"):
        return "Out"
    return "Other"


def main():
    _load_env()
    engine = create_engine(_db_url())

    df = pd.read_sql(
        """
        SELECT pitch_type, events
        FROM clean_statcast_with_batter
        WHERE pitch_type IS NOT NULL AND pitch_type != ''
          AND launch_speed IS NOT NULL AND launch_angle IS NOT NULL
        """,
        engine,
    )
    df["outcome"] = df["events"].apply(_outcome_category)

    # Pivot: count per (pitch_type, outcome)
    counts = df.groupby(["pitch_type", "outcome"], dropna=False).size().unstack(fill_value=0)
    counts.index = counts.index.map(lambda x: PITCH_LABELS.get(x, x) if x else "Other")

    # Order: Home Run, Hit, Out, Other
    outcome_order = ["Home Run", "Hit", "Out", "Other"]
    cols = [c for c in outcome_order if c in counts.columns]
    counts = counts[cols] if cols else counts

    ax = counts.plot(kind="bar", stacked=False, rot=45, alpha=0.85)
    ax.set_xlabel("Pitch Type")
    ax.set_ylabel("Count (Batted Balls)")
    ax.set_title("Hit Frequency by Pitch Type: Home Run vs Hit vs Out")
    ax.legend(title="Outcome")
    plt.tight_layout()

    out_path = _EVIDENCE_DIR / "type_ball_hit_frequency.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
