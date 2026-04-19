"""
Time-ordered train/val/test splits for pitch-level data.

Several seasons: train on all but the last two years, val on the second-to-last year, test on
the last. Two seasons: train on the first year, val+test from the second by date order.
One season: cut ordered rows into train/val/test. If year is missing, falls back to a random
stratified split (same train_frac / val_frac as sklearn train_test_split).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _ensure_year_column(df: pd.DataFrame, date_col: str, year_col: str) -> pd.DataFrame:
    out = df.copy()
    if year_col not in out.columns and date_col in out.columns:
        out[year_col] = pd.to_datetime(out[date_col], errors="coerce").dt.year
    return out


def sort_chronologically(df: pd.DataFrame, date_col: str = "game_date", year_col: str = "game_year") -> pd.DataFrame:
    """Stable time ordering: year, date, game_pk, at_bat_number, pitch_number when available."""
    d = _ensure_year_column(df, date_col, year_col)
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if year_col in d.columns:
        sort_cols.append(year_col)
        ascending.append(True)
    if date_col in d.columns:
        sort_cols.append(date_col)
        ascending.append(True)
    for c in ("game_pk", "at_bat_number", "pitch_number"):
        if c in d.columns:
            sort_cols.append(c)
            ascending.append(True)
    if not sort_cols:
        return d.reset_index(drop=True)
    return d.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)


def temporal_train_val_test(
    df: pd.DataFrame,
    *,
    date_col: str = "game_date",
    year_col: str = "game_year",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return train, val, test, and meta (split_kind, years or reason, row counts)."""
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1.0:
        raise ValueError("train_frac and val_frac must be positive and train_frac + val_frac < 1.0")

    d = _ensure_year_column(df, date_col, year_col)
    meta: dict[str, Any] = {}

    if year_col not in d.columns or d[year_col].isna().all():
        meta["split_kind"] = "random_fallback"
        meta["reason"] = f"missing {year_col} and could not derive from {date_col}"
        from sklearn.model_selection import train_test_split

        strat = d.get("pitch_type") or d.get("pitch_result")
        try:
            train_df, rest = train_test_split(d, train_size=train_frac, random_state=42, stratify=strat)
        except (ValueError, TypeError):
            train_df, rest = train_test_split(d, train_size=train_frac, random_state=42)
        val_ratio = val_frac / (1.0 - train_frac)
        try:
            val_df, test_df = train_test_split(rest, train_size=val_ratio, random_state=42, stratify=strat)
        except (ValueError, TypeError):
            val_df, test_df = train_test_split(rest, train_size=val_ratio, random_state=42)
        meta["n_train"], meta["n_val"], meta["n_test"] = len(train_df), len(val_df), len(test_df)
        return train_df, val_df, test_df, meta

    years = sorted(pd.Series(d[year_col]).dropna().astype(int).unique().tolist())
    meta["years_present"] = years

    if len(years) >= 3:
        val_year = years[-2]
        test_year = years[-1]
        train_df = d[d[year_col].astype(float) < val_year].copy()
        val_df = d[d[year_col] == val_year].copy()
        test_df = d[d[year_col] == test_year].copy()
        meta["split_kind"] = "multi_season"
        meta["val_year"] = int(val_year)
        meta["test_year"] = int(test_year)
    elif len(years) == 2:
        y0, y1 = years[0], years[1]
        train_df = d[d[year_col] == y0].copy()
        rest = d[d[year_col] == y1].copy()
        rest = sort_chronologically(rest, date_col=date_col, year_col=year_col)
        n = len(rest)
        if n == 0:
            val_df, test_df = rest, rest
        elif n == 1:
            val_df, test_df = rest.copy(), rest.iloc[0:0].copy()
        else:
            cut = int(n * train_frac)
            cut = min(max(cut, 1), n - 1)
            val_df = rest.iloc[:cut].copy()
            test_df = rest.iloc[cut:].copy()
        meta["split_kind"] = "two_season"
        meta["train_year"] = int(y0)
        meta["later_year"] = int(y1)
    else:
        d_sorted = sort_chronologically(d, date_col=date_col, year_col=year_col)
        n = len(d_sorted)
        t1 = int(n * train_frac)
        t2 = int(n * (train_frac + val_frac))
        t1 = min(max(t1, 1), n - 2) if n > 2 else max(1, min(t1, n))
        t2 = min(max(t2, t1 + 1), n - 1) if n > 1 else n
        train_df = d_sorted.iloc[:t1].copy()
        val_df = d_sorted.iloc[t1:t2].copy()
        test_df = d_sorted.iloc[t2:].copy()
        meta["split_kind"] = "single_season"

    meta["n_train"], meta["n_val"], meta["n_test"] = len(train_df), len(val_df), len(test_df)
    return train_df, val_df, test_df, meta
