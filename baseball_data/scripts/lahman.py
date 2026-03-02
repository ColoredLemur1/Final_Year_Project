#!/usr/bin/env python3
"""
Lahman loader: migrates the Lahman CSV files into the PostgreSQL database.

Reads People.csv, Teams.csv, Batting.csv, Pitching.csv, Fielding.csv from data

PostgreSQL: set DATABASE_URL or PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

_base = Path(__file__).resolve().parent.parent
load_dotenv(_base / ".env")
load_dotenv(_base.parent / ".env")

_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _BASE_DIR / "data" / "lahman_1871-2025_csv"
_BATCH = 5000

# CSV uses 2B/3B; schema uses "X2B"/"X3B". Map when reading.
_COL_ALIAS = {"2B": "X2B", "3B": "X3B"}
_REV_ALIAS = {"X2B": "2B", "X3B": "3B"}
_QUOTED_COLS = {"rank", "X2B", "X3B"}  # rank: reserved word, stored lowercase


def _norm(s):
    return (s or "").strip()


def _empty_to_none(s):
    t = _norm(s)
    return None if t == "" else t


def _int_or_none(s):
    t = _empty_to_none(s)
    if t is None:
        return None
    try:
        return int(float(t))
    except (ValueError, TypeError):
        return None


def _float_or_none(s):
    t = _empty_to_none(s)
    if t is None:
        return None
    try:
        return float(t)
    except (ValueError, TypeError):
        return None


def _cap_float(v: float | None, lo: float, hi: float):
    if v is None:
        return None
    return max(lo, min(hi, float(v)))


def _date_or_none(s):
    t = _empty_to_none(s)
    if t is None:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(t[:10], fmt).date() if len(t) >= 10 else datetime.strptime(t, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _get(row, key, conv: Callable[[str], Any] = _empty_to_none):
    v = row.get(key)
    if v is None and key in _REV_ALIAS:
        v = row.get(_REV_ALIAS[key])
    if v is None and key in _COL_ALIAS:
        v = row.get(_COL_ALIAS[key])
    if v is None:
        return conv("")
    return conv(v)



def _people_rows(path: Path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            yield (
                _get(row, "playerID") or "",
                _get(row, "birthYear", _int_or_none),
                _get(row, "birthMonth", _int_or_none),
                _get(row, "birthDay", _int_or_none),
                _get(row, "birthCountry"),
                _get(row, "birthState"),
                _get(row, "birthCity"),
                _get(row, "deathYear", _int_or_none),
                _get(row, "deathMonth", _int_or_none),
                _get(row, "deathDay", _int_or_none),
                _get(row, "deathCountry"),
                _get(row, "deathState"),
                _get(row, "deathCity"),
                _get(row, "nameFirst"),
                _get(row, "nameLast"),
                _get(row, "nameGiven"),
                _get(row, "weight", _int_or_none),
                _get(row, "height", _int_or_none),
                _get(row, "bats"),
                _get(row, "throws"),
                _get(row, "debut", _date_or_none),
                _get(row, "finalGame", _date_or_none),
                _get(row, "retroID"),
                _get(row, "bbrefID"),
            )


def _teams_rows(path: Path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            lg = (_get(row, "lgID") or "")[:2]
            div = (_get(row, "divID") or "")[:1] or None
            tid = (_get(row, "teamID") or "")[:3]
            fid = (_get(row, "franchID") or "")[:3] or None
            yield (
                _get(row, "yearID", _int_or_none) or 0,
                lg,
                tid,
                fid,
                div,
                _get(row, "Rank", _int_or_none),
                _get(row, "G", _int_or_none),
                _get(row, "Ghome", _int_or_none),
                _get(row, "W", _int_or_none),
                _get(row, "L", _int_or_none),
                _get(row, "DivWin"),
                _get(row, "WCWin"),
                _get(row, "LgWin"),
                _get(row, "WSWin"),
                _get(row, "R", _int_or_none),
                _get(row, "AB", _int_or_none),
                _get(row, "H", _int_or_none),
                _get(row, "X2B", _int_or_none),
                _get(row, "X3B", _int_or_none),
                _get(row, "HR", _int_or_none),
                _get(row, "BB", _int_or_none),
                _get(row, "SO", _int_or_none),
                _get(row, "SB", _int_or_none),
                _get(row, "CS", _int_or_none),
                _get(row, "HBP", _int_or_none),
                _get(row, "SF", _int_or_none),
                _get(row, "RA", _int_or_none),
                _get(row, "ER", _int_or_none),
                _get(row, "ERA", _float_or_none),
                _get(row, "CG", _int_or_none),
                _get(row, "SHO", _int_or_none),
                _get(row, "SV", _int_or_none),
                _get(row, "IPouts", _int_or_none),
                _get(row, "HA", _int_or_none),
                _get(row, "HRA", _int_or_none),
                _get(row, "BBA", _int_or_none),
                _get(row, "SOA", _int_or_none),
                _get(row, "E", _int_or_none),
                _get(row, "DP", _int_or_none),
                _get(row, "FP", _float_or_none),
                _get(row, "name"),
                _get(row, "park"),
                _get(row, "attendance", _int_or_none),
                _get(row, "BPF", _int_or_none),
                _get(row, "PPF", _int_or_none),
                _get(row, "teamIDBR"),
                _get(row, "teamIDlahman45"),
                _get(row, "teamIDretro"),
            )


def _batting_rows(path: Path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            tid = (_get(row, "teamID") or "")[:3]
            lg = (_get(row, "lgID") or "")[:2]
            yield (
                _get(row, "playerID") or "",
                _get(row, "yearID", _int_or_none) or 0,
                _get(row, "stint", _int_or_none) or 1,
                tid,
                lg,
                _get(row, "G", _int_or_none),
                _get(row, "AB", _int_or_none),
                _get(row, "R", _int_or_none),
                _get(row, "H", _int_or_none),
                _get(row, "X2B", _int_or_none),
                _get(row, "X3B", _int_or_none),
                _get(row, "HR", _int_or_none),
                _get(row, "RBI", _int_or_none),
                _get(row, "SB", _int_or_none),
                _get(row, "CS", _int_or_none),
                _get(row, "BB", _int_or_none),
                _get(row, "SO", _int_or_none),
                _get(row, "IBB", _int_or_none),
                _get(row, "HBP", _int_or_none),
                _get(row, "SH", _int_or_none),
                _get(row, "SF", _int_or_none),
                _get(row, "GIDP", _int_or_none),
            )


def _pitching_rows(path: Path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            tid = (_get(row, "teamID") or "")[:3]
            lg = (_get(row, "lgID") or "")[:2]
            yield (
                _get(row, "playerID") or "",
                _get(row, "yearID", _int_or_none) or 0,
                _get(row, "stint", _int_or_none) or 1,
                tid,
                lg,
                _get(row, "W", _int_or_none),
                _get(row, "L", _int_or_none),
                _get(row, "G", _int_or_none),
                _get(row, "GS", _int_or_none),
                _get(row, "CG", _int_or_none),
                _get(row, "SHO", _int_or_none),
                _get(row, "SV", _int_or_none),
                _get(row, "IPouts", _int_or_none),
                _get(row, "H", _int_or_none),
                _get(row, "ER", _int_or_none),
                _get(row, "HR", _int_or_none),
                _get(row, "BB", _int_or_none),
                _get(row, "SO", _int_or_none),
                _cap_float(_get(row, "BAOpp", _float_or_none), 0, 9.999),
                _cap_float(_get(row, "ERA", _float_or_none), 0, 999.99),
                _get(row, "IBB", _int_or_none),
                _get(row, "WP", _int_or_none),
                _get(row, "HBP", _int_or_none),
                _get(row, "BK", _int_or_none),
                _get(row, "BFP", _int_or_none),
                _get(row, "GF", _int_or_none),
                _get(row, "R", _int_or_none),
                _get(row, "SH", _int_or_none),
                _get(row, "SF", _int_or_none),
                _get(row, "GIDP", _int_or_none),
            )


def _fielding_rows(path: Path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            tid = (_get(row, "teamID") or "")[:3]
            lg = (_get(row, "lgID") or "")[:2]
            pos = (_get(row, "POS") or "")[:2]
            yield (
                _get(row, "playerID") or "",
                _get(row, "yearID", _int_or_none) or 0,
                _get(row, "stint", _int_or_none) or 1,
                tid,
                lg,
                pos,
                _get(row, "G", _int_or_none),
                _get(row, "GS", _int_or_none),
                _get(row, "PO", _int_or_none),
                _get(row, "A", _int_or_none),
                _get(row, "E", _int_or_none),
                _get(row, "DP", _int_or_none),
                _get(row, "PB", _int_or_none),
                _get(row, "WP", _int_or_none),
                _get(row, "SB", _int_or_none),
                _get(row, "CS", _int_or_none),
                _get(row, "ZR", _float_or_none),
            )



def _pg_conn():
    url = os.getenv("DATABASE_URL")
    if url:
        return psycopg2.connect(url)
    return psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        dbname=os.getenv("PGDATABASE"),
    )


def _truncate(conn):
    cur = conn.cursor()
    cur.execute(sql.SQL("TRUNCATE TABLE people, teams CASCADE"))
    conn.commit()
    cur.close()


def _csv_path(data_dir: Path, name: str):
    candidates = [data_dir / f"{name}.csv"]
    if name == "people":
        candidates.append(data_dir / "Master.csv")
    for p in candidates:
        if p.is_file():
            return p
    return None


def _insert_batch(cur, table: str, cols: list[str], rows: list[tuple[Any, ...]]):
    if not rows:
        return
    placeholders = ",".join(["%s"] * len(cols))
    quoted = [f'"{c}"' if c in _QUOTED_COLS else c for c in cols]
    stmt = f'INSERT INTO {table} ({",".join(quoted)}) VALUES ({placeholders})'
    cur.executemany(stmt, rows)


_TABLES = [
    ("people", "people", _people_rows, [
        "playerID", "birthYear", "birthMonth", "birthDay", "birthCountry", "birthState", "birthCity",
        "deathYear", "deathMonth", "deathDay", "deathCountry", "deathState", "deathCity",
        "nameFirst", "nameLast", "nameGiven", "weight", "height", "bats", "throws",
        "debut", "finalGame", "retroID", "bbrefID",
    ]),
    ("teams", "teams", _teams_rows, [
        "yearID", "lgID", "teamID", "franchID", "divID", "rank", "G", "Ghome", "W", "L",
        "DivWin", "WCWin", "LgWin", "WSWin", "R", "AB", "H", "X2B", "X3B", "HR", "BB", "SO",
        "SB", "CS", "HBP", "SF", "RA", "ER", "ERA", "CG", "SHO", "SV", "IPouts", "HA", "HRA",
        "BBA", "SOA", "E", "DP", "FP", "name", "park", "attendance", "BPF", "PPF",
        "teamIDBR", "teamIDlahman45", "teamIDretro",
    ]),
    ("batting", "batting", _batting_rows, [
        "playerID", "yearID", "stint", "teamID", "lgID", "G", "AB", "R", "H", "X2B", "X3B",
        "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP",
    ]),
    ("pitching", "pitching", _pitching_rows, [
        "playerID", "yearID", "stint", "teamID", "lgID", "W", "L", "G", "GS", "CG", "SHO", "SV",
        "IPouts", "H", "ER", "HR", "BB", "SO", "BAOpp", "ERA", "IBB", "WP", "HBP", "BK", "BFP",
        "GF", "R", "SH", "SF", "GIDP",
    ]),
    ("fielding", "fielding", _fielding_rows, [
        "playerID", "yearID", "stint", "teamID", "lgID", "POS", "G", "GS", "PO", "A", "E", "DP",
        "PB", "WP", "SB", "CS", "ZR",
    ]),
]


def _load(conn, data_dir: Path):
    counts: dict[str, int] = {}
    cur = conn.cursor()
    _truncate(conn)

    for csv_name, table, row_gen, cols in _TABLES:
        path = _csv_path(data_dir, csv_name)
        if not path:
            print(f"  skip {table}: missing {csv_name}.csv in {data_dir}", file=sys.stderr)
            continue
        batch: list[tuple[Any, ...]] = []
        n = 0
        for tup in row_gen(path):
            batch.append(tup)
            if len(batch) >= _BATCH:
                _insert_batch(cur, table, cols, batch)
                n += len(batch)
                batch = []
                conn.commit()
        if batch:
            _insert_batch(cur, table, cols, batch)
            n += len(batch)
        conn.commit()
        counts[table] = n
        print(f"  {table}: {n} rows")

    cur.close()
    return counts


def main():
    ap = argparse.ArgumentParser(description="Load Lahman CSV into PostgreSQL")
    ap.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR, help="Directory containing Lahman CSV files")
    args = ap.parse_args()
    data_dir = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"Data dir not found: {data_dir}")

    print("Connecting to PostgreSQL...")
    conn = _pg_conn()
    print("Truncating people, teams (CASCADE)...")
    print("Loading Lahman CSV...")
    counts = _load(conn, data_dir)
    conn.close()
    print("Done.")
    for t, c in counts.items():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
