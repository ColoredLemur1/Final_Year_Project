# An Analytical and Simulation Based Approach to Baseball Matchup Evaluation

A baseball matchup simulator that predicts the probable outcome of a plate appearance between any two MLB players. The system combines a five-stage machine-learning pipeline (pitcher → batter → launch regressor → physics engine → final outcome) with a Streamlit web interface, backed by a PostgreSQL database that integrates pitch-level Statcast telemetry with season-level Lahman career statistics.

This is the codebase for a BSc Computer Science Final Year Project (University of Leeds, 2025/26).

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Quick start](#quick-start)
3. [Step-by-step user manual](#step-by-step-user-manual)
   - [Step 1 — Prerequisites](#step-1--prerequisites)
   - [Step 2 — Database container](#step-2--database-container)
   - [Step 3 — Python environments](#step-3--python-environments)
   - [Step 4 — Data ingestion](#step-4--data-ingestion)
   - [Step 5 — Schema enrichment](#step-5--schema-enrichment)
   - [Step 6 — Training the pipeline](#step-6--training-the-pipeline)
   - [Step 7 — Running the web app](#step-7--running-the-web-app)
   - [Step 8 — Reproducing the evaluation figures](#step-8--reproducing-the-evaluation-figures)
4. [Pipeline architecture](#pipeline-architecture)
5. [Libraries and dependencies](#libraries-and-dependencies)
6. [Troubleshooting](#troubleshooting)

---

## Repository layout

```
Final_Year_Project/
├── baseball_data/          # Database schema and ingestion scripts
│   ├── data/               # Lahman CSV releases land here
│   ├── scripts/            # Loader scripts (lahman.py, statcast.py, …)
│   ├── sql/                # schema.sql, views.sql, multi-season SQL
│   └── requirements.txt
├── Models/
│   ├── Training/           # Training notebooks, data prep, physics engine
│   │   ├── training_pitchers.ipynb
│   │   ├── training_batters.ipynb
│   │   ├── outcome_regresor.ipynb
│   │   ├── final_outcome.ipynb
│   │   ├── physics_engine.py
│   │   ├── data_prep.py
│   │   ├── data_prep_batters.py
│   │   ├── batter_calibration.py
│   │   ├── temporal_split.py
│   │   └── saved_models/   # Trained artefacts (also mirrored at Models/saved_models/)
│   ├── Evaluation/         # Per-stage evaluation notebooks + shared inference utils
│   │   ├── ohtani_evaluation.ipynb
│   │   ├── batter_evaluation.ipynb
│   │   ├── outcome_evaluation.ipynb
│   │   └── utils.py        # Shared inference (used by web app and notebooks)
│   ├── saved_models/       # Pickled XGBoost models + JSON metadata
│   └── requirements.txt
├── web_app/
│   ├── app.py              # Streamlit landing page
│   ├── pages/              # Multi-page Streamlit views
│   │   ├── 1_Stats.py      # Career stats with radar profile
│   │   └── 2_Simulator.py  # Matchup simulator
│   └── requirements.txt
├── Evidence/               # SQL-backed exploratory figures
├── metrics/                # Evaluation-figure scripts and notebooks
├── report/                 # LaTeX report sources (report.tex)
├── docker-compose.yml      # PostgreSQL container
└── README.md               # This file
```

---

## Quick start

If you only want to run the simulator against the already-trained models:

```bash
# 1. Start the database
docker compose up -d

# 2. Install web app dependencies
cd web_app
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

Open <http://localhost:8501> in your browser. The simulator will not work without an ingested database — see [Step 4](#step-4--data-ingestion) below if the database is empty.

---

## Step-by-step user manual

### Step 1 — Prerequisites

Install the following on your host machine:

| Tool       | Version  | Purpose                                                  |
| ---------- | -------- | -------------------------------------------------------- |
| Docker     | ≥ 24.x   | Runs the PostgreSQL container                            |
| Python     | ≥ 3.10   | Runs the loaders, training notebooks, and Streamlit app  |
| Git        | any      | Clone the repository                                     |
| ~10 GB disk | —       | Database volume for three seasons of Statcast data       |

A virtual environment is strongly recommended (`python -m venv .venv && source .venv/bin/activate`).

Clone the repo:

```bash
git clone <repository-url> Final_Year_Project
cd Final_Year_Project
```

### Step 2 — Database container

The project uses PostgreSQL 16 inside Docker. The compose file mounts `baseball_data/sql/` into the container's init directory, so a fresh container automatically runs `schema.sql` and `views.sql` on first startup.

**Create the external volume once** (the volume is intentionally external so `docker compose down` cannot delete the ingested data):

```bash
docker volume create baseball_data_pgdata
```

**Start the container:**

```bash
docker compose up -d
```

Default credentials (set in `docker-compose.yml`):

| Variable              | Value     |
| --------------------- | --------- |
| `POSTGRES_USER`       | postgres  |
| `POSTGRES_PASSWORD`   | postgres  |
| `POSTGRES_DB`         | baseball  |
| Host port             | 5432      |

Optionally create a `.env` file at the project root so the loader scripts pick up these credentials automatically:

```dotenv
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=postgres
PGDATABASE=baseball
```

### Step 3 — Python environments

The project is split into four loosely coupled components, each with its own `requirements.txt`. You can install them all into one virtual environment, or use separate ones if you want to keep them isolated.

```bash
# Database / ingestion stack
pip install -r baseball_data/requirements.txt

# Modelling stack (training notebooks, evaluation notebooks, physics engine)
pip install -r Models/requirements.txt

# Web app stack
pip install -r web_app/requirements.txt

# Exploratory data analysis stack
pip install -r Evidence/requirements.txt
```

The `Models/` and `web_app/` stacks share the heavy ML libraries (XGBoost, scikit-learn). Installing both into the same environment is the simplest workflow.

### Step 4 — Data ingestion

The loader scripts pull from three external sources and write into the PostgreSQL container.

**4.1 Lahman season-level statistics**

Download the [Lahman 1871–2025 CSV release](https://github.com/chadwickbureau/baseballdatabank) and unzip it into `baseball_data/data/lahman_1871-2025_csv/`. Then:

```bash
cd baseball_data
python scripts/lahman.py
```

This populates `people`, `teams`, `batting`, `pitching`, `fielding`.

**4.2 Statcast pitch-level data (current season)**

```bash
python scripts/statcast.py --year 2025
```

Statcast ingestion is bandwidth-intensive — a single season is roughly 700,000 pitches and takes 20–40 minutes depending on connection. The loader uses `pybaseball` under the hood and respects Baseball Savant's rate limits.

**4.3 Statcast pitch-level data (previous seasons)**

For temporal validation you also need 2023 and 2024:

```bash
# Run the multi-season SQL once (creates per-season tables and clean views)
docker exec -i baseball_postgres psql -U postgres -d baseball \
  < baseball_data/sql/z_statcast_multiseason.sql

# Then ingest each season
python scripts/statcast_seasons.py --year 2023
python scripts/statcast_seasons.py --year 2024
```

**4.4 Player ID bridge (Chadwick Register)**

```bash
python scripts/unify_ids.py
```

This populates the `player_map` table that bridges Statcast `MLBAM` ids to Lahman `bbrefID`s. Without it, none of the cross-source joins will return rows.

**4.5 Plate-coordinate backfill (optional)**

Some older Statcast rows are missing `plate_x`/`plate_z`. Run the backfill script if your evaluation depends on those columns being populated:

```bash
python scripts/statcast_plate_backfill.py --year 2023
python scripts/statcast_plate_backfill.py --year 2024
```

### Step 5 — Schema enrichment

After ingestion is complete, apply two SQL patches that add columns the batter model expects:

```bash
docker exec -i baseball_postgres psql -U postgres -d baseball \
  < baseball_data/sql/add_spin_rate.sql

docker exec -i baseball_postgres psql -U postgres -d baseball \
  < baseball_data/sql/add_statcast_movement.sql
```

These add `release_spin_rate` and the Statcast pitch-movement columns (`pfx_x`, `pfx_z`, `spin_axis`) to the cleaned views.


### Step 6 — Training the pipeline

The five pipeline stages are trained in order. Each stage saves its artefacts to `Models/saved_models/` and writes a JSON metadata file alongside the pickle so the web app can load it without code changes.

Start a Jupyter server inside the project root:

```bash
jupyter lab
```

**Run the notebooks in this order:**

| Order | Notebook                                          | Stage                              |
| ----- | ------------------------------------------------- | ---------------------------------- |
| 1     | `Models/Training/training_pitchers.ipynb`         | Pitcher classifier + 4 regressors  |
| 2     | `Models/Training/training_batters.ipynb`          | Batter pitch-result classifier     |
| 3     | `Models/Training/outcome_regresor.ipynb`          | Launch-speed / launch-angle regressor |
| 4     | `Models/Training/final_outcome.ipynb`             | Two-stage final-outcome cascade    |

Each notebook can be re-run independently once its inputs exist on disk. The training pipeline runs offline in a few hours on a CPU; no GPU is required because XGBoost is the only learner.

**Saved artefacts** live in `Models/saved_models/`:

- `pitcher_*.json` — XGBoost models in native JSON format
- `pitcher_encoders.json` — categorical encoders
- `residual_stds.json` — per pitch-type residual standard deviations (noise injection at inference)
- `pitcher_repertoire.json` — per-pitcher pitch-type masks
- `batter_calibrated.joblib` — isotonic-calibrated classifier
- `bip_launch_regressor*.json/joblib` — launch regressor
- `final_outcome_*` — two-stage cascade and metadata

### Step 7 — Running the web app

With trained models in place:

```bash
cd web_app
streamlit run app.py
```

The app opens at <http://localhost:8501>. Two pages are available:

- **Stats** — career-statistics view with a five-axis radar profile (contact, power, patience, speed, durability).
- **Simulator** — pick any pitcher and any batter, then run a simulated plate appearance through the full pipeline. Each pitch displays its predicted type, location, exit velocity, launch angle, and trajectory; each plate appearance returns one of the final-outcome classes (single, double, home run, strikeout, walk, etc.).

The web app reads the same `Models/Evaluation/utils.py` inference functions the evaluation notebooks use, so a code fix in either place propagates to both.

### Step 8 — Reproducing the evaluation figures

The figures in the report's Results and Evaluation chapter are produced by:

- `Models/Evaluation/*.ipynb` — per-stage evaluation notebooks (Ohtani repertoire fidelity, batter classifier metrics, outcome cascade).
- `metrics/` — standalone scripts and notebooks that emit the league-average comparison tables, physics ablation tables, distribution overlays, and per-park residual charts. Outputs are written to `metrics/outputs/`.
- `Evidence/` — exploratory data analysis figures referenced in the Implementation chapter (pitch speed/spin distributions, plate-location heatmaps, hit-probability heatmaps).

To regenerate a single figure:

```bash
# Example: launch-regressor distribution overlay
cd metrics
jupyter execute launch_regressor_metrics.ipynb

# Example: physics-ablation distance table
python physics_ablation_distance_table.py
```

---

## Pipeline architecture

The simulator runs as five stages, each consuming the previous stage's output:

```
┌─────────────────┐
│  Pitcher stage  │  XGBoost classifier + 4 regressors
│                 │  → pitch_type, release_speed, release_spin_rate, plate_x, plate_z
└────────┬────────┘
         ▼
┌─────────────────┐
│  Batter stage   │  XGBoost classifier (calibrated, isotonic)
│                 │  → ball, called_strike, swinging_strike, foul, in_play, hit
└────────┬────────┘
         ▼   (only if in-play)
┌─────────────────┐
│ Launch regressor│  XGBoost regressors with residual-std noise injection
│                 │  → launch_speed (mph), launch_angle (deg)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Physics engine │  Explicit Euler, Δt = 0.005 s, drag + Magnus + ISA altitude
│                 │  → trajectory, hit_distance_ft
└────────┬────────┘
         ▼
┌─────────────────┐
│ Final outcome   │  Two-stage cascade: coarse → fine classifier
│                 │  → single, double, triple, home_run, field_out, …
└─────────────────┘
```

Each stage is documented in detail in the Implementation chapter of the report (`report/report.tex`).

---

## Libraries and dependencies

### Database and ingestion (`baseball_data/`)

| Library             | Purpose                                                                   |
| ------------------- | ------------------------------------------------------------------------- |
| `psycopg2-binary`   | PostgreSQL driver. Used directly by the Lahman loader for `COPY` ingest.  |
| `sqlalchemy`        | ORM and connection pooling. Used by the Statcast loaders and notebooks.   |
| `pybaseball`        | Wraps Baseball Savant and Lahman public endpoints behind a Python API.    |
| `pandas`            | DataFrame container for the `pybaseball` output before insertion.         |
| `numpy`             | Numerical primitives used by the cleaning steps.                          |
| `python-dotenv`     | Loads database credentials from a `.env` file at project root.            |
| `seaborn`, `matplotlib` | Used by ad-hoc inspection scripts during ingestion debugging.         |

### Modelling (`Models/`)

| Library             | Purpose                                                                                |
| ------------------- | -------------------------------------------------------------------------------------- |
| `xgboost`           | Gradient-boosted decision trees. The single learner used for every classifier and regressor in the pipeline. |
| `scikit-learn`      | `CalibratedClassifierCV` (isotonic calibration), `train_test_split`, scoring metrics, `LogisticRegression` baseline. |
| `optuna`            | Hyperparameter search. Drives the Bayesian optimisation in `training_pitchers.ipynb` and `training_batters.ipynb`. |
| `joblib`            | Pickle wrapper for saving/loading scikit-learn pipelines (calibrated classifiers, regressors). |
| `numpy`, `pandas`   | Feature matrices, label arrays, and all tabular plumbing.                              |
| `matplotlib`, `seaborn` | Reliability diagrams, confusion matrices, residual histograms in the evaluation notebooks. |
| `jupyter`, `ipykernel` | Notebook runtime for the training and evaluation workflows.                         |

The physics engine (`Models/Training/physics_engine.py`) is intentionally written in plain NumPy so it has no machine-learning dependency at all — it uses only `numpy` for vector arithmetic and explicit Euler integration.

### Web app (`web_app/`)

| Library             | Purpose                                                                          |
| ------------------- | -------------------------------------------------------------------------------- |
| `streamlit`         | Interactive UI framework. Provides `@st.cache_resource` for one-time model loads, multi-page navigation via the `pages/` convention, and reactive widgets. |
| `plotly`            | Interactive trajectory and zone plots in the simulator view.                     |
| `matplotlib`        | Radar profile chart on the stats page.                                           |
| `xgboost`, `scikit-learn`, `joblib` | Required to deserialise the trained models at app startup.       |
| `pandas`, `numpy`   | Feature-vector construction at inference time.                                   |
| `sqlalchemy`, `psycopg2-binary` | Read pitcher/batter lookups directly from the database.              |

### Exploratory data analysis (`Evidence/`, `metrics/`)

| Library             | Purpose                                                                          |
| ------------------- | -------------------------------------------------------------------------------- |
| `pandas`, `numpy`   | Aggregate Statcast pulls into the figures shown in the report.                   |
| `matplotlib`, `seaborn` | Pitch-speed distributions, hit-probability heatmaps, plate-location density. |
| `sqlalchemy`        | Read the `clean_statcast_with_batter` view directly without going through `pybaseball`. |

### Database

| Component                | Version          |
| ------------------------ | ---------------- |
| PostgreSQL               | 16-alpine        |
| Docker / docker-compose  | ≥ 24.x           |

---

## Troubleshooting

**`docker compose up` says the volume already exists**
Expected — the volume is declared `external: true` so it persists across container rebuilds. If you actually want a clean start, run `docker volume rm baseball_data_pgdata` first.

**`pybaseball` rate-limit errors during Statcast ingest**
Re-run the loader; it resumes by year. The `statcast_seasons.py` script logs progress so you can identify which game date it stopped on.

**`KeyError: 'spin_axis'` when training the batter model**
You skipped Step 5. Run `add_statcast_movement.sql` and re-load the cleaned views.

**Streamlit shows "Model not found" on startup**
The web app expects `Models/saved_models/` to contain the pickled artefacts. Either re-run the training notebooks (Step 6) or copy a pre-trained set into that directory.

**`docker exec` cannot find `psql`**
The container hasn't finished starting up. Wait for `docker compose ps` to show `healthy`, then retry.

**Out-of-memory during training**
The 11-class batter classifier with the full 2023–2025 feature matrix peaks around 8 GB of RAM. Switch to the 5-class or 4-class schema by changing the `TARGET_SCHEMA` constant near the top of `training_batters.ipynb`.

## Author

Javier Duarte Macias — BSc Computer Science, University of Leeds, Session 2025/26 (COMP3931 Individual Project).
