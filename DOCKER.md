# PostgreSQL via Docker

Docker Compose for Postgres lives in the **project root**. Run all `docker compose` commands from there.

## Quick start

1. **Start Postgres** (from project root):

   ```bash
   docker compose up -d
   ```

2. **Optional – use .env for DB connection**

   Copy `.env.example` to `.env` in the project root (or in `baseball_data/` if your scripts load it from there). Defaults: `localhost:5432`, user `postgres`, password `postgres`, database `baseball`.

3. **Schema + views (automatic on first run)**  
   On first start, the container runs `baseball_data/sql/schema.sql` and `baseball_data/sql/views.sql` via `docker-entrypoint-initdb.d`. The `baseball` database is created with tables and views.

4. **Load Lahman CSV data** (from `baseball_data/`):

   ```bash
   cd baseball_data
   python -m scripts.lahman --csv --data-dir "data/lahman_1871-2025_csv"
   ```

## Useful commands

| Command | Description |
|--------|-------------|
| `docker compose up -d` | Start Postgres in background |
| `docker compose down` | Stop container (volume kept) |
| `docker compose down -v` | Stop and remove compose‑managed volumes (pgdata is **external**, so it is **not** removed) |
| `docker compose logs -f postgres` | Stream Postgres logs |

**Volume:** Data is stored in the external volume `baseball_data_pgdata`. To remove it (and wipe the DB), run `docker volume rm baseball_data_pgdata` after `docker compose down`.

## Connection

- **Host:** `localhost`
- **Port:** `5432`
- **User:** `postgres`
- **Password:** `postgres`
- **Database:** `baseball`

Scripts (e.g. lahman loader) use `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE` or `DATABASE_URL`. Set them in `.env` or your environment.
