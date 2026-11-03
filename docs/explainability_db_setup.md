# Explainability database — setup

Design: [`explainability_db_design.md`](explainability_db_design.md).
Code: `src/xdb/` (models, Alembic migrations, ingestion, queries).
Tests: `tests/test_xdb.py`.

The database is **optional**. Without `DATABASE_URL` the website behaves
exactly as before. With it, every Step-4 run report (which the page requests
after each simulation) is also written to the database in a background
thread; a database error never breaks the page.

---

## 1. Connection string

Neon console → project → **Connection Details** → copy the **pooled**
connection string (host contains `-pooler` and `eu-central-1`).

Put it in a file named `.env` in the project root (this file is gitignored):

```
DATABASE_URL=postgresql://USER:PASSWORD@ep-xxxx-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require
```

Never commit it, never paste it into chats or issues. If it leaks, reset the
password in the Neon console.

## 2. Install dependencies

```bash
pip install -r requirements-core.txt
```

(adds SQLAlchemy, psycopg2-binary, alembic).

## 3. Create the tables

```bash
python -m src.xdb upgrade
```

This also runs `CREATE EXTENSION IF NOT EXISTS vector` (pgvector). Check:

```bash
python -m src.xdb current
```

```bash
python -m src.xdb stats
```

Run `upgrade` again whenever a new migration is pulled from git; it only
applies what is missing.

## 4. Use the website as usual

```bash
python main.py
```

Go through Steps 0–4. After the simulation the terminal prints a line like

```
🗄️  explainability DB: {'ingested': True, 'new': True, 'scenarios_new': 10, 'checks': 67, ...}
```

and `python -m src.xdb stats` shows the new rows.

Runs that were simulated earlier can be added by hand:

```bash
python -m src.xdb ingest --testbench output/testbench/<llm>/alu_tb.v --log output/simulation/<file>.log --bdd output/bdd/<llm>/<file>.feature --duv output/dut/<llm>/<file>.v --run-id <run_id>
```

Only testbenches generated after scenario tagging (they have a
`*_tb.scenarios.json` next to them) can be ingested.

## 5. Render

Render dashboard → service → **Environment** → add `DATABASE_URL` with the
same pooled connection string. Run the migration once from your machine
(step 3) before the first deploy that uses it.

## 6. Running the tests (developers)

The tests need a PostgreSQL they may wipe (never point them at Neon) and
iverilog:

```bash
XDB_TEST_URL=postgresql://USER:PASS@localhost/bdx_test python -m pytest tests/test_xdb.py -q
```

## Notes on Neon's free plan

Region is fixed at project creation (ours: AWS Frankfurt). The compute
suspends after about 5 minutes without traffic; the first request afterwards
takes a moment longer, which is why the website writes in the background.
Storage is limited (0.5 GB per project per Neon's docs at the time of
writing); waveforms are never stored, only KB-sized text artifacts.
