# Explainability database — setup

Design: [`explainability_db_design.md`](explainability_db_design.md).
Code: `src/xdb/` (models, Alembic migrations, ingestion, queries).
Tests: `tests/test_xdb.py`.

The database is **optional**. Without `DATABASE_URL` the website behaves
exactly as before. With it, every Verilog design (Step 1) and feature file
(Step 2) is stored as soon as it is generated, and every Step-4 run report
(which the page requests after each simulation) adds the testbench, the
simulation results and a verdict per assertion (who is wrong: design or
scenario). All writes happen in background threads; a database error never
breaks the page.

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
applies what is missing. `stats` ends with a storage line (database size
against the 512 MB free-plan limit; about 7–8 MB of it is PostgreSQL's own
baseline).

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

## 5. Explanations in the terminal

Until the website shows them (milestone M3), explanations can be printed:

```bash
python -m src.xdb runs
```

lists recent simulations with their verdict counts.

```bash
python -m src.xdb explain --failures
```

explains every failing check of the latest simulation (`--run N` for another).

```bash
python -m src.xdb explain ALU-SUB-001/r7
```

explains one check (any row, passing or failing).

```bash
python -m src.xdb explain --global overflow=True
```

answers "when does the specification say overflow = 1?" for the latest
feature (`--feature N` for another).

```bash
python -m src.xdb explain --requirement REQ-2
```

summarises one Step-0 requirement.

Runs stored before this version have no step mappings yet; map them once:

```bash
python -m src.xdb remap
```

## 6. Render

Render dashboard → service → **Environment** → add `DATABASE_URL` with the
same pooled connection string. Run the migration once from your machine
(step 3) before the first deploy that uses it.

## 7. Running the tests (developers)

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
