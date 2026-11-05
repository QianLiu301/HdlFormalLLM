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

## 5. Explanations on the website

With `DATABASE_URL` set, Step 4 opens with a **Problems** card a few seconds
after each simulation (the run is written to the database in the background
first): "No problems found" or a list of problems in plain words, each with
who is at fault and a **Details** button, plus one line on what was checked.
Everything below it — the former sections — is folded into **Technical
details**:

- a headline and counts per verdict (design wrong, scenario wrong, …);
- a specification check: for each flag the scenarios state (zero, negative,
  overflow), how many rows contradict the rule — flags are not compared by the
  testbench, so this is the only place such errors show up;
- a **Why?** button on every scenario row, every requirement and every flag
  line, opening the explanation;
- the flag probe: the testbench compares only `result`, so a second short
  simulation (same stimulus, same design) records the flags and checks them
  against the rules. It never changes the pass rate;
- requirement status by the output a requirement is about: a requirement on
  the zero flag is verified only if the probe saw the design's zero flag
  right, not because the result checks passed.

Migration 0004 is needed for the probe: run `python -m src.xdb upgrade` after
pulling.

For ALU designs, Step 0 now always asks what "overflow" means; the answer is
used by both generators and by the rule library.

### Repair (M4)

Below the problems the card offers what can be done about them:

- **Correct the scenarios** — when scenarios state values the reference
  model disagrees with. The values are set to the reference values, saved as
  a new feature file (`<name>_fix1.feature`; the old file is kept), and the
  testbench is generated and simulated again. No model is called. Step 2 and
  Step 3 then show the new files.
- **Regenerate the design with <LLM>** — when the design fails checks (ALU,
  generated on this page). The model that wrote the design gets the failing
  checks with the correct values, the rules, and the current design, and
  writes a new version (a new file; the old one is kept). The testbench is
  then generated and simulated again. This calls the model's API.

Both ask for confirmation first, show a note on what was changed above the
new result, and are stored in the database as feedback with the new file as
the next version (`reason = feedback:<id>`).

### Register files (M5a)

The same card works for register-file designs (Step 1 module type
"Register File"). The testbench runs the Examples rows in order and keeps the
register contents between rows, so each explanation says where the value of a
read comes from — which example wrote it, or the reset that cleared it — and,
when a scenario expects an old value, which later write replaced it. Read
port 2 is never read by the testbench; the card says so under "What was
checked". "Regenerate the design" is available for register files too.

To try it without a model: `python tests/ui/stage_run.py regfile` writes a
sample feature and a design with a register-0 bug under `output/`.

### RISC-V CPUs (M5b)

For CPU designs each example is judged the way the CPU executes it, and the
card separates three kinds of problem: the design computes an instruction
wrong (with a hint such as "seems to compare as unsigned numbers"), the
scenario expects a wrong value or contradicts itself (one register given two
values, a result expected in x0), and examples the testbench skips, with the
reason (an instruction that uses other registers than its columns, x29/x30,
unaligned addresses, two examples using the same memory word). Branches are
shown as taken / not taken. `python tests/ui/stage_run.py cpu` stages a
sample.

Browser checks (developers; app running with `DATABASE_URL`, iverilog):

```bash
mkdir -p /tmp/ui && python tests/ui/stage_run.py > /tmp/ui/state.json    # or: stage_run.py regfile|cpu
```

```bash
NODE_PATH=$(npm root -g) node tests/ui/repair_flow.js /tmp/ui
```

The staging script writes files named `xdb_ui_*` under `output/`; delete
them afterwards if you like (`output/` is not in git).

## 6. Batch trials (many runs without the browser)

To find out how the website copes with real model output, run the whole flow
many times: Step 0 (defaults accepted), Steps 1–4, the explanations, then the
repairs the Problems card offers (up to two rounds). Every run is stored in the
database like a web run; its session is marked `trial:<tag>`.

First a dry run with the built-in mock model (no API key, no cost, about 15 s):

```bash
python -m src.xdb.trial run --llms mock --tag dryrun
```

Then real models, for example two providers, all three design types, two runs each:

```bash
python -m src.xdb.trial run --llms deepseek,gemini --designs alu,regfile,cpu --reps 2 --tag first
```

Read the result:

```bash
open output/trials/first/summary.md
```

`output/trials/<tag>/` holds `summary.md` (per model × design type: how far
the runs got, the verdict shares, how many scenario steps the rule library
could read, how many runs were clear before and after repair, the problems
seen and the steps that could not be read), `runs.csv` (one row per run),
`runs.jsonl` (everything, per run), `logs/` (the server output of each run)
and `server.log`. An interrupted trial continues where it stopped when the
same command is run again. Options: `--reps N`, `--repair N` (rounds, 0 =
none), `--no-step0`, `--model gemini=gemini-2.5-flash`, `--designs alu`.
A real run makes about 4–8 model calls (Step 0, design, scenarios, one or two
design repairs) and stores a few hundred KB in the database.

A design that does not compile, or whose ports the testbench cannot connect,
is sent back to the model with the compiler's messages (up to `--compile-fix N`
attempts, default 2; 0 = off), as the buttons "Fix the compile error with …"
(Step 1) and "Ask … to fix the design" (Step 4) do on the page. To see that
path without an API key: `--llms mock --mock-fault compile` (or `interface`).

To try the buttons in the browser without an API key, start the website with
the mock model instead of `python main.py`, choose "Mock (no API key)" in
Steps 0–2 (`compile`: the first design of each type does not compile;
`interface`: its ports differ from the testbench's):

```bash
python tests/ui/mock_server.py compile
```

To share a trial (results and every generated file) through git:

```bash
python -m src.xdb.trial export --tag first --to data/trials/first/files
```

## 6b. Keeping keys out of git

Once per clone, let git refuse commits that contain credentials:

```bash
git config core.hooksPath .githooks
```

The hook (`tools/check_secrets.py`) refuses `.env*`, `config/llm_config.json`
and any added line that holds a key configured on this machine or looks like
one. To check every tracked file: `python tools/check_secrets.py --all`.
Trial outputs are redacted when they are written, and the server redacts its
error messages. Still, add only what you mean to share
(`git add data/trials/<tag>`) and look at `git status` before committing.

If a key ever reaches GitHub, deleting the file is not enough (it stays in the
history): revoke the key at the provider and put the new one into
`config/llm_config.json` or `.env`.

## 6a. Explanations in the terminal

The same explanations can be printed:

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

## 7. Render

Render dashboard → service → **Environment** → add `DATABASE_URL` with the
same pooled connection string. Run the migration once from your machine
(step 3) before the first deploy that uses it.

## 8. Running the tests (developers)

The tests need a PostgreSQL they may wipe (never point them at Neon) and
iverilog:

```bash
XDB_TEST_URL=postgresql://USER:PASS@localhost/bdx_test python -m pytest tests -q
```

`tests/conftest.py` keeps every test away from the `.env` database. Without
`XDB_TEST_URL` pointing at a running server, `tests/test_xdb.py` is skipped.

On a Mac (Homebrew), a throwaway server that needs no background service:

```bash
brew install postgresql@17 pgvector
```

```bash
LC_ALL=en_US.UTF-8 /opt/homebrew/opt/postgresql@17/bin/initdb -D /tmp/bdx_pg -U bdx --auth=trust -E UTF-8 --locale=en_US.UTF-8
```

```bash
LC_ALL=en_US.UTF-8 /opt/homebrew/opt/postgresql@17/bin/pg_ctl -D /tmp/bdx_pg -w -l /tmp/bdx_pg/server.log -o "-p 5544 -c listen_addresses=127.0.0.1 -k /tmp" start
```

```bash
/opt/homebrew/opt/postgresql@17/bin/createdb -h 127.0.0.1 -p 5544 -U bdx bdx_test
```

```bash
XDB_TEST_URL=postgresql://bdx@127.0.0.1:5544/bdx_test python -m pytest tests -q
```

```bash
/opt/homebrew/opt/postgresql@17/bin/pg_ctl -D /tmp/bdx_pg stop
```

`LC_ALL` is needed: without it the server stops at startup with "postmaster
became multithreaded during startup".

## Notes on Neon's free plan

Region is fixed at project creation (ours: AWS Frankfurt). The compute
suspends after about 5 minutes without traffic; the first request afterwards
takes a moment longer, which is why the website writes in the background.
Storage is limited (0.5 GB per project per Neon's docs at the time of
writing); waveforms are never stored, only KB-sized text artifacts.
