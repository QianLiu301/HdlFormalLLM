# Artifact: LLM-Generated BDD Scenarios for Hardware Verification

Anonymized replication package. Everything here is machine-generated output from
the experiment runner plus the prompt templates that produced it. No API keys,
author names, institution names, or local filesystem paths are included; see
[Anonymization](#anonymization) for what was removed.

```
artifact/
├── README.md                  this file
├── results.csv                320 rows — the main experiment matrix
├── regfile_results.csv        16 rows  — the oracle-isolation experiment
├── ablation/                  feature-file ablation (6 variants + results.md)
└── prompts/                   all prompt templates, by stage and version
```

---

## 1. The main experiment matrix

`results.csv` has one row per run. The matrix is fully crossed:

| Factor | Levels | n |
|---|---|---|
| Workflow | `implementation`, `specification` | 2 |
| Design | `counter`, `alu`, `alu_comb`, `cpu` | 4 |
| Model | 8 (see below) | 8 |
| Seed | 1–5 | 5 |

2 × 4 × 8 × 5 = **320 rows**, and all 320 cells are present — there are no
missing cells and no placeholder rows.

### The two workflows

Both workflows produce a DUV (design under verification) and a testbench, and
both are scored by the same simulator. They differ in *what the DUV generator is
shown*:

- **`implementation`** — the DUV is generated from the natural-language
  specification alone. The BDD feature file is generated separately and the DUV
  generator never sees it. (`duv_prompt_has_bdd = 0`)
- **`specification`** — the BDD feature file is generated first and is included
  in the DUV generation prompt. (`duv_prompt_has_bdd = 1`)

This is the manipulation the paper's RQ1 tests. The split is exactly 160/160.

### The four designs

| `module_type` | What it is | `prompt_version` |
|---|---|---|
| `counter` | parameterized up/down counter with load and enable | v1 |
| `alu` | 32-bit ALU, registered (clocked) output | v1 |
| `alu_comb` | same ALU, purely combinational output | v2 |
| `cpu` | 5-stage RV32I pipeline with forwarding and hazard handling | v1 |

`alu` and `alu_comb` are the same functional specification differing only in
whether the result is registered; they exist to separate "sequential logic is
hard" from "this design is large". `prompt_version` refers to the file under
`prompts/duv_<design>/`. Note that `prompts/duv_alu/v3.yaml` is present in this
package but was **not** used for any row in `results.csv` — it is a later
revision used by the interactive tool, included here only for completeness.

### The eight models

`gpt-5-mini`, `gpt-5.1`, `codestral-latest`, `mistral-medium-3-5`,
`deepseek-v4-flash`, `deepseek-v4-pro`, `qwen3-coder-next`,
`meta-llama/Llama-3.3-70B-Instruct-Turbo`.

Each contributes 40 rows (4 designs × 5 seeds × 2 workflows).

Sampling parameters were constant across every row: `step1_temp = 0.1` (DUV
generation), `step2_temp = 0.7` (BDD scenario generation).

---

## 2. Column reference for `results.csv`

### Identification

| Column | Meaning |
|---|---|
| `run_id` | Unique id for this run. Contains a UTC timestamp and a random suffix. |
| `workflow` | `implementation` or `specification` — see above. |
| `module_type` | `counter`, `alu`, `alu_comb`, or `cpu`. |
| `provider` | API provider the model was served through. |
| `model_effective` | The model identifier **requested** from the provider. See the caveat below. |
| `seed` | Repetition index 1–5, passed to the provider as a sampling seed where the API supports one. |
| `step1_temp`, `step2_temp` | Sampling temperature for DUV generation and for BDD scenario generation. |
| `prompt_version` | Which template version under `prompts/duv_<design>/` was used. |
| `duv_prompt_has_bdd` | 1 if the BDD feature file was included in the DUV prompt (i.e. `workflow = specification`). |

> **Caveat on `model_effective`.** This field records the model name that was
> *sent to the provider*, not a name the provider echoed back. Providers can
> silently re-point an alias such as `-latest` to a different underlying
> checkpoint, so this column should be read as "what was asked for", not as a
> verified served version. Two of these identifiers are aliases
> (`codestral-latest`, `qwen3-coder-next`) and are therefore not pinned to a
> fixed checkpoint.

### Pipeline stage outcomes

Each row walks a fixed pipeline. A stage can only be attempted if the previous
one succeeded, so a `0` in an early column forces `0` in later ones.

| Column | 1 means |
|---|---|
| `duv_success` | The DUV generation call returned usable Verilog (not an API error, not an empty or truncated response). |
| `duv_compile` | The generated DUV **compiles and elaborates** under Icarus Verilog. |
| `bdd_success` | The BDD scenario generation call returned a parseable `.feature` file. |
| `tb_success` | A testbench was produced from the feature file. |
| `tb_compile` | The testbench compiles together with the DUV. |
| `sim_run` | Simulation **executed to completion** (simulator exit status 0). |
| `sim_all_pass` | **Every** assertion in the testbench passed. |

| Column | Meaning |
|---|---|
| `sim_pass_rate` | Percentage of assertions that passed, 0–100. Defined whenever `sim_run = 1`. |
| `duv_attempts`, `bdd_attempts` | Number of API calls made for that stage, including retries after transient errors. `1` means it succeeded first try. |
| `tokens_in`, `tokens_out`, `latency_ms` | Totals across every API call made for the row. |
| `duv_sha256`, `bdd_sha256`, `tb_sha256` | SHA-256 of the generated DUV, feature file, and testbench. Use these to check whether two rows produced identical artifacts. |
| `duv_error`, `sim_error` | First few lines of the compiler/simulator diagnostic, truncated to 400 characters, with filesystem paths replaced. Empty when the stage succeeded. |

### `sim_run` vs `sim_all_pass` — the distinction the paper turns on

These two columns are **not** the same measurement, and conflating them is the
main way published numbers in this area become incomparable.

- `sim_run = 1` means the simulator *ran*. The design compiled, the testbench
  compiled, and the simulation reached the end without the simulator itself
  aborting.
- `sim_all_pass = 1` means the design was actually *correct* with respect to
  every assertion the testbench checked.

The generated testbenches report assertion failures and keep going; they do not
abort the simulation on a mismatch. **A testbench in which every single
assertion fails still yields `sim_run = 1`.** So `sim_run` is close to a
compile-success metric, while `sim_all_pass` is a functional-correctness metric.

In this dataset the gap is large:

| | count |
|---|---|
| `sim_run = 0` (never got to simulate) | 75 |
| `sim_run = 1`, `sim_all_pass = 0` | **130** |
| `sim_run = 1`, `sim_all_pass = 1` | 115 |

130 of the 320 rows — 41% — would be scored as successes by a "it simulated"
metric and as failures by a correctness metric. Any comparison against prior
work must state which of the two it is using. `sim_pass_rate` lets you see how
near-miss each of those 130 rows was.

---

## 3. `regfile_results.csv` — the oracle-isolation experiment

16 rows: 8 models × 2 workflows × 1 seed, on a register-file design.

This experiment exists to answer a threat to validity: when a test fails, is the
*design* wrong or is the *expected value* wrong? The same stimulus is run
against two independently derived oracles:

- **`*_bdd_arm`** — expected values come from the LLM-generated BDD scenarios.
- **`*_spec_arm`** — expected values are recomputed from the specification by a
  reference model written independently of the LLM output.

Both arms drive identical stimulus. A row that fails the BDD arm but passes the
spec arm indicates a **faulty oracle** (the LLM wrote a wrong expected value),
not a faulty design. A row that fails both indicates a genuine design defect.

Columns mirror `results.csv`, with the simulation outcome split per arm:
`sim_run_bdd_arm`, `pass_rate_bdd_arm`, `all_pass_bdd_arm` and the corresponding
`*_spec_arm` triple.

In this set all 16 designs elaborate and all 16 rows pass the spec arm, while
**5 fail the BDD arm** (pass rates 76.0–96.9). Every failure observed on this
design class is therefore attributable to a wrong expected value in the
generated scenarios, not to the design.

---

## 4. `ablation/` — what in a feature file actually matters

Six variants of one 32-bit ALU feature file (67 scenarios): the original plus
five progressively stripped versions. Each variant's testbench is regenerated
and hashed; an unchanged hash means the removed content carried no information.

See `ablation/results.md` for the table and the full hashes. Summary: exactly
one of the five variants changes the generated testbench.

---

## 5. `prompts/`

Every prompt template, organized as `<stage>_<design>/<version>.yaml`:

```
prompts/duv_alu/{v1,v2,v3}.yaml     prompts/bdd_alu/v1.yaml
prompts/duv_counter/v1.yaml         prompts/bdd_counter/v1.yaml
prompts/duv_cpu/v1.yaml             prompts/bdd_cpu/v1.yaml
prompts/duv_register/v1.yaml        prompts/bdd_regfile/v1.yaml
```

`duv_*` templates generate the design; `bdd_*` templates generate the BDD
scenarios. Placeholders are written `{{ name }}` and are substituted at run time
with the design parameters (bit width, module name, port list). Templates are
reproduced verbatim — no rewording, since the wording is the experimental
condition.

---

## 6. Known limitations

**46 of the 320 rows were not produced in the original batch run.** They were
regenerated later, under tooling that differs from the first pass. Grouped by
cause:

| n | When | Why | Which rows |
|---|---|---|---|
| 13 | +1 day | A token-budget defect truncated long responses; the budget was raised and these rows regenerated. | `deepseek-v4-pro` on `alu_comb` and `counter` (all 5 seeds each, `specification`), plus `deepseek-v4-flash` / `counter` / `specification` seeds 2, 3, 5. |
| 30 | +6 days | The originally used Mistral model was retired by the provider mid-study. All of its rows were regenerated with `mistral-medium-3-5`. | `mistral-medium-3-5` on `alu`, `alu_comb`, `counter` (10 rows each). |
| 3 | +6 days | Transient API read-timeouts left three cells with no generated design. Re-requested with the same parameters. | `codestral-latest` / `cpu` / `implementation` seeds 4–5; `deepseek-v4-flash` / `cpu` / `specification` seed 3. |

The `cpu` design class (80 rows) was run six days after the other three. The
runner, prompts, and scoring code for the other three designs were unchanged
between the two dates, but this is a temporal confound worth stating: provider-
side model behavior can drift over that interval, and two of the eight model
identifiers are unpinned aliases.

**A ninth model was excluded.** One additional model was run on three of the
four designs but not on `cpu`, so it does not form a complete cell in the
matrix. Its 30 rows are omitted here rather than reported as a partial row.

**Seeds do not guarantee reproducibility.** Not every provider API honors a
sampling seed, and those that do make no cross-version determinism guarantee.
Re-running these prompts will not reproduce the SHA-256 values in the CSVs. The
hashes are for checking equality *within* this dataset, not for reproducing it.

**Single simulator.** All compile and simulation results come from Icarus
Verilog. A design rejected here might elaborate under a commercial simulator
with different strictness, particularly for the SystemVerilog constructs that
appear in the failure data.

---

## Anonymization

The following were removed or checked for across every file in this package:
absolute filesystem paths (replaced with `<REPO>` / `<PATH>`), user and host
names, API keys and bearer tokens, author names, institution names, email
addresses, and repository URLs. The `duv_error` and `sim_error` columns are the
only free-text fields carried over from tooling output; they were path-scrubbed
and truncated. `run_id` values retain a UTC timestamp, which is what links a row
to the rerun groups described above.
