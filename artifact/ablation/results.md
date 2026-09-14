# Ablation: which parts of a `.feature` file actually drive testbench generation

One BDD feature file (32-bit ALU, 67 scenarios) is progressively stripped of
its natural-language content. After each step the testbench is regenerated and
hashed. If the hash is unchanged, that part of the feature file carried no
information into the testbench.

## Hashing convention

`sha256_body` is computed after stripping `//` comments and blank lines, because
the generator stamps a timestamp into the header comment on every run. Two runs
that agree on `sha256_body` produced byte-identical testbench *code*.
`sha256_full` is over the raw file and is included only for completeness.

## Results

| Variant | What was removed | tests | unmapped | sha256_body (first 16) | vs. baseline |
|---|---|---|---|---|---|
| A | nothing (baseline) | 67 | 0 | `26671546061159f5` | **identical** |
| B | step text blanked; tags, titles and Examples tables kept | 67 | 0 | `26671546061159f5` | **identical** |
| C | @tags removed; step text kept | 67 | 0 | `26671546061159f5` | **identical** |
| D | @tags removed **and** step text blanked | 67 | 0 | `26671546061159f5` | **identical** |
| E | @tags removed, step text blanked, scenario titles blanked | 67 | 0 | `644398ea72d584b6` | **DIFFERENT** |
| F | scenario titles blanked; @tags kept | 67 | 0 | `26671546061159f5` | **identical** |

## Which variant differs

Exactly one variant differs from the baseline: **E**. Variants B, C, D, F are byte-identical to A.

Reading the chain in order:

- **B** (step text blanked, tags kept) = A → the `Given/When/Then` prose is not read.
- **C** (tags removed, step text kept) = A → the tags alone are not the only source.
- **D** (tags removed *and* step text blanked) = A → even with both gone the
  generator still recovers the operation, because the scenario *title* still
  names it.
- **E** (tags + step text + titles all blanked) ≠ A → only now does the generator
  lose the mapping.
- **F** (titles blanked, tags kept) = A → the tag is sufficient on its own.

So the generator resolves each scenario through a fallback chain
`@tag → step text → scenario title`, and any single surviving link reproduces the
baseline testbench exactly. The Examples table (the stimulus values and expected
results) is never blanked in any variant and is what supplies the actual test
vectors.

## Full hashes

| Variant | sha256_body | sha256_full |
|---|---|---|
| A | `26671546061159f50a13db38b4b9d72c299807a822a367c34562208d53a50a19` | `4e5c304e2f8d7549507d051fd5addaf28e515addaa96143add59d64c25cfe9de` |
| B | `26671546061159f50a13db38b4b9d72c299807a822a367c34562208d53a50a19` | `4e5c304e2f8d7549507d051fd5addaf28e515addaa96143add59d64c25cfe9de` |
| C | `26671546061159f50a13db38b4b9d72c299807a822a367c34562208d53a50a19` | `4e5c304e2f8d7549507d051fd5addaf28e515addaa96143add59d64c25cfe9de` |
| D | `26671546061159f50a13db38b4b9d72c299807a822a367c34562208d53a50a19` | `816b900b8b09c56de77f37f37f923f4575b0ad9d88b0a14ce3d4d1c3ee76f704` |
| E | `644398ea72d584b6b1b2a85611c3bd7e77054b6422462c54dc29070ed4022dbe` | `472eacdb12a09cdad6ddfd860e3857d005b6c3e554801b30b97e97a674778e35` |
| F | `26671546061159f50a13db38b4b9d72c299807a822a367c34562208d53a50a19` | `816b900b8b09c56de77f37f37f923f4575b0ad9d88b0a14ce3d4d1c3ee76f704` |
