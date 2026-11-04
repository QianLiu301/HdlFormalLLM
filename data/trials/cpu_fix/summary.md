# Trial summary

| model | design | runs | reached explanations | checks | pass | design wrong | scenario wrong | other | steps read | skipped rows | clear at start | clear at end | design repairs | compile fixes | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek | cpu | 3 | 1 | 32 | 94% | 6% | 0% | 0% | 100% | 1 | 0/1 | 0/1 | 2 | — | design |
| gemini | cpu | 3 | 0 | 0 | — | — | — | — | — | 0 | 0/0 | 0/0 | 0 | — | design |

Problems shown at the first simulation (how many runs):

- [design] The design executes LW wrong: 1
- [test] Partly tested: “Load and store, one instruction per row”: 1
