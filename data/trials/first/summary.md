# Trial summary

| model | design | runs | reached explanations | checks | pass | design wrong | scenario wrong | other | steps read | skipped rows | clear at start | clear at end | design repairs | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek | alu | 2 | 2 | 122 | 99% | 0% | 1% | 0% | 100% | 0 | 1/2 | 2/2 | 0 | — |
| deepseek | cpu | 2 | 1 | 66 | 92% | 8% | 0% | 0% | 98% | 6 | 0/1 | 0/1 | 1 | design does not compile |
| deepseek | regfile | 2 | 2 | 46 | 89% | 0% | 11% | 0% | 67% | 0 | 1/2 | 2/2 | 0 | — |
| gemini | alu | 2 | 2 | 77 | 92% | 3% | 5% | 0% | 99% | 0 | 1/2 | 2/2 | 1 | — |
| gemini | cpu | 2 | 0 | 0 | — | — | — | — | — | 0 | 0/0 | 0/0 | 0 | design does not compile |
| gemini | regfile | 2 | 2 | 104 | 90% | 0% | 10% | 0% | 82% | 7 | 0/2 | 2/2 | 0 | — |

Scenario steps the rule library could not read (up to 20 per run):

- `I attempt to write <data> to register <addr> with write enable deasserted`
- `I disable write and attempt to write <data> to register <addr>`
- `I perform the <Operation> operation with opcode <Opcode>`
- `I read register <addr> on port 2`
- `I read register <addr> on read port A`
- `I simultaneously write <data> to register <addr> and read register <addr>`
- `I simultaneously write <new_data> to register <addr> and read register <addr>`
- `I write <data> to register <addr> while reading register <addr> on port 1`
- `I write <data> to register <addr> with write enable asserted on the rising clock edge`
- `memory at that address should contain <expected>`
- `register 0 is hardwired to zero (REQ-7, REQ-8)`
- `register <addr> holds value <initial_data>`
- `register <addr> initially holds <initial_data>`
- `reset has completed after writes`
- `reset is active`
- `the register file has completed simultaneous operations`
- `the register file has two independent read ports and a single write port`
- `the register file holds all current writes`
- `the register file holds current writes`
- `the register file holds values from earlier writes`
- `the register file is under active reset`
- `the value in register <addr> should remain <expected>`
- `the value should still be <expected>`

Problems shown at the first simulation (how many runs):

- [coverage] A requirement is not tested by any scenario: 12
- [scenario] N scenario examples expect a wrong register value: 3
- [scenario] A scenario expects a wrong SRL result: 1
- [test] Not tested: “Read back first write to register N and other registers”: 1
- [design] The design executes LW wrong: 1
- [test] Not tested: “Writes to xN are discarded and xN always reads as N”: 1
- [design] The design computes SRA wrong: 1
- [scenario] A scenario expects a wrong AND result: 1
- [scenario] A scenario expects a wrong OR result: 1
- [scenario] A scenario expects a wrong SUB result: 1
- [scenario] A scenario expects a wrong XOR result: 1
- [test] Partly tested: “Verify ADD operation”: 1
- [test] Partly tested: “Verify SUB operation”: 1
- [test] Partly tested: “Verify AND operation”: 1
- [test] Partly tested: “Verify OR operation”: 1
- [test] Partly tested: “Verify XOR operation”: 1
- [test] Partly tested: “Verify SLL operation”: 1
- [test] Partly tested: “Verify SRL operation”: 1
- [test] Partly tested: “Verify SRA operation”: 1
- [test] Partly tested: “Verify SLT operation (Signed Less Than)”: 1
- [test] Partly tested: “Verify SLTU operation (Unsigned Less Than)”: 1
- [test] Not tested: “Distinguish SLT and SLTU behavior”: 1
- [test] Not tested: “Read-before-write behavior (simultaneous read/write)”: 1
- [coverage] A requirement was not simulated: 1
