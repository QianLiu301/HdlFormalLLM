# Email to Professor — Research Direction Proposal

---

**Subject:** Research Direction Proposal: LLM-Based BDD for Hardware Verification — Next Steps & Target Venues

---

Dear Professor [Name],

I hope this email finds you well. I would like to share an update on my research progress and propose the next steps for our LLM-based hardware verification project. I have also attached a research roadmap diagram for your reference.

## 1. Current Progress

I have developed a web-based platform (**HdlFormalLLM**) that supports an end-to-end LLM-driven hardware verification pipeline:

- **Multi-LLM support**: Integration with 9 LLM providers (GPT, Gemini, Claude, Groq, DeepSeek, Mistral, Qwen, Together AI, xAI)
- **Hardware generation**: Automated Verilog generation for ALU, Counter, Register File, and RISC-V CPU modules from natural language specifications
- **BDD test generation**: LLM-driven Behavior-Driven Development (BDD) scenario generation from hardware specifications
- **Testbench synthesis**: Automated conversion from BDD scenarios to executable Verilog testbenches
- **Quality analysis**: Custom quality metrics including functional coverage, input space coverage, test uniqueness, and corner case detection
- **Simulation & synthesis**: Integration with Icarus Verilog (iverilog) and Yosys for simulation execution and synthesis analysis
- **Bug detection**: Interface-level bug detection with severity classification (CRITICAL/WARNING)

The platform is functional and deployed, but currently lacks the rigorous experimental framework needed for publication.

## 2. Literature Review — Key Findings

After conducting an extensive survey of 2024–2026 publications, I identified the following key observations:

1. **The field is growing rapidly**: LLM mentions in hardware design literature grew from 12 (2023) to 274+ (2025), a >20x increase. Major venues (DAC, ICCAD, ASP-DAC, DATE) now have dedicated LLM tracks.

2. **BDD for hardware is a blue ocean**: Only one paper directly addresses LLM-based BDD for hardware — *"LLM-based Behaviour Driven Development for Hardware Design"* (University of Bremen & DFKI, arXiv: 2512.17814). This confirms both the novelty and timeliness of our direction.

3. **The field is moving toward closed-loop systems**: State-of-the-art work like *LLM4Cov* (VTS 2026) uses simulation feedback to iteratively improve testbench coverage. *VeriReason* (NeurIPS 2025) applies reinforcement learning with testbench rewards for RTL generation.

4. **Standardized benchmarks are becoming essential**: *CVDP* (NVIDIA, 783 problems), *FVEval* (UC Berkeley + NVIDIA), and *VerilogEval-v2* have established evaluation norms. Papers without rigorous benchmarks face increasing difficulty at top venues.

5. **Multi-agent architectures are emerging**: *VeriMind* (5-agent system), *RTLSquad*, and *CircuitLM* demonstrate that multi-agent collaboration improves design quality.

## 3. Proposed Research Plan

Based on the literature analysis and our existing infrastructure, I propose a **phased approach** with three progressive stages:

### Phase 1: BDD-Hardware Benchmark & Multi-LLM Evaluation (Months 1–3)
**Priority: HIGH — This is our most differentiated contribution**

- **Construct a BDD-Hardware Benchmark Suite**: 50–100 hardware modules spanning diverse types (ALU, FSM, FIFO, bus protocols, pipeline stages), each with reference specifications, BDD scenarios, testbenches, and DUTs
- **Define BDD-specific quality metrics**:
  - Scenario Completeness: coverage of specification functional points
  - BDD-to-Testbench Conversion Rate: successful transformation ratio
  - Simulation Pass Rate: correctness on golden DUTs
  - Mutation Score: bug detection rate on fault-injected DUTs (leveraging our existing bug taxonomy)
- **Systematic multi-LLM comparison**: Controlled experiments across all 9 providers with fixed prompts, temperature settings, and ≥5 repetitions per configuration; statistical significance testing (Wilcoxon test)
- **Deliverable**: A benchmark dataset (published on HuggingFace/Zenodo) + evaluation framework

### Phase 2: Simulation-Feedback-Driven Iterative BDD Generation (Months 3–5)
**Priority: MEDIUM — Adds technical depth**

- After initial BDD → Testbench → Simulation, feed coverage reports and failure logs back to the LLM
- LLM generates supplementary BDD scenarios to address coverage gaps
- Measure the coverage improvement curve across iterations
- Compare with direct testbench generation (AutoBench-style) to demonstrate the structural advantage of BDD as an intermediate representation

### Phase 3: Multi-Agent BDD Verification Pipeline (Months 5–8)
**Priority: EXPLORATORY — Higher risk, higher reward**

- Design a multi-agent architecture with specialized roles:
  - Spec Agent: Extract functional points from natural language specifications
  - BDD Agent: Generate behavioral scenarios
  - Testbench Agent: Synthesize executable Verilog testbenches
  - Simulation Agent: Execute and analyze simulation results
  - Reviewer Agent: Identify coverage gaps and drive iteration
- Evaluate against single-LLM baseline to quantify collaboration benefits

## 4. Target Venues (for discussion)

| Venue | Deadline (typical) | Fit | Notes |
|-------|-------------------|-----|-------|
| **ASP-DAC 2026** | Jul–Aug 2025 | High | Dedicated LLM track; Phase 1 results |
| **DATE 2026** | Sep 2025 | High | Verification methodology focus |
| **ISVLSI 2026** | Mar 2026 | Medium | Phase 1+2 combined results |
| **VTS 2026** | Nov 2025 | Medium | Testing/verification automation |
| **MLCAD / ICLAD** | Varies | Medium | Workshop paper for early results |
| **TCAD** (journal) | Rolling | High | Full system paper (Phase 1+2+3) |

I would appreciate your advice on which venue(s) to prioritize and whether the proposed timeline is realistic given the upcoming deadlines.

## 5. Immediate Next Steps

If you agree with this direction, I plan to start with the following tasks this month:

1. Curate the first batch of benchmark modules (20 modules across 5 categories)
2. Implement experiment logging infrastructure (recording all LLM calls, parameters, and outputs)
3. Design the mutation testing framework (systematic bug injection into reference DUTs)
4. Run pilot experiments with 3 LLMs on 10 modules to validate the evaluation methodology

I have attached a **research roadmap diagram** that visualizes the overall plan. I would be happy to discuss this in our next meeting.

Best regards,
[Your Name]

---

*Attachments: Research Roadmap Diagram (see roadmap.html)*
