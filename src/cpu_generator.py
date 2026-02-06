"""
RISC-V CPU Generator - Generate Pipelined RISC-V CPU using LLM
===============================================================

Generates a 5-stage pipelined RISC-V CPU with:
- RV32I instruction subset support
- 5-stage pipeline: IF, ID, EX, MEM, WB
- Data hazard handling with forwarding
- Control hazard handling with branch prediction
- 32 x 32-bit register file (x0 hardwired to 0)

Part of the Hardware Generator Pipeline.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List


class CPUGenerator:
    """
    Generate RISC-V CPU Verilog design using LLM.
    """

    def __init__(
        self,
        llm_provider: str = 'groq',
        output_dir: Optional[str] = None,
        project_root: Optional[str] = None,
        debug: bool = True
    ):
        """
        Initialize CPU generator.

        Args:
            llm_provider: LLM to use ('groq', 'deepseek', 'openai', etc.)
            output_dir: Output directory for CPU files
            project_root: Project root directory
            debug: Enable debug output
        """
        self.llm_provider = llm_provider.lower()
        self.debug = debug

        # Setup LLM
        self.llm = self._setup_llm()

        # Setup output directory
        self.output_dir = self._setup_output_dir(output_dir, project_root)

        print(f"🔧 RISC-V CPU Generator initialized")
        print(f"   LLM Provider: {self.llm_provider}")
        print(f"   Output directory: {self.output_dir}")

    def _setup_llm(self):
        """Setup LLM provider"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))

            from llm_providers import (
                GeminiProvider,
                OpenAIProvider,
                ClaudeProvider,
                GroqProvider,
                DeepSeekProvider,
                GrokProvider,
                QwenProvider,
                MistralProvider,
                TogetherProvider
            )

            providers = {
                'gemini': GeminiProvider,
                'openai': OpenAIProvider,
                'gpt': OpenAIProvider,
                'claude': ClaudeProvider,
                'groq': GroqProvider,
                'deepseek': DeepSeekProvider,
                'grok': GrokProvider,
                'xai': GrokProvider,
                'qwen': QwenProvider,
                'mistral': MistralProvider,
                'codestral': MistralProvider,
                'together': TogetherProvider,
            }

            if self.llm_provider not in providers:
                print(f"⚠️  Unknown LLM provider: {self.llm_provider}")
                print(f"   Available: {', '.join(providers.keys())}")
                print(f"   Falling back to Groq")
                self.llm_provider = 'groq'

            provider_class = providers[self.llm_provider]
            llm = provider_class()

            print(f"✅ LLM provider loaded: {provider_class.__name__}")
            return llm

        except ImportError as e:
            print(f"❌ Failed to import LLM providers: {e}")
            return None

    def _setup_output_dir(self, output_dir: Optional[str], project_root: Optional[str]) -> Path:
        """Setup output directory for DUT, organized by LLM provider"""
        if output_dir:
            base_dir = Path(output_dir)
        elif project_root:
            base_dir = Path(project_root) / "output" / "dut"
        else:
            current = Path.cwd()
            possible_paths = [
                current / "output" / "dut",
                current / "outputs" / "dut",
                current.parent / "output" / "dut",
            ]
            for path in possible_paths:
                if path.parent.exists():
                    base_dir = path
                    break
            else:
                base_dir = current / "output" / "dut"

        # Create LLM-specific subdirectory
        llm_dir = base_dir / self.llm_provider
        llm_dir.mkdir(parents=True, exist_ok=True)

        return llm_dir

    def generate_cpu(
        self,
        bitwidth: int = 32,
        pipeline_stages: int = 5,
        module_name: str = "riscv_cpu"
    ) -> str:
        """
        Generate RISC-V CPU design.

        Args:
            bitwidth: Data bitwidth (32 for RV32I)
            pipeline_stages: Number of pipeline stages (5)
            module_name: Verilog module name

        Returns:
            Path to generated CPU file
        """
        print("\n" + "=" * 80)
        print(f"🔧 Generating {pipeline_stages}-stage RISC-V CPU using {self.llm_provider.upper()}")
        print("=" * 80)

        # Create prompt
        prompt = self._create_cpu_prompt(bitwidth, pipeline_stages, module_name)

        if self.debug:
            print(f"\n📝 Prompt preview:")
            print(prompt[:800] + "...")

        # Call LLM
        print(f"\n🤖 Calling {self.llm_provider.upper()} API...")

        try:
            if hasattr(self.llm, '_call_api'):
                response = self.llm._call_api(
                    prompt,
                    max_tokens=20000,  # CPU needs more tokens
                    system_prompt="You are an expert CPU architect and Verilog designer. Generate high-quality, synthesizable RTL code for a RISC-V processor."
                )
            else:
                print(f"❌ LLM does not have _call_api method")
                return None

            if not response:
                print(f"❌ LLM returned empty response")
                return None

            print(f"✅ Received response ({len(response)} chars)")

            # Extract Verilog code
            verilog_code = self._extract_verilog(response)

            if not verilog_code:
                #  Check if it's a truncation issue
                if 'module' in response and 'endmodule' not in response:
                    print(f"❌ Code was truncated! Response length: {len(response)}")
                    print(f"   Try increasing max_tokens (current limit may be too low)")
                else:
                    print(f"❌ Could not extract valid Verilog code")
                return None

            # Fix common syntax errors (missing begin/end in case branches)
            verilog_code = self._fix_verilog_syntax(verilog_code)

            # Validate
            if self._validate_verilog(verilog_code, bitwidth, pipeline_stages):
                print(f"✅ Verilog validation passed")
            else:
                print(f"⚠️  Verilog validation had warnings (continuing anyway)")

            # Save
            cpu_path = self._save_cpu(verilog_code, module_name, bitwidth, pipeline_stages)
            print(f"\n💾 CPU saved to: {cpu_path}")

            return str(cpu_path)

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_cpu_prompt(self, bitwidth: int, pipeline_stages: int, module_name: str) -> str:
        """Create prompt for CPU generation"""

        prompt = f"""Generate a synthesizable Verilog module for a {pipeline_stages}-stage pipelined RISC-V RV32I CPU.

## Architecture Overview

### Pipeline Stages
1. **IF (Instruction Fetch)**: Fetch instruction from memory using PC
2. **ID (Instruction Decode)**: Decode instruction, read registers
3. **EX (Execute)**: ALU operations, branch target calculation
4. **MEM (Memory Access)**: Load/Store operations
5. **WB (Write Back)**: Write results back to register file

### Module Interface

module {module_name} (
    input  wire        clk,
    input  wire        rst_n,
    
    // Instruction Memory Interface
    output wire [31:0] imem_addr,
    input  wire [31:0] imem_data,
    
    // Data Memory Interface
    output wire [31:0] dmem_addr,
    output wire [31:0] dmem_wdata,
    output wire        dmem_wen,
    output wire [3:0]  dmem_be,      // Byte enable
    input  wire [31:0] dmem_rdata,
    
    // Debug Interface (optional)
    output wire [31:0] debug_pc,
    output wire [31:0] debug_inst
);


## Supported Instructions (RV32I Subset)

### R-Type (Register-Register)
| Instruction | Opcode  | funct3 | funct7  | Operation |
|-------------|---------|--------|---------|-----------|
| ADD         | 0110011 | 000    | 0000000 | rd = rs1 + rs2 |
| SUB         | 0110011 | 000    | 0100000 | rd = rs1 - rs2 |
| AND         | 0110011 | 111    | 0000000 | rd = rs1 & rs2 |
| OR          | 0110011 | 110    | 0000000 | rd = rs1 | rs2 |
| XOR         | 0110011 | 100    | 0000000 | rd = rs1 ^ rs2 |
| SLT         | 0110011 | 010    | 0000000 | rd = (rs1 < rs2) ? 1 : 0 |

### I-Type (Immediate)
| Instruction | Opcode  | funct3 | Operation |
|-------------|---------|--------|-----------|
| ADDI        | 0010011 | 000    | rd = rs1 + imm |
| ANDI        | 0010011 | 111    | rd = rs1 & imm |
| ORI         | 0010011 | 110    | rd = rs1 | imm |
| XORI        | 0010011 | 100    | rd = rs1 ^ imm |
| SLTI        | 0010011 | 010    | rd = (rs1 < imm) ? 1 : 0 |
| LW          | 0000011 | 010    | rd = mem[rs1 + imm] |

### S-Type (Store)
| Instruction | Opcode  | funct3 | Operation |
|-------------|---------|--------|-----------|
| SW          | 0100011 | 010    | mem[rs1 + imm] = rs2 |

### B-Type (Branch)
| Instruction | Opcode  | funct3 | Operation |
|-------------|---------|--------|-----------|
| BEQ         | 1100011 | 000    | if (rs1 == rs2) PC += imm |
| BNE         | 1100011 | 001    | if (rs1 != rs2) PC += imm |
| BLT         | 1100011 | 100    | if (rs1 < rs2) PC += imm |
| BGE         | 1100011 | 101    | if (rs1 >= rs2) PC += imm |

### J-Type (Jump)
| Instruction | Opcode  | Operation |
|-------------|---------|-----------|
| JAL         | 1101111 | rd = PC + 4; PC += imm |
| JALR        | 1100111 | rd = PC + 4; PC = rs1 + imm |

## Implementation Requirements

### 1. Pipeline Registers
Create pipeline registers between each stage. EVERY signal listed below MUST be declared as `reg` and propagated through the pipeline:
- IF/ID register: instruction (ifid_inst), PC (ifid_pc)
- ID/EX register: rs1_data, rs2_data, rs1, rs2, rd, imm, control signals (regwrite, memread, memwrite, alu_src, branch, jump, alu_op)
- EX/MEM register: alu_result, mem_data (rs2 forwarded), rd, control signals (regwrite, memread, memwrite)
- MEM/WB register: mem_data, alu_result, rd, control signals (regwrite, memread)

CRITICAL: In the WB stage, `memwb_memread` is needed to select between memory data and ALU result:
```verilog
reg memwb_memread;  // MUST be declared
// In pipeline update: memwb_memread <= exmem_memread;
// In writeback: result = memwb_memread ? memwb_mem_data : memwb_alu_result;
```

### 2. Register File
- 32 registers x 32 bits
- x0 hardwired to 0
- 2 read ports, 1 write port
- Write in WB stage

### 3. ALU Operations
Support: ADD, SUB, AND, OR, XOR, SLT, shift operations

### 4. Hazard Handling

**Data Hazards (Forwarding):**
```verilog
// Forward from EX/MEM
if (exmem_rd == idex_rs1 && exmem_regwrite) 
    forward_a = exmem_alu_result;
// Forward from MEM/WB
if (memwb_rd == idex_rs1 && memwb_regwrite)
    forward_a = memwb_result;
```

**Load-Use Hazard (Stall):**
```verilog
// Stall if previous instruction is LW and current needs the data
if (idex_memread && (idex_rd == ifid_rs1 || idex_rd == ifid_rs2))
    stall = 1;
```

**Control Hazards (Flush):**
```verilog
// Flush IF/ID and ID/EX on branch taken
if (branch_taken)
    flush = 1;
```

### 5. Control Signals
Generate control signals in ID stage based on opcode:
- RegWrite: write to register file
- MemRead: read from data memory
- MemWrite: write to data memory
- ALUSrc: ALU source (register or immediate)
- Branch: branch instruction
- Jump: jump instruction

## Output Format
Generate ONLY the Verilog code. No explanations.
The code should be complete and synthesizable.
Include all pipeline stages, hazard detection, and forwarding logic.
Start with `module` and end with `endmodule`.

## IMPORTANT Verilog Syntax Rules (MUST FOLLOW)
1. Signals assigned in 'always' blocks MUST be declared as 'reg', NOT 'wire'
2. Signals assigned with 'assign' statements should be 'wire'
3. Forwarding signals (forward_a, forward_b) set in always blocks MUST be 'reg'
4. Control signals (stall, flush, branch_taken) set in always blocks MUST be 'reg'
5. Pipeline registers MUST be 'reg' type
6. NEVER assign to a 'wire' inside an 'always' block
7. When a case branch has MORE THAN ONE statement, you MUST wrap them in begin/end blocks
8. Use Verilog-2001 ONLY. Do NOT use SystemVerilog (no `logic`, no inline for-loop declarations)
9. For loops: declare `integer i;` BEFORE the for statement, then use `for (i = 0; ...)`

## Verilog Number Format Rules (MUST FOLLOW)
- funct3 is 3 bits: use 3'b000, 3'b001, 3'b100, 3'b110, 3'b111 (binary format)
- funct7 is 7 bits: use 7'b0000000, 7'b0100000 (binary format)
- opcode is 7 bits: use 7'b0110011, 7'b0010011 (binary format)
- Do NOT use decimal format like 4'd100 for binary bit patterns
- WRONG: 4'd100, 4'd110, 4'd111 (these are decimal numbers)
- CORRECT: 3'b100, 3'b110, 3'b111 (these are 3-bit binary values)

## Signal Width Rules
- funct3: declare as [2:0], NOT [3:0]
- funct7: declare as [6:0]
- opcode: declare as [6:0]
"""

        return prompt

    def _fix_verilog_syntax(self, verilog_code: str) -> str:
        """Fix common Verilog syntax errors from LLM generation (missing begin/end in case branches)"""
        verilog_code = re.sub(r'```verilog\s*', '', verilog_code)
        verilog_code = re.sub(r'```v\s*', '', verilog_code)
        verilog_code = re.sub(r'```\s*', '', verilog_code)

        lines = verilog_code.split('\n')
        fixed_lines = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if re.match(r'\s*case\s*\(', stripped):
                case_block = [lines[i]]
                i += 1
                depth = 1
                while i < len(lines) and depth > 0:
                    case_block.append(lines[i])
                    s = lines[i].strip()
                    if re.match(r'case\s*\(', s):
                        depth += 1
                    elif s.startswith('endcase'):
                        depth -= 1
                    i += 1
                fixed_block = self._fix_case_block(case_block)
                fixed_lines.extend(fixed_block)
                continue
            fixed_lines.append(lines[i])
            i += 1
        result = '\n'.join(fixed_lines)

        # 修复 SystemVerilog 内联 for 循环变量声明
        result = self._fix_for_loop_declarations(result)

        # 修复 always 块内部的 integer 声明（移到模块级别）
        result = self._fix_integer_in_always(result)

        # 修复未声明的流水线寄存器信号
        result = self._fix_undeclared_pipeline_signals(result)

        return result

    def _fix_integer_in_always(self, verilog_code: str) -> str:
        """Move integer declarations from inside always blocks to module level.

        Verilog-2001 doesn't allow variable declarations inside always blocks
        (unless it's a named block). This fixes LLM-generated code that does:
            always @(...) begin
                if (!rst_n) begin
                    integer i;  // ILLEGAL
                    for (i = 0; ...)
        """
        lines = verilog_code.split('\n')
        fixed_lines = []
        integers_to_declare = []
        in_always = False
        always_depth = 0
        module_end_idx = -1

        # First pass: find integers inside always blocks and mark them for removal
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Track module header end (after ports, before logic)
            if module_end_idx == -1 and (stripped.startswith('reg ') or
                                          stripped.startswith('wire ') or
                                          stripped.startswith('assign ') or
                                          stripped.startswith('always') or
                                          stripped.startswith('localparam') or
                                          stripped.startswith('parameter')):
                module_end_idx = len(fixed_lines)

            # Track always blocks
            if re.match(r'\s*always\s*@', stripped):
                in_always = True
                always_depth = 0

            if in_always:
                # Count begin/end depth
                always_depth += stripped.count('begin')
                always_depth -= stripped.count('end')
                if always_depth <= 0 and 'end' in stripped:
                    in_always = False

                # Check for integer declaration inside always
                int_match = re.match(r'^(\s*)integer\s+(\w+)\s*;', line)
                if int_match:
                    var_name = int_match.group(2)
                    if var_name not in integers_to_declare:
                        integers_to_declare.append(var_name)
                    # Skip this line (don't add to fixed_lines)
                    i += 1
                    continue

            fixed_lines.append(line)
            i += 1

        # Second pass: insert integer declarations at module level
        if integers_to_declare and module_end_idx > 0:
            declarations = [f"    integer {var};" for var in integers_to_declare]
            for j, decl in enumerate(declarations):
                fixed_lines.insert(module_end_idx + j, decl)

        return '\n'.join(fixed_lines)

    def _fix_for_loop_declarations(self, verilog_code: str) -> str:
        """Fix SystemVerilog inline for-loop variable declarations for Verilog-2001 compatibility.

        Converts:  for (integer i = 0; i < N; i = i + 1)
        To:        integer i;
                   for (i = 0; i < N; i = i + 1)
        """
        lines = verilog_code.split('\n')
        fixed_lines = []
        declared_vars = set()

        for line in lines:
            m = re.search(r'\bfor\s*\(\s*(integer|int|genvar)\s+(\w+)\s*=', line)
            if m:
                var_type = m.group(1)
                var_name = m.group(2)
                indent_match = re.match(r'^(\s*)', line)
                indent = indent_match.group(1) if indent_match else ''
                verilog_type = 'integer' if var_type in ('integer', 'int') else var_type
                if var_name not in declared_vars:
                    fixed_lines.append(f"{indent}{verilog_type} {var_name};")
                    declared_vars.add(var_name)
                fixed_line = re.sub(
                    r'\bfor\s*\(\s*(integer|int|genvar)\s+(\w+)\s*=',
                    r'for (\2 =',
                    line
                )
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_undeclared_pipeline_signals(self, verilog_code: str) -> str:
        """Fix undeclared pipeline register signals.

        LLMs sometimes use pipeline signals (e.g., memwb_memread) in logic
        but forget to declare them as reg or propagate them through the pipeline.
        This detects such cases and adds missing declarations and assignments.
        """
        # Pipeline stage prefixes and their source stages
        pipeline_chain = {
            'memwb': 'exmem',
            'exmem': 'idex',
        }

        # Find all declared reg/wire signals
        declared = set()
        for m in re.finditer(r'\b(?:reg|wire)\b[^;]*\b(\w+)\s*(?:,\s*(\w+)\s*)*;', verilog_code):
            declared.add(m.group(1))
            if m.group(2):
                declared.add(m.group(2))
        # Also catch multi-signal declarations like "reg a, b, c;"
        for m in re.finditer(r'\b(?:reg|wire)\b(?:\s*\[[^\]]*\])?\s+([\w\s,]+);', verilog_code):
            for name in m.group(1).split(','):
                name = name.strip()
                if name and re.match(r'^\w+$', name):
                    declared.add(name)

        # Find all used pipeline signals (prefixed with pipeline stage names)
        pipeline_prefixes = list(pipeline_chain.keys()) + list(pipeline_chain.values())
        used_pipeline_signals = set()
        for prefix in pipeline_prefixes:
            for m in re.finditer(rf'\b({prefix}_\w+)\b', verilog_code):
                used_pipeline_signals.add(m.group(1))

        # Find undeclared pipeline signals
        undeclared = used_pipeline_signals - declared
        if not undeclared:
            return verilog_code

        print(f"   🔧 Fixing undeclared pipeline signals: {undeclared}")

        lines = verilog_code.split('\n')

        # Find where to insert declarations (after last reg/wire declaration before first always)
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r'\s*(reg|wire)\s+', stripped):
                insert_idx = i + 1
            if re.match(r'\s*always\s*@', stripped):
                break

        # For each undeclared signal, determine width and add declaration
        new_declarations = []
        new_assignments = []
        for sig in sorted(undeclared):
            # Determine bit width by looking at similar signals in same pipeline stage
            prefix = sig.split('_')[0]
            suffix = '_'.join(sig.split('_')[1:])

            # Check if a corresponding signal exists in another pipeline stage
            width = ''
            for other_prefix in pipeline_prefixes:
                other_sig = f"{other_prefix}_{suffix}"
                if other_sig in declared and other_sig != sig:
                    # Find its width declaration
                    width_match = re.search(rf'\b(?:reg|wire)\s+(\[[^\]]+\])\s+[^;]*\b{re.escape(other_sig)}\b', verilog_code)
                    if width_match:
                        width = width_match.group(1) + ' '
                    break

            new_declarations.append(f"    reg {width}{sig};")

            # Add pipeline propagation assignment if source stage signal exists
            if prefix in pipeline_chain:
                source_prefix = pipeline_chain[prefix]
                source_sig = f"{source_prefix}_{suffix}"
                if source_sig in declared or source_sig in used_pipeline_signals:
                    new_assignments.append((sig, source_sig))

        # Insert declarations
        if new_declarations:
            for j, decl in enumerate(new_declarations):
                lines.insert(insert_idx + j, decl)

        result = '\n'.join(lines)

        # Add pipeline assignments: find the block where other signals of same stage are assigned
        for target_sig, source_sig in new_assignments:
            prefix = target_sig.split('_')[0]
            # Find an existing assignment like "prefix_xxx <= source_prefix_xxx;"
            pattern = rf'({re.escape(prefix)}_\w+\s*<=\s*{re.escape(pipeline_chain[prefix])}_\w+\s*;)'
            match = re.search(pattern, result)
            if match:
                existing_line = match.group(1)
                indent_match = re.search(rf'^(\s*){re.escape(existing_line)}', result, re.MULTILINE)
                indent = indent_match.group(1) if indent_match else '            '
                assignment = f"{indent}{target_sig} <= {source_sig};"
                # Insert after the matched line
                result = result.replace(existing_line, existing_line + '\n' + assignment, 1)

            # Also add reset: find reset block with "prefix_xxx <= N'b0;"
            reset_pattern = rf'({re.escape(prefix)}_\w+\s*<=\s*\d+\'b0\s*;)'
            reset_match = re.search(reset_pattern, result)
            if reset_match:
                existing_reset = reset_match.group(1)
                indent_match = re.search(rf'^(\s*){re.escape(existing_reset)}', result, re.MULTILINE)
                indent = indent_match.group(1) if indent_match else '            '
                reset_line = f"{indent}{target_sig} <= 1'b0;"
                result = result.replace(existing_reset, existing_reset + '\n' + reset_line, 1)

        return result

    def _fix_case_block(self, case_lines: list) -> list:
        """Fix begin/end in a case...endcase block"""
        if len(case_lines) < 3:
            return case_lines
        label_re = re.compile(
            r"^(\s*)"
            r"(\d+'[bBhHdD][0-9a-fA-F_xXzZ]+|default)"
            r"\s*:\s*(begin\b)?(.*?)$"
        )
        result = [case_lines[0]]
        i = 1
        while i < len(case_lines):
            line = case_lines[i]
            stripped = line.strip()
            if stripped.startswith('endcase'):
                result.append(line)
                i += 1
                continue
            m = label_re.match(line)
            if m:
                indent, label, has_begin, rest = m.group(1), m.group(2), m.group(3), m.group(4).strip()
                if has_begin:
                    result.append(line)
                    i += 1
                    bd = 1
                    while i < len(case_lines) and bd > 0:
                        s = case_lines[i].strip()
                        if s == 'begin' or s.startswith('begin '):
                            bd += 1
                        elif s == 'end' or s.startswith('end ') or s.startswith('end//'):
                            bd -= 1
                        result.append(case_lines[i])
                        i += 1
                    continue
                is_comment = rest.startswith('//')
                inline_stmt = rest if rest and not is_comment else None
                comment = rest if is_comment else ''
                branch_stmts = []
                if inline_stmt:
                    branch_stmts.append(f"{indent}    {inline_stmt}")
                j = i + 1
                while j < len(case_lines):
                    check = case_lines[j].strip()
                    if check.startswith('endcase') or label_re.match(case_lines[j]):
                        break
                    if check:
                        branch_stmts.append(case_lines[j])
                    j += 1
                stmt_count = sum(1 for s in branch_stmts if s.strip() and not s.strip().startswith('//'))
                if stmt_count > 1:
                    result.append(f"{indent}{label}: begin {comment}".rstrip())
                    for bl in branch_stmts:
                        result.append(bl)
                    result.append(f"{indent}end")
                elif branch_stmts:
                    if inline_stmt:
                        result.append(line)
                    else:
                        result.append(f"{indent}{label}: {comment}".rstrip())
                        for bl in branch_stmts:
                            result.append(bl)
                else:
                    result.append(line)
                i = j
                continue
            result.append(line)
            i += 1
        return result

    def _extract_verilog(self, response: str) -> Optional[str]:
        """Extract Verilog code from LLM response"""

        # ★★★ 首先尝试提取完整的代码块 ★★★

        # Try ```verilog ... ```
        if '```verilog' in response:
            match = re.search(r'```verilog\s*(.*?)```', response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if 'module' in code and 'endmodule' in code:
                    return code

        # Try ```v ... ```
        if '```v' in response:
            match = re.search(r'```v\s*(.*?)```', response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if 'module' in code and 'endmodule' in code:
                    return code

        # Try ``` ... ```
        if '```' in response:
            match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if 'module' in code and 'endmodule' in code:
                    return code

        # ★★★ 处理被截断的情况（没有结尾的 ```）★★★
        if '```verilog' in response or '```v' in response:
            # 移除开头的 markdown 标记，提取后面的内容
            cleaned = re.sub(r'```verilog\s*', '', response)
            cleaned = re.sub(r'```v\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)

            # 检查是否有完整的 module
            if 'module' in cleaned and 'endmodule' in cleaned:
                match = re.search(r'(module\s+.*?endmodule)', cleaned, re.DOTALL)
                if match:
                    return match.group(1).strip()

        # Try to find module directly
        match = re.search(r'(module\s+.*?endmodule)', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # If response contains module/endmodule, return cleaned
        if 'module' in response and 'endmodule' in response:
            cleaned = response.replace('```verilog', '').replace('```v', '').replace('```', '')
            return cleaned.strip()

        # ★★★ 最后：如果代码被截断（有 module 但没有 endmodule），返回 None ★★★
        # 这样调用方知道生成失败了
        return None

    def _validate_verilog(self, verilog_code: str, bitwidth: int, pipeline_stages: int) -> bool:
        """Validate generated Verilog code"""

        print(f"\n🔍 Validating Verilog code...")

        checks = []

        # Check 1: Has module declaration
        has_module = 'module' in verilog_code and 'endmodule' in verilog_code
        checks.append(('Module structure', has_module))

        # Check 2: Has pipeline stages (check for pipeline register names)
        has_ifid = 'if_id' in verilog_code.lower() or 'ifid' in verilog_code.lower() or 'IF_ID' in verilog_code
        has_idex = 'id_ex' in verilog_code.lower() or 'idex' in verilog_code.lower() or 'ID_EX' in verilog_code
        has_exmem = 'ex_mem' in verilog_code.lower() or 'exmem' in verilog_code.lower() or 'EX_MEM' in verilog_code
        has_memwb = 'mem_wb' in verilog_code.lower() or 'memwb' in verilog_code.lower() or 'MEM_WB' in verilog_code
        has_pipeline = has_ifid or has_idex or has_exmem or has_memwb
        checks.append(('Pipeline registers', has_pipeline))

        # Check 3: Has PC (program counter)
        has_pc = 'pc' in verilog_code.lower() or 'PC' in verilog_code
        checks.append(('Program Counter', has_pc))

        # Check 4: Has ALU
        has_alu = 'alu' in verilog_code.lower() or 'ALU' in verilog_code
        checks.append(('ALU', has_alu))

        # Check 5: Has register file or registers
        has_regfile = 'reg' in verilog_code.lower() and ('file' in verilog_code.lower() or '[31:0]' in verilog_code)
        checks.append(('Register File', has_regfile))

        # Check 6: Has memory interface
        has_mem = 'mem' in verilog_code.lower() or 'imem' in verilog_code.lower() or 'dmem' in verilog_code.lower()
        checks.append(('Memory Interface', has_mem))

        # Check 7: Has forwarding or hazard handling
        has_forward = 'forward' in verilog_code.lower() or 'hazard' in verilog_code.lower() or 'stall' in verilog_code.lower()
        checks.append(('Hazard Handling', has_forward))

        # Print validation results
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "⚠️"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False

        return all_passed

    def _save_cpu(self, verilog_code: str, module_name: str, bitwidth: int, pipeline_stages: int) -> Path:
        """Save CPU to file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{module_name}_{pipeline_stages}stage_{timestamp}.v"

        header = f"""//==============================================================================
// RISC-V CPU Design - Design Under Test (DUT)
//
// Project: LLM-based Hardware Verification Pipeline
// Authors: Rolf Drechsler, Qian Liu
// Paper: https://arxiv.org/abs/2512.17814
//
// Generated by: cpu_generator.py
// LLM Provider: {self.llm_provider}
// Generated at: {timestamp}
// Architecture: RV32I, {pipeline_stages}-stage pipeline
//
// Features:
//   - {pipeline_stages}-stage pipeline (IF, ID, EX, MEM, WB)
//   - RV32I instruction subset
//   - Data forwarding for hazard resolution
//   - Branch prediction with flush
//   - 32 x 32-bit register file
//==============================================================================

"""

        full_code = header + verilog_code

        cpu_path = self.output_dir / filename
        with open(cpu_path, 'w', encoding='utf-8') as f:
            f.write(full_code)

        # Save metadata
        metadata = {
            'module_name': module_name,
            'module_type': 'cpu',
            'architecture': 'RV32I',
            'bitwidth': bitwidth,
            'pipeline_stages': pipeline_stages,
            'llm_provider': self.llm_provider,
            'timestamp': timestamp,
            'filepath': str(cpu_path),
            'supported_instructions': [
                'ADD', 'SUB', 'AND', 'OR', 'XOR', 'SLT',
                'ADDI', 'ANDI', 'ORI', 'XORI', 'SLTI',
                'LW', 'SW',
                'BEQ', 'BNE', 'BLT', 'BGE',
                'JAL', 'JALR'
            ]
        }

        metadata_path = self.output_dir / f"{module_name}_{pipeline_stages}stage_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"   💾 Metadata saved: {metadata_path}")

        return cpu_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate RISC-V CPU Verilog design using LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate 5-stage pipelined CPU using Groq (default)
  python cpu_generator.py

  # Use specific LLM provider
  python cpu_generator.py --llm deepseek

  # Specify output directory
  python cpu_generator.py --output output/dut/
        '''
    )

    parser.add_argument('--llm', default='groq',
                       help='LLM provider (groq, deepseek, openai, claude, gemini)')
    parser.add_argument('--pipeline-stages', type=int, default=5,
                       help='Number of pipeline stages (default: 5)')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--module-name', default='riscv_cpu', help='Verilog module name')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output')

    args = parser.parse_args()

    print("=" * 80)
    print("🔧 RISC-V CPU Generator")
    print("=" * 80)

    generator = CPUGenerator(
        llm_provider=args.llm,
        output_dir=args.output,
        project_root=args.project_root,
        debug=not args.no_debug
    )

    cpu_path = generator.generate_cpu(
        pipeline_stages=args.pipeline_stages,
        module_name=args.module_name
    )

    if cpu_path:
        print("\n" + "=" * 80)
        print("✅ RISC-V CPU Generation Complete")
        print("=" * 80)
        print(f"\n📁 CPU file: {cpu_path}")
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ RISC-V CPU Generation Failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())