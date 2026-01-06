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
                DeepSeekProvider
            )

            providers = {
                'gemini': GeminiProvider,
                'openai': OpenAIProvider,
                'gpt': OpenAIProvider,
                'claude': ClaudeProvider,
                'groq': GroqProvider,
                'deepseek': DeepSeekProvider,
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
                    max_tokens=10000,  # CPU needs more tokens
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
                print(f"❌ Could not extract valid Verilog code")
                print(f"   Raw response preview: {response[:300]}...")
                return None

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
Create pipeline registers between each stage:
- IF/ID register: instruction, PC
- ID/EX register: control signals, register data, immediates
- EX/MEM register: ALU result, write data, control signals
- MEM/WB register: memory data, ALU result, control signals

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

    def _extract_verilog(self, response: str) -> Optional[str]:
        """Extract Verilog code from LLM response"""

        # Remove markdown code blocks
        if '```verilog' in response:
            # Extract content between ```verilog and ```
            match = re.search(r'```verilog\s*(.*?)```', response, re.DOTALL)
            if match:
                return match.group(1).strip()

        if '```v' in response:
            match = re.search(r'```v\s*(.*?)```', response, re.DOTALL)
            if match:
                return match.group(1).strip()

        if '```' in response:
            match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if 'module' in code and 'endmodule' in code:
                    return code

        # Try to find module directly
        match = re.search(r'(module\s+.*?endmodule)', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # If response contains module/endmodule, return as-is
        if 'module' in response and 'endmodule' in response:
            # Remove any remaining ``` markers
            cleaned = response.replace('```verilog', '').replace('```v', '').replace('```', '')
            return cleaned.strip()

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