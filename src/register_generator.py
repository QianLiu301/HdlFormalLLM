"""
Register File Generator - Generate Register File Verilog using LLM
===================================================================

Generates parameterized register file designs with:
- Configurable bitwidth (8/16/32/64-bit)
- Configurable depth (8/16/32 registers)
- Dual read ports, single write port
- RISC-V style: R0/x0 always returns 0

Part of the Hardware Generator Pipeline.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List


class RegFileGenerator:
    """
    Generate Register File Verilog design using LLM.
    """

    def __init__(
        self,
        llm_provider: str = 'groq',
        output_dir: Optional[str] = None,
        project_root: Optional[str] = None,
        debug: bool = True
    ):
        """
        Initialize Register File generator.

        Args:
            llm_provider: LLM to use ('groq', 'deepseek', 'openai', etc.)
            output_dir: Output directory for Register File
            project_root: Project root directory
            debug: Enable debug output
        """
        self.llm_provider = llm_provider.lower()
        self.debug = debug

        # Setup LLM
        self.llm = self._setup_llm()

        # Setup output directory
        self.output_dir = self._setup_output_dir(output_dir, project_root)

        print(f"🔧 Register File Generator initialized")
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
        """Setup output directory for DUT"""
        if output_dir:
            dut_dir = Path(output_dir)
        elif project_root:
            dut_dir = Path(project_root) / "output" / "dut"
        else:
            current = Path.cwd()
            possible_paths = [
                current / "output" / "dut",
                current / "outputs" / "dut",
                current.parent / "output" / "dut",
            ]

            for path in possible_paths:
                if path.parent.exists():
                    dut_dir = path
                    break
            else:
                dut_dir = current / "output" / "dut"

        dut_dir.mkdir(parents=True, exist_ok=True)
        return dut_dir

    def generate_regfile(
        self,
        bitwidth: int = 32,
        depth: int = 32,
        module_name: str = "regfile"
    ) -> str:
        """
        Generate Register File design.

        Args:
            bitwidth: Data bitwidth (8, 16, 32, 64)
            depth: Number of registers (8, 16, 32)
            module_name: Verilog module name

        Returns:
            Path to generated Register File
        """
        print("\n" + "=" * 80)
        print(f"🔧 Generating {depth}x{bitwidth}-bit Register File using {self.llm_provider.upper()}")
        print("=" * 80)

        # Create prompt
        prompt = self._create_regfile_prompt(bitwidth, depth, module_name)

        if self.debug:
            print(f"\n📝 Prompt preview:")
            print(prompt[:500] + "...")

        # Call LLM
        print(f"\n🤖 Calling {self.llm_provider.upper()} API...")

        try:
            if hasattr(self.llm, '_call_api'):
                response = self.llm._call_api(
                    prompt,
                    max_tokens=3000,
                    system_prompt="You are an expert Verilog hardware designer. Generate high-quality, synthesizable RTL code."
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
                print(f"   Raw response preview: {response[:200]}...")
                return None

            # Validate
            if self._validate_verilog(verilog_code, bitwidth, depth):
                print(f"✅ Verilog validation passed")
            else:
                print(f"⚠️  Verilog validation had warnings (continuing anyway)")

            # Save
            regfile_path = self._save_regfile(verilog_code, module_name, bitwidth, depth)
            print(f"\n💾 Register File saved to: {regfile_path}")

            return str(regfile_path)

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_regfile_prompt(self, bitwidth: int, depth: int, module_name: str) -> str:
        """Create prompt for Register File generation"""

        # Calculate address width
        import math
        addr_width = max(1, int(math.ceil(math.log2(depth))))

        prompt = f"""Generate a synthesizable Verilog module for a {depth}x{bitwidth}-bit Register File.

## Requirements

### Module Interface
```verilog
module {module_name} #(
    parameter WIDTH = {bitwidth},
    parameter DEPTH = {depth},
    parameter ADDR_WIDTH = {addr_width}
)(
    input  wire                  clk,        // Clock
    input  wire                  rst_n,      // Active-low reset
    // Read port 1
    input  wire [ADDR_WIDTH-1:0] raddr1,     // Read address 1
    output wire [WIDTH-1:0]      rdata1,     // Read data 1
    // Read port 2
    input  wire [ADDR_WIDTH-1:0] raddr2,     // Read address 2
    output wire [WIDTH-1:0]      rdata2,     // Read data 2
    // Write port
    input  wire                  wen,        // Write enable
    input  wire [ADDR_WIDTH-1:0] waddr,      // Write address
    input  wire [WIDTH-1:0]      wdata       // Write data
);
```

### Functional Requirements

1. **Register Array**: {depth} registers, each {bitwidth} bits wide

2. **Read Ports** (Combinational/Asynchronous):
   - Two independent read ports
   - Read is combinational (no clock delay)
   - Can read any register at any time

3. **Write Port** (Synchronous):
   - Single write port
   - Write happens on rising edge of clk when wen=1
   - Write has priority over reset for normal registers

4. **RISC-V Convention** (IMPORTANT):
   - Register 0 (x0) MUST always read as 0
   - Writes to register 0 are ignored
   - This is hardwired, not just initialized

5. **Reset Behavior**:
   - When rst_n is low, all registers reset to 0
   - Register 0 stays 0 regardless

6. **Read-During-Write**:
   - If reading and writing same address simultaneously
   - Read should return the OLD value (before write)
   - This is standard register file behavior

### Implementation Notes
- Use `reg [WIDTH-1:0] registers [0:DEPTH-1]` for storage
- Read ports are combinational assigns
- Write port uses always @(posedge clk)
- Handle x0 specially in both read and write

### Expected Behavior Example
```
// Write 0x1234 to register 5
wen=1, waddr=5, wdata=0x1234
// Next cycle: registers[5] = 0x1234

// Read from register 5 and register 0
raddr1=5, raddr2=0
// Immediately: rdata1=0x1234, rdata2=0x0000 (x0 always 0)

// Try to write to register 0
wen=1, waddr=0, wdata=0xFFFF
// Next cycle: registers[0] still = 0 (write ignored)
```

## Output Format
Generate ONLY the Verilog code. No explanations.
Start with `module` and end with `endmodule`.
"""

        return prompt

    def _extract_verilog(self, response: str) -> Optional[str]:
        """Extract Verilog code from LLM response"""

        patterns = [
            r'```verilog\n(.*?)```',
            r'```v\n(.*?)```',
            r'```\n(.*?)```',
            r'(module\s+.*?endmodule)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
                if 'module' in code and 'endmodule' in code:
                    return code

        if 'module' in response and 'endmodule' in response:
            return response.strip()

        return None

    def _validate_verilog(self, verilog_code: str, bitwidth: int, depth: int) -> bool:
        """Validate generated Verilog code"""

        print(f"\n🔍 Validating Verilog code...")

        checks = []

        # Check 1: Has module declaration
        has_module = 'module' in verilog_code and 'endmodule' in verilog_code
        checks.append(('Module structure', has_module))

        # Check 2: Has register array
        has_reg_array = 'reg' in verilog_code and '[' in verilog_code
        checks.append(('Register array', has_reg_array))

        # Check 3: Has read ports
        has_rdata1 = 'rdata1' in verilog_code
        has_rdata2 = 'rdata2' in verilog_code
        checks.append(('Read ports', has_rdata1 and has_rdata2))

        # Check 4: Has write logic
        has_wen = 'wen' in verilog_code
        has_wdata = 'wdata' in verilog_code
        checks.append(('Write logic', has_wen and has_wdata))

        # Check 5: Has x0 handling (register 0 always 0)
        has_x0_handling = '0' in verilog_code and ('raddr' in verilog_code or 'waddr' in verilog_code)
        checks.append(('x0 handling', has_x0_handling))

        # Check 6: Has always block for write
        has_always = 'always' in verilog_code
        checks.append(('Always block', has_always))

        # Print validation results
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False

        return all_passed

    def _save_regfile(self, verilog_code: str, module_name: str, bitwidth: int, depth: int) -> Path:
        """Save Register File to file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{module_name}_{depth}x{bitwidth}bit.v"

        header = f"""//==============================================================================
// Register File Design - Design Under Test (DUT)
//
// Project: LLM-based Hardware Verification Pipeline
// Authors: Rolf Drechsler, Qian Liu
// Paper: https://arxiv.org/abs/2512.17814
//
// Generated by: regfile_generator.py
// LLM Provider: {self.llm_provider}
// Generated at: {timestamp}
// Configuration: {depth} registers x {bitwidth} bits
//
// Features:
//   - Dual read ports (combinational)
//   - Single write port (synchronous)
//   - RISC-V style: x0 always returns 0
//   - Synchronous reset (active low)
//==============================================================================

"""

        full_code = header + verilog_code

        regfile_path = self.output_dir / filename
        with open(regfile_path, 'w', encoding='utf-8') as f:
            f.write(full_code)

        # Save metadata
        metadata = {
            'module_name': module_name,
            'module_type': 'regfile',
            'bitwidth': bitwidth,
            'depth': depth,
            'llm_provider': self.llm_provider,
            'timestamp': timestamp,
            'filepath': str(regfile_path),
        }

        metadata_path = self.output_dir / f"{module_name}_{depth}x{bitwidth}bit_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"   💾 Metadata saved: {metadata_path}")

        return regfile_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate Register File Verilog design using LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate 32x32-bit register file using Groq (default)
  python regfile_generator.py

  # Use specific LLM provider
  python regfile_generator.py --llm deepseek

  # Custom configuration
  python regfile_generator.py --bitwidth 64 --depth 16

  # Specify output directory
  python regfile_generator.py --output output/dut/
        '''
    )

    parser.add_argument('--llm', default='groq',
                       help='LLM provider (groq, deepseek, openai, claude, gemini)')
    parser.add_argument('--bitwidth', type=int, default=32,
                       help='Data bitwidth (default: 32)')
    parser.add_argument('--depth', type=int, default=32,
                       help='Number of registers (default: 32)')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--module-name', default='regfile', help='Verilog module name')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output')

    args = parser.parse_args()

    print("=" * 80)
    print("🔧 Register File Generator")
    print("=" * 80)

    generator = RegFileGenerator(
        llm_provider=args.llm,
        output_dir=args.output,
        project_root=args.project_root,
        debug=not args.no_debug
    )

    regfile_path = generator.generate_regfile(
        bitwidth=args.bitwidth,
        depth=args.depth,
        module_name=args.module_name
    )

    if regfile_path:
        print("\n" + "=" * 80)
        print("✅ Register File Generation Complete")
        print("=" * 80)
        print(f"\n📁 Register File: {regfile_path}")
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ Register File Generation Failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())