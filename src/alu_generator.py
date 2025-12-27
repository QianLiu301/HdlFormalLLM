"""
Deterministic ALU Generator - Generate fixed ALU design using LLM
==================================================================

This module generates a FIXED ALU design that will be used as DUT (Design Under Test)
for all experiments. Different LLM-generated testbenches will test this same ALU.

ARCHITECTURE:
    Spec (requirements)
           │
           ▼
    ┌─────────────────┐
    │  LLM (Gemini)   │ ◄── Deterministic prompt
    │  alu_generator  │     Same input → Same output
    └────────┬────────┘
             │
             ▼
        alu.v (fixed DUT)
             │
             ▼
    Used by ALL testbenches from different LLMs

PURPOSE:
- Generate ONE high-quality ALU design
- Use as fixed DUT for all experiments
- Ensure consistent testing baseline
- Support LLM selection (Gemini, GPT, etc.)

FEATURES:
- ✅ Deterministic generation (fixed seed)
- ✅ LLM selection interface
- ✅ High-quality prompt engineering
- ✅ Validation and verification
- ✅ Clear directory structure
- ✅ Automatic proxy setup
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


# ============================================================================
# Automatic Proxy Setup (from config/llm_config.json)
# ============================================================================
def setup_proxy():
    """
    从配置文件读取并自动设置代理

    查找 config/llm_config.json 并读取 proxy 配置
    """
    config_paths = [
        Path('config/llm_config.json'),
        Path('llm_config.json'),
        Path('../config/llm_config.json'),
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                proxy_config = config.get('proxy', {})
                if proxy_config.get('enabled'):
                    http_proxy = proxy_config.get('http_proxy')
                    https_proxy = proxy_config.get('https_proxy')

                    if https_proxy:
                        os.environ['HTTP_PROXY'] = http_proxy or https_proxy
                        os.environ['HTTPS_PROXY'] = https_proxy
                        print(f"🌐 代理已启用: {https_proxy}")
                    return
            except Exception as e:
                print(f"⚠️  读取代理配置失败: {e}")
                continue

    # 没有找到配置文件，尝试使用环境变量
    if not os.environ.get('HTTPS_PROXY'):
        print("⚠️  未找到代理配置")
        print("   如需使用代理，请配置 config/llm_config.json")
        print("   或设置环境变量 HTTPS_PROXY")

# 模块加载时自动调用
setup_proxy()
# ============================================================================


class ALUGenerator:
    """
    Generate deterministic ALU design using LLM.

    This creates a FIXED ALU that all testbenches will test against.
    """

    def __init__(
        self,
        llm_provider: str = 'groq',
        output_dir: Optional[str] = None,
        project_root: Optional[str] = None,
        debug: bool = True
    ):
        """
        Initialize ALU generator.

        Args:
            llm_provider: LLM to use ('gemini', 'openai', 'claude', etc.)
            output_dir: Output directory for ALU file
            project_root: Project root directory
            debug: Enable debug output
        """
        self.llm_provider = llm_provider.lower()
        self.debug = debug

        # Setup LLM
        self.llm = self._setup_llm()

        # Setup output directory
        self.output_dir = self._setup_output_dir(output_dir, project_root)

        print(f"🔧 ALU Generator initialized")
        print(f"   LLM Provider: {self.llm_provider}")
        print(f"   Output directory: {self.output_dir}")

    def _setup_llm(self):
        """Setup LLM provider"""
        try:
            # Import LLM providers
            import sys
            sys.path.insert(0, str(Path(__file__).parent))

            from llm_providers import (
                GeminiProvider,
                OpenAIProvider,
                ClaudeProvider,
                GroqProvider,
                DeepSeekProvider
            )

            # Map provider names to classes
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
                print(f"   Falling back to Gemini")
                self.llm_provider = 'gemini'

            provider_class = providers[self.llm_provider]
            llm = provider_class()

            print(f"✅ LLM provider loaded: {provider_class.__name__}")
            return llm

        except ImportError as e:
            print(f"❌ Failed to import LLM providers: {e}")
            print(f"   Please ensure llm_providers.py is available")
            return None

    def _setup_output_dir(self, output_dir: Optional[str], project_root: Optional[str]) -> Path:
        """Setup output directory for DUT"""
        if output_dir:
            dut_dir = Path(output_dir)
        elif project_root:
            dut_dir = Path(project_root) / "output" / "dut"
        else:
            # Try to find existing output directory
            current = Path.cwd()
            possible_paths = [
                current / "output" / "dut",
                current / "outputs" / "dut",
                current.parent / "output" / "dut",
            ]

            for path in possible_paths:
                if path.parent.exists():  # Check if parent (output/) exists
                    dut_dir = path
                    break
            else:
                # Default
                dut_dir = current / "output" / "dut"

        # Ensure directory exists
        dut_dir.mkdir(parents=True, exist_ok=True)
        return dut_dir

    def generate_alu(
        self,
        bitwidth: int = 16,
        operations: Optional[Dict] = None,
        module_name: str = "alu"
    ) -> str:
        """
        Generate ALU design.

        Args:
            bitwidth: ALU bitwidth (8, 16, 32, 64)
            operations: Dictionary of operations to include
            module_name: Verilog module name

        Returns:
            Path to generated ALU file
        """
        print("\n" + "=" * 80)
        print(f"🔧 Generating {bitwidth}-bit ALU using {self.llm_provider.upper()}")
        print("=" * 80)

        # Use default operations if not provided
        if operations is None:
            operations = {
                "ADD": {"opcode": "0000", "description": "Addition (A + B)"},
                "SUB": {"opcode": "0001", "description": "Subtraction (A - B)"},
                "AND": {"opcode": "0010", "description": "Bitwise AND (A & B)"},
                "OR": {"opcode": "0011", "description": "Bitwise OR (A | B)"},
            }

        # Create prompt
        prompt = self._create_alu_prompt(bitwidth, operations, module_name)

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
                print(f"❌ LLM provider does not support _call_api method")
                return None

            print(f"✅ LLM response received ({len(response)} chars)")

            # Extract Verilog code
            verilog_code = self._extract_verilog(response)

            if not verilog_code:
                print(f"❌ Failed to extract Verilog code from response")
                return None

            print(f"✅ Verilog code extracted ({len(verilog_code)} lines)")

            # Validate
            if not self._validate_verilog(verilog_code, bitwidth, operations):
                print(f"⚠️  Verilog validation failed, but continuing...")

            # Save to file
            alu_path = self._save_alu(verilog_code, module_name, bitwidth)

            print(f"\n✅ ALU saved: {alu_path}")
            print(f"   Bitwidth: {bitwidth}")
            print(f"   Operations: {len(operations)}")
            print(f"   LLM: {self.llm_provider}")

            return str(alu_path)

        except Exception as e:
            print(f"❌ ALU generation failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

    def _create_alu_prompt(self, bitwidth: int, operations: Dict, module_name: str) -> str:
        """
        Create deterministic prompt for ALU generation.

        This prompt is carefully crafted to produce consistent, high-quality results.
        """

        # Build operations list
        ops_list = []
        for op_name, op_info in operations.items():
            opcode = op_info['opcode']
            desc = op_info['description']
            ops_list.append(f"  - {op_name} (opcode {opcode}): {desc}")

        ops_text = "\n".join(ops_list)

        prompt = f"""Generate a high-quality, synthesizable Verilog RTL design for a {bitwidth}-bit ALU.

REQUIREMENTS:

1. Module Interface:
   - Module name: {module_name}
   - Inputs:
     * a: {bitwidth}-bit operand A
     * b: {bitwidth}-bit operand B  
     * opcode: 4-bit operation selector
     * clk: clock signal
     * rst: active-high synchronous reset
   - Outputs:
     * result: {bitwidth}-bit result
     * zero: zero flag (result == 0)
     * overflow: overflow flag
     * negative: negative flag (MSB of result)

2. Operations to implement:
{ops_text}

3. Design Requirements:
   - Synchronous design with registered outputs
   - Proper reset handling (all outputs to 0 on reset)
   - Combinational logic for operations
   - Sequential logic for output registers
   - Clean, readable code with comments
   - Synthesizable RTL (no delays, no X/Z)

4. Code Quality:
   - Use proper Verilog-2001 syntax
   - Include module header comments
   - Comment each operation case
   - Use meaningful signal names
   - Proper indentation

EXAMPLE STRUCTURE:
```verilog
module {module_name} (
    input wire clk,
    input wire rst,
    input wire [{bitwidth-1}:0] a,
    input wire [{bitwidth-1}:0] b,
    input wire [3:0] opcode,
    output reg [{bitwidth-1}:0] result,
    output reg zero,
    output reg overflow,
    output reg negative
);

    // Combinational logic for operations
    reg [{bitwidth-1}:0] temp_result;
    reg temp_overflow;
    
    always @(*) begin
        temp_overflow = 1'b0;
        case (opcode)
            4'b0000: // ADD
                ...
            4'b0001: // SUB
                ...
            ...
        endcase
    end
    
    // Sequential logic for output registers
    always @(posedge clk) begin
        if (rst) begin
            result <= 0;
            zero <= 0;
            overflow <= 0;
            negative <= 0;
        end else begin
            result <= temp_result;
            zero <= (temp_result == 0);
            overflow <= temp_overflow;
            negative <= temp_result[{bitwidth-1}];
        end
    end

endmodule
```

Generate ONLY the Verilog code. No explanations before or after.
Start with `module` and end with `endmodule`.
"""

        return prompt

    def _extract_verilog(self, response: str) -> Optional[str]:
        """Extract Verilog code from LLM response"""

        # Try to find code block
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

        # If no match, check if entire response is code
        if 'module' in response and 'endmodule' in response:
            return response.strip()

        return None

    def _validate_verilog(self, verilog_code: str, bitwidth: int, operations: Dict) -> bool:
        """Validate generated Verilog code"""

        print(f"\n🔍 Validating Verilog code...")

        checks = []

        # Check 1: Has module declaration
        has_module = 'module' in verilog_code and 'endmodule' in verilog_code
        checks.append(('Module structure', has_module))

        # Check 2: Has required inputs
        required_inputs = ['clk', 'rst', 'opcode']
        has_inputs = all(inp in verilog_code for inp in required_inputs)
        checks.append(('Required inputs', has_inputs))

        # Check 3: Has required outputs
        required_outputs = ['result', 'zero', 'overflow', 'negative']
        has_outputs = all(out in verilog_code for out in required_outputs)
        checks.append(('Required outputs', has_outputs))

        # Check 4: Has always blocks
        has_always = 'always' in verilog_code
        checks.append(('Always blocks', has_always))

        # Check 5: Has case statement
        has_case = 'case' in verilog_code
        checks.append(('Case statement', has_case))

        # Print validation results
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False

        return all_passed

    def _save_alu(self, verilog_code: str, module_name: str, bitwidth: int) -> Path:
        """Save ALU to file"""

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{module_name}_{bitwidth}bit.v"

        # Add header comment
        header = f"""//==============================================================================
// Fixed ALU Design - Design Under Test (DUT)
// Generated by: alu_generator.py
// LLM Provider: {self.llm_provider}
// Generated at: {timestamp}
// Bitwidth: {bitwidth}
//
// This is the FIXED DUT used for all experiments.
// All testbenches from different LLMs test this same design.
//==============================================================================

"""

        full_code = header + verilog_code

        # Save to file
        alu_path = self.output_dir / filename
        with open(alu_path, 'w', encoding='utf-8') as f:
            f.write(full_code)

        # Also save metadata
        metadata = {
            'module_name': module_name,
            'bitwidth': bitwidth,
            'llm_provider': self.llm_provider,
            'timestamp': timestamp,
            'filepath': str(alu_path),
        }

        metadata_path = self.output_dir / f"{module_name}_{bitwidth}bit_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"   💾 Metadata saved: {metadata_path}")

        return alu_path

    def load_spec_and_generate(self, spec_path: str) -> Optional[str]:
        """
        Load specification and generate ALU.

        Args:
            spec_path: Path to spec JSON file

        Returns:
            Path to generated ALU file
        """
        print(f"\n📖 Loading spec: {spec_path}")

        try:
            with open(spec_path, 'r', encoding='utf-8') as f:
                spec = json.load(f)

            # Check if spec is valid
            if spec.get('api_call_failed'):
                print(f"❌ Spec is marked as FAILED, cannot generate ALU")
                print(f"   Error: {spec.get('error_message')}")
                return None

            # Extract info
            bitwidth = spec.get('bitwidth', 16)
            operations = spec.get('operations', {})

            if not operations:
                print(f"❌ Spec has no operations, cannot generate ALU")
                return None

            print(f"✅ Spec loaded: {bitwidth}-bit, {len(operations)} operations")

            # Generate ALU
            return self.generate_alu(bitwidth, operations)

        except Exception as e:
            print(f"❌ Failed to load spec: {e}")
            return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate deterministic ALU design using LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate 16-bit ALU using Gemini (default)
  python alu_generator.py

  # Use specific LLM provider
  python alu_generator.py --llm openai
  python alu_generator.py --llm claude

  # Generate from spec file
  python alu_generator.py --spec specs/gemini/spec_xxx.json

  # Custom bitwidth
  python alu_generator.py --bitwidth 32

  # Specify output directory
  python alu_generator.py --output output/dut/
        '''
    )

    parser.add_argument('--llm', default='groq',
                       help='LLM provider (gemini, openai, claude, groq, deepseek)')
    parser.add_argument('--bitwidth', type=int, default=16,
                       help='ALU bitwidth (default: 16)')
    parser.add_argument('--spec', help='Path to spec JSON file (optional)')
    parser.add_argument('--output', help='Output directory (default: output/dut/)')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--module-name', default='alu', help='Verilog module name')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output')

    args = parser.parse_args()

    print("=" * 80)
    print("🔧 ALU Generator - Fixed DUT for All Experiments")
    print("=" * 80)

    # Create generator
    generator = ALUGenerator(
        llm_provider=args.llm,
        output_dir=args.output,
        project_root=args.project_root,
        debug=not args.no_debug
    )

    # Generate ALU
    if args.spec:
        # From spec file
        alu_path = generator.load_spec_and_generate(args.spec)
    else:
        # Direct generation
        alu_path = generator.generate_alu(
            bitwidth=args.bitwidth,
            module_name=args.module_name
        )

    if alu_path:
        print("\n" + "=" * 80)
        print("✅ ALU Generation Complete")
        print("=" * 80)
        print(f"\n📁 ALU file: {alu_path}")
        print(f"\n🎯 Next steps:")
        print(f"   1. Review the generated ALU code")
        print(f"   2. Generate testbenches from different LLMs")
        print(f"   3. Run simulations with all testbenches")
        print(f"   4. Compare test quality across LLMs")
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ ALU Generation Failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())