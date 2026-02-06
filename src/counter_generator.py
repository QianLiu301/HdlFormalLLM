"""
Counter Generator - Generate Counter Verilog using LLM
=======================================================

Generates parameterized counter designs with:
- Configurable bitwidth (8/16/32-bit)
- Multiple modes: Up, Down, Up-Down
- Features: Enable, Load, Reset
- Outputs: Count, Overflow, Zero flags

Part of the Hardware Generator Pipeline.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List


class CounterGenerator:
    """
    Generate Counter Verilog design using LLM.
    """

    def __init__(
        self,
        llm_provider: str = 'groq',
        output_dir: Optional[str] = None,
        project_root: Optional[str] = None,
        debug: bool = True
    ):
        """
        Initialize Counter generator.

        Args:
            llm_provider: LLM to use ('groq', 'deepseek', 'openai', etc.)
            output_dir: Output directory for Counter file
            project_root: Project root directory
            debug: Enable debug output
        """
        self.llm_provider = llm_provider.lower()
        self.debug = debug

        # Setup LLM
        self.llm = self._setup_llm()

        # Setup output directory
        self.output_dir = self._setup_output_dir(output_dir, project_root)

        print(f"🔧 Counter Generator initialized")
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

    def generate_counter(
        self,
        bitwidth: int = 16,
        modes: Optional[List[str]] = None,
        module_name: str = "counter"
    ) -> str:
        """
        Generate Counter design.

        Args:
            bitwidth: Counter bitwidth (8, 16, 32)
            modes: List of modes to support ['up', 'down', 'updown']
            module_name: Verilog module name

        Returns:
            Path to generated Counter file
        """
        print("\n" + "=" * 80)
        print(f"🔧 Generating {bitwidth}-bit Counter using {self.llm_provider.upper()}")
        print("=" * 80)

        # Default modes
        if modes is None:
            modes = ['up', 'down', 'updown']

        # Create prompt
        prompt = self._create_counter_prompt(bitwidth, modes, module_name)

        if self.debug:
            print(f"\n📝 Prompt preview:")
            print(prompt[:500] + "...")

        # Call LLM
        print(f"\n🤖 Calling {self.llm_provider.upper()} API...")

        try:
            # 🔧 根据位宽动态计算 max_tokens（修复截断问题）
            base_tokens = 4000 + (bitwidth // 16) * 1000  # 64位约6000
            max_tokens = min(base_tokens, 12000)

            if hasattr(self.llm, '_call_api'):
                response = self.llm._call_api(
                    prompt,
                    max_tokens=max_tokens,
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

            # 🔧 新增：截断检测和自动重试
            if not verilog_code and 'module' in response and 'endmodule' not in response:
                print(f"⚠️ Code appears truncated! Retrying with more tokens...")
                retry_tokens = min(max_tokens * 2, 16000)
                response = self.llm._call_api(
                    prompt,
                    max_tokens=retry_tokens,
                    system_prompt="You are an expert Verilog hardware designer. Generate high-quality, synthesizable RTL code."
                )
                if response:
                    verilog_code = self._extract_verilog(response)

            if not verilog_code:
                print(f"❌ Could not extract valid Verilog code")
                print(f"   Raw response preview: {response[:200]}...")
                return None

            # Fix common syntax errors (missing begin/end in case branches)
            verilog_code = self._fix_verilog_syntax(verilog_code)

            # Validate
            if self._validate_verilog(verilog_code, bitwidth, modes):
                print(f"✅ Verilog validation passed")
            else:
                print(f"⚠️  Verilog validation had warnings (continuing anyway)")

            # Save
            counter_path = self._save_counter(verilog_code, module_name, bitwidth, modes)
            print(f"\n💾 Counter saved to: {counter_path}")

            return str(counter_path)

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_counter_prompt(self, bitwidth: int, modes: List[str], module_name: str) -> str:
        """Create prompt for Counter generation"""

        modes_desc = []
        if 'up' in modes:
            modes_desc.append("UP (2'b00): Count upward")
        if 'down' in modes:
            modes_desc.append("DOWN (2'b01): Count downward")
        if 'updown' in modes:
            modes_desc.append("UP-DOWN (2'b10): Count up then down (ping-pong)")

        prompt = f"""Generate a synthesizable Verilog module for a {bitwidth}-bit counter.

## Requirements

### Module Interface
```verilog
module {module_name} #(
    parameter WIDTH = {bitwidth}
)(
    input  wire             clk,        // Clock
    input  wire             rst_n,      // Active-low reset
    input  wire             enable,     // Count enable
    input  wire             load,       // Load preset value
    input  wire [WIDTH-1:0] load_value, // Preset value to load
    input  wire [1:0]       mode,       // Counter mode
    output reg  [WIDTH-1:0] count,      // Current count value
    output reg              overflow,   // Overflow flag
    output wire             zero        // Zero flag
);
```

### Counter Modes
{chr(10).join('- ' + m for m in modes_desc)}

### Functional Requirements
1. **Reset**: When rst_n is low, count resets to 0
2. **Load**: When load is high, count loads from load_value
3. **Enable**: Counter only counts when enable is high
4. **Overflow**: 
   - In UP mode: overflow when count transitions from MAX to 0
   - In DOWN mode: overflow when count transitions from 0 to MAX
   - In UP-DOWN mode: overflow at both boundaries
5. **Zero**: High when count equals 0

### Implementation Notes
- Use synchronous design (all updates on posedge clk)
- Priority: rst_n > load > enable
- For UP-DOWN mode, use internal direction register
- Ensure clean, synthesizable code

### Expected Behavior Example (8-bit UP mode)
```
count: 0xFD -> 0xFE -> 0xFF -> 0x00 (overflow=1) -> 0x01
```
### CRITICAL Verilog Rules (MUST follow)
1. Signals assigned inside `always` blocks MUST be declared as `reg`, not `wire`
2. Use blocking assignment (=) in combinational always @(*) blocks
3. Use non-blocking assignment (<=) in sequential always @(posedge clk) blocks
4. The `direction` register should ONLY be updated in the sequential always block
5. Ensure all cases in combinational logic have default values to avoid latches
6. When a case branch has MORE THAN ONE statement, you MUST wrap them in begin/end blocks
7. Do NOT mix blocking and non-blocking assignments for the same signal
7. NEVER use `assign` for `reg` signals - `assign` is ONLY for `wire` types
8. Output ports declared as `reg` should be assigned directly in the always block, no extra `assign` needed
9. Use Verilog-2001 ONLY. Do NOT use SystemVerilog (no `logic`, no inline for-loop declarations)
10. For loops: declare `integer i;` BEFORE the for, then use `for (i = 0; ...)`

### Required Implementation for zero flag
The `zero` flag MUST be implemented using assign statement (NOT inside always block):
```verilog
assign zero = (count == 0);
```
Do NOT put zero assignment inside any always block.

## Output Format
Generate ONLY the Verilog code. No explanations.
Start with `module` and end with `endmodule`.
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

        return result

    def _fix_integer_in_always(self, verilog_code: str) -> str:
        """Move integer declarations from inside always blocks to module level."""
        lines = verilog_code.split('\n')
        fixed_lines = []
        integers_to_declare = []
        in_always = False
        always_depth = 0
        module_end_idx = -1

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if module_end_idx == -1 and (stripped.startswith('reg ') or
                                          stripped.startswith('wire ') or
                                          stripped.startswith('assign ') or
                                          stripped.startswith('always') or
                                          stripped.startswith('localparam') or
                                          stripped.startswith('parameter')):
                module_end_idx = len(fixed_lines)

            if re.match(r'\s*always\s*@', stripped):
                in_always = True
                always_depth = 0

            if in_always:
                always_depth += stripped.count('begin')
                always_depth -= stripped.count('end')
                if always_depth <= 0 and 'end' in stripped:
                    in_always = False

                int_match = re.match(r'^(\s*)integer\s+(\w+)\s*;', line)
                if int_match:
                    var_name = int_match.group(2)
                    if var_name not in integers_to_declare:
                        integers_to_declare.append(var_name)
                    i += 1
                    continue

            fixed_lines.append(line)
            i += 1

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
        """Extract Verilog code from LLM response (with truncation handling)"""

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

        # 🔧 新增：处理截断情况 - 强制补全
        if 'module' in response and 'endmodule' not in response:
            print(f"⚠️ Verilog code truncated (missing endmodule), attempting force complete...")

            code = response.strip()
            # 移除 markdown 标记
            if code.endswith('```'):
                code = code[:-3].strip()

            # 补全 endcase
            case_count = len(re.findall(r'\bcase\b', code))
            endcase_count = code.count('endcase')
            while endcase_count < case_count:
                code += '\n            endcase'
                endcase_count += 1

            # 补全 end
            begin_count = code.count('begin')
            end_count = len(re.findall(r'\bend\b(?!\w)', code))
            while end_count < begin_count:
                code += '\n    end'
                end_count += 1

            # 补全 endmodule
            code += '\n\nendmodule'

            print(f"   ✅ Force completed (added {begin_count - end_count + case_count - endcase_count + 1} closing statements)")
            return code

        return None

    def _validate_verilog(self, verilog_code: str, bitwidth: int, modes: List[str]) -> bool:
        """Validate generated Verilog code"""

        print(f"\n🔍 Validating Verilog code...")

        checks = []

        # Check 1: Has module declaration
        has_module = 'module' in verilog_code and 'endmodule' in verilog_code
        checks.append(('Module structure', has_module))

        # Check 2: Has required inputs
        required_inputs = ['clk', 'rst_n', 'enable', 'load', 'mode']
        has_inputs = all(inp in verilog_code for inp in required_inputs)
        checks.append(('Required inputs', has_inputs))

        # Check 3: Has required outputs
        required_outputs = ['count', 'overflow', 'zero']
        has_outputs = all(out in verilog_code for out in required_outputs)
        checks.append(('Required outputs', has_outputs))

        # Check 4: Has always block
        has_always = 'always' in verilog_code
        checks.append(('Always block', has_always))

        # Check 5: Has case or if for mode handling
        has_mode_logic = 'case' in verilog_code or ('mode' in verilog_code and 'if' in verilog_code)
        checks.append(('Mode handling', has_mode_logic))

        # Print validation results
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False

        return all_passed

    def _save_counter(self, verilog_code: str, module_name: str, bitwidth: int, modes: List[str]) -> Path:
        """Save Counter to file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{module_name}_{timestamp}.v"

        modes_str = ', '.join(modes).upper()
        header = f"""//==============================================================================
// Counter Design - Design Under Test (DUT)
//
// Project: LLM-based Hardware Verification Pipeline
// Authors: Rolf Drechsler, Qian Liu
// Paper: https://arxiv.org/abs/2512.17814
//
// Generated by: counter_generator.py
// LLM Provider: {self.llm_provider}
// Generated at: {timestamp}
// Bitwidth: {bitwidth}
// Modes: {modes_str}
//
// Features:
//   - Synchronous reset (active low)
//   - Preset/Load capability
//   - Enable control
//   - Overflow and Zero flags
//==============================================================================

"""

        full_code = header + verilog_code

        counter_path = self.output_dir / filename
        with open(counter_path, 'w', encoding='utf-8') as f:
            f.write(full_code)

        # Save metadata
        metadata = {
            'module_name': module_name,
            'module_type': 'counter',
            'bitwidth': bitwidth,
            'modes': modes,
            'llm_provider': self.llm_provider,
            'timestamp': timestamp,
            'filepath': str(counter_path),
        }

        metadata_path = self.output_dir / f"{module_name}_{bitwidth}bit_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"   💾 Metadata saved: {metadata_path}")

        return counter_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate Counter Verilog design using LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate 16-bit counter using Groq (default)
  python counter_generator.py

  # Use specific LLM provider
  python counter_generator.py --llm deepseek

  # Custom bitwidth
  python counter_generator.py --bitwidth 32

  # Specify output directory
  python counter_generator.py --output output/dut/
        '''
    )

    parser.add_argument('--llm', default='groq',
                       help='LLM provider (groq, deepseek, openai, claude, gemini)')
    parser.add_argument('--bitwidth', type=int, default=16,
                       help='Counter bitwidth (default: 16)')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--module-name', default='counter', help='Verilog module name')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output')

    args = parser.parse_args()

    print("=" * 80)
    print("🔧 Counter Generator")
    print("=" * 80)

    generator = CounterGenerator(
        llm_provider=args.llm,
        output_dir=args.output,
        project_root=args.project_root,
        debug=not args.no_debug
    )

    counter_path = generator.generate_counter(
        bitwidth=args.bitwidth,
        module_name=args.module_name
    )

    if counter_path:
        print("\n" + "=" * 80)
        print("✅ Counter Generation Complete")
        print("=" * 80)
        print(f"\n📁 Counter file: {counter_path}")
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ Counter Generation Failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())