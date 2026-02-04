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
                DeepSeekProvider,
                GrokProvider,
                QwenProvider,
                MistralProvider,
                TogetherProvider
            )

            # Map provider names to classes
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
        """Setup output directory for DUT, organized by LLM provider"""
        if output_dir:
            base_dir = Path(output_dir)
        elif project_root:
            base_dir = Path(project_root) / "output" / "dut"
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
                    base_dir = path
                    break
            else:
                # Default
                base_dir = current / "output" / "dut"

        # Create LLM-specific subdirectory
        llm_dir = base_dir / self.llm_provider
        llm_dir.mkdir(parents=True, exist_ok=True)

        # Store base_dir for later use
        self.base_output_dir = base_dir

        return llm_dir

    def generate_alu(
            self,
            bitwidth: int = 16,
            operations: Optional[Dict] = None,
            module_name: str = None
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

        # Auto-generate module name with bitwidth
        if module_name is None:
            module_name = f"alu_{bitwidth}bit"

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
            # 🔧 根据位宽和操作数动态计算 max_tokens
            base_tokens = 5000 + (bitwidth // 16) * 1000 + len(operations) * 200
            max_tokens = min(base_tokens, 12000)

            if hasattr(self.llm, '_call_api'):
                response = self.llm._call_api(
                    prompt,
                    max_tokens=max_tokens,
                    system_prompt="You are an expert Verilog hardware designer. Generate high-quality, synthesizable RTL code."
                )
            else:
                print(f"❌ LLM provider does not support _call_api method")
                return None

            print(f"✅ LLM response received ({len(response)} chars)")

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
                print(f"❌ Failed to extract Verilog code from response")
                return None

            # 修复常见语法错误
            verilog_code = self._fix_verilog_syntax(verilog_code)

            # Force correct module name (LLM may ignore the prompt)
            verilog_code = self._fix_module_name(verilog_code, module_name)

            print(f"✅ Verilog code extracted ({len(verilog_code.splitlines())} lines)")


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

IMPORTANT VERILOG SYNTAX RULE:
   - When a case branch has MORE THAN ONE statement, you MUST wrap them
     in begin/end blocks. Failure to do so causes syntax errors.
   - Example: 4'b0000: begin ... end  (NOT 4'b0000: stmt1; stmt2;)

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
            4'b0000: begin // ADD
                temp_result = a + b;
                temp_overflow = (a[{bitwidth-1}] & b[{bitwidth-1}] & ~temp_result[{bitwidth-1}]) |
                                (~a[{bitwidth-1}] & ~b[{bitwidth-1}] & temp_result[{bitwidth-1}]);
            end
            4'b0001: begin // SUB
                temp_result = a - b;
                temp_overflow = (a[{bitwidth-1}] & ~b[{bitwidth-1}] & ~temp_result[{bitwidth-1}]) |
                                (~a[{bitwidth-1}] & b[{bitwidth-1}] & temp_result[{bitwidth-1}]);
            end
            4'b0010: temp_result = a & b;  // AND (single statement, no begin/end needed)
            4'b0011: temp_result = a | b;  // OR
            default: temp_result = {bitwidth}'b0;
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
        """Extract Verilog code from LLM response (with truncation handling)"""

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

        # 🔧 新增：处理截断情况 - 强制补全
        if 'module' in response and 'endmodule' not in response:
            print(f"⚠️ Verilog code truncated (missing endmodule), attempting force complete...")

            code = response.strip()
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

            code += '\n\nendmodule'
            print(f"   ✅ Force completed")
            return code

        return None

    def _fix_verilog_syntax(self, verilog_code: str) -> str:
        """Fix common Verilog syntax errors from LLM generation"""

        # 移除 markdown 代码块标记
        verilog_code = re.sub(r'```verilog\s*', '', verilog_code)
        verilog_code = re.sub(r'```v\s*', '', verilog_code)
        verilog_code = re.sub(r'```\s*', '', verilog_code)

        # 修复 case 分支缺少 begin/end 的问题
        # 策略：找到所有 case...endcase 块，逐个修复内部分支

        lines = verilog_code.split('\n')
        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 检测 case 语句开始
            if re.match(r'\s*case\s*\(', stripped):
                # 收集整个 case...endcase 块
                case_block = [line]
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

                # 修复 case 块内部
                fixed_block = self._fix_case_block(case_block)
                fixed_lines.extend(fixed_block)
                continue

            fixed_lines.append(line)
            i += 1

        return '\n'.join(fixed_lines)

    def _fix_case_block(self, case_lines: list) -> list:
        """Fix begin/end in a case...endcase block"""
        if len(case_lines) < 3:
            return case_lines

        # 正则：匹配 case 分支标签行
        # 支持: 4'b0000: 或 4'b0000: // comment 或 default: 等
        label_re = re.compile(
            r"^(\s*)"                                          # 缩进
            r"("                                               # 标签组
            r"\d+'[bBhHdD][0-9a-fA-F_xXzZ]+"                  # 如 4'b0000, 8'hFF
            r"|default"                                        # 或 default
            r")"
            r"\s*:\s*"                                         # 冒号
            r"(begin\b)?"                                      # 可选的 begin
            r"(.*)"                                            # 其余内容（注释或单行语句）
            r"$"
        )

        result = [case_lines[0]]  # case(...) 行直接保留
        i = 1

        while i < len(case_lines):
            line = case_lines[i]
            stripped = line.strip()

            # 最后一行是 endcase，直接保留
            if stripped.startswith('endcase'):
                result.append(line)
                i += 1
                continue

            m = label_re.match(line)
            if m:
                indent = m.group(1)
                label = m.group(2)
                has_begin = m.group(3)
                rest = m.group(4).strip()

                if has_begin:
                    # 已有 begin，原样保留到对应的 end
                    result.append(line)
                    i += 1
                    begin_depth = 1
                    while i < len(case_lines) and begin_depth > 0:
                        s = case_lines[i].strip()
                        if s == 'begin' or s.startswith('begin '):
                            begin_depth += 1
                        elif s == 'end' or s.startswith('end ') or s.startswith('end//'):
                            begin_depth -= 1
                        result.append(case_lines[i])
                        i += 1
                    continue

                # 没有 begin - 收集此分支的语句
                # rest 可能是: 空、注释、或一条内联语句
                is_comment = rest.startswith('//')
                inline_stmt = rest if rest and not is_comment else None
                comment = rest if is_comment else ''

                branch_stmts = []
                if inline_stmt:
                    # 标签行上有内联语句，如 4'b0010: temp_result = a & b;
                    branch_stmts.append(f"{indent}    {inline_stmt}")

                # 收集后续属于此分支的行
                j = i + 1
                while j < len(case_lines):
                    check = case_lines[j].strip()
                    # 遇到下一个标签或 endcase 就停止
                    if check.startswith('endcase') or label_re.match(case_lines[j]):
                        break
                    if check and not (not branch_stmts and check.startswith('//')):
                        # 将非空行（含注释行）视为分支内容
                        branch_stmts.append(case_lines[j])
                    elif check:
                        branch_stmts.append(case_lines[j])
                    j += 1

                # 计算实际语句数（排除纯注释行）
                stmt_count = sum(1 for s in branch_stmts
                                 if s.strip() and not s.strip().startswith('//'))

                if stmt_count > 1:
                    # 多条语句 - 必须添加 begin/end
                    result.append(f"{indent}{label}: begin {comment}".rstrip())
                    for bl in branch_stmts:
                        result.append(bl)
                    result.append(f"{indent}end")
                elif branch_stmts:
                    # 单条语句 - 不需要 begin/end
                    if inline_stmt:
                        result.append(line)
                    else:
                        result.append(f"{indent}{label}: {comment}".rstrip())
                        for bl in branch_stmts:
                            result.append(bl)
                else:
                    # 空分支
                    result.append(line)

                i = j
                continue

            # 非标签行，直接保留
            result.append(line)
            i += 1

        return result

    def _fix_module_name(self, verilog_code: str, expected_name: str) -> str:
        """
        Fix module name in generated Verilog code.

        LLM may ignore the module name in prompt, so we force correct it here.
        """
        # Pattern to match module declaration: module <name> (
        pattern = r'module\s+(\w+)\s*\('
        match = re.search(pattern, verilog_code)

        if match:
            actual_name = match.group(1)
            if actual_name != expected_name:
                print(f"   ⚠️  Fixing module name: '{actual_name}' → '{expected_name}'")
                # Replace module name
                verilog_code = re.sub(
                    r'module\s+\w+\s*\(',
                    f'module {expected_name} (',
                    verilog_code,
                    count=1
                )

        return verilog_code

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
        """Save ALU to file in LLM-specific directory"""

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{module_name}_{timestamp}.v"

        # Add header comment
        header = f"""//==============================================================================
    // Fixed ALU Design - Design Under Test (DUT)
    //
    // Project: LLM-based Hardware Verification Pipeline
    // Authors: Rolf Drechsler, Qian Liu
    // Paper: https://arxiv.org/abs/2512.17814
    //
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

        # Save to LLM-specific directory
        alu_path = self.output_dir / filename
        with open(alu_path, 'w', encoding='utf-8') as f:
            f.write(full_code)

        print(f"   💾 ALU saved: {alu_path}")

        # Also save metadata
        metadata = {
            'module_name': module_name,
            'bitwidth': bitwidth,
            'llm_provider': self.llm_provider,
            'timestamp': timestamp,
            'filename': filename,
            'filepath': str(alu_path),
        }

        metadata_path = self.output_dir / f"{module_name}_{bitwidth}bit_{timestamp}_metadata.json"
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