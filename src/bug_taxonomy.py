#!/usr/bin/env python3
"""
ALU Bug Taxonomy and Fault Injection Framework
用于LLM-based Hardware Verification研究的系统性Bug分类

这个框架定义了ALU中可能出现的各类Bug，以及每类Bug的：
1. 检测方法（静态分析 vs 动态仿真 vs 形式化验证）
2. LLM-based验证能否检测
3. 需要什么样的测试才能暴露
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import re


class BugCategory(Enum):
    """Bug大类"""
    INTERFACE = "Interface-level"  # 接口级
    FUNCTIONAL = "Functional"  # 功能级
    TIMING = "Timing"  # 时序级
    STRUCTURAL = "Structural"  # 结构级


class DetectionMethod(Enum):
    """检测方法"""
    STATIC_ANALYSIS = "Static Analysis"
    DYNAMIC_SIMULATION = "Dynamic Simulation"
    FORMAL_VERIFICATION = "Formal Verification"
    LINT_CHECK = "Lint/Synthesis Check"


@dataclass
class BugType:
    """Bug类型定义"""
    name: str
    category: BugCategory
    description: str
    verilog_pattern: str  # 注入此Bug的代码模式
    detection_methods: List[DetectionMethod]
    llm_tb_detectable: bool  # LLM生成的testbench能否检测
    required_test_strategy: str  # 需要什么测试策略
    formal_property: Optional[str]  # 用于形式化验证的性质
    escape_reason: Optional[str]  # 如果能逃逸，原因是什么


# ALU Bug分类体系
ALU_BUG_TAXONOMY = {

    # ========== 接口级Bug ==========
    "PORT_WIDTH_EXPANSION": BugType(
        name="Port Width Expansion",
        category=BugCategory.INTERFACE,
        description="端口位宽大于规范要求 (e.g., [8:0] instead of [7:0])",
        verilog_pattern="input wire [N:0] signal  // N > expected",
        detection_methods=[DetectionMethod.STATIC_ANALYSIS, DetectionMethod.LINT_CHECK],
        llm_tb_detectable=False,
        required_test_strategy="Interface specification comparison + values requiring full extended width",
        formal_property="width(port) == spec_width",
        escape_reason="Verilog零扩展机制使8位值在9位端口上功能等价"
    ),

    "PORT_WIDTH_TRUNCATION": BugType(
        name="Port Width Truncation",
        category=BugCategory.INTERFACE,
        description="端口位宽小于规范要求 (e.g., [2:0] instead of [3:0])",
        verilog_pattern="input wire [N:0] signal  // N < expected",
        detection_methods=[DetectionMethod.STATIC_ANALYSIS, DetectionMethod.DYNAMIC_SIMULATION],
        llm_tb_detectable=True,  # 如果测试用到被截断的位
        required_test_strategy="测试需要使用被截断位的值 (e.g., opcode > 7)",
        formal_property="width(port) >= spec_width",
        escape_reason="若测试值恰好不使用高位则无法检测"
    ),

    "PORT_DIRECTION_ERROR": BugType(
        name="Port Direction Error",
        category=BugCategory.INTERFACE,
        description="端口方向错误 (input declared as output)",
        verilog_pattern="output wire signal  // should be input",
        detection_methods=[DetectionMethod.LINT_CHECK, DetectionMethod.STATIC_ANALYSIS],
        llm_tb_detectable=False,  # 通常会导致编译错误
        required_test_strategy="静态分析即可检测",
        formal_property="direction(port) == spec_direction",
        escape_reason=None  # 通常会被编译器捕获
    ),

    # ========== 功能级Bug ==========
    "OPCODE_MAPPING_ERROR": BugType(
        name="Opcode Mapping Error",
        category=BugCategory.FUNCTIONAL,
        description="操作码映射错误 (e.g., ADD mapped to 0001 instead of 0000)",
        verilog_pattern="4'b0001: temp_result = a + b;  // wrong opcode for ADD",
        detection_methods=[DetectionMethod.DYNAMIC_SIMULATION, DetectionMethod.FORMAL_VERIFICATION],
        llm_tb_detectable=True,  # 如果LLM使用正确的opcode
        required_test_strategy="使用规范定义的opcode执行所有操作",
        formal_property="(opcode == ADD_CODE) |-> (result == a + b)",
        escape_reason="若LLM的opcode假设与DUT一致则无法检测"
    ),

    "OPERATOR_SUBSTITUTION": BugType(
        name="Operator Substitution",
        category=BugCategory.FUNCTIONAL,
        description="运算符替换 (e.g., - instead of +)",
        verilog_pattern="temp_result = a - b;  // should be a + b",
        detection_methods=[DetectionMethod.DYNAMIC_SIMULATION, DetectionMethod.FORMAL_VERIFICATION],
        llm_tb_detectable=True,
        required_test_strategy="使用a≠b的测试值",
        formal_property="(opcode == ADD) |-> (result == a + b)",
        escape_reason=None  # 除非a==b或a==0
    ),

    "OFF_BY_ONE": BugType(
        name="Off-by-One Error",
        category=BugCategory.FUNCTIONAL,
        description="差一错误 (e.g., a + b + 1 instead of a + b)",
        verilog_pattern="temp_result = a + b + 1;",
        detection_methods=[DetectionMethod.DYNAMIC_SIMULATION, DetectionMethod.FORMAL_VERIFICATION],
        llm_tb_detectable=True,
        required_test_strategy="任何非特殊值的测试",
        formal_property="result == expected_result",
        escape_reason=None
    ),

    "OVERFLOW_LOGIC_ERROR": BugType(
        name="Overflow Logic Error",
        category=BugCategory.FUNCTIONAL,
        description="溢出检测逻辑错误",
        verilog_pattern="temp_overflow = (a[7] == b[7]);  // incomplete condition",
        detection_methods=[DetectionMethod.DYNAMIC_SIMULATION, DetectionMethod.FORMAL_VERIFICATION],
        llm_tb_detectable=True,  # 如果测试包含溢出用例
        required_test_strategy="边界值测试：127+1, -128-1, 等",
        formal_property="overflow == ((a[7]==b[7]) && (a[7]!=result[7]))",
        escape_reason="若测试不包含溢出情况则无法检测"
    ),

    "MISSING_DEFAULT_CASE": BugType(
        name="Missing Default Case",
        category=BugCategory.FUNCTIONAL,
        description="case语句缺少default分支",
        verilog_pattern="case(opcode) ... endcase  // no default",
        detection_methods=[DetectionMethod.LINT_CHECK, DetectionMethod.DYNAMIC_SIMULATION],
        llm_tb_detectable=True,  # 如果测试未定义的opcode
        required_test_strategy="测试所有可能的opcode值，包括未定义的",
        formal_property="full_case property",
        escape_reason="若测试仅使用已定义opcode则无法检测"
    ),

    # ========== 时序级Bug ==========
    "COMBINATIONAL_INSTEAD_OF_SEQUENTIAL": BugType(
        name="Combinational Instead of Sequential",
        category=BugCategory.TIMING,
        description="应该是时序逻辑但实现为组合逻辑",
        verilog_pattern="always @(*) result = ...  // should be @(posedge clk)",
        detection_methods=[DetectionMethod.DYNAMIC_SIMULATION, DetectionMethod.FORMAL_VERIFICATION],
        llm_tb_detectable=True,  # 如果testbench检查时序
        required_test_strategy="检查输出是否在时钟边沿变化",
        formal_property="$rose(clk) |-> result == $past(expected)",
        escape_reason="功能测试可能恰好在稳态检查"
    ),

    "ASYNC_RESET_INSTEAD_OF_SYNC": BugType(
        name="Async Reset Instead of Sync",
        category=BugCategory.TIMING,
        description="异步复位应为同步复位",
        verilog_pattern="always @(posedge clk or posedge rst)",
        detection_methods=[DetectionMethod.DYNAMIC_SIMULATION, DetectionMethod.FORMAL_VERIFICATION],
        llm_tb_detectable=False,  # 功能测试通常不区分
        required_test_strategy="在非时钟边沿assert reset",
        formal_property="reset_synchronous property",
        escape_reason="若reset仅在时钟边沿测试则无法区分"
    ),

    # ========== 结构级Bug ==========
    "SIGNAL_NOT_REGISTERED": BugType(
        name="Signal Not Registered",
        category=BugCategory.STRUCTURAL,
        description="输出信号未经寄存器",
        verilog_pattern="assign result = temp_result;  // should be registered",
        detection_methods=[DetectionMethod.STATIC_ANALYSIS, DetectionMethod.FORMAL_VERIFICATION],
        llm_tb_detectable=False,
        required_test_strategy="检查输出的寄存器属性",
        formal_property="output is registered",
        escape_reason="功能等价但时序特性不同"
    ),

    "LATCH_INFERENCE": BugType(
        name="Unintended Latch Inference",
        category=BugCategory.STRUCTURAL,
        description="意外的锁存器推断",
        verilog_pattern="always @(*) if(cond) out = val;  // latch on !cond",
        detection_methods=[DetectionMethod.LINT_CHECK, DetectionMethod.STATIC_ANALYSIS],
        llm_tb_detectable=False,
        required_test_strategy="Synthesis report / lint check",
        formal_property="no_latch property",
        escape_reason="锁存器在仿真中可能功能正确"
    ),
}


def analyze_detection_gaps() -> str:
    """分析LLM-based验证的检测盲区"""

    lines = []
    lines.append("=" * 80)
    lines.append("LLM-BASED VERIFICATION: DETECTION GAP ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # 统计
    total = len(ALU_BUG_TAXONOMY)
    detectable = sum(1 for b in ALU_BUG_TAXONOMY.values() if b.llm_tb_detectable)
    undetectable = total - detectable

    lines.append(f"Total Bug Types Analyzed: {total}")
    lines.append(f"LLM-TB Detectable: {detectable} ({100 * detectable / total:.1f}%)")
    lines.append(f"LLM-TB Undetectable: {undetectable} ({100 * undetectable / total:.1f}%)")
    lines.append("")

    # 按类别分析
    lines.append("BY CATEGORY:")
    lines.append("-" * 40)

    for category in BugCategory:
        bugs_in_cat = [b for b in ALU_BUG_TAXONOMY.values() if b.category == category]
        detect_in_cat = sum(1 for b in bugs_in_cat if b.llm_tb_detectable)
        lines.append(f"\n  {category.value}:")
        lines.append(f"    Total: {len(bugs_in_cat)}, Detectable: {detect_in_cat}")

        for name, bug in ALU_BUG_TAXONOMY.items():
            if bug.category == category:
                status = "✓" if bug.llm_tb_detectable else "✗"
                lines.append(f"    {status} {name}")
                if not bug.llm_tb_detectable and bug.escape_reason:
                    lines.append(f"        └─ Escape: {bug.escape_reason}")

    # 关键发现
    lines.append("")
    lines.append("KEY FINDINGS FOR RESEARCH:")
    lines.append("-" * 40)
    lines.append("""
  1. Interface-level bugs have the HIGHEST escape rate
     - LLM assumes canonical interface specification
     - Verilog's implicit type conversion masks mismatches

  2. Timing/Structural bugs are systematically UNDETECTABLE
     - Functional equivalence ≠ Implementation correctness
     - Requires formal property checking or synthesis analysis

  3. Functional bugs are MOSTLY detectable, BUT:
     - Requires comprehensive test vectors
     - Edge cases (overflow, boundary) need explicit targeting

  4. RECOMMENDED HYBRID APPROACH:
     - LLM-generated TB for functional coverage
     - Static analysis for interface/structural bugs
     - Formal verification for timing properties
""")

    lines.append("=" * 80)
    return "\n".join(lines)


def generate_fault_injection_suite(dut_code: str) -> Dict[str, str]:
    """
    为给定的DUT生成故障注入变体

    Returns:
        Dict[bug_name, mutated_code]
    """
    mutations = {}

    # 接口级Bug注入
    # 1. 端口扩展
    pattern = r'input\s+wire\s+\[7:0\]\s+a'
    if re.search(pattern, dut_code):
        mutations["PORT_WIDTH_EXPANSION_A"] = re.sub(
            pattern,
            'input wire [8:0] a',
            dut_code
        )

    # 2. Opcode截断
    pattern = r'input\s+wire\s+\[3:0\]\s+opcode'
    if re.search(pattern, dut_code):
        mutations["PORT_WIDTH_TRUNCATION_OPCODE"] = re.sub(
            pattern,
            'input wire [2:0] opcode',
            dut_code
        )

    # 功能级Bug注入
    # 3. 运算符替换
    pattern = r'temp_result\s*=\s*a\s*\+\s*b'
    if re.search(pattern, dut_code):
        mutations["OPERATOR_SUB_ADD_TO_SUB"] = re.sub(
            pattern,
            'temp_result = a - b',
            dut_code
        )

    # 4. 溢出逻辑错误
    pattern = r'temp_overflow\s*=\s*\(a\[7\]\s*==\s*b\[7\]\)\s*&&\s*\(a\[7\]\s*!=\s*temp_result\[7\]\)'
    if re.search(pattern, dut_code):
        mutations["OVERFLOW_LOGIC_INCOMPLETE"] = re.sub(
            pattern,
            'temp_overflow = (a[7] == b[7])',  # 移除第二个条件
            dut_code
        )

    # 5. Off-by-one
    pattern = r'temp_result\s*=\s*a\s*\+\s*b;'
    if re.search(pattern, dut_code):
        mutations["OFF_BY_ONE_ADD"] = re.sub(
            pattern,
            'temp_result = a + b + 1;',
            dut_code,
            count=1  # 只替换第一个
        )

    return mutations


if __name__ == "__main__":
    # 打印检测盲区分析
    print(analyze_detection_gaps())