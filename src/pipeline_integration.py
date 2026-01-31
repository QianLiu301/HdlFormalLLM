#!/usr/bin/env python3
"""
Pipeline Integration Module
展示如何将接口Bug检测集成到LLM-based Hardware Verification Pipeline

流程变化:
  原流程: DUT → LLM → .feature → tb.v → simulation
  新流程: DUT → [Interface Check] → LLM → .feature → tb.v → [Consistency Check] → simulation
                    ↓                                              ↓
              Bug Report (if any)                           Mismatch Report
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

# 导入接口检查器
from dut_interface_checker import (
    InterfaceBugDetector,
    VerilogInterfaceParser,
    ALUSpecification,
    BugSeverity,
    InterfaceBug
)


class PipelineStage(Enum):
    DUT_PARSING = "DUT Parsing"
    INTERFACE_CHECK = "Interface Validation"
    FEATURE_GENERATION = "BDD Feature Generation"
    TB_GENERATION = "Testbench Generation"
    CONSISTENCY_CHECK = "TB-DUT Consistency Check"
    SIMULATION = "Simulation"


@dataclass
class PipelineResult:
    stage: PipelineStage
    passed: bool
    message: str
    details: Optional[dict] = None
    blocking: bool = True  # 是否阻止后续流程


class VerificationPipeline:
    """带Bug检测的验证Pipeline"""

    def __init__(self, dut_path: str, bitwidth: int = 8):
        self.dut_path = Path(dut_path)
        self.bitwidth = bitwidth
        self.results: List[PipelineResult] = []
        self.dut_code: Optional[str] = None
        self.dut_ports: Dict = {}
        self.interface_bugs: List[InterfaceBug] = []

    def run_stage_1_interface_check(self) -> PipelineResult:
        """Stage 1: 解析DUT并检查接口规范"""

        # 读取DUT
        try:
            with open(self.dut_path, 'r') as f:
                self.dut_code = f.read()
        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.DUT_PARSING,
                passed=False,
                message=f"Failed to read DUT file: {e}",
                blocking=True
            )

        # 接口检查
        detector = InterfaceBugDetector(self.dut_code, self.bitwidth)
        self.interface_bugs = detector.check_all()
        self.dut_ports = detector.actual_ports

        critical_bugs = [b for b in self.interface_bugs if b.severity == BugSeverity.CRITICAL]

        if critical_bugs:
            details = {
                'total_bugs': len(self.interface_bugs),
                'critical_bugs': len(critical_bugs),
                'bugs': [
                    {
                        'port': b.port_name,
                        'line': b.line_number,
                        'type': b.bug_type,
                        'expected': b.expected,
                        'actual': b.actual,
                        'code': b.raw_line,
                        'impact': b.impact
                    }
                    for b in critical_bugs
                ]
            }

            return PipelineResult(
                stage=PipelineStage.INTERFACE_CHECK,
                passed=False,
                message=f"Found {len(critical_bugs)} critical interface bugs",
                details=details,
                blocking=True
            )

        return PipelineResult(
            stage=PipelineStage.INTERFACE_CHECK,
            passed=True,
            message="Interface validation passed",
            blocking=False
        )

    def run_stage_2_tb_consistency_check(self, tb_path: str) -> PipelineResult:
        """Stage 2: 检查Testbench与DUT的一致性"""

        try:
            with open(tb_path, 'r') as f:
                tb_code = f.read()
        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.CONSISTENCY_CHECK,
                passed=False,
                message=f"Failed to read testbench file: {e}",
                blocking=True
            )

        # 解析testbench中的信号声明
        tb_signals = self._extract_tb_signals(tb_code)
        mismatches = []

        for port_name, dut_port in self.dut_ports.items():
            if port_name in tb_signals:
                tb_width = tb_signals[port_name]['width']
                if tb_width != dut_port.width:
                    mismatches.append({
                        'signal': port_name,
                        'dut_width': dut_port.width,
                        'tb_width': tb_width,
                        'dut_line': dut_port.line_number,
                        'dut_code': dut_port.raw_declaration,
                        'tb_code': tb_signals[port_name]['declaration']
                    })

        if mismatches:
            return PipelineResult(
                stage=PipelineStage.CONSISTENCY_CHECK,
                passed=False,
                message=f"Found {len(mismatches)} DUT-TB port width mismatches",
                details={'mismatches': mismatches},
                blocking=True
            )

        return PipelineResult(
            stage=PipelineStage.CONSISTENCY_CHECK,
            passed=True,
            message="Testbench-DUT consistency check passed",
            blocking=False
        )

    def _extract_tb_signals(self, tb_code: str) -> Dict:
        """从testbench提取信号声明"""
        import re
        signals = {}

        # 匹配 reg/wire 声明
        pattern = re.compile(
            r'(reg|wire)\s*(?:\[(\d+):(\d+)\])?\s*(\w+)\s*[;,]',
            re.IGNORECASE
        )

        for line in tb_code.split('\n'):
            line_clean = re.sub(r'//.*$', '', line)
            for match in pattern.finditer(line_clean):
                sig_type, msb, lsb, name = match.groups()
                width = int(msb) - int(lsb) + 1 if msb else 1
                signals[name] = {
                    'type': sig_type,
                    'width': width,
                    'declaration': line.strip()
                }

        return signals

    def generate_full_report(self) -> str:
        """生成完整的Pipeline检测报告"""
        lines = []
        lines.append("=" * 80)
        lines.append("LLM-BASED HARDWARE VERIFICATION PIPELINE REPORT")
        lines.append("=" * 80)
        lines.append(f"DUT File: {self.dut_path}")
        lines.append(f"Target Bitwidth: {self.bitwidth}")
        lines.append("")

        # 运行接口检查
        interface_result = self.run_stage_1_interface_check()
        self.results.append(interface_result)

        lines.append(f"[Stage 1] {interface_result.stage.value}")
        lines.append(f"  Status: {'✓ PASSED' if interface_result.passed else '✗ FAILED'}")
        lines.append(f"  Message: {interface_result.message}")

        if not interface_result.passed and interface_result.details:
            lines.append("")
            lines.append("  DETECTED INTERFACE BUGS:")
            lines.append("  " + "-" * 60)

            for i, bug in enumerate(interface_result.details['bugs'], 1):
                lines.append(f"")
                lines.append(f"  [{i}] Port: {bug['port']}")
                lines.append(f"      Line {bug['line']}: {bug['code']}")
                lines.append(f"      Expected: {bug['expected']}")
                lines.append(f"      Actual:   {bug['actual']}")
                lines.append(f"      Impact:   {bug['impact']}")

            lines.append("")
            lines.append("  " + "=" * 60)
            lines.append("  ⚠️  PIPELINE BLOCKED: Fix interface bugs before proceeding")
            lines.append("  " + "=" * 60)

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


def analyze_bug_impact_for_research(bugs: List[dict]) -> str:
    """
    为研究论文生成Bug影响分析

    这个函数帮助你在论文中讨论：
    1. Bug的类型分类
    2. 检测难度分析
    3. 对验证流程的影响
    """

    lines = []
    lines.append("=" * 80)
    lines.append("BUG IMPACT ANALYSIS FOR RESEARCH")
    lines.append("=" * 80)
    lines.append("")

    # 1. Bug分类
    lines.append("1. BUG TAXONOMY")
    lines.append("-" * 40)

    bug_categories = {
        'WIDTH_MISMATCH': {
            'description': 'Port bit-width differs from specification',
            'detection_method': 'Static analysis / AST comparison',
            'formal_property': '∀p ∈ Ports: width(DUT.p) = width(Spec.p)',
            'escapes_simulation': True,
            'reason': 'Verilog implicit type conversion masks the mismatch'
        },
        'DIRECTION_MISMATCH': {
            'description': 'Port direction (input/output) is incorrect',
            'detection_method': 'Static analysis',
            'formal_property': '∀p ∈ Ports: dir(DUT.p) = dir(Spec.p)',
            'escapes_simulation': False,
            'reason': 'Usually causes elaboration/compilation error'
        },
        'MISSING_PORT': {
            'description': 'Required port is absent from module',
            'detection_method': 'Port enumeration',
            'formal_property': 'Spec.Ports ⊆ DUT.Ports',
            'escapes_simulation': False,
            'reason': 'Causes compilation error on instantiation'
        }
    }

    for bug_type, info in bug_categories.items():
        lines.append(f"\n  {bug_type}:")
        lines.append(f"    Description: {info['description']}")
        lines.append(f"    Detection: {info['detection_method']}")
        lines.append(f"    Formal: {info['formal_property']}")
        lines.append(f"    Escapes Simulation: {'Yes' if info['escapes_simulation'] else 'No'}")
        if info['escapes_simulation']:
            lines.append(f"    Why: {info['reason']}")

    # 2. 研究意义
    lines.append("")
    lines.append("2. RESEARCH IMPLICATIONS")
    lines.append("-" * 40)
    lines.append("""
  This bug injection experiment demonstrates a critical limitation in
  LLM-based hardware verification:

  Problem: LLM-generated testbenches assume a canonical interface
           specification without parsing the actual DUT implementation.

  Consequence: Interface-level bugs that don't cause compilation errors
               can escape detection entirely.

  Metric Impact: 100% test pass rate becomes a FALSE NEGATIVE indicator
                 when DUT-TB interface mismatch exists.

  Proposed Solution: Add interface consistency checking as a mandatory
                     pre-simulation gate in the verification pipeline.
""")

    # 3. 论文中的呈现建议
    lines.append("3. SUGGESTED PAPER PRESENTATION")
    lines.append("-" * 40)
    lines.append("""
  Section: "Limitations and Failure Mode Analysis"

  Key Points to Discuss:

  a) Oracle Problem in LLM-based Verification
     - LLM generates expected values based on assumed specification
     - No ground truth comparison against actual DUT interface

  b) Implicit Type Conversion as Bug Mask
     - Verilog's permissive width matching hides interface bugs
     - Standard simulation tools don't flag these as errors

  c) Proposed Mitigation
     - Formal interface specification as pipeline input
     - Mandatory DUT parsing before testbench generation
     - SVA assertions for runtime interface checking

  d) Evaluation Metric Refinement
     - Pass rate alone is insufficient
     - Need: Bug Detection Rate under fault injection
     - Need: Interface Conformance Score
""")

    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    # 测试Pipeline
    dut_file = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/alu_8bit_20260128_test.v"

    pipeline = VerificationPipeline(dut_file, bitwidth=8)
    report = pipeline.generate_full_report()
    print(report)

    # 生成研究分析
    if pipeline.results and not pipeline.results[0].passed:
        print("\n")
        analysis = analyze_bug_impact_for_research(pipeline.results[0].details['bugs'])
        print(analysis)