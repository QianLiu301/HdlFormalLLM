#!/usr/bin/env python3
"""
DUT Interface Checker - 检测Verilog模块接口的Bug
用于LLM-based Hardware Verification Pipeline

功能：
1. 从DUT源码提取实际接口定义
2. 与预期规范（golden spec）对比
3. 定位并报告具体的不一致位置
"""

import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class PortDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class BugSeverity(Enum):
    CRITICAL = "CRITICAL"  # 会导致功能错误
    WARNING = "WARNING"  # 可能导致问题
    INFO = "INFO"  # 风格/规范问题


@dataclass
class PortDefinition:
    name: str
    direction: PortDirection
    width: int
    msb: int
    lsb: int
    line_number: int
    raw_declaration: str


@dataclass
class InterfaceBug:
    bug_type: str
    severity: BugSeverity
    port_name: str
    expected: str
    actual: str
    line_number: int
    raw_line: str
    explanation: str
    impact: str


class VerilogInterfaceParser:
    """解析Verilog模块接口"""

    def __init__(self, verilog_code: str):
        self.code = verilog_code
        self.lines = verilog_code.split('\n')

    def extract_module_name(self) -> Optional[str]:
        """提取模块名"""
        match = re.search(r'module\s+(\w+)\s*[\(#]', self.code)
        return match.group(1) if match else None

    def extract_ports(self) -> Dict[str, PortDefinition]:
        """提取所有端口定义"""
        ports = {}

        # 匹配端口声明的正则表达式
        # 支持格式: input wire [7:0] a, input [7:0] b, input a
        port_pattern = re.compile(
            r'(input|output|inout)\s+'
            r'(?:wire|reg)?\s*'
            r'(?:\[(\d+):(\d+)\])?\s*'
            r'(\w+)',
            re.IGNORECASE
        )

        for line_num, line in enumerate(self.lines, 1):
            # 跳过注释
            line_clean = re.sub(r'//.*$', '', line)

            for match in port_pattern.finditer(line_clean):
                direction_str, msb_str, lsb_str, port_name = match.groups()

                direction = PortDirection(direction_str.lower())
                msb = int(msb_str) if msb_str else 0
                lsb = int(lsb_str) if lsb_str else 0
                width = msb - lsb + 1 if msb_str else 1

                ports[port_name] = PortDefinition(
                    name=port_name,
                    direction=direction,
                    width=width,
                    msb=msb,
                    lsb=lsb,
                    line_number=line_num,
                    raw_declaration=line.strip()
                )

        return ports


class ALUSpecification:
    """ALU的Golden Specification"""

    @staticmethod
    def get_expected_interface(bitwidth: int) -> Dict[str, dict]:
        """返回预期的ALU接口规范"""
        return {
            'clk': {
                'direction': PortDirection.INPUT,
                'width': 1,
                'description': 'Clock signal'
            },
            'rst': {
                'direction': PortDirection.INPUT,
                'width': 1,
                'description': 'Reset signal'
            },
            'a': {
                'direction': PortDirection.INPUT,
                'width': bitwidth,
                'description': f'Operand A ({bitwidth}-bit)'
            },
            'b': {
                'direction': PortDirection.INPUT,
                'width': bitwidth,
                'description': f'Operand B ({bitwidth}-bit)'
            },
            'opcode': {
                'direction': PortDirection.INPUT,
                'width': 4,  # 支持16种操作
                'description': 'Operation code (4-bit)'
            },
            'result': {
                'direction': PortDirection.OUTPUT,
                'width': bitwidth,
                'description': f'Result ({bitwidth}-bit)'
            },
            'zero': {
                'direction': PortDirection.OUTPUT,
                'width': 1,
                'description': 'Zero flag'
            },
            'overflow': {
                'direction': PortDirection.OUTPUT,
                'width': 1,
                'description': 'Overflow flag'
            },
            'negative': {
                'direction': PortDirection.OUTPUT,
                'width': 1,
                'description': 'Negative flag'
            }
        }


class CounterSpecification:
    """Counter的Golden Specification"""

    @staticmethod
    def get_expected_interface(bitwidth: int) -> Dict[str, dict]:
        """返回预期的Counter接口规范"""
        return {
            'clk': {
                'direction': PortDirection.INPUT,
                'width': 1,
                'description': 'Clock signal'
            },
            'rst_n': {
                'direction': PortDirection.INPUT,
                'width': 1,
                'description': 'Active-low reset'
            },
            'enable': {
                'direction': PortDirection.INPUT,
                'width': 1,
                'description': 'Count enable'
            },
            'load': {
                'direction': PortDirection.INPUT,
                'width': 1,
                'description': 'Load preset value'
            },
            'load_value': {
                'direction': PortDirection.INPUT,
                'width': bitwidth,
                'description': f'Preset value to load ({bitwidth}-bit)'
            },
            'mode': {
                'direction': PortDirection.INPUT,
                'width': 2,
                'description': 'Counter mode (2-bit)'
            },
            'count': {
                'direction': PortDirection.OUTPUT,
                'width': bitwidth,
                'description': f'Current count value ({bitwidth}-bit)'
            },
            'overflow': {
                'direction': PortDirection.OUTPUT,
                'width': 1,
                'description': 'Overflow flag'
            },
            'zero': {
                'direction': PortDirection.OUTPUT,
                'width': 1,
                'description': 'Zero flag'
            }
        }


class RegFileSpecification:
    """Register File的Golden Specification"""

    @staticmethod
    def get_expected_interface(bitwidth: int, depth: int = 32) -> Dict[str, dict]:
        """返回预期的Register File接口规范"""
        import math
        addr_width = max(1, int(math.ceil(math.log2(depth))))

        return {
            'clk': {
                'direction': PortDirection.INPUT,
                'width': 1,
                'description': 'Clock signal'
            },
            'rst_n': {
                'direction': PortDirection.INPUT,
                'width': 1,
                'description': 'Active-low reset'
            },
            'raddr1': {
                'direction': PortDirection.INPUT,
                'width': addr_width,
                'description': f'Read address 1 ({addr_width}-bit)'
            },
            'rdata1': {
                'direction': PortDirection.OUTPUT,
                'width': bitwidth,
                'description': f'Read data 1 ({bitwidth}-bit)'
            },
            'raddr2': {
                'direction': PortDirection.INPUT,
                'width': addr_width,
                'description': f'Read address 2 ({addr_width}-bit)'
            },
            'rdata2': {
                'direction': PortDirection.OUTPUT,
                'width': bitwidth,
                'description': f'Read data 2 ({bitwidth}-bit)'
            },
            'wen': {
                'direction': PortDirection.INPUT,
                'width': 1,
                'description': 'Write enable'
            },
            'waddr': {
                'direction': PortDirection.INPUT,
                'width': addr_width,
                'description': f'Write address ({addr_width}-bit)'
            },
            'wdata': {
                'direction': PortDirection.INPUT,
                'width': bitwidth,
                'description': f'Write data ({bitwidth}-bit)'
            }
        }


class CPUSpecification:
    """RISC-V CPU的Golden Specification"""

    @staticmethod
    def get_expected_interface(bitwidth: int = 32) -> Dict[str, dict]:
        """返回预期的CPU接口规范 (RV32I)"""
        return {
            'clk': {
                'direction': PortDirection.INPUT,
                'width': 1,
                'description': 'Clock signal'
            },
            'rst_n': {
                'direction': PortDirection.INPUT,
                'width': 1,
                'description': 'Active-low reset'
            },
            # Instruction Memory Interface
            'imem_addr': {
                'direction': PortDirection.OUTPUT,
                'width': 32,
                'description': 'Instruction memory address'
            },
            'imem_data': {
                'direction': PortDirection.INPUT,
                'width': 32,
                'description': 'Instruction memory data'
            },
            # Data Memory Interface
            'dmem_addr': {
                'direction': PortDirection.OUTPUT,
                'width': 32,
                'description': 'Data memory address'
            },
            'dmem_wdata': {
                'direction': PortDirection.OUTPUT,
                'width': 32,
                'description': 'Data memory write data'
            },
            'dmem_wen': {
                'direction': PortDirection.OUTPUT,
                'width': 1,
                'description': 'Data memory write enable'
            },
            'dmem_be': {
                'direction': PortDirection.OUTPUT,
                'width': 4,
                'description': 'Data memory byte enable'
            },
            'dmem_rdata': {
                'direction': PortDirection.INPUT,
                'width': 32,
                'description': 'Data memory read data'
            },
            # Debug Interface (optional, so marked as WARNING not CRITICAL if missing)
            'debug_pc': {
                'direction': PortDirection.OUTPUT,
                'width': 32,
                'description': 'Debug: Program Counter',
                'optional': True
            },
            'debug_inst': {
                'direction': PortDirection.OUTPUT,
                'width': 32,
                'description': 'Debug: Current Instruction',
                'optional': True
            }
        }


class InterfaceBugDetector:
    """接口Bug检测器 - 支持ALU/Counter/RegFile/CPU"""

    def __init__(self, dut_code: str, bitwidth: int = 8,
                 module_type: str = 'alu', depth: int = 32):
        self.parser = VerilogInterfaceParser(dut_code)
        self.bitwidth = bitwidth
        self.module_type = module_type
        self.depth = depth
        self.expected_spec = self._get_spec(module_type, bitwidth, depth)
        self.actual_ports = self.parser.extract_ports()
        self.bugs: List[InterfaceBug] = []

    @staticmethod
    def _get_spec(module_type: str, bitwidth: int, depth: int = 32) -> Dict[str, dict]:
        """根据模块类型获取对应的Golden Specification"""
        specs = {
            'alu': lambda: ALUSpecification.get_expected_interface(bitwidth),
            'counter': lambda: CounterSpecification.get_expected_interface(bitwidth),
            'regfile': lambda: RegFileSpecification.get_expected_interface(bitwidth, depth),
            'cpu': lambda: CPUSpecification.get_expected_interface(bitwidth),
        }
        getter = specs.get(module_type)
        if getter:
            return getter()
        return {}

    def check_all(self) -> List[InterfaceBug]:
        """执行所有检查"""
        self.bugs = []
        self._check_port_widths()
        self._check_missing_ports()
        self._check_extra_ports()
        self._check_direction_mismatch()
        return self.bugs

    def _check_port_widths(self):
        """检查端口宽度是否匹配"""
        for port_name, expected in self.expected_spec.items():
            if port_name in self.actual_ports:
                actual = self.actual_ports[port_name]
                expected_width = expected['width']

                if actual.width != expected_width:
                    # 确定严重程度和影响
                    if actual.width > expected_width:
                        impact = (
                            f"DUT expects {actual.width}-bit input, but testbench will provide "
                            f"{expected_width}-bit value. Upper {actual.width - expected_width} bits "
                            f"will be implicitly zero-extended, potentially masking bugs when "
                            f"testing values that should use the full {actual.width}-bit range."
                        )
                    else:
                        impact = (
                            f"DUT only accepts {actual.width}-bit input, but specification requires "
                            f"{expected_width}-bit. Upper {expected_width - actual.width} bits of "
                            f"test values will be truncated, causing incorrect operation selection "
                            f"or data loss."
                        )

                    self.bugs.append(InterfaceBug(
                        bug_type="WIDTH_MISMATCH",
                        severity=BugSeverity.CRITICAL,
                        port_name=port_name,
                        expected=f"{expected_width}-bit [{expected_width - 1}:0]",
                        actual=f"{actual.width}-bit [{actual.msb}:{actual.lsb}]",
                        line_number=actual.line_number,
                        raw_line=actual.raw_declaration,
                        explanation=f"Port '{port_name}' width mismatch: expected {expected_width}-bit, found {actual.width}-bit",
                        impact=impact
                    ))

    def _check_missing_ports(self):
        """检查缺失的端口"""
        for port_name, expected in self.expected_spec.items():
            if port_name not in self.actual_ports:
                is_optional = expected.get('optional', False)
                self.bugs.append(InterfaceBug(
                    bug_type="MISSING_PORT",
                    severity=BugSeverity.WARNING if is_optional else BugSeverity.CRITICAL,
                    port_name=port_name,
                    expected=f"{expected['direction'].value} [{expected['width'] - 1}:0] {port_name}",
                    actual="(not found)",
                    line_number=-1,
                    raw_line="",
                    explanation=f"{'Optional' if is_optional else 'Required'} port '{port_name}' is missing from DUT",
                    impact=f"Testbench cannot connect to {port_name}, simulation will {'have warnings' if is_optional else 'fail'}"
                ))

    def _check_extra_ports(self):
        """检查多余的端口"""
        for port_name, actual in self.actual_ports.items():
            if port_name not in self.expected_spec:
                self.bugs.append(InterfaceBug(
                    bug_type="EXTRA_PORT",
                    severity=BugSeverity.WARNING,
                    port_name=port_name,
                    expected="(not in specification)",
                    actual=actual.raw_declaration,
                    line_number=actual.line_number,
                    raw_line=actual.raw_declaration,
                    explanation=f"Unexpected port '{port_name}' found in DUT",
                    impact="Port will be unconnected in testbench, may cause simulation warnings"
                ))

    def _check_direction_mismatch(self):
        """检查端口方向是否匹配"""
        for port_name, expected in self.expected_spec.items():
            if port_name in self.actual_ports:
                actual = self.actual_ports[port_name]
                if actual.direction != expected['direction']:
                    self.bugs.append(InterfaceBug(
                        bug_type="DIRECTION_MISMATCH",
                        severity=BugSeverity.CRITICAL,
                        port_name=port_name,
                        expected=expected['direction'].value,
                        actual=actual.direction.value,
                        line_number=actual.line_number,
                        raw_line=actual.raw_declaration,
                        explanation=f"Port '{port_name}' direction mismatch",
                        impact="Signal direction conflict will cause simulation/synthesis errors"
                    ))

    def generate_report(self) -> str:
        """生成检测报告"""
        if not self.bugs:
            self.check_all()

        module_name = self.parser.extract_module_name()

        lines = []
        lines.append("=" * 80)
        lines.append("DUT INTERFACE BUG DETECTION REPORT")
        lines.append("=" * 80)
        lines.append(f"Module: {module_name}")
        lines.append(f"Expected Bitwidth: {self.bitwidth}")
        lines.append(f"Total Bugs Found: {len(self.bugs)}")
        lines.append("")

        if not self.bugs:
            lines.append("✓ No interface bugs detected. DUT matches specification.")
        else:
            # 按严重程度分组
            critical = [b for b in self.bugs if b.severity == BugSeverity.CRITICAL]
            warnings = [b for b in self.bugs if b.severity == BugSeverity.WARNING]

            if critical:
                lines.append(f"CRITICAL BUGS ({len(critical)}):")
                lines.append("-" * 40)
                for i, bug in enumerate(critical, 1):
                    lines.append(f"\n[BUG #{i}] {bug.bug_type}")
                    lines.append(f"  Port: {bug.port_name}")
                    lines.append(f"  Line: {bug.line_number}")
                    lines.append(f"  Code: {bug.raw_line}")
                    lines.append(f"  Expected: {bug.expected}")
                    lines.append(f"  Actual:   {bug.actual}")
                    lines.append(f"  Impact: {bug.impact}")

            if warnings:
                lines.append(f"\nWARNINGS ({len(warnings)}):")
                lines.append("-" * 40)
                for bug in warnings:
                    lines.append(f"  - {bug.explanation}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_json_report(self) -> dict:
        """生成JSON格式的报告，便于集成到pipeline"""
        if not self.bugs:
            self.check_all()

        return {
            'module_name': self.parser.extract_module_name(),
            'expected_bitwidth': self.bitwidth,
            'total_bugs': len(self.bugs),
            'has_critical': any(b.severity == BugSeverity.CRITICAL for b in self.bugs),
            'bugs': [
                {
                    'type': bug.bug_type,
                    'severity': bug.severity.value,
                    'port': bug.port_name,
                    'line': bug.line_number,
                    'expected': bug.expected,
                    'actual': bug.actual,
                    'impact': bug.impact
                }
                for bug in self.bugs
            ]
        }


def check_dut_file(filepath: str, bitwidth: int = 8) -> Tuple[bool, str, dict]:
    """
    检查DUT文件的接口是否符合规范

    Returns:
        (is_valid, text_report, json_report)
    """
    with open(filepath, 'r') as f:
        code = f.read()

    detector = InterfaceBugDetector(code, bitwidth)
    bugs = detector.check_all()

    is_valid = not any(b.severity == BugSeverity.CRITICAL for b in bugs)
    text_report = detector.generate_report()
    json_report = detector.generate_json_report()

    return is_valid, text_report, json_report


# 直接测试
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        bitwidth = int(sys.argv[2]) if len(sys.argv) > 2 else 8

        is_valid, report, json_data = check_dut_file(filepath, bitwidth)
        print(report)

        if not is_valid:
            print("\n⚠️  VERIFICATION BLOCKED: Critical interface bugs detected.")
            print("    Fix the above issues before proceeding with testbench generation.")
            sys.exit(1)
    else:
        print("Usage: python dut_interface_checker.py <dut_file.v> [bitwidth]")