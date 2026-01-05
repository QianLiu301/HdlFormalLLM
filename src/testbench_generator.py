"""
Testbench Generator - Enhanced Version
=======================================

Supports:
- Single file generation (for Web UI)
- Batch generation (for CLI)
- Multiple module types (ALU, Counter, RegFile, CPU)
- Quality analysis with metrics
- Comparison reports across LLMs

Quality Metrics:
1. Functional Coverage - which operations are tested
2. Input Space Coverage - positive, negative, zero, boundary values
3. Test Uniqueness - duplicate detection
4. Corner Case Coverage - edge cases and overflow scenarios
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set
from enum import Enum
from datetime import datetime
from collections import defaultdict


class NumberFormat(Enum):
    """Number format enumeration"""
    DECIMAL = "decimal"
    HEXADECIMAL = "hexadecimal"
    BINARY = "binary"


class ModuleType(Enum):
    """Hardware module type enumeration"""
    ALU = "alu"
    COUNTER = "counter"
    REGFILE = "regfile"
    CPU = "cpu"
    OTHER = "other"


# ============================================================================
# Quality Analyzer
# ============================================================================
class TestQualityAnalyzer:
    """
    Analyze testbench quality with comprehensive metrics.
    """

    def __init__(self, bitwidth: int = 16, module_type: str = "alu"):
        self.bitwidth = bitwidth
        self.module_type = module_type
        self.max_value = (1 << bitwidth) - 1
        self.min_signed = -(1 << (bitwidth - 1))
        self.max_signed = (1 << (bitwidth - 1)) - 1

    def analyze(self, scenarios: List[Dict], operations: Dict = None) -> Dict:
        """
        Comprehensive quality analysis of test scenarios.
        Returns dictionary with quality metrics.
        """
        analysis = {
            'total_tests': len(scenarios),
            'functional_coverage': self._analyze_functional_coverage(scenarios, operations),
            'input_space_coverage': self._analyze_input_space_coverage(scenarios),
            'test_uniqueness': self._analyze_test_uniqueness(scenarios),
            'corner_case_coverage': self._analyze_corner_cases(scenarios),
            'quality_score': 0.0
        }

        analysis['quality_score'] = self._calculate_quality_score(analysis)
        return analysis

    def _analyze_functional_coverage(self, scenarios: List[Dict], operations: Dict = None) -> Dict:
        """Analyze which operations are covered by tests"""
        op_counts = defaultdict(int)
        unique_ops = set()

        for scenario in scenarios:
            op = scenario.get('opcode') or scenario.get('operation') or scenario.get('mode')

            if isinstance(op, int):
                op = f"{op:04b}"
            elif isinstance(op, str):
                if op.startswith('0x') or op.startswith('0X'):
                    op_int = int(op, 16)
                    op = f"{op_int:04b}"

            if op:
                op_counts[op] += 1
                unique_ops.add(op)

        # Expected operations based on module type
        expected_ops = self._get_expected_operations()
        covered_ops = len(unique_ops)

        coverage_percentage = min((covered_ops / expected_ops * 100) if expected_ops > 0 else 0, 100.0)

        return {
            'operations_tested': dict(op_counts),
            'unique_operations': covered_ops,
            'expected_operations': expected_ops,
            'coverage_percentage': coverage_percentage,
            'tests_per_operation': {op: count for op, count in op_counts.items()}
        }

    def _get_expected_operations(self) -> int:
        """Get expected number of operations based on module type"""
        expected = {
            'alu': 4,  # ADD, SUB, AND, OR
            'counter': 3,  # UP, DOWN, UPDOWN
            'regfile': 2,  # READ, WRITE
            'cpu': 10,  # Various instructions
            'other': 4
        }
        return expected.get(self.module_type, 4)

    def _analyze_input_space_coverage(self, scenarios: List[Dict]) -> Dict:
        """Analyze coverage of input space"""
        a_values = []
        b_values = []

        for scenario in scenarios:
            a = scenario.get('a') or scenario.get('A') or scenario.get('data') or 0
            b = scenario.get('b') or scenario.get('B') or scenario.get('addr') or 0

            if isinstance(a, int):
                if a > self.max_signed:
                    a = a - (1 << self.bitwidth)
                a_values.append(a)
            if isinstance(b, int):
                if b > self.max_signed:
                    b = b - (1 << self.bitwidth)
                b_values.append(b)

        def categorize_values(values):
            categories = {
                'zero': 0,
                'positive_small': 0,
                'positive_medium': 0,
                'positive_large': 0,
                'negative_small': 0,
                'negative_medium': 0,
                'negative_large': 0,
                'boundary_values': 0
            }

            for v in values:
                if v == 0:
                    categories['zero'] += 1
                elif v > 0:
                    if v <= 100:
                        categories['positive_small'] += 1
                    elif v <= 1000:
                        categories['positive_medium'] += 1
                    else:
                        categories['positive_large'] += 1
                else:
                    if v >= -100:
                        categories['negative_small'] += 1
                    elif v >= -1000:
                        categories['negative_medium'] += 1
                    else:
                        categories['negative_large'] += 1

                if v in [self.max_signed, self.min_signed,
                         self.max_signed - 1, self.min_signed + 1,
                         self.max_value, 0]:
                    categories['boundary_values'] += 1

            return categories

        a_categories = categorize_values(a_values)
        b_categories = categorize_values(b_values)

        total_categories = 8
        covered_a = sum(1 for v in a_categories.values() if v > 0)
        covered_b = sum(1 for v in b_categories.values() if v > 0)
        diversity_score = min(((covered_a + covered_b) / (2 * total_categories)) * 100, 100.0)

        return {
            'input_a_distribution': a_categories,
            'input_b_distribution': b_categories,
            'diversity_score': diversity_score,
            'has_zero': a_categories['zero'] > 0 or b_categories['zero'] > 0,
            'has_negative': any(k.startswith('negative') for k in a_categories if a_categories[k] > 0) or \
                            any(k.startswith('negative') for k in b_categories if b_categories[k] > 0),
            'has_boundary': a_categories['boundary_values'] > 0 or b_categories['boundary_values'] > 0
        }

    def _analyze_test_uniqueness(self, scenarios: List[Dict]) -> Dict:
        """Analyze test uniqueness and detect duplicates"""
        test_signatures = []
        duplicates = []

        for i, scenario in enumerate(scenarios):
            a = scenario.get('a') or scenario.get('A') or 0
            b = scenario.get('b') or scenario.get('B') or 0
            op = scenario.get('opcode') or scenario.get('operation') or '0000'

            signature = (a, b, str(op))

            if signature in test_signatures:
                dup_index = test_signatures.index(signature)
                duplicates.append({
                    'original_index': dup_index + 1,
                    'duplicate_index': i + 1,
                    'signature': f"a={a}, b={b}, op={op}"
                })
            else:
                test_signatures.append(signature)

        unique_count = len(set(test_signatures))
        total_count = len(scenarios)
        uniqueness_rate = min((unique_count / total_count * 100) if total_count > 0 else 0, 100.0)

        return {
            'total_tests': total_count,
            'unique_tests': unique_count,
            'duplicate_tests': len(duplicates),
            'uniqueness_percentage': uniqueness_rate,
            'duplicates': duplicates[:5]
        }

    def _analyze_corner_cases(self, scenarios: List[Dict]) -> Dict:
        """Analyze coverage of corner cases and edge conditions"""
        corner_cases = {
            'zero_operands': False,
            'max_values': False,
            'min_values': False,
            'overflow_potential': False,
            'underflow_potential': False,
            'sign_boundary': False,
            'all_ones': False,
            'alternating_bits': False,
        }

        corner_case_tests = []

        for scenario in scenarios:
            a = scenario.get('a') or scenario.get('A') or 0
            b = scenario.get('b') or scenario.get('B') or 0
            op = scenario.get('opcode') or scenario.get('operation') or '0000'

            if not isinstance(a, int) or not isinstance(b, int):
                continue

            a_signed = a if a <= self.max_signed else a - (1 << self.bitwidth)
            b_signed = b if b <= self.max_signed else b - (1 << self.bitwidth)

            if a == 0 and b == 0:
                corner_cases['zero_operands'] = True
                corner_case_tests.append('zero_operands')

            if a == self.max_value or b == self.max_value:
                corner_cases['all_ones'] = True
                corner_case_tests.append('all_ones')

            if abs(a_signed) == self.max_signed or abs(b_signed) == self.max_signed:
                corner_cases['max_values'] = True
                corner_case_tests.append('max_values')

            if a_signed == self.min_signed or b_signed == self.min_signed:
                corner_cases['min_values'] = True
                corner_case_tests.append('min_values')

            if str(op) in ['0000', '0x0'] and a_signed > 0 and b_signed > 0:
                if a_signed + b_signed > self.max_signed:
                    corner_cases['overflow_potential'] = True
                    corner_case_tests.append('overflow_potential')

            if str(op) in ['0001', '0x1'] and a_signed < b_signed:
                if a_signed - b_signed < self.min_signed:
                    corner_cases['underflow_potential'] = True
                    corner_case_tests.append('underflow_potential')

            if abs(a_signed - 0) < 10 or abs(b_signed - 0) < 10:
                corner_cases['sign_boundary'] = True

            if self.bitwidth == 8:
                if a in [0xAA, 0x55] or b in [0xAA, 0x55]:
                    corner_cases['alternating_bits'] = True
                    corner_case_tests.append('alternating_bits')
            elif self.bitwidth == 16:
                if a in [0xAAAA, 0x5555] or b in [0xAAAA, 0x5555]:
                    corner_cases['alternating_bits'] = True
                    corner_case_tests.append('alternating_bits')

        covered = sum(1 for v in corner_cases.values() if v)
        total = len(corner_cases)

        return {
            'corner_cases_covered': corner_cases,
            'coverage_count': covered,
            'total_corner_cases': total,
            'coverage_percentage': min((covered / total * 100) if total > 0 else 0, 100.0),
            'corner_case_tests': list(set(corner_case_tests))
        }

    def _calculate_quality_score(self, analysis: Dict) -> float:
        """Calculate overall quality score (0-100)"""
        weights = {
            'functional_coverage': 0.30,
            'input_diversity': 0.25,
            'uniqueness': 0.20,
            'corner_cases': 0.25,
        }

        scores = {
            'functional_coverage': min(analysis['functional_coverage']['coverage_percentage'], 100.0),
            'input_diversity': min(analysis['input_space_coverage']['diversity_score'], 100.0),
            'uniqueness': min(analysis['test_uniqueness']['uniqueness_percentage'], 100.0),
            'corner_cases': min(analysis['corner_case_coverage']['coverage_percentage'], 100.0),
        }

        total_score = sum(scores[k] * weights[k] for k in weights)
        return round(min(total_score, 100.0), 2)

    def get_summary(self, analysis: Dict) -> Dict:
        """Get simplified summary for Web UI"""
        score = analysis['quality_score']

        if score >= 80:
            grade = 'Excellent'
            emoji = '🎉'
        elif score >= 60:
            grade = 'Good'
            emoji = '✅'
        elif score >= 40:
            grade = 'Fair'
            emoji = '⚠️'
        else:
            grade = 'Poor'
            emoji = '❌'

        return {
            'score': score,
            'grade': grade,
            'emoji': emoji,
            'total_tests': analysis['total_tests'],
            'operations_covered': f"{analysis['functional_coverage']['unique_operations']}/{analysis['functional_coverage']['expected_operations']}",
            'operations_percentage': analysis['functional_coverage']['coverage_percentage'],
            'corner_cases_covered': f"{analysis['corner_case_coverage']['coverage_count']}/{analysis['corner_case_coverage']['total_corner_cases']}",
            'corner_cases_percentage': analysis['corner_case_coverage']['coverage_percentage'],
            'unique_tests': analysis['test_uniqueness']['unique_tests'],
            'duplicate_tests': analysis['test_uniqueness']['duplicate_tests']
        }

    def generate_report(self, analysis: Dict, llm_name: str = "Unknown") -> str:
        """Generate human-readable quality report"""
        lines = []
        lines.append("=" * 80)
        lines.append(f" Test Quality Analysis Report - {llm_name}")
        lines.append("=" * 80)
        lines.append("")

        score = analysis['quality_score']
        score_emoji = "🎉" if score >= 80 else "✅" if score >= 60 else "⚠️" if score >= 40 else "❌"
        lines.append(f"📊 Overall Quality Score: {score:.1f}/100 {score_emoji}")
        lines.append("")

        fc = analysis['functional_coverage']
        lines.append("1️⃣  Functional Coverage")
        lines.append("-" * 80)
        lines.append(
            f"   Operations Tested: {fc['unique_operations']}/{fc['expected_operations']} ({fc['coverage_percentage']:.1f}%)")
        lines.append(f"   Total Tests: {analysis['total_tests']}")
        lines.append("")
        lines.append("   Tests per Operation:")
        for op, count in sorted(fc['tests_per_operation'].items()):
            lines.append(f"      • {op}: {count} tests")
        lines.append("")

        isc = analysis['input_space_coverage']
        lines.append("2️⃣  Input Space Coverage")
        lines.append("-" * 80)
        lines.append(f"   Diversity Score: {isc['diversity_score']:.1f}%")
        lines.append(f"   ✓ Zero values: {'Yes' if isc['has_zero'] else 'No'}")
        lines.append(f"   ✓ Negative values: {'Yes' if isc['has_negative'] else 'No'}")
        lines.append(f"   ✓ Boundary values: {'Yes' if isc['has_boundary'] else 'No'}")
        lines.append("")

        tu = analysis['test_uniqueness']
        lines.append("3️⃣  Test Uniqueness")
        lines.append("-" * 80)
        lines.append(f"   Unique Tests: {tu['unique_tests']}/{tu['total_tests']} ({tu['uniqueness_percentage']:.1f}%)")
        lines.append(f"   Duplicate Tests: {tu['duplicate_tests']}")
        lines.append("")

        cc = analysis['corner_case_coverage']
        lines.append("4️⃣  Corner Case Coverage")
        lines.append("-" * 80)
        lines.append(
            f"   Coverage: {cc['coverage_count']}/{cc['total_corner_cases']} ({cc['coverage_percentage']:.1f}%)")
        lines.append("   Covered corner cases:")
        for case, covered in sorted(cc['corner_cases_covered'].items()):
            emoji = "✅" if covered else "❌"
            lines.append(f"      {emoji} {case.replace('_', ' ').title()}")
        lines.append("")

        lines.append("=" * 80)
        return '\n'.join(lines)


# ============================================================================
# Feature Parser
# ============================================================================
class FeatureParser:
    """Parse .feature files and extract test scenarios"""

    def __init__(self, feature_file: str, debug: bool = False):
        self.feature_file = feature_file
        self.debug = debug
        self.bitwidth = 16
        self.module_type = 'alu'
        self.operations = {}
        self.scenarios = []
        self.number_format = NumberFormat.DECIMAL

    def parse(self) -> Dict:
        """Parse a .feature file"""
        with open(self.feature_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Detect bitwidth
        bitwidth_match = re.search(r'(\d+)[-_]?bit', content, re.IGNORECASE)
        if bitwidth_match:
            self.bitwidth = int(bitwidth_match.group(1))

        # Detect module type
        self._detect_module_type(content)
        self._detect_number_format(content)
        self._extract_operations(content)
        self._extract_scenarios(content)

        if not bitwidth_match and self.scenarios:
            self.bitwidth = self._infer_bitwidth_from_scenarios()

        return {
            'bitwidth': self.bitwidth,
            'module_type': self.module_type,
            'operations': self.operations,
            'scenarios': self.scenarios,
            'number_format': self.number_format.value
        }

    def _detect_module_type(self, content: str):
        """Detect module type from feature content"""
        content_lower = content.lower()

        if 'counter' in content_lower or 'count' in content_lower:
            self.module_type = 'counter'
        elif 'regfile' in content_lower or 'register file' in content_lower or 'register bank' in content_lower:
            self.module_type = 'regfile'
        elif 'cpu' in content_lower or 'risc' in content_lower or 'processor' in content_lower:
            self.module_type = 'cpu'
        elif 'alu' in content_lower or 'arithmetic' in content_lower:
            self.module_type = 'alu'
        else:
            self.module_type = 'other'

    def _infer_bitwidth_from_scenarios(self) -> int:
        """Infer bitwidth from value range"""
        max_value = 0
        for scenario in self.scenarios:
            for key in ['a', 'b', 'result', 'expected_result', 'data', 'value']:
                if key in scenario:
                    val = scenario[key]
                    if isinstance(val, int) and val > max_value:
                        max_value = val

        if max_value <= 255:
            return 8
        elif max_value <= 65535:
            return 16
        elif max_value <= 4294967295:
            return 32
        else:
            return 64

    def _detect_number_format(self, content: str):
        """Detect number format used in feature file"""
        if re.search(r'\b0x[0-9A-Fa-f]+\b', content):
            self.number_format = NumberFormat.HEXADECIMAL
        elif re.search(r'\b0b[01]+\b', content):
            self.number_format = NumberFormat.BINARY
        else:
            self.number_format = NumberFormat.DECIMAL

    def _extract_operations(self, content: str):
        """Extract operation definitions"""
        opcode_pattern = r'(\w+)\s*(?:operation|opcode)?\s*(?:with\s+)?opcode\s+([0-9a-fA-Fx]+)'

        for match in re.finditer(opcode_pattern, content, re.IGNORECASE):
            op_name = match.group(1).upper()
            opcode = match.group(2)

            if opcode.startswith('0x') or opcode.startswith('0X'):
                opcode_int = int(opcode, 16)
                opcode = format(opcode_int, '04b')
            elif all(c in '01' for c in opcode):
                opcode = opcode.zfill(4)

            self.operations[op_name] = opcode

    def _extract_scenarios(self, content: str):
        """Extract test scenarios from Examples tables"""
        examples_pattern = r'Examples?:\s*\n((?:\s*\|.*\n)+)'

        for match in re.finditer(examples_pattern, content, re.MULTILINE):
            table_text = match.group(1)
            rows = [row.strip() for row in table_text.strip().split('\n')]

            if len(rows) < 2:
                continue

            header = [col.strip() for col in rows[0].split('|') if col.strip()]

            for row in rows[1:]:
                cols = [col.strip() for col in row.split('|') if col.strip()]

                if len(cols) != len(header):
                    continue

                scenario = {}
                for col_name, col_value in zip(header, cols):
                    parsed_value = self._parse_value(col_value)
                    if parsed_value is not None:
                        scenario[col_name.lower()] = parsed_value
                    else:
                        scenario[col_name.lower()] = col_value

                if scenario:
                    self.scenarios.append(scenario)

    def _parse_value(self, value_str: str) -> Optional[int]:
        """Parse value string to integer"""
        try:
            if value_str.startswith('0b') or value_str.startswith('0B'):
                return int(value_str, 2)
            if value_str.startswith('0x') or value_str.startswith('0X'):
                return int(value_str, 16)
            return int(value_str)
        except (ValueError, AttributeError):
            return None


# ============================================================================
# Testbench Generator
# ============================================================================
class TestbenchGenerator:
    """Generate Verilog testbench with quality analysis"""

    def __init__(
            self,
            feature_dir: Optional[str] = None,
            output_dir: Optional[str] = None,
            dut_dir: Optional[str] = None,
            project_root: Optional[str] = None,
            dut_module_name: Optional[str] = None,
            debug: bool = False
    ):
        self.debug = debug
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.dut_module_name = dut_module_name

        self.feature_dir = self._find_feature_dir(feature_dir)
        self.output_base_dir = self._setup_output_base_dir(output_dir)
        self.dut_dir = self._find_dut_dir(dut_dir)
        self.quality_reports_dir = self.output_base_dir.parent / "quality_reports"
        self.quality_reports_dir.mkdir(parents=True, exist_ok=True)

        if self.debug:
            print(f"📁 Feature directory: {self.feature_dir}")
            print(f"📁 Output base directory: {self.output_base_dir}")
            print(f"📁 DUT directory: {self.dut_dir}")
            print(f"📁 Quality reports: {self.quality_reports_dir}")

    def _find_feature_dir(self, feature_dir: Optional[str]) -> Path:
        """Find .feature files directory"""
        if feature_dir:
            path = Path(feature_dir)
            if path.exists():
                return path

        path = self.project_root / "output" / "bdd"
        if path.exists():
            return path

        path.mkdir(parents=True, exist_ok=True)
        return path

    def _setup_output_base_dir(self, output_dir: Optional[str]) -> Path:
        """Setup output directory"""
        if output_dir:
            path = Path(output_dir)
        else:
            path = self.project_root / "output" / "testbench"

        path.mkdir(parents=True, exist_ok=True)
        return path

    def _find_dut_dir(self, dut_dir: Optional[str]) -> Path:
        """Find DUT directory"""
        if dut_dir:
            path = Path(dut_dir)
            if path.exists():
                return path

        path = self.project_root / "output" / "dut"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ========================================================================
    # Single File Generation (for Web UI)
    # ========================================================================
    def generate_single(
            self,
            bdd_filepath: str,
            dut_info: Dict,
            output_filename: Optional[str] = None
    ) -> Dict:
        """
        Generate testbench for a single BDD file.

        Args:
            bdd_filepath: Path to the .feature file
            dut_info: Dictionary with DUT information:
                - module_type: 'alu', 'counter', 'regfile', 'cpu'
                - module_name: Verilog module name (e.g., 'alu_16bit')
                - bitwidth: Bit width of the design
                - filename: DUT filename
            output_filename: Optional custom output filename

        Returns:
            Dictionary with:
                - success: bool
                - filepath: Output testbench path
                - filename: Output filename
                - quality: Quality analysis results
                - quality_summary: Simplified quality summary
                - content: Testbench content (preview)
        """
        try:
            bdd_path = Path(bdd_filepath)
            if not bdd_path.exists():
                return {'success': False, 'error': f'BDD file not found: {bdd_filepath}'}

            # Parse feature file
            parser = FeatureParser(str(bdd_path), debug=self.debug)
            spec = parser.parse()

            if not spec['scenarios']:
                return {'success': False, 'error': 'No test scenarios found in BDD file'}

            # Get DUT info
            module_type = dut_info.get('module_type', spec.get('module_type', 'alu'))
            module_name = dut_info.get('module_name', f"{module_type}_{spec['bitwidth']}bit")
            bitwidth = dut_info.get('bitwidth', spec['bitwidth'])

            # Override spec with DUT info
            spec['bitwidth'] = bitwidth
            spec['module_type'] = module_type

            # Determine output path
            # Try to get LLM name from path
            llm_name = 'default'
            if bdd_path.parent.name in ['groq', 'deepseek', 'openai', 'claude', 'gemini']:
                llm_name = bdd_path.parent.name

            output_dir = self.output_base_dir / llm_name
            output_dir.mkdir(parents=True, exist_ok=True)

            if output_filename:
                tb_filename = output_filename
            else:
                feature_name = bdd_path.stem
                tb_filename = f"{feature_name}_tb.v"

            tb_path = output_dir / tb_filename

            # Generate testbench
            tb_content, quality_analysis = self._generate_testbench_content(
                spec, module_name, llm_name, module_type
            )

            # Save testbench
            with open(tb_path, 'w', encoding='utf-8') as f:
                f.write(tb_content)

            # Get quality summary
            analyzer = TestQualityAnalyzer(bitwidth=bitwidth, module_type=module_type)
            quality_summary = analyzer.get_summary(quality_analysis)

            # Save quality report
            quality_dir = self.quality_reports_dir / llm_name
            quality_dir.mkdir(parents=True, exist_ok=True)
            quality_report = analyzer.generate_report(quality_analysis, llm_name)
            quality_report_path = quality_dir / f"{bdd_path.stem}_quality.txt"
            with open(quality_report_path, 'w', encoding='utf-8') as f:
                f.write(quality_report)

            return {
                'success': True,
                'filepath': str(tb_path),
                'filename': tb_filename,
                'quality': quality_analysis,
                'quality_summary': quality_summary,
                'quality_report_path': str(quality_report_path),
                'content': tb_content[:2000] + ('...' if len(tb_content) > 2000 else ''),
                'full_content': tb_content,
                'test_count': len(spec['scenarios']),
                'llm': llm_name
            }

        except Exception as e:
            import traceback
            if self.debug:
                traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _generate_testbench_content(
            self,
            spec: Dict,
            module_name: str,
            llm_name: str,
            module_type: str
    ) -> Tuple[str, Dict]:
        """Generate testbench content based on module type"""

        if module_type == 'counter':
            return self._generate_counter_testbench(spec, module_name, llm_name)
        elif module_type == 'regfile':
            return self._generate_regfile_testbench(spec, module_name, llm_name)
        elif module_type == 'cpu':
            return self._generate_cpu_testbench(spec, module_name, llm_name)
        else:
            # Default to ALU
            return self._generate_alu_testbench(spec, module_name, llm_name)

    def _generate_alu_testbench(
            self,
            spec: Dict,
            module_name: str,
            llm_name: str
    ) -> Tuple[str, Dict]:
        """Generate ALU testbench"""
        bitwidth = spec['bitwidth']
        scenarios = spec['scenarios']

        # Quality analysis
        analyzer = TestQualityAnalyzer(bitwidth=bitwidth, module_type='alu')
        quality_analysis = analyzer.analyze(scenarios, spec.get('operations'))

        lines = []
        lines.append(f"// ==========================================================================")
        lines.append(f"// ALU Testbench - Generated from BDD")
        lines.append(f"// ==========================================================================")
        lines.append(f"// LLM Provider: {llm_name}")
        lines.append(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"// Bitwidth: {bitwidth}")
        lines.append(f"// Test Cases: {len(scenarios)}")
        lines.append(f"// Quality Score: {quality_analysis['quality_score']:.1f}/100")
        lines.append(f"// ==========================================================================")
        lines.append("")
        lines.append("`timescale 1ns/1ps")
        lines.append("")
        lines.append(f"module {module_name}_tb;")
        lines.append("")
        lines.append(f"    // Parameters")
        lines.append(f"    parameter WIDTH = {bitwidth};")
        lines.append("")
        lines.append(f"    // Signals")
        lines.append(f"    reg clk;")
        lines.append(f"    reg rst;")
        lines.append(f"    reg [WIDTH-1:0] a;")
        lines.append(f"    reg [WIDTH-1:0] b;")
        lines.append(f"    reg [3:0] opcode;")
        lines.append(f"    wire [WIDTH-1:0] result;")
        lines.append(f"    wire zero;")
        lines.append(f"    wire overflow;")
        lines.append(f"    wire negative;")
        lines.append("")
        lines.append(f"    // Test counters")
        lines.append(f"    integer total, passed, failed;")
        lines.append("")
        lines.append(f"    // DUT instantiation")
        lines.append(f"    {module_name} dut (")
        lines.append(f"        .clk(clk),")
        lines.append(f"        .rst(rst),")
        lines.append(f"        .a(a),")
        lines.append(f"        .b(b),")
        lines.append(f"        .opcode(opcode),")
        lines.append(f"        .result(result),")
        lines.append(f"        .zero(zero),")
        lines.append(f"        .overflow(overflow),")
        lines.append(f"        .negative(negative)")
        lines.append(f"    );")
        lines.append("")
        lines.append(f"    // Clock generation")
        lines.append(f"    initial clk = 0;")
        lines.append(f"    always #5 clk = ~clk;")
        lines.append("")
        lines.append(f"    // Test stimulus")
        lines.append(f"    initial begin")
        lines.append(f"        // Initialize")
        lines.append(f"        total = 0;")
        lines.append(f"        passed = 0;")
        lines.append(f"        failed = 0;")
        lines.append(f"        rst = 1;")
        lines.append(f"        a = 0;")
        lines.append(f"        b = 0;")
        lines.append(f"        opcode = 0;")
        lines.append(f"        #20 rst = 0;")
        lines.append("")
        lines.append(f"        $display(\"\\n{'=' * 70}\");")
        lines.append(f"        $display(\"ALU Testbench - {llm_name}\");")
        lines.append(f"        $display(\"{'=' * 70}\\n\");")
        lines.append("")

        # Generate test cases
        for i, scenario in enumerate(scenarios, 1):
            a_val = scenario.get('a', scenario.get('A', 0))
            b_val = scenario.get('b', scenario.get('B', 0))
            op = scenario.get('opcode', scenario.get('operation', '0000'))
            expected = scenario.get('expected_result', scenario.get('result', 0))

            if isinstance(op, str) and op.startswith('0x'):
                op = int(op, 16)
            elif isinstance(op, str) and all(c in '01' for c in op):
                op = int(op, 2)

            op_str = f"{op:04b}" if isinstance(op, int) else str(op).zfill(4)

            lines.append(f"        // Test {i}")
            lines.append(f"        a = {bitwidth}'d{a_val};")
            lines.append(f"        b = {bitwidth}'d{b_val};")
            lines.append(f"        opcode = 4'b{op_str};")
            lines.append(f"        #10;")
            lines.append(f"        total = total + 1;")
            lines.append(f"        if (result == {bitwidth}'d{expected}) begin")
            lines.append(f"            passed = passed + 1;")
            lines.append(f"            $display(\"✓ Test {i}: PASS\");")
            lines.append(f"        end else begin")
            lines.append(f"            failed = failed + 1;")
            lines.append(
                f"            $display(\"✗ Test {i}: FAIL - Expected %d, Got %d\", {bitwidth}'d{expected}, result);")
            lines.append(f"        end")
            lines.append("")

        # Summary
        lines.append(f"        // Summary")
        lines.append(f"        $display(\"\\n{'=' * 70}\");")
        lines.append(f"        $display(\"Test Summary\");")
        lines.append(f"        $display(\"{'=' * 70}\");")
        lines.append(f"        $display(\"Total:  %0d\", total);")
        lines.append(f"        $display(\"Passed: %0d\", passed);")
        lines.append(f"        $display(\"Failed: %0d\", failed);")
        lines.append("")
        lines.append(f"        if (failed == 0)")
        lines.append(f"            $display(\"\\n🎉 ALL TESTS PASSED!\");")
        lines.append(f"        else")
        lines.append(f"            $display(\"\\n⚠️  SOME TESTS FAILED\");")
        lines.append("")
        lines.append(f"        $display(\"{'=' * 70}\\n\");")
        lines.append(f"        $finish;")
        lines.append(f"    end")
        lines.append("")
        lines.append(f"endmodule")

        return '\n'.join(lines), quality_analysis

    def _generate_counter_testbench(
            self,
            spec: Dict,
            module_name: str,
            llm_name: str
    ) -> Tuple[str, Dict]:
        """Generate Counter testbench"""
        bitwidth = spec['bitwidth']
        scenarios = spec['scenarios']

        analyzer = TestQualityAnalyzer(bitwidth=bitwidth, module_type='counter')
        quality_analysis = analyzer.analyze(scenarios)

        lines = []
        lines.append(f"// ==========================================================================")
        lines.append(f"// Counter Testbench - Generated from BDD")
        lines.append(f"// ==========================================================================")
        lines.append(f"// LLM Provider: {llm_name}")
        lines.append(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"// Bitwidth: {bitwidth}")
        lines.append(f"// Test Cases: {len(scenarios)}")
        lines.append(f"// ==========================================================================")
        lines.append("")
        lines.append("`timescale 1ns/1ps")
        lines.append("")
        lines.append(f"module {module_name}_tb;")
        lines.append("")
        lines.append(f"    parameter WIDTH = {bitwidth};")
        lines.append("")
        lines.append(f"    reg clk;")
        lines.append(f"    reg rst_n;")
        lines.append(f"    reg enable;")
        lines.append(f"    reg load;")
        lines.append(f"    reg [1:0] mode;  // 00=hold, 01=up, 10=down, 11=load")
        lines.append(f"    reg [WIDTH-1:0] load_value;")
        lines.append(f"    wire [WIDTH-1:0] count;")
        lines.append(f"    wire overflow;")
        lines.append(f"    wire zero;")
        lines.append("")
        lines.append(f"    integer total, passed, failed;")
        lines.append("")
        lines.append(f"    {module_name} dut (")
        lines.append(f"        .clk(clk),")
        lines.append(f"        .rst_n(rst_n),")
        lines.append(f"        .enable(enable),")
        lines.append(f"        .load(load),")
        lines.append(f"        .load_value(load_value),")
        lines.append(f"        .mode(mode),")
        lines.append(f"        .count(count),")
        lines.append(f"        .overflow(overflow),")
        lines.append(f"        .zero(zero)")
        lines.append(f"    );")
        lines.append("")
        lines.append(f"    initial clk = 0;")
        lines.append(f"    always #5 clk = ~clk;")
        lines.append("")
        lines.append(f"    initial begin")
        lines.append(f"        total = 0; passed = 0; failed = 0;")
        lines.append(f"        rst_n = 0; enable = 0; load = 0; mode = 0; load_value = 0;")
        lines.append(f"        #20 rst_n = 1;")
        lines.append("")
        lines.append(f"        $display(\"\\nCounter Testbench - {llm_name}\\n\");")
        lines.append("")

        # Generate test cases
        for i, scenario in enumerate(scenarios, 1):
            mode_val = scenario.get('mode', 1)
            data_val = scenario.get('data_in', scenario.get('value', 0))
            expected = scenario.get('expected', scenario.get('count', 0))

            lines.append(f"        // Test {i}")
            lines.append(f"        enable = 1;")
            lines.append(f"        mode = 2'd{mode_val};")
            lines.append(f"        load_value = {bitwidth}'d{data_val};")
            lines.append(f"        #10;")
            lines.append(f"        total = total + 1;")
            lines.append(f"        $display(\"Test {i}: mode=%d, count=%d\", mode, count);")
            lines.append("")

        lines.append(f"        $display(\"\\nTotal: %0d, Passed: %0d, Failed: %0d\", total, passed, failed);")
        lines.append(f"        $finish;")
        lines.append(f"    end")
        lines.append("")
        lines.append(f"endmodule")

        return '\n'.join(lines), quality_analysis

    def _generate_regfile_testbench(
            self,
            spec: Dict,
            module_name: str,
            llm_name: str
    ) -> Tuple[str, Dict]:
        """Generate Register File testbench"""
        bitwidth = spec['bitwidth']
        scenarios = spec['scenarios']

        analyzer = TestQualityAnalyzer(bitwidth=bitwidth, module_type='regfile')
        quality_analysis = analyzer.analyze(scenarios)

        lines = []
        lines.append(f"// ==========================================================================")
        lines.append(f"// Register File Testbench - Generated from BDD")
        lines.append(f"// ==========================================================================")
        lines.append(f"// LLM Provider: {llm_name}")
        lines.append(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"// ==========================================================================")
        lines.append("")
        lines.append("`timescale 1ns/1ps")
        lines.append("")
        lines.append(f"module {module_name}_tb;")
        lines.append("")
        lines.append(f"    parameter WIDTH = {bitwidth};")
        lines.append(f"    parameter DEPTH = 32;")
        lines.append("")
        lines.append(f"    reg clk;")
        lines.append(f"    reg rst_n;")
        lines.append(f"    reg we;")
        lines.append(f"    reg [4:0] rs1, rs2, rd;")
        lines.append(f"    reg [WIDTH-1:0] wd;")
        lines.append(f"    wire [WIDTH-1:0] rd1, rd2;")
        lines.append("")
        lines.append(f"    integer total, passed, failed;")
        lines.append("")
        lines.append(f"    {module_name} dut (")
        lines.append(f"        .clk(clk),")
        lines.append(f"        .rst_n(rst_n),")
        lines.append(f"        .we(we),")
        lines.append(f"        .rs1(rs1),")
        lines.append(f"        .rs2(rs2),")
        lines.append(f"        .rd(rd),")
        lines.append(f"        .wd(wd),")
        lines.append(f"        .rd1(rd1),")
        lines.append(f"        .rd2(rd2)")
        lines.append(f"    );")
        lines.append("")
        lines.append(f"    initial clk = 0;")
        lines.append(f"    always #5 clk = ~clk;")
        lines.append("")
        lines.append(f"    initial begin")
        lines.append(f"        total = 0; passed = 0; failed = 0;")
        lines.append(f"        rst_n = 0; we = 0; rs1 = 0; rs2 = 0; rd = 0; wd = 0;")
        lines.append(f"        #20 rst_n = 1;")
        lines.append("")
        lines.append(f"        $display(\"\\nRegister File Testbench - {llm_name}\\n\");")
        lines.append("")

        for i, scenario in enumerate(scenarios, 1):
            lines.append(f"        // Test {i}")
            lines.append(f"        we = 1; rd = 5'd{i % 32}; wd = {bitwidth}'d{i * 100};")
            lines.append(f"        #10;")
            lines.append(f"        we = 0; rs1 = 5'd{i % 32};")
            lines.append(f"        #10;")
            lines.append(f"        total = total + 1;")
            lines.append(
                f"        $display(\"Test {i}: Write %d to R%d, Read %d\", {bitwidth}'d{i * 100}, 5'd{i % 32}, rd1);")
            lines.append("")

        lines.append(f"        $display(\"\\nTotal: %0d tests\", total);")
        lines.append(f"        $finish;")
        lines.append(f"    end")
        lines.append("")
        lines.append(f"endmodule")

        return '\n'.join(lines), quality_analysis

    def _generate_cpu_testbench(
            self,
            spec: Dict,
            module_name: str,
            llm_name: str
    ) -> Tuple[str, Dict]:
        """Generate CPU testbench"""
        bitwidth = spec.get('bitwidth', 32)
        scenarios = spec['scenarios']

        analyzer = TestQualityAnalyzer(bitwidth=bitwidth, module_type='cpu')
        quality_analysis = analyzer.analyze(scenarios)

        lines = []
        lines.append(f"// ==========================================================================")
        lines.append(f"// CPU Testbench - Generated from BDD")
        lines.append(f"// ==========================================================================")
        lines.append(f"// LLM Provider: {llm_name}")
        lines.append(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"// ==========================================================================")
        lines.append("")
        lines.append("`timescale 1ns/1ps")
        lines.append("")
        lines.append(f"module {module_name}_tb;")
        lines.append("")
        lines.append(f"    reg clk;")
        lines.append(f"    reg rst_n;")
        lines.append(f"    wire [31:0] pc;")
        lines.append(f"    wire [31:0] instr;")
        lines.append("")
        lines.append(f"    integer total, passed, failed;")
        lines.append("")
        lines.append(f"    {module_name} dut (")
        lines.append(f"        .clk(clk),")
        lines.append(f"        .rst_n(rst_n)")
        lines.append(f"    );")
        lines.append("")
        lines.append(f"    initial clk = 0;")
        lines.append(f"    always #5 clk = ~clk;")
        lines.append("")
        lines.append(f"    initial begin")
        lines.append(f"        total = 0; passed = 0; failed = 0;")
        lines.append(f"        rst_n = 0;")
        lines.append(f"        #20 rst_n = 1;")
        lines.append("")
        lines.append(f"        $display(\"\\nCPU Testbench - {llm_name}\\n\");")
        lines.append("")
        lines.append(f"        // Run for some cycles")
        lines.append(f"        repeat(100) @(posedge clk);")
        lines.append("")
        lines.append(f"        $display(\"\\nCPU test complete\");")
        lines.append(f"        $finish;")
        lines.append(f"    end")
        lines.append("")
        lines.append(f"endmodule")

        return '\n'.join(lines), quality_analysis

    # ========================================================================
    # Batch Generation (for CLI)
    # ========================================================================
    def scan_features(self) -> List[Tuple[Path, str]]:
        """Scan for .feature files"""
        print(f"\n🔍 Scanning for .feature files...")

        feature_files = []

        for f in self.feature_dir.glob("*.feature"):
            feature_files.append((f, "default"))

        for subdir in self.feature_dir.iterdir():
            if subdir.is_dir():
                llm_name = subdir.name
                for f in subdir.glob("*.feature"):
                    feature_files.append((f, llm_name))

        if not feature_files:
            print(f"   ⚠️  No .feature files found")
            return []

        print(f"   ✅ Found {len(feature_files)} feature file(s)")
        return feature_files

    def generate_all(self) -> Dict[str, List[Path]]:
        """Generate testbenches for all .feature files (batch mode)"""
        print("\n" + "=" * 70)
        print("🚀 Testbench Generator - Batch Mode")
        print("=" * 70)

        feature_files = self.scan_features()

        if not feature_files:
            return {}

        generated_by_llm = {}
        quality_by_llm = {}

        for feature_path, llm_name in feature_files:
            try:
                print(f"\n📖 Processing: {llm_name}/{feature_path.name}")

                parser = FeatureParser(str(feature_path), debug=self.debug)
                spec = parser.parse()

                module_name = f"{spec['module_type']}_{spec['bitwidth']}bit"

                tb_content, quality_analysis = self._generate_testbench_content(
                    spec, module_name, llm_name, spec['module_type']
                )

                # Save testbench
                llm_output_dir = self.output_base_dir / llm_name
                llm_output_dir.mkdir(parents=True, exist_ok=True)

                feature_name = feature_path.stem
                tb_path = llm_output_dir / f"{feature_name}_tb.v"
                with open(tb_path, 'w', encoding='utf-8') as f:
                    f.write(tb_content)

                print(f"   ✅ Testbench: {tb_path.name}")

                # Save quality report
                analyzer = TestQualityAnalyzer(bitwidth=spec['bitwidth'], module_type=spec['module_type'])
                quality_report = analyzer.generate_report(quality_analysis, llm_name)

                llm_quality_dir = self.quality_reports_dir / llm_name
                llm_quality_dir.mkdir(parents=True, exist_ok=True)

                quality_path = llm_quality_dir / f"{feature_name}_quality.txt"
                with open(quality_path, 'w', encoding='utf-8') as f:
                    f.write(quality_report)

                print(f"   📊 Quality: {quality_analysis['quality_score']:.1f}/100")

                if llm_name not in generated_by_llm:
                    generated_by_llm[llm_name] = []
                    quality_by_llm[llm_name] = []

                generated_by_llm[llm_name].append(tb_path)
                quality_by_llm[llm_name].append(quality_analysis)

            except Exception as e:
                print(f"   ❌ Error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

        # Generate comparison report
        if quality_by_llm:
            self._generate_quality_comparison(quality_by_llm)

        # Summary
        print("\n" + "=" * 70)
        print("✨ Generation Complete!")
        print("=" * 70)

        total = sum(len(files) for files in generated_by_llm.values())
        print(f"\n📊 Summary:")
        print(f"   Total testbenches: {total}")
        print(f"   LLM providers: {len(generated_by_llm)}")
        print()

        for llm_name, files in sorted(generated_by_llm.items()):
            avg_quality = sum(q['quality_score'] for q in quality_by_llm[llm_name]) / len(quality_by_llm[llm_name])
            print(f"   📂 {llm_name}: {len(files)} testbench(es), Avg Quality: {avg_quality:.1f}/100")

        return generated_by_llm

    def _generate_quality_comparison(self, quality_by_llm: Dict[str, List[Dict]]):
        """Generate comparison report across all LLMs"""
        comparison_path = self.quality_reports_dir / "quality_comparison.txt"

        lines = []
        lines.append("=" * 80)
        lines.append(" Multi-LLM Testbench Quality Comparison")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        lines.append("=" * 80)
        lines.append(" Overall Quality Scores")
        lines.append("=" * 80)
        lines.append(f"{'LLM Provider':<15} {'Testbenches':<12} {'Avg Quality':<12} {'Best':<8} {'Worst':<8}")
        lines.append("-" * 80)

        all_scores = []
        for llm_name, analyses in sorted(quality_by_llm.items()):
            scores = [a['quality_score'] for a in analyses]
            avg_score = sum(scores) / len(scores)
            best_score = max(scores)
            worst_score = min(scores)

            all_scores.append((llm_name, avg_score))
            lines.append(
                f"{llm_name:<15} {len(analyses):<12} {avg_score:>10.1f}% {best_score:>6.1f}% {worst_score:>6.1f}%")

        lines.append("")
        lines.append("=" * 80)
        lines.append(" Rankings")
        lines.append("=" * 80)
        lines.append("")

        all_scores.sort(key=lambda x: x[1], reverse=True)
        lines.append("🏆 By Average Quality Score:")
        for i, (llm, score) in enumerate(all_scores, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            lines.append(f"   {medal} {i}. {llm}: {score:.1f}%")

        lines.append("")
        lines.append("=" * 80)

        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\n📊 Quality comparison: {comparison_path}")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description='Testbench Generator with Quality Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--feature-dir', help='Directory containing .feature files')
    parser.add_argument('--output-dir', help='Output directory for testbench files')
    parser.add_argument('--dut-dir', help='Directory containing DUT files')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--dut-module', help='DUT module name')
    parser.add_argument('--single', help='Generate single testbench from specified .feature file')
    parser.add_argument('--batch', action='store_true', help='Batch generate all testbenches')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    generator = TestbenchGenerator(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        dut_dir=args.dut_dir,
        project_root=args.project_root,
        dut_module_name=args.dut_module,
        debug=args.debug
    )

    if args.single:
        # Single file mode
        result = generator.generate_single(
            bdd_filepath=args.single,
            dut_info={'module_name': args.dut_module or 'alu_16bit'}
        )
        if result['success']:
            print(f"\n✅ Generated: {result['filename']}")
            print(f"📊 Quality Score: {result['quality_summary']['score']}/100 {result['quality_summary']['emoji']}")
        else:
            print(f"\n❌ Error: {result['error']}")
    else:
        # Batch mode (default)
        generated_by_llm = generator.generate_all()

        if generated_by_llm:
            print("\n📋 NEXT STEPS:")
            print("=" * 70)
            print("1. Review quality reports in: output/quality_reports/")
            print("2. Run simulations with your preferred simulator")
            print("=" * 70)


if __name__ == "__main__":
    main()