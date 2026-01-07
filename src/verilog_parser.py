"""
Verilog Parser - Parse and analyze Verilog/SystemVerilog files
===============================================================

This module parses uploaded Verilog files to extract:
- Module names
- Port definitions (inputs/outputs)
- Bitwidth information
- Auto-detect module type (ALU, Counter, RegFile, CPU)

Authors: Rolf Drechsler, Qian Liu
Paper: https://arxiv.org/abs/2512.17814
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class VerilogParser:
    """
    Parse Verilog/SystemVerilog files to extract module information.
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.v', '.sv', '.vh'}

    # Max file size (1MB)
    MAX_FILE_SIZE = 1 * 1024 * 1024

    # Keywords for module type detection
    MODULE_TYPE_KEYWORDS = {
        'alu': {
            'names': ['alu', 'arithmetic', 'logic_unit'],
            'ports': ['opcode', 'op_code', 'op', 'operation'],
            'signals': ['add', 'sub', 'and', 'or', 'xor', 'overflow', 'carry']
        },
        'counter': {
            'names': ['counter', 'cnt', 'count'],
            'ports': ['up', 'down', 'up_down', 'load', 'preset', 'mode'],
            'signals': ['increment', 'decrement', 'wrap']
        },
        'regfile': {
            'names': ['regfile', 'register_file', 'reg_file', 'rf', 'gpr'],
            'ports': ['rs1', 'rs2', 'rd', 'raddr', 'waddr', 'read_addr', 'write_addr'],
            'signals': ['register', 'regs', 'x0', 'r0']
        },
        'cpu': {
            'names': ['cpu', 'processor', 'core', 'riscv', 'risc_v', 'mips'],
            'ports': ['pc', 'instruction', 'inst', 'imem', 'dmem', 'mem_addr'],
            'signals': ['pipeline', 'fetch', 'decode', 'execute', 'writeback', 'alu_op']
        }
    }

    # BDD template suggestions
    BDD_TEMPLATES = {
        'alu': lambda bw, ports: f"""{bw}-bit ALU with:
- Arithmetic operations (ADD, SUB)
- Logic operations (AND, OR, XOR)
- Flag detection (overflow, zero, negative)
- Boundary value tests (0x{'0' * max(bw // 4, 1)}, 0x{'F' * max(bw // 4, 1)})
- Random test cases""",

        'counter': lambda bw, ports: f"""{bw}-bit Counter with:
- UP mode (increment from 0 to MAX)
- DOWN mode (decrement from MAX to 0)
- Load preset value functionality
- Enable/disable control
- Overflow and zero flag detection
- Boundary tests""",

        'regfile': lambda bw, ports: f"""{bw}-bit Register File with:
- Read port verification
- Write port verification
- R0/x0 always returns 0 (if RISC-V style)
- Write then read same register
- Concurrent read and write tests
- All registers write/read test""",

        'cpu': lambda bw, ports: f"""RISC-V CPU with:
- R-type instructions: ADD, SUB, AND, OR, XOR
- I-type instructions: ADDI, LW
- S-type instructions: SW
- B-type instructions: BEQ, BNE
- Data hazard tests
- Branch tests""",

        'other': lambda bw, ports: f"""{bw}-bit module with:
- Input ports: {', '.join(ports.get('inputs', [])[:5])}
- Output ports: {', '.join(ports.get('outputs', [])[:5])}
- Basic functionality tests
- Boundary value tests
- Reset behavior tests"""
    }

    def __init__(self, upload_dir: Optional[str] = None):
        """
        Initialize parser.

        Args:
            upload_dir: Directory to store uploaded files
        """
        if upload_dir:
            self.upload_dir = Path(upload_dir)
        else:
            self.upload_dir = Path("output/dut/uploaded")

        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def validate_file(self, filename: str, file_content: bytes) -> Tuple[bool, str]:
        """
        Validate uploaded file.

        Args:
            filename: Original filename
            file_content: File content as bytes

        Returns:
            (is_valid, error_message)
        """
        # Check extension
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type: {ext}. Supported: .v, .sv, .vh"

        # Check file size
        if len(file_content) > self.MAX_FILE_SIZE:
            size_mb = len(file_content) / (1024 * 1024)
            return False, f"File too large: {size_mb:.2f}MB. Maximum: 1MB"

        # Check if empty
        if len(file_content) == 0:
            return False, "File is empty"

        # Try to decode as text
        try:
            text_content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text_content = file_content.decode('latin-1')
            except:
                return False, "Cannot decode file. Please use UTF-8 or ASCII encoding."

        # Check for module keyword
        if 'module' not in text_content.lower():
            return False, "No 'module' keyword found. Please upload a valid Verilog file."

        # Check for endmodule
        if 'endmodule' not in text_content.lower():
            return False, "No 'endmodule' keyword found. File may be incomplete."

        # Security check - dangerous includes
        dangerous_patterns = [
            r'`include\s*["\'].*\.\./',  # Path traversal
            r'`include\s*["\']/etc/',  # System files
            r'\$system\s*\(',  # System calls
            r'\$fopen\s*\(',  # File operations (suspicious)
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                return False, "File contains potentially dangerous content."

        return True, ""

    def parse_file(self, filename: str, file_content: bytes) -> Dict:
        """
        Parse Verilog file and extract module information.

        Args:
            filename: Original filename
            file_content: File content as bytes

        Returns:
            Dict with parsed information
        """
        # Validate first
        is_valid, error_msg = self.validate_file(filename, file_content)
        if not is_valid:
            return {
                'success': False,
                'error': error_msg
            }

        # Decode content
        try:
            text_content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            text_content = file_content.decode('latin-1')

        # Remove comments for parsing
        clean_content = self._remove_comments(text_content)

        # Extract all modules
        modules = self._extract_modules(clean_content)

        if not modules:
            return {
                'success': False,
                'error': 'No valid module found in file.'
            }

        # Process each module
        parsed_modules = []
        for module_name, module_content in modules:
            module_info = self._parse_module(module_name, module_content, filename)
            parsed_modules.append(module_info)

        # Save file
        saved_path = self._save_file(filename, file_content)

        return {
            'success': True,
            'filename': filename,
            'saved_path': str(saved_path),
            'modules': parsed_modules,
            'module_count': len(parsed_modules)
        }

    def _remove_comments(self, content: str) -> str:
        """Remove Verilog comments from content."""
        # Remove single-line comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        return content

    def _extract_modules(self, content: str) -> List[Tuple[str, str]]:
        """
        Extract all module definitions from content.

        Returns:
            List of (module_name, module_content) tuples
        """
        modules = []

        # Pattern to match module definition
        # Handles both Verilog-1995 and Verilog-2001 styles
        pattern = r'module\s+(\w+)\s*(?:#\s*\([^)]*\))?\s*\([^;]*\)\s*;(.*?)endmodule'

        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

        for match in matches:
            module_name = match[0]
            module_body = match[1] if len(match) > 1 else ""

            # Get the full module text for port parsing
            full_pattern = rf'module\s+{re.escape(module_name)}\s*(?:#\s*\([^)]*\))?\s*\(([^;]*)\)\s*;'
            port_match = re.search(full_pattern, content, re.DOTALL | re.IGNORECASE)
            port_list = port_match.group(1) if port_match else ""

            modules.append((module_name, port_list + module_body))

        return modules

    def _parse_module(self, module_name: str, module_content: str, filename: str = "") -> Dict:
        """
        Parse a single module to extract detailed information.
        """
        # Extract ports
        ports = self._extract_ports(module_content)

        # Detect module type first (needed for other detections)
        detected_type, confidence = self._detect_module_type(module_name, ports, module_content)

        # Detect bitwidth
        bitwidth = self._detect_bitwidth(module_content, ports, module_name)

        # ★★★ 新增：检测 CPU stages ★★★
        stages = self._detect_stages(module_name, module_content) if detected_type == 'cpu' else None

        # ★★★ 新增：检测 Register 数量 ★★★
        num_registers = self._detect_num_registers(module_name, module_content, ports, filename) if detected_type == 'regfile' else None

        # Generate BDD suggestion
        bdd_template = self.BDD_TEMPLATES.get(detected_type, self.BDD_TEMPLATES['other'])
        suggested_bdd = bdd_template(bitwidth, ports)

        result = {
            'name': module_name,
            'ports': ports,
            'bitwidth': bitwidth,
            'detected_type': detected_type,
            'type_confidence': confidence,
            'suggested_bdd': suggested_bdd
        }

        # add stages and num_registers to return results
        if stages is not None:
            result['stages'] = stages
        if num_registers is not None:
            result['num_registers'] = num_registers

        return result

    def _extract_ports(self, content: str) -> Dict[str, List[Dict]]:
        """
        Extract input and output ports from module content.
        """
        inputs = []
        outputs = []
        inouts = []

        # Pattern for Verilog-2001 style: input wire [7:0] data
        patterns = [
            # input/output [width] name
            (r'input\s+(?:wire|reg|logic)?\s*(?:\[(\d+):(\d+)\])?\s*(\w+)', 'input'),
            (r'output\s+(?:wire|reg|logic)?\s*(?:\[(\d+):(\d+)\])?\s*(\w+)', 'output'),
            (r'inout\s+(?:wire|reg|logic)?\s*(?:\[(\d+):(\d+)\])?\s*(\w+)', 'inout'),
        ]

        for pattern, port_type in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                msb, lsb, name = match
                if msb and lsb:
                    width = abs(int(msb) - int(lsb)) + 1
                else:
                    width = 1

                port_info = {'name': name, 'width': width}

                if port_type == 'input':
                    inputs.append(port_info)
                elif port_type == 'output':
                    outputs.append(port_info)
                else:
                    inouts.append(port_info)

        # Also try to find port names from module declaration
        # module name (a, b, c);
        port_list_match = re.search(r'^\s*(\w+(?:\s*,\s*\w+)*)\s*(?:,|$)', content, re.MULTILINE)

        return {
            'inputs': inputs,
            'outputs': outputs,
            'inouts': inouts,
            'input_names': [p['name'] for p in inputs],
            'output_names': [p['name'] for p in outputs]
        }

    def _detect_bitwidth(self, content: str, ports: Dict, module_name: str = "") -> int:
        """
        Detect the primary bitwidth of the module.

        Priority:
        1. Extract from module name (e.g., counter_8bit -> 8)
        2. Check key output ports (result, count, data, out, q)
        3. Check key input ports (a, b, data, din)
        4. Find most common width >= 4 (filter control signals)
        5. Default to 8
        """
        import re

        # ★ Priority 1: Extract from module name ★
        if module_name:
            # Match patterns like: counter_8bit, alu_32bit, 16bit_counter
            patterns = [
                r'(\d+)\s*bit',  # 8bit, 32bit
                r'_(\d+)(?:_|$)',  # _32_, _16
                r'^(\d+)_',  # 16_xxx
            ]
            for pattern in patterns:
                match = re.search(pattern, module_name.lower())
                if match:
                    width = int(match.group(1))
                    if width in [4, 8, 16, 32, 64, 128]:
                        return width

            # For register files: 32x32 means 32-bit data width
            regfile_match = re.search(r'\d+x(\d+)', module_name.lower())
            if regfile_match:
                width = int(regfile_match.group(1))
                if width in [8, 16, 32, 64]:
                    return width

        # ★ Priority 2: Check key output ports ★
        key_outputs = ['result', 'data', 'out', 'q', 'count', 'dout', 'rdata', 'read_data']
        for port in ports.get('outputs', []):
            if port['name'].lower() in key_outputs:
                if port['width'] > 1:
                    return port['width']

        # ★ Priority 3: Check key input ports ★
        key_inputs = ['a', 'b', 'data', 'in', 'd', 'din', 'wdata', 'write_data', 'load_value']
        for port in ports.get('inputs', []):
            if port['name'].lower() in key_inputs:
                if port['width'] > 1:
                    return port['width']

        # ★ Priority 4: Find most common width >= 4 ★
        widths = []
        for port_list in [ports.get('inputs', []), ports.get('outputs', [])]:
            for port in port_list:
                if port['width'] >= 4:  # Filter out small control signals
                    widths.append(port['width'])

        if widths:
            return max(set(widths), key=widths.count)

        # ★ Priority 5: Any width > 1 ★
        for port_list in [ports.get('outputs', []), ports.get('inputs', [])]:
            for port in port_list:
                if port['width'] > 1:
                    return port['width']

        return 8  # Default

    def _detect_stages(self, module_name: str, content: str) -> int:
        """
        Detect CPU pipeline stages from module name or content.

        Examples:
            - riscv_cpu_3stage.v -> 3
            - cpu_5stage.v -> 5
            - 内容中有 IF/ID/EX/MEM/WB -> 5
        """
        import re

        # ★ Priority 1: From module name ★
        # Match patterns like: 3stage, 5_stage, 7-stage
        patterns = [
            r'(\d+)\s*stage',  # 3stage, 5stage, 5 stage
            r'(\d+)_stage',  # 3_stage, 5_stage
            r'(\d+)-stage',  # 3-stage, 5-stage
        ]

        for pattern in patterns:
            match = re.search(pattern, module_name.lower())
            if match:
                stages = int(match.group(1))
                if stages in [3, 5, 7]:  # Common pipeline stages
                    return stages

        # ★ Priority 2: From content - count pipeline stage registers/signals ★
        content_lower = content.lower()

        # 5-stage pipeline indicators
        five_stage_signals = ['if_id', 'id_ex', 'ex_mem', 'mem_wb']
        if sum(1 for s in five_stage_signals if s in content_lower) >= 3:
            return 5

        # Check for stage names
        stage_names = {
            'fetch': 0, 'if_': 0,
            'decode': 0, 'id_': 0,
            'execute': 0, 'ex_': 0,
            'memory': 0, 'mem_': 0,
            'writeback': 0, 'wb_': 0
        }

        for stage in stage_names:
            if stage in content_lower:
                stage_names[stage] = 1

        # Count unique stages
        stage_count = 0
        if stage_names['fetch'] or stage_names['if_']:
            stage_count += 1
        if stage_names['decode'] or stage_names['id_']:
            stage_count += 1
        if stage_names['execute'] or stage_names['ex_']:
            stage_count += 1
        if stage_names['memory'] or stage_names['mem_']:
            stage_count += 1
        if stage_names['writeback'] or stage_names['wb_']:
            stage_count += 1

        if stage_count >= 4:
            return 5
        elif stage_count >= 2:
            return 3

        return 5  # Default for CPU

    def _detect_num_registers(self, module_name: str, content: str, ports: Dict, filename: str = "") -> int:
        """
        Detect number of registers in a register file.
        """
        import re

        # ★★★ 同时检查 filename 和 module_name ★★★
        names_to_check = [filename, module_name]

        for name in names_to_check:
            if not name:
                continue

            patterns = [
                r'_(\d+)x(?:_|\d|$)',  # _8x_, _16x_, _8x_20260107
                r'(\d+)x\d+',  # 32x32
                r'_(\d+)(?:regs?|registers?)',  # _32regs
            ]

            for pattern in patterns:
                match = re.search(pattern, name.lower())
                if match:
                    num = int(match.group(1))
                    if num in [4, 8, 16, 32, 64]:
                        return num

        # Priority 2: From content
        array_patterns = [
            r'reg\s*\[\d+:\d+\]\s*\w+\s*\[(?:\d+:)?(\d+)\]',
            r'reg\s*\[\d+:\d+\]\s*\w+\s*\[(\d+):0\]',
        ]

        for pattern in array_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                num = int(match.group(1)) + 1
                if num in [4, 8, 16, 32, 64]:
                    return num

        # Priority 3: Check address width
        addr_ports = ['rs1', 'rs2', 'rd', 'raddr', 'waddr', 'read_addr', 'write_addr', 'addr']

        for port in ports.get('inputs', []):
            if port['name'].lower() in addr_ports:
                addr_width = port['width']
                return 2 ** addr_width  # 3-bit addr = 8 regs, 4-bit = 16, 5-bit = 32

        return 32  # Default

    def _detect_module_type(self, module_name: str, ports: Dict, content: str) -> Tuple[str, str]:
        """
        Auto-detect module type based on name, ports, and content.

        Returns:
            (module_type, confidence) - confidence is 'high', 'medium', or 'low'
        """
        scores = {t: 0 for t in self.MODULE_TYPE_KEYWORDS.keys()}

        module_name_lower = module_name.lower()
        content_lower = content.lower()
        input_names = [p.lower() for p in ports.get('input_names', [])]
        output_names = [p.lower() for p in ports.get('output_names', [])]
        all_port_names = input_names + output_names

        for module_type, keywords in self.MODULE_TYPE_KEYWORDS.items():
            # Check module name (highest weight)
            for name_kw in keywords['names']:
                if name_kw in module_name_lower:
                    scores[module_type] += 10

            # Check port names (medium weight)
            for port_kw in keywords['ports']:
                if any(port_kw in p for p in all_port_names):
                    scores[module_type] += 3

            # Check content for signals (lower weight)
            for signal_kw in keywords['signals']:
                if signal_kw in content_lower:
                    scores[module_type] += 1

        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Determine confidence
        if best_score >= 10:
            confidence = 'high'
        elif best_score >= 5:
            confidence = 'medium'
        elif best_score >= 2:
            confidence = 'low'
        else:
            best_type = 'other'
            confidence = 'low'

        return best_type, confidence

    def _save_file(self, filename: str, file_content: bytes) -> Path:
        """
        Save uploaded file to storage directory.
        """
        # Add timestamp to prevent overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(filename).stem
        suffix = Path(filename).suffix

        new_filename = f"{stem}_{timestamp}{suffix}"
        save_path = self.upload_dir / new_filename

        with open(save_path, 'wb') as f:
            f.write(file_content)

        return save_path

    def get_module_type_options(self) -> List[Dict]:
        """
        Get list of module type options for UI dropdown.
        """
        return [
            {'value': 'alu', 'label': 'ALU - Arithmetic Logic Unit'},
            {'value': 'counter', 'label': 'Counter - Up/Down Counter'},
            {'value': 'regfile', 'label': 'Register File'},
            {'value': 'cpu', 'label': 'CPU - Processor'},
            {'value': 'other', 'label': 'Other - Custom Module'}
        ]


# Test function
def test_parser():
    """Test the parser with sample Verilog code."""

    sample_verilog = b"""
    // Sample ALU module
    module alu_16bit (
        input  wire        clk,
        input  wire        rst_n,
        input  wire [15:0] A,
        input  wire [15:0] B,
        input  wire [3:0]  opcode,
        output reg  [15:0] result,
        output reg         zero,
        output reg         overflow,
        output reg         negative
    );

        // ALU operations
        localparam ADD = 4'b0000;
        localparam SUB = 4'b0001;
        localparam AND = 4'b0010;
        localparam OR  = 4'b0011;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                result <= 16'b0;
            end else begin
                case (opcode)
                    ADD: result <= A + B;
                    SUB: result <= A - B;
                    AND: result <= A & B;
                    OR:  result <= A | B;
                endcase
            end
        end

    endmodule
    """

    parser = VerilogParser()
    result = parser.parse_file("alu_16bit.v", sample_verilog)

    print("=" * 60)
    print("Verilog Parser Test")
    print("=" * 60)
    print(f"Success: {result['success']}")

    if result['success']:
        print(f"Modules found: {result['module_count']}")
        for module in result['modules']:
            print(f"\n  Module: {module['name']}")
            print(f"  Detected Type: {module['detected_type']} ({module['type_confidence']} confidence)")
            print(f"  Bitwidth: {module['bitwidth']}")
            print(f"  Inputs: {module['ports']['input_names']}")
            print(f"  Outputs: {module['ports']['output_names']}")
            print(f"\n  Suggested BDD:")
            print("  " + module['suggested_bdd'].replace('\n', '\n  '))
    else:
        print(f"Error: {result.get('error')}")


if __name__ == "__main__":
    test_parser()