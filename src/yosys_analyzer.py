"""
Yosys Analyzer - Synthesize and Visualize Verilog DUT
=====================================================

Uses Yosys to:
1. Synthesize Verilog code and check for errors
2. Generate circuit schematic (SVG via Graphviz)
3. Extract resource statistics (cells, wires, etc.)
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional


class YosysAnalyzer:
    """Analyze Verilog DUT files using Yosys synthesis tool."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.output_dir = self.project_root / "output" / "yosys"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.yosys_available = self._check_yosys()

    def _check_yosys(self) -> bool:
        """Check if yosys and graphviz (dot) are available."""
        try:
            result = subprocess.run(
                ['yosys', '-V'],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _check_graphviz(self) -> bool:
        """Check if graphviz dot is available."""
        try:
            result = subprocess.run(
                ['dot', '-V'],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def analyze(self, verilog_path: str, module_name: Optional[str] = None) -> Dict:
        """
        Full analysis of a Verilog DUT file.

        Args:
            verilog_path: Path to the .v file
            module_name: Top module name (auto-detected if not provided)

        Returns:
            Dictionary with synthesis results, stats, and SVG diagram path
        """
        if not self.yosys_available:
            return {
                'success': False,
                'error': 'Yosys is not installed. Install with: apt-get install yosys graphviz'
            }

        vpath = Path(verilog_path)
        if not vpath.exists():
            return {'success': False, 'error': f'File not found: {verilog_path}'}

        # Auto-detect module name from file
        if not module_name:
            module_name = self._detect_module_name(vpath)

        if not module_name:
            return {'success': False, 'error': 'Could not detect module name'}

        # Create output directory for this analysis
        stem = vpath.stem
        analysis_dir = self.output_dir / stem
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Run Yosys synthesis + stats + diagram generation
        result = self._run_yosys(vpath, module_name, analysis_dir)
        return result

    def _detect_module_name(self, vpath: Path) -> Optional[str]:
        """Detect top module name from Verilog file."""
        try:
            content = vpath.read_text(encoding='utf-8')
            match = re.search(r'module\s+(\w+)\s*[\(#]', content)
            if match:
                return match.group(1)
        except Exception:
            pass
        return None

    def _run_yosys(self, vpath: Path, module_name: str, out_dir: Path) -> Dict:
        """Run Yosys synthesis pipeline."""

        json_path = out_dir / "netlist.json"
        dot_path = out_dir / "circuit.dot"
        svg_path = out_dir / "circuit.svg"
        stat_path = out_dir / "stats.txt"

        # Build Yosys script
        yosys_script = f"""
read_verilog -sv {vpath}
hierarchy -top {module_name}
proc
opt
fsm
opt
memory
opt
techmap
opt
stat -top {module_name}
show -format dot -prefix {out_dir / 'circuit'} -width {module_name}
write_json {json_path}
"""

        try:
            # Run Yosys
            result = subprocess.run(
                ['yosys', '-p', yosys_script],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.project_root)
            )

            yosys_log = result.stdout + result.stderr

            # Check for errors
            if result.returncode != 0:
                # Extract meaningful error from log
                error_lines = [
                    l for l in yosys_log.split('\n')
                    if 'error' in l.lower() or 'ERROR' in l
                ]
                error_msg = '\n'.join(error_lines[-5:]) if error_lines else 'Synthesis failed'
                return {
                    'success': False,
                    'error': error_msg,
                    'log': yosys_log[-2000:]
                }

            # Parse stats from log
            stats = self._parse_stats(yosys_log)

            # Save stats
            stat_path.write_text(yosys_log, encoding='utf-8')

            # Convert DOT to SVG using graphviz
            svg_content = None
            if dot_path.exists() and self._check_graphviz():
                try:
                    dot_result = subprocess.run(
                        ['dot', '-Tsvg', str(dot_path)],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if dot_result.returncode == 0:
                        svg_content = dot_result.stdout
                        svg_path.write_text(svg_content, encoding='utf-8')
                except Exception:
                    pass

            # If no SVG from graphviz, try alternative
            if not svg_content and dot_path.exists():
                svg_content = self._dot_to_simple_svg(dot_path)
                if svg_content:
                    svg_path.write_text(svg_content, encoding='utf-8')

            # Parse warnings from log
            warnings = self._parse_warnings(yosys_log)

            return {
                'success': True,
                'module_name': module_name,
                'stats': stats,
                'warnings': warnings,
                'svg_path': str(svg_path) if svg_path.exists() else None,
                'svg_content': svg_content,
                'json_path': str(json_path) if json_path.exists() else None,
                'log_path': str(stat_path),
            }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Yosys synthesis timed out (60s limit)'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _parse_stats(self, log: str) -> Dict:
        """Parse resource statistics from Yosys log output."""
        stats = {
            'wires': 0,
            'wire_bits': 0,
            'memories': 0,
            'memory_bits': 0,
            'processes': 0,
            'cells': {},
            'cell_total': 0,
        }

        in_stat_section = False
        for line in log.split('\n'):
            stripped = line.strip()

            if 'Printing statistics.' in line or 'Statistics for' in line:
                in_stat_section = True
                continue

            if in_stat_section:
                # Parse: "   Number of wires:               15"
                m = re.match(r'Number of wires:\s+(\d+)', stripped)
                if m:
                    stats['wires'] = int(m.group(1))
                    continue

                m = re.match(r'Number of wire bits:\s+(\d+)', stripped)
                if m:
                    stats['wire_bits'] = int(m.group(1))
                    continue

                m = re.match(r'Number of memories:\s+(\d+)', stripped)
                if m:
                    stats['memories'] = int(m.group(1))
                    continue

                m = re.match(r'Number of memory bits:\s+(\d+)', stripped)
                if m:
                    stats['memory_bits'] = int(m.group(1))
                    continue

                m = re.match(r'Number of processes:\s+(\d+)', stripped)
                if m:
                    stats['processes'] = int(m.group(1))
                    continue

                m = re.match(r'Number of cells:\s+(\d+)', stripped)
                if m:
                    stats['cell_total'] = int(m.group(1))
                    continue

                # Individual cell types: "$_AND_          12"
                m = re.match(r'(\$?\w+)\s+(\d+)', stripped)
                if m and m.group(1).startswith('$'):
                    stats['cells'][m.group(1)] = int(m.group(2))
                    continue

                # End of stat section (blank line or new section)
                if stripped == '' and stats['cell_total'] > 0:
                    in_stat_section = False

        return stats

    def _parse_warnings(self, log: str) -> list:
        """Extract warnings from Yosys log."""
        warnings = []
        for line in log.split('\n'):
            if 'Warning:' in line or 'warning:' in line:
                # Clean up the line
                clean = line.strip()
                if clean and clean not in warnings:
                    warnings.append(clean)
        return warnings[:10]  # Limit to 10 warnings

    def _dot_to_simple_svg(self, dot_path: Path) -> Optional[str]:
        """Fallback: read DOT and return it as-is if graphviz fails."""
        # Just return None - the frontend can handle missing SVG
        return None
