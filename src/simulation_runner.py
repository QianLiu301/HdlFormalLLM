"""
Simulation Runner for Web Interface
====================================
Runs Verilog simulations using iverilog/vvp on the server.
"""

import subprocess
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class WebSimulationRunner:
    """Web-friendly simulation runner"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.tools_available = self._check_tools()

    def _check_tools(self) -> Dict[str, bool]:
        """Check available simulation tools"""
        tools = {
            'iverilog': False,
            'vvp': False
        }

        if shutil.which('iverilog'):
            try:
                subprocess.run(['iverilog', '-V'], capture_output=True, timeout=5)
                tools['iverilog'] = True
            except:
                pass

        if shutil.which('vvp'):
            try:
                subprocess.run(['vvp', '-V'], capture_output=True, timeout=5)
                tools['vvp'] = True
            except:
                pass

        return tools

    def can_run_simulation(self) -> bool:
        """Check if simulation can be run"""
        return self.tools_available.get('iverilog', False) and \
               self.tools_available.get('vvp', False)

    def get_tools_status(self) -> Dict:
        """Get tools status for API response"""
        return {
            'can_simulate': self.can_run_simulation(),
            'tools': self.tools_available
        }

    def run_single(self, testbench_path: str, dut_path: str) -> Dict:
        """
        Run simulation for a single testbench.

        Args:
            testbench_path: Path to testbench .v file
            dut_path: Path to DUT .v file

        Returns:
            Dict with simulation results
        """
        if not self.can_run_simulation():
            return {
                'success': False,
                'error': 'Simulation tools (iverilog/vvp) not available on server',
                'tools_available': self.tools_available
            }

        tb_path = Path(testbench_path)
        dut_file = Path(dut_path)

        if not tb_path.exists():
            return {'success': False, 'error': f'Testbench not found: {testbench_path}'}

        if not dut_file.exists():
            return {'success': False, 'error': f'DUT not found: {dut_path}'}

        # Setup output directory
        results_dir = self.project_root / 'output' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        vvp_file = results_dir / f'{tb_path.stem}_{timestamp}.vvp'
        log_file = results_dir / f'{tb_path.stem}_{timestamp}.log'

        result = {
            'success': False,
            'testbench': tb_path.name,
            'dut': dut_file.name,
            'compile_time': 0,
            'sim_time': 0,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'pass_rate': 0,
            'log_preview': '',
            'log_file': None,
            'vcd_file': None,
            'waveform_data': None,
            'errors': []
        }

        # Step 1: Compile with iverilog
        try:
            import time
            start_time = time.time()

            compile_cmd = [
                'iverilog',
                '-g2012',
                '-o', str(vvp_file),
                str(dut_file),
                str(tb_path)
            ]

            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )

            result['compile_time'] = round(time.time() - start_time, 2)

            if compile_result.returncode != 0:
                result['errors'].append(f'Compilation failed: {compile_result.stderr}')
                return result

        except subprocess.TimeoutExpired:
            result['errors'].append('Compilation timeout (30s)')
            return result
        except Exception as e:
            result['errors'].append(f'Compilation error: {str(e)}')
            return result

        # Step 2: Run simulation with vvp
        try:
            start_time = time.time()

            sim_result = subprocess.run(
                ['vvp', str(vvp_file)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(results_dir),
                encoding='utf-8',
                errors='replace'
            )

            result['sim_time'] = round(time.time() - start_time, 2)

            if sim_result.returncode == 0:
                result['success'] = True

                # Save log
                log_content = sim_result.stdout
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                result['log_file'] = str(log_file.relative_to(self.project_root))
                result['log_preview'] = log_content[:3000] + ('...' if len(log_content) > 3000 else '')
                result['full_log'] = log_content

                # Parse results
                self._parse_output(result, log_content)

                # Check for VCD file
                vcd_files = list(results_dir.glob('*.vcd'))
                if vcd_files:
                    latest_vcd = max(vcd_files, key=lambda x: x.stat().st_mtime)
                    result['vcd_file'] = str(latest_vcd.relative_to(self.project_root))

                    # Parse VCD for waveform data
                    result['waveform_data'] = self._parse_vcd_simple(latest_vcd)
            else:
                result['errors'].append(f'Simulation failed: {sim_result.stderr}')

        except subprocess.TimeoutExpired:
            result['errors'].append('Simulation timeout (60s)')
        except Exception as e:
            result['errors'].append(f'Simulation error: {str(e)}')

        # Cleanup vvp file
        try:
            if vvp_file.exists():
                vvp_file.unlink()
        except:
            pass

        return result

    def _parse_output(self, result: Dict, output: str):
        """Parse simulation output for test statistics"""
        total_match = re.search(r'Total:\s*(\d+)', output)
        passed_match = re.search(r'Passed:\s*(\d+)', output)
        failed_match = re.search(r'Failed:\s*(\d+)', output)

        if total_match:
            result['total_tests'] = int(total_match.group(1))
        if passed_match:
            result['passed_tests'] = int(passed_match.group(1))
        if failed_match:
            result['failed_tests'] = int(failed_match.group(1))

        # Fallback: count PASS/FAIL occurrences
        if result['total_tests'] == 0:
            passed = len(re.findall(r'PASS|✓|✅', output, re.IGNORECASE))
            failed = len(re.findall(r'FAIL|✗|❌', output, re.IGNORECASE))
            result['passed_tests'] = passed
            result['failed_tests'] = failed
            result['total_tests'] = passed + failed

        if result['total_tests'] > 0:
            result['pass_rate'] = round(
                (result['passed_tests'] / result['total_tests']) * 100, 1
            )

    def _parse_vcd_simple(self, vcd_path: Path, max_signals: int = 10) -> Optional[Dict]:
        """Parse VCD file and extract simplified waveform data"""
        try:
            with open(vcd_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(50000)  # Read first 50KB

            signals = {}
            signal_map = {}
            current_time = 0
            max_time = 0

            lines = content.split('\n')

            for line in lines:
                line = line.strip()

                # Parse signal definitions
                if line.startswith('$var'):
                    parts = line.split()
                    if len(parts) >= 5:
                        var_type = parts[1]
                        width = int(parts[2]) if parts[2].isdigit() else 1
                        identifier = parts[3]
                        name = parts[4].split('[')[0]  # Remove bit range

                        signal_map[identifier] = name
                        if name not in signals:
                            signals[name] = {
                                'width': width,
                                'type': var_type,
                                'changes': []
                            }

                # Parse time changes
                elif line.startswith('#'):
                    try:
                        current_time = int(line[1:])
                        max_time = max(max_time, current_time)
                    except:
                        pass

                # Parse value changes (single bit)
                elif len(line) >= 2 and line[0] in '01xXzZ':
                    value = line[0]
                    identifier = line[1:]
                    if identifier in signal_map:
                        name = signal_map[identifier]
                        if name in signals and len(signals[name]['changes']) < 100:
                            signals[name]['changes'].append({
                                'time': current_time,
                                'value': 1 if value == '1' else 0
                            })

                # Parse value changes (multi-bit)
                elif line.startswith('b'):
                    parts = line.split()
                    if len(parts) >= 2:
                        value = parts[0][1:]
                        identifier = parts[1]
                        if identifier in signal_map:
                            name = signal_map[identifier]
                            if name in signals and len(signals[name]['changes']) < 100:
                                try:
                                    int_value = int(value.replace('x', '0').replace('z', '0'), 2)
                                except:
                                    int_value = 0
                                signals[name]['changes'].append({
                                    'time': current_time,
                                    'value': int_value
                                })

            # Select most important signals
            priority = ['clk', 'rst', 'a', 'b', 'opcode', 'result', 'out', 'zero', 'carry']
            sorted_names = sorted(
                signals.keys(),
                key=lambda x: (
                    0 if any(p in x.lower() for p in priority) else 1,
                    x
                )
            )[:max_signals]

            return {
                'signals': {k: signals[k] for k in sorted_names if k in signals},
                'signal_names': sorted_names,
                'max_time': max_time
            }

        except Exception as e:
            print(f"VCD parse error: {e}")
            return None

    def run_batch(self) -> Dict:
        """Run simulations for all testbenches"""
        if not self.can_run_simulation():
            return {
                'success': False,
                'error': 'Simulation tools not available',
                'tools_available': self.tools_available
            }

        testbench_dir = self.project_root / 'output' / 'testbench'
        dut_dir = self.project_root / 'output' / 'dut'

        if not testbench_dir.exists():
            return {'success': False, 'error': 'No testbench directory found'}

        results_by_llm = {}

        # Scan for testbenches in LLM subdirectories
        for item in testbench_dir.iterdir():
            if item.is_dir():
                llm_name = item.name
                results_by_llm[llm_name] = []

                for tb_file in item.glob('*_tb.v'):
                    # Extract bitwidth from filename
                    match = re.search(r'(\d+)bit', tb_file.name)
                    if match:
                        bitwidth = match.group(1)

                        # Look for DUT in LLM-specific directory first
                        llm_dut_dir = dut_dir / llm_name
                        dut_file = None

                        if llm_dut_dir.exists():
                            # Find DUT file with matching bitwidth (may have timestamp)
                            dut_files = list(llm_dut_dir.glob(f'alu_{bitwidth}bit*.v'))
                            if dut_files:
                                dut_file = max(dut_files, key=lambda x: x.stat().st_mtime)

                        # Fallback: check root dut directory
                        if not dut_file:
                            fallback_files = list(dut_dir.glob(f'alu_{bitwidth}bit*.v'))
                            if fallback_files:
                                dut_file = max(fallback_files, key=lambda x: x.stat().st_mtime)

                        if dut_file and dut_file.exists():
                            result = self.run_single(str(tb_file), str(dut_file))
                            result['llm'] = llm_name
                            results_by_llm[llm_name].append(result)
                    # Extract bitwidth from filename
                    match = re.search(r'(\d+)bit', tb_file.name)
                    if match:
                        bitwidth = match.group(1)
                        dut_file = dut_dir / f'alu_{bitwidth}bit.v'

                        if dut_file.exists():
                            result = self.run_single(str(tb_file), str(dut_file))
                            result['llm'] = llm_name
                            results_by_llm[llm_name].append(result)

        # Also check root testbench directory
        for tb_file in testbench_dir.glob('*_tb.v'):
            match = re.search(r'(\d+)bit', tb_file.name)
            if match:
                bitwidth = match.group(1)
                dut_file = dut_dir / f'alu_{bitwidth}bit.v'

                if dut_file.exists():
                    if 'default' not in results_by_llm:
                        results_by_llm['default'] = []
                    result = self.run_single(str(tb_file), str(dut_file))
                    result['llm'] = 'default'
                    results_by_llm['default'].append(result)

        if not results_by_llm:
            return {'success': False, 'error': 'No testbenches found'}

        # Generate summary
        summary = self._generate_summary(results_by_llm)

        return {
            'success': True,
            'results_by_llm': results_by_llm,
            'summary': summary,
            'llm_count': len(results_by_llm),
            'total_simulations': summary['total_simulations']
        }

    def _generate_summary(self, results_by_llm: Dict) -> Dict:
        """Generate summary statistics"""
        summary = {
            'total_simulations': 0,
            'successful': 0,
            'failed': 0,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'overall_pass_rate': 0,
            'by_llm': {}
        }

        for llm_name, results in results_by_llm.items():
            llm_summary = {
                'simulations': len(results),
                'successful': sum(1 for r in results if r.get('success')),
                'total_tests': sum(r.get('total_tests', 0) for r in results),
                'passed_tests': sum(r.get('passed_tests', 0) for r in results),
                'failed_tests': sum(r.get('failed_tests', 0) for r in results),
                'pass_rate': 0
            }

            if llm_summary['total_tests'] > 0:
                llm_summary['pass_rate'] = round(
                    (llm_summary['passed_tests'] / llm_summary['total_tests']) * 100, 1
                )

            summary['by_llm'][llm_name] = llm_summary
            summary['total_simulations'] += llm_summary['simulations']
            summary['successful'] += llm_summary['successful']
            summary['total_tests'] += llm_summary['total_tests']
            summary['passed_tests'] += llm_summary['passed_tests']
            summary['failed_tests'] += llm_summary['failed_tests']

        summary['failed'] = summary['total_simulations'] - summary['successful']

        if summary['total_tests'] > 0:
            summary['overall_pass_rate'] = round(
                (summary['passed_tests'] / summary['total_tests']) * 100, 1
            )

        return summary