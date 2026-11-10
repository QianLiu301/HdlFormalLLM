"""Step 3 writes the testbench where Step 4 looks for it, for every model.

A real run (2026-11-10, Gemma, 16-bit ALU, tests/data/gemma_alu_16): Step 3 wrote
the testbench under output/testbench/default/ because 'gemma' was not on a fixed
list of folder names, and Step 4, which rebuilt the path from the model name,
reported "Testbench not found: output/testbench/gemma/...".
"""
import contextlib
import io
import shutil
from pathlib import Path

import pytest

DATA = Path(__file__).parent / 'data' / 'gemma_alu_16'


@pytest.fixture
def app(monkeypatch, tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        import main
    from src.simulation_runner import WebSimulationRunner
    monkeypatch.setattr(main, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(main, 'simulation_runner', WebSimulationRunner(project_root=str(tmp_path)))
    monkeypatch.setattr('src.experiment_logger.DB_PATH', str(tmp_path / 'calls.db'))
    return main, tmp_path


@pytest.mark.parametrize('model', ['gemma', 'glm', 'qwen', 'a_model_added_later'])
def test_step3_reports_the_path_step4_uses(app, model):
    main, root = app
    for d in (f'output/bdd/{model}', f'output/dut/{model}'):
        (root / d).mkdir(parents=True)
    shutil.copy(DATA / 'alu.feature', root / f'output/bdd/{model}/alu_16bit.feature')
    shutil.copy(DATA / 'alu.v', root / f'output/dut/{model}/alu_16bit.v')
    c = main.app.test_client()
    with contextlib.redirect_stdout(io.StringIO()):
        tb = c.post('/api/generate-testbench', json={
            'bdd_filepath': f'output/bdd/{model}/alu_16bit.feature',
            'dut_filepath': f'output/dut/{model}/alu_16bit.v',
            'dut_info': {'module_type': 'alu', 'bitwidth': 16,
                         'dut_filepath': f'output/dut/{model}/alu_16bit.v'}}).get_json()
    assert tb['success'], tb
    assert tb['relpath'] == f'output/testbench/{model}/{tb["filename"]}'
    if shutil.which('iverilog'):
        with contextlib.redirect_stdout(io.StringIO()):
            sim = c.post('/api/run-simulation', json={
                'testbench_path': tb['relpath'], 'dut_path': f'output/dut/{model}/alu_16bit.v'}).get_json()
        assert 'Testbench not found' not in str(sim.get('error')), sim


def test_the_page_uses_the_reported_path():
    page = (Path(__file__).parent.parent / 'static' / 'bdd_generator.html').read_text(encoding='utf-8')
    assert 'currentTbPath = data.relpath' in page
    assert 'const tbPath = tbFilePath();' in page
