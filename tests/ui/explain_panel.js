// Browser check of the Step-4 explanations panel (milestone M3).
// Needs: the app running on localhost:5000 with DATABASE_URL set, and the
// files of one run under output/xdb_ui/ (see docs/explainability_db_setup.md).
//   NODE_PATH=$(npm root -g) node tests/ui/explain_panel.js <screenshot dir>
const { chromium } = require('playwright');
(async () => {
  const out = process.argv[2];
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto('http://localhost:5000/', { waitUntil: 'networkidle' });
  const paths = {
    requirements_filepath: 'output/xdb_ui/requirements/req.json',
    bdd_filepath: 'output/xdb_ui/bdd/alu.feature',
    duv_filepath: 'output/xdb_ui/dut/alu_32bit.v',
    testbench_path: 'output/xdb_ui/testbench/default/alu_tb.v',
    log_file: 'output/xdb_ui/simulation/alu.log',
    run_id: 'web_ui_test', workflow_mode: 'implementation'
  };
  // what the page does after "Run Simulation": show the result, then load the run report
  const state = await page.evaluate(async (paths) => {
    document.getElementById('main-app').style.display = 'block';   // as after entering the access password
    document.querySelectorAll('.step-section').forEach(s => s.classList.remove('collapsed'));
    const rep = await (await fetch('/api/run-report', { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(paths) })).json();
    currentSimResult = { paths, log_file: paths.log_file };
    document.getElementById('sim-result').classList.add('visible');
    renderScenarioResults(rep.scenario_results);
    await loadRunReport();
    return { db: rep.explainability_db };
  }, paths);
  console.log('run-report ->', JSON.stringify(state));
  await page.waitForSelector('#sim-explain', { state: 'visible', timeout: 30000 });
  await page.waitForSelector('.xp-slot .why-btn', { timeout: 10000 });
  const explain = await page.$('#sim-explain');
  await explain.scrollIntoViewIfNeeded();
  await explain.screenshot({ path: `${out}/m3_summary.png` });
  console.log('headline:', await page.textContent('#xp-headline'));
  console.log('chips:', (await page.$$eval('#xp-verdicts .xp-chip', e => e.map(x => x.textContent))).join(' | '));
  console.log('spec:', (await page.$$eval('.xp-spec-item', e => e.map(x => x.textContent.trim()))).join(' || '));
  // the failing SUB scenario is open by default; its row 7 has a verdict chip and a Why? button
  const subRow = page.locator('.xp-slot[data-tag="BDD:ALU-SUB-001/r7"]');
  console.log('SUB r7 slot:', await subRow.textContent());
  const scn = page.locator('details.scn', { hasText: 'ALU-SUB-001' });
  await scn.screenshot({ path: `${out}/m3_rows.png` });
  await subRow.locator('.why-btn').click();
  await page.waitForFunction(() => !document.getElementById('xp-text').textContent.startsWith('Loading'));
  await page.locator('.xp-dialog').screenshot({ path: `${out}/m3_why_check.png` });
  console.log('modal:', (await page.textContent('#xp-text')).split('\n').slice(0, 3).join(' / '));
  await page.keyboard.press('Escape');
  await page.locator('.xp-spec-item', { hasText: 'zero flag' }).locator('.why-btn').click();
  await page.waitForFunction(() => !document.getElementById('xp-text').textContent.startsWith('Loading'));
  await page.locator('.xp-dialog').screenshot({ path: `${out}/m3_why_global.png` });
  console.log('global:', (await page.textContent('#xp-text')).split('\n').filter(l => l.startsWith('Check')).join(''));
  await page.keyboard.press('Escape');
  const reqBtn = page.locator('.xp-slot[data-req="REQ-2"] .why-btn');
  await reqBtn.click();
  await page.waitForFunction(() => !document.getElementById('xp-text').textContent.startsWith('Loading'));
  console.log('requirement:', (await page.textContent('#xp-text')).split('\n').slice(0, 3).join(' / '));
  await page.keyboard.press('Escape');
  // mobile width: no horizontal scroll
  await page.setViewportSize({ width: 390, height: 800 });
  await page.locator('#sim-explain').screenshot({ path: `${out}/m3_mobile.png` });
  const overflowX = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  console.log('horizontal overflow at 390px:', overflowX);
  console.log('page errors:', errors.length ? errors : 'none');
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
