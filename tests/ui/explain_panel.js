// Browser check of the Step-4 explanations (Problems card, technical details, Details popup).
// Needs: the app running on localhost:5000 with DATABASE_URL set, and the files of
// one run under output/ (paths below; see docs/explainability_db_setup.md).
//   NODE_PATH=$(npm root -g) node tests/ui/explain_panel.js <screenshot dir>
const { chromium } = require('playwright');
(async () => {
  const out = process.argv[2] || '.';
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
    testbench_path: 'output/testbench/default/xdb_ui_alu_tb.v',
    log_file: 'output/xdb_ui/simulation/alu.log',
    run_id: 'web_ui_test', workflow_mode: 'implementation'
  };
  // what the page does after "Run Simulation": show the result, then load the run report
  await page.evaluate(async (paths) => {
    document.getElementById('main-app').style.display = 'block';   // as after entering the access password
    document.querySelectorAll('.step-section').forEach(s => s.classList.remove('collapsed'));
    const rep = await (await fetch('/api/run-report', { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(paths) })).json();
    currentSimResult = { paths, log_file: paths.log_file };
    document.getElementById('sim-result').classList.add('visible');
    renderScenarioResults(rep.scenario_results);
    await loadRunReport();
  }, paths);

  await page.waitForSelector('#sim-problems', { state: 'visible', timeout: 30000 });
  console.log('banner:', (await page.textContent('#pb-banner')).trim());
  const items = await page.$$eval('.pb-item', es => es.map(e =>
    `[${e.querySelector('.pb-tag').textContent}] ${e.querySelector('.pb-title').textContent} — ${e.querySelector('.pb-detail').textContent}`));
  items.forEach(t => console.log('  ' + t));
  console.log('checked:', await page.textContent('#pb-checked'));
  console.log('technical details open:', await page.$eval('#sim-tech', d => d.open));
  await page.locator('#sim-problems').screenshot({ path: `${out}/ui_problems.png` });

  // Details of the first problem: plain text first, technical text collapsed
  await page.locator('.pb-item .why-btn').first().click();
  await page.waitForFunction(() => !document.getElementById('xp-plain').textContent.startsWith('Loading'));
  console.log('popup title:', await page.textContent('#xp-title'));
  console.log('popup plain:', (await page.$$eval('#xp-plain p', ps => ps.map(p => p.textContent))).join(' / '));
  console.log('technical text present, collapsed:', (await page.textContent('#xp-text')).startsWith('❓'),
              !(await page.$eval('#xp-tech', d => d.open)));
  await page.locator('.xp-dialog').screenshot({ path: `${out}/ui_details.png` });
  await page.keyboard.press('Escape');

  // the technical sections are still there, one click away
  await page.locator('#sim-tech > summary').click();
  await page.waitForSelector('#sim-explain', { state: 'visible' });
  console.log('coverage rows:', (await page.$$eval('#sim-cov-list tbody tr', rs => rs.map(r =>
    r.cells[0].textContent + ' ' + r.cells[3].textContent.replace(/\s+/g, ' ').trim()))).join(' | '));

  await page.setViewportSize({ width: 390, height: 800 });
  await page.locator('#sim-problems').screenshot({ path: `${out}/ui_problems_mobile.png` });
  console.log('horizontal overflow at 390px:',
              await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1));
  console.log('page errors:', errors.length ? errors : 'none');
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
