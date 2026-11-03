// Browser check of the M4 repair buttons on the Step-4 Problems card.
// Runs the page's own Step 3 and Step 4 on a staged run, then
//   1. "Correct the scenarios"  (real: /api/repair/scenarios, no model)
//   2. "Regenerate the design"  (the model call is replaced by a staged fixed
//      design via request interception; the rerun after it is real)
// Needs: the app on localhost:5000 with DATABASE_URL set, iverilog, and
//   python tests/ui/stage_run.py [regfile] > <dir>/state.json
//   NODE_PATH=$(npm root -g) node tests/ui/repair_flow.js <dir>
const fs = require('fs');
const { chromium } = require('playwright');
(async () => {
  const out = process.argv[2] || '.';
  const state = JSON.parse(fs.readFileSync(`${out}/state.json`, 'utf-8'));
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [], dialogs = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('dialog', d => { dialogs.push(d.message()); d.accept(); });
  await page.goto('http://localhost:5000/', { waitUntil: 'networkidle' });

  // as after Steps 0-2 (generated with GEMINI), then the page's own Step 3 and Step 4
  await page.evaluate(async (st) => {
    document.getElementById('main-app').style.display = 'block';   // as after entering the access password
    document.querySelectorAll('.step-section').forEach(s => s.classList.remove('collapsed'));
    currentWorkflowMode = 'implementation';
    currentRunId = 'web_ui_repair_' + Date.now();
    currentRequirements = st.requirements;
    currentDUT = st.dut;
    currentBDD = st.bdd;
    await generateTestbench();
    await runSimulation();
  }, state);

  const problemsReady = async (label) => {
    await page.waitForFunction(() => explainState && document.getElementById('sim-problems').style.display === 'block',
                               null, { timeout: 60000 });
    console.log(`\n== ${label}`);
    const note = await page.$eval('#pb-note', n => n.style.display === 'none' ? '' : n.innerText);
    if (note) console.log('note:', note.replace(/\n+/g, ' / '));
    console.log('banner:', (await page.textContent('#pb-banner')).trim());
    const items = await page.$$eval('.pb-item', es => es.map(e =>
      `[${e.querySelector('.pb-tag').textContent}] ${e.querySelector('.pb-title').textContent}`));
    items.forEach(t => console.log('  ' + t));
    const acts = await page.$$eval('#pb-actions .pb-action', as => as.map(a =>
      `<${a.querySelector('button').textContent}> ${a.querySelector('.pb-action-text').textContent}`));
    acts.forEach(t => console.log('  action: ' + t));
    return { items, acts };
  };

  const first = await problemsReady('after Step 4');
  await page.locator('#sim-problems').screenshot({ path: `${out}/repair_1_problems.png` });

  // 1. correct the scenarios
  await page.getByRole('button', { name: 'Correct the scenarios' }).click();
  await page.waitForFunction(() => repairNotice && repairNotice.html.includes('Scenarios corrected')
                                   && explainState && explainState.repairs.scenario_values === 0,
                             null, { timeout: 60000 });
  const second = await problemsReady('after "Correct the scenarios"');
  console.log('Step 2 shows:', await page.textContent('#bdd-filename'), '| Step 3 shows:', await page.textContent('#tb-filename'));
  await page.locator('#sim-problems').screenshot({ path: `${out}/repair_2_scenarios.png` });

  // 2. regenerate the design; the model call is replaced by the staged fixed design
  await page.route('**/api/generate-hardware', async route => {
    const req = route.request().postDataJSON();
    console.log('\nrequest to Step 1:', JSON.stringify({ llm: req.llm, module_type: req.module_type,
      run_id: req.run_id === await page.evaluate(() => currentRunId), repair_of: req.repair_of }));
    await route.fulfill({ json: { success: true, filename: state.fixed_design.filename,
      full_content: state.fixed_design.content, llm: 'gemini', bitwidth: 32, module_type: state.dut.moduleType,
      run_id: req.run_id, duv_compile: { ok: true, error: null },
      repair: { feedback_id: 0, ...state.fake_repair } } });
  });
  await page.getByRole('button', { name: /Regenerate the design with GEMINI/ }).click();
  await page.waitForFunction(() => repairNotice && repairNotice.html.includes('Design regenerated')
                                   && explainState && explainState.repairs.design_failures === 0,
                             null, { timeout: 60000 });
  const third = await problemsReady('after "Regenerate the design"');
  console.log('Step 1 shows:', await page.textContent('#hw-filename'));
  await page.locator('#sim-problems').screenshot({ path: `${out}/repair_3_design.png` });

  await page.setViewportSize({ width: 390, height: 800 });
  await page.locator('#sim-problems').screenshot({ path: `${out}/repair_3_design_mobile.png` });
  console.log('\nhorizontal overflow at 390px:',
              await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1));
  console.log('confirm dialogs:', dialogs.length);
  dialogs.forEach(d => console.log('  ' + d.replace(/\n+/g, ' / ')));
  console.log('page errors:', errors.length ? errors : 'none');
  const ok = first.acts.length === 2 && second.acts.length === 1 && third.acts.length === 0
             && !second.items.some(t => t.startsWith('[Scenario]'))
             && !third.items.some(t => t.startsWith('[Design]')) && !errors.length;
  console.log(ok ? 'RESULT: OK' : 'RESULT: FAILED');
  await browser.close();
  process.exit(ok ? 0 : 1);
})().catch(e => { console.error(e); process.exit(1); });
