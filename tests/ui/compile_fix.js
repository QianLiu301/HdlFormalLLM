// Browser check of the compile-error repair buttons.
//   1. Step 1: the page's own generation with the mock model returns a register
//      file that does not compile -> "Fix the compile error with MOCK" -> the
//      fixed design compiles (real endpoint, mock model).
//   2. Step 4: a staged register file whose ports the testbench cannot connect
//      -> Step 4 fails to compile -> "Ask MOCK to fix the design" -> new design,
//      new testbench, simulation, Problems card.
// Needs iverilog and the app with the mock model:
//   DATABASE_URL=... python tests/ui/mock_server.py compile &
//   python tests/ui/stage_run.py regfile > <dir>/state.json
//   NODE_PATH=$(npm root -g) node tests/ui/compile_fix.js <dir>
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');
(async () => {
  const out = process.argv[2] || '.';
  const state = JSON.parse(fs.readFileSync(`${out}/state.json`, 'utf-8'));
  // the staged design with other port names (what a model sometimes writes)
  const root = path.resolve(__dirname, '../..');
  const staged = fs.readFileSync(`${root}/output/dut/gemini/${state.dut.filename}`, 'utf-8');
  fs.mkdirSync(`${root}/output/dut/mock`, { recursive: true });
  fs.writeFileSync(`${root}/output/dut/mock/xdb_ui_rf_ports.v`,
                   staged.replace(/\bwen\b/g, 'we_n').replace(/\brst_n\b/g, 'reset_n'), 'utf-8');

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [], dialogs = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('dialog', d => { dialogs.push(d.message()); d.accept(); });
  await page.goto('http://localhost:5000/', { waitUntil: 'networkidle' });
  await page.evaluate(() => {
    document.getElementById('main-app').style.display = 'block';   // as after entering the access password
    document.querySelectorAll('.step-section').forEach(s => s.classList.remove('collapsed'));
    const sel = document.getElementById('hw-llm');
    if (![...sel.options].some(o => o.value === 'mock')) sel.add(new Option('Mock', 'mock'));
    sel.value = 'mock';
    document.getElementById('hw-module').value = 'regfile';
    document.getElementById('hw-module').dispatchEvent(new Event('change'));
    document.getElementById('hw-stream').checked = false;
    currentWorkflowMode = 'implementation';
  });

  // ---- 1. Step 1
  await page.evaluate(() => generateHardware());
  await page.waitForSelector('#hw-compile-status.compile-bad', { timeout: 60000 });
  const bad = (await page.textContent('#hw-compile-status')).replace(/\s+/g, ' ').trim();
  console.log('== Step 1 after generation\n' + bad);
  await page.locator('#hw-result').screenshot({ path: `${out}/compile_1_broken.png` });
  const firstFile = await page.textContent('#hw-filename');
  await page.getByRole('button', { name: 'Fix the compile error with MOCK' }).click();
  await page.waitForSelector('#hw-compile-status.compile-ok', { timeout: 60000 });
  const fixedStatus = (await page.textContent('#hw-compile-status')).trim();
  const fixedFile = await page.textContent('#hw-filename');
  console.log('\n== Step 1 after the fix\n' + fixedStatus + ' | file: ' + firstFile + ' -> ' + fixedFile
              + ' | ' + await page.textContent('#hw-info'));
  await page.locator('#hw-result').screenshot({ path: `${out}/compile_2_fixed.png` });
  const step1ok = /does not compile/.test(bad) && /Compiles standalone/.test(fixedStatus) && fixedFile !== firstFile;

  // ---- 2. Step 4 with a design the testbench cannot connect
  await page.evaluate(async (st) => {
    currentRunId = 'web_ui_compile_' + Date.now();
    currentRequirements = st.requirements;
    currentDUT = { ...st.dut, filename: 'xdb_ui_rf_ports.v', llm: 'mock' };
    currentBDD = st.bdd;
    await generateTestbench();
    await runSimulation();
  }, state);
  await page.waitForSelector('#sim-fix', { state: 'visible', timeout: 60000 });
  console.log('\n== Step 4 error\n' + (await page.textContent('#sim-error')).replace(/\s+/g, ' ').trim().slice(0, 300));
  console.log((await page.textContent('#sim-fix')).replace(/\s+/g, ' ').trim());
  await page.locator('#sim-fix').screenshot({ path: `${out}/compile_3_sim_error.png` });
  await page.getByRole('button', { name: 'Ask MOCK to fix the design' }).click();
  await page.waitForFunction(() => repairNotice && repairNotice.html.includes('Design fixed by MOCK')
                                   && explainState && document.getElementById('sim-problems').style.display === 'block',
                             null, { timeout: 90000 });
  const note = await page.$eval('#pb-note', n => n.innerText.replace(/\s+/g, ' ').trim());
  const items = await page.$$eval('.pb-item', es => es.map(e =>
    `[${e.querySelector('.pb-tag').textContent}] ${e.querySelector('.pb-title').textContent}`));
  console.log('\n== Step 4 after the fix\nnote: ' + note);
  items.forEach(t => console.log('  ' + t));
  console.log('sim-fix hidden:', await page.$eval('#sim-fix', n => n.style.display === 'none'),
              '| Step 1 shows:', await page.textContent('#hw-filename'));
  await page.locator('#sim-problems').screenshot({ path: `${out}/compile_4_after.png` });
  const step4ok = /Design fixed by MOCK/.test(note) && items.length > 0;

  await page.setViewportSize({ width: 390, height: 800 });
  console.log('\nhorizontal overflow at 390px:',
              await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1));
  console.log('confirm dialogs:', dialogs.length);
  dialogs.forEach(d => console.log('  ' + d.replace(/\n+/g, ' / ')));
  console.log('page errors:', errors.length ? errors : 'none');
  const ok = step1ok && step4ok && dialogs.length === 2 && !errors.length;
  console.log(ok ? 'RESULT: OK' : 'RESULT: FAILED');
  await browser.close();
  process.exit(ok ? 0 : 1);
})().catch(e => { console.error(e); process.exit(1); });
