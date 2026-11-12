import './styles/main.css';
import './styles/components.css';
import './styles/responsive.css';

import { RecognitionClient } from './ai/recognition-client.js';
import { LABEL_BASE_URL, OCR } from './config.js';
import { loadImage, releaseImage } from './image/image-loader.js';
import { Progress } from './ui/progress.js';
import { renderOcr } from './ui/ocr-panel.js';
import { describe, renderError, renderResult } from './ui/results.js';
import { initTheme } from './ui/theme.js';
import { initUpload } from './ui/upload.js';
import { formatBytes } from './utils/formatting.js';

const $ = (id) => document.getElementById(id);
const els = {
  dropzone: $('dropzone'),
  fileInput: $('file-input'),
  cameraInput: $('camera-input'),
  browseBtn: $('browse-btn'),
  cameraBtn: $('camera-btn'),
  uploadSection: $('upload-section'),
  uploadError: $('upload-error'),
  workspace: $('workspace'),
  previewImg: $('preview-img'),
  boxLayer: $('box-layer'),
  boxHint: $('box-hint'),
  anotherBtn: $('another-btn'),
  removeBtn: $('remove-btn'),
  imageInfo: $('image-info'),
  resultPanel: $('result-panel'),
  results: $('results'),
  envInfo: $('env-info'),
  ocrSection: $('ocr-section'),
  ocrBody: $('ocr-body'),
  ocrToggle: $('ocr-toggle'),
};

initTheme($('theme-toggle'));
const progress = new Progress({ root: $('progress'), label: $('progress-label'), bar: $('progress-bar'), detail: $('progress-detail') });
const params = new URLSearchParams(location.search);
// Debug/support switches: ?backend=wasm forces the CPU (WebAssembly) backend, ?worker=off runs inference on the main thread.
const client = new RecognitionClient({ disableWorker: params.get('worker') === 'off' });
const forceWasm = params.get('backend') === 'wasm';
if (forceWasm) client.configure({ forceWasm: true });

/** @type {{bitmap, previewUrl, info}|null} */
let current = null;
let lastResult = null;
let runToken = 0;
let idleTimer = null;
const IDLE_RELEASE_MS = 10 * 60 * 1000;

// ---------- helpers ----------
function showUploadError(message) {
  els.uploadError.textContent = message;
  els.uploadError.hidden = !message;
}

function renderInfo(info, result) {
  const rows = [
    ['Dimensions', `${info.width} × ${info.height} px`],
    ['Format', info.format],
    ['File size', formatBytes(info.bytes)],
  ];
  if (result?.timings) rows.push(['Processing time', `${(result.timings.total / 1000).toFixed(2)} s`]);
  if (result?.models?.length) rows.push(['AI models used', result.models.map((m) => m.title.split(' ')[0] + (m.key === 'detector' ? ' detector' : '')).join(', ')]);
  els.imageInfo.replaceChildren(
    ...rows.map(([k, v]) => {
      const div = document.createElement('div');
      const dt = document.createElement('dt');
      dt.textContent = k;
      const dd = document.createElement('dd');
      dd.textContent = v;
      div.append(dt, dd);
      return div;
    }),
  );
}

function drawBoxes(result) {
  els.boxLayer.replaceChildren();
  const { width, height } = result.image;
  result.detections.forEach((d, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'vbox' + (i === result.selected ? ' is-selected' : '');
    b.style.left = `${(d.x1 / width) * 100}%`;
    b.style.top = `${(d.y1 / height) * 100}%`;
    b.style.width = `${((d.x2 - d.x1) / width) * 100}%`;
    b.style.height = `${((d.y2 - d.y1) / height) * 100}%`;
    b.setAttribute('aria-label', `Identify ${d.label} ${i + 1} (detected with ${Math.round(d.score * 100)}% certainty)`);
    b.setAttribute('aria-pressed', String(i === result.selected));
    const tag = document.createElement('span');
    tag.className = 'vbox-tag';
    tag.textContent = `${d.label} ${i + 1}`;
    b.append(tag);
    b.addEventListener('click', () => run({ box: d, detections: result.detections }));
    els.boxLayer.append(b);
  });
  els.boxHint.hidden = result.detections.length < 2;
}

function scheduleIdleRelease() {
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => client.releaseModels().catch(() => {}), IDLE_RELEASE_MS);
}

// ---------- OCR (badge text) ----------
function ocrEnabled() {
  try {
    return localStorage.getItem(OCR.storageKey) !== '0';
  } catch {
    return true;
  }
}
els.ocrToggle.checked = ocrEnabled();
els.ocrToggle.addEventListener('change', () => {
  try {
    localStorage.setItem(OCR.storageKey, els.ocrToggle.checked ? '1' : '0');
  } catch {
    /* storage blocked: the choice just won't persist */
  }
  if (els.ocrToggle.checked && lastResult) runOcr(lastResult, runToken);
  else els.ocrSection.hidden = true;
});

/** Read text on the selected vehicle and match it against the local JSON vehicle database. Never blocks classification. */
async function runOcr(result, token) {
  els.ocrSection.hidden = true;
  if (!ocrEnabled() || result.status !== 'ok' || !current) return;
  const alive = () => token === runToken && current;
  renderOcr(els.ocrBody, { phase: 'loading', label: 'Preparing text recognition…' });
  els.ocrSection.hidden = false;
  try {
    const [{ readText }, { loadVehicleDatabase }, { compareEvidence }] = await Promise.all([
      import('./ai/ocr-engine.js'),
      import('./ai/vehicle-database.js'),
      import('./ai/ocr-evidence.js'),
    ]);
    const db = await loadVehicleDatabase();
    const { words } = await readText(current.bitmap, result.box, (m) => {
      if (alive()) renderOcr(els.ocrBody, { phase: 'loading', label: `Reading text: ${m.status}…`, progress: typeof m.progress === 'number' ? m.progress : null });
    });
    if (!alive()) return;
    const match = db.match(words);
    const title = result.primary ? describe(result.primary).title : null;
    renderOcr(els.ocrBody, { phase: 'done', words, match, evidence: compareEvidence(db, match, title) });
  } catch (err) {
    if (!alive()) return;
    console.error('[CarVision] OCR failed', err);
    renderOcr(els.ocrBody, { phase: 'error', message: 'Text recognition is not available right now (' + (err?.message || 'unknown error') + '). The image recognition result above is unaffected.' });
  }
}

// ---------- recognition ----------
async function run(options = {}) {
  if (!current) return;
  const token = ++runToken;
  clearTimeout(idleTimer);
  els.resultPanel.setAttribute('aria-busy', 'true');
  els.results.hidden = true;
  els.ocrSection.hidden = true;
  progress.show('Preparing…');
  try {
    const result = await client.recognize(options, (e) => token === runToken && progress.handle(e));
    if (token !== runToken) return; // a newer request superseded this one
    lastResult = result;
    progress.hide();
    renderResult(els.results, result, { onWholeImage: () => run({ wholeImage: true, detections: result.detections }) });
    drawBoxes(result);
    renderInfo(current.info, result);
    runOcr(result, token); // not awaited: OCR never delays the classification result
  } catch (err) {
    if (token !== runToken) return;
    console.error('[CarVision] recognition failed', err);
    progress.hide();
    renderError(els.results, err?.message || 'Something went wrong while analysing the image.', () => run(options));
  } finally {
    if (token === runToken) {
      els.resultPanel.setAttribute('aria-busy', 'false');
      scheduleIdleRelease();
    }
  }
}

// ---------- image lifecycle ----------
function clearImage() {
  runToken++; // abandon any in-flight run's UI updates
  releaseImage(current);
  current = null;
  lastResult = null;
  els.previewImg.removeAttribute('src');
  els.boxLayer.replaceChildren();
  els.results.replaceChildren();
  els.ocrSection.hidden = true;
  progress.hide();
  els.workspace.hidden = true;
  els.uploadSection.hidden = false;
  client.clearImage().catch(() => {});
}

async function handleFile(file) {
  showUploadError('');
  let loaded;
  try {
    loaded = await loadImage(file);
  } catch (err) {
    showUploadError(err.message);
    return;
  }
  clearImage();
  current = loaded;
  els.previewImg.src = loaded.previewUrl;
  els.previewImg.alt = `Uploaded photo: ${loaded.info.name}`;
  els.uploadSection.hidden = true;
  els.workspace.hidden = false;
  renderInfo(loaded.info, null);
  els.workspace.scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    // Ownership of the bitmap moves to the worker; the preview uses the object URL instead.
    const copy = await createImageBitmap(loaded.bitmap);
    await client.setImage(copy);
  } catch (err) {
    renderError(els.results, err.message);
    return;
  }
  run();
}

els.removeBtn.addEventListener('click', clearImage);
els.anotherBtn.addEventListener('click', () => {
  els.fileInput.value = '';
  els.fileInput.click();
});
initUpload({ ...els, onFile: handleFile });

// Offline support (production only): cache the app shell so the page opens without a network after one visit.
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`, { scope: import.meta.env.BASE_URL }).catch((e) => console.warn('[CarVision] service worker not registered', e));
  });
}

if (import.meta.env.DEV) {
  // Dev-only inspection hook used by manual/automated browser tests.
  Object.defineProperty(window, '__cv', { get: () => ({ result: lastResult, client }) });
}

window.addEventListener('pagehide', () => {
  client.releaseModels().catch(() => {});
  import('./ai/ocr-engine.js').then((m) => m.releaseOcr()).catch(() => {});
});

// ---------- static content ----------
async function fillChips() {
  const intl = ['Toyota Camry', 'Honda Accord', 'Hyundai Sonata', 'Volkswagen Golf', 'Porsche Panamera', 'Tesla Model S', 'Ford Mustang', 'BMW 3 Series Sedan', 'Mercedes-Benz S-Class', 'Audi A5', 'Chevrolet Camaro', 'Nissan Leaf'];
  const fill = (ul, items) => ul.replaceChildren(...items.map((t) => Object.assign(document.createElement('li'), { textContent: t })));
  try {
    const [ir, it] = await Promise.all([
      fetch(LABEL_BASE_URL + 'iranian-labels.json').then((r) => r.json()),
      fetch(LABEL_BASE_URL + 'international-labels.json').then((r) => r.json()),
    ]);
    fill($('chips-iran'), ir.filter((l) => l.verified).map((l) => l.title));
    fill(
      $('chips-intl'),
      intl.map((q) => it.find((l) => l.raw.startsWith(q))).filter(Boolean).map((l) => `${l.manufacturer} ${l.model}`),
    );
  } catch {
    /* decorative only */
  }
}
fillChips();

(window.requestIdleCallback || ((f) => setTimeout(f, 800)))(async () => {
  try {
    const caps = await client.capabilities();
    els.envInfo.textContent = `Your browser: ${forceWasm ? 'WebAssembly (CPU) backend forced with ?backend=wasm' : caps.webgpu ? 'WebGPU available (used for the small models)' : 'WebGPU not available – using the WebAssembly (CPU) backend'}${client.mode === 'main-thread' ? ' · running on the main thread' : ''}.`;
  } catch {
    /* informational only */
  }
});
