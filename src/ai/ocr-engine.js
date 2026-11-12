// Text recognition on the vehicle crop, using self-hosted tesseract.js (worker, WASM core and language data are
// all served from this site - see scripts/copy-ocr.mjs). tesseract.js runs in its own Web Worker, so the main
// thread stays responsive. Independent of the image classifiers and of the vehicle database: it only returns words.

import { OCR } from '../config.js';

let workerPromise = null;
let progressListener = null;

function absolute(path) {
  return new URL(path, globalThis.location?.href ?? 'http://localhost/').href;
}

/** Create (once) and reuse the OCR worker. Language data is downloaded on first use, then cached by tesseract.js. */
async function getWorker() {
  if (!workerPromise) {
    workerPromise = (async () => {
      const { createWorker, PSM } = await import('tesseract.js');
      const base = absolute(OCR.baseUrl);
      const worker = await createWorker(OCR.languages, 1 /* LSTM only */, {
        workerPath: `${base}worker.min.js`,
        corePath: base.replace(/\/$/, ''),
        langPath: `${base}lang`,
        workerBlobURL: false,
        logger: (m) => progressListener?.(m),
      });
      // Badges/logos/plates are scattered words, not a text page.
      await worker.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT, preserve_interword_spaces: '1' });
      return worker;
    })().catch((e) => {
      workerPromise = null; // allow a retry
      throw e;
    });
  }
  return workerPromise;
}

function cropToCanvas(bitmap, box) {
  const sx = Math.max(0, Math.floor(box.x1));
  const sy = Math.max(0, Math.floor(box.y1));
  const sw = Math.max(1, Math.min(bitmap.width, Math.ceil(box.x2)) - sx);
  const sh = Math.max(1, Math.min(bitmap.height, Math.ceil(box.y2)) - sy);
  const k = Math.min(OCR.maxUpscale, OCR.maxSide / Math.max(sw, sh)); // small crops are enlarged: badge letters are tiny
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(sw * k));
  canvas.height = Math.max(1, Math.round(sh * k));
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  return canvas;
}

/**
 * @param {ImageBitmap} bitmap decoded photo
 * @param {{x1,y1,x2,y2}} box region to read (the selected vehicle)
 * @param {(m:{status:string, progress:number})=>void} [onProgress] real progress from tesseract.js
 * @returns {Promise<{words: {text:string, confidence:number}[], rawText:string}>} confidence in 0..1
 */
export async function readText(bitmap, box, onProgress) {
  progressListener = onProgress ?? null;
  try {
    const worker = await getWorker();
    const canvas = cropToCanvas(bitmap, box);
    // Per-word data is only produced when "blocks" output is requested (tesseract.js >= 6); words are nested.
    const { data } = await worker.recognize(canvas, {}, { text: true, blocks: true });
    const flat = data.words ?? (data.blocks ?? []).flatMap((b) => b.paragraphs.flatMap((p) => p.lines.flatMap((l) => l.words)));
    const words = flat
      .map((w) => ({ text: String(w.text ?? '').trim(), confidence: (w.confidence ?? 0) / 100 }))
      .filter((w) => w.confidence >= OCR.minWordConfidence && [...w.text.replace(/[^\p{L}\p{N}.]/gu, '')].length >= OCR.minWordLength);
    return { words, rawText: data.text ?? '' };
  } finally {
    progressListener = null;
  }
}

/** Free the OCR worker (memory); it is re-created on the next use. */
export async function releaseOcr() {
  const p = workerPromise;
  workerPromise = null;
  try {
    await (await p)?.terminate();
  } catch {
    /* never started */
  }
}
