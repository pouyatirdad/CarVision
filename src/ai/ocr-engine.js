// Text recognition on the vehicle crop, using self-hosted tesseract.js (worker, WASM core and language data are
// all served from this site - see scripts/copy-ocr.mjs). tesseract.js runs in its own Web Worker, so the main
// thread stays responsive. Independent of the image classifiers and of the vehicle database: it only returns words.
//
// Rear badges are small, so several attempts are made (see ai/ocr-passes.js): original, contrast-stretched and
// sharpened crops, upscaled rear badge zones, and upscaled re-reads of words that were seen with low confidence.

import { OCR } from '../config.js';
import { grayToRGBA, sharpen, stretchContrast, toGray } from '../utils/image-ops.js';
import { mergePassWords, planPasses, refinePasses } from './ocr-passes.js';

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

/** Draw `region` of the bitmap scaled by `scale` and apply the pass variant. */
function renderPass(bitmap, { region, scale, variant }) {
  const sw = Math.max(1, region.x2 - region.x1);
  const sh = Math.max(1, region.y2 - region.y1);
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(sw * scale));
  canvas.height = Math.max(1, Math.round(sh * scale));
  const ctx = canvas.getContext('2d', { willReadFrequently: variant !== 'original' });
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, region.x1, region.y1, sw, sh, 0, 0, canvas.width, canvas.height);
  if (variant === 'original') return canvas;
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  let gray = stretchContrast(toGray(img.data));
  if (variant === 'sharpen') gray = sharpen(gray, canvas.width, canvas.height, 1.2);
  img.data.set(grayToRGBA(gray));
  ctx.putImageData(img, 0, 0);
  return canvas;
}

const cleanText = (t) => [...String(t ?? '').replace(/[^\p{L}\p{N}.]/gu, '')].length;

async function recognisePass(worker, bitmap, pass) {
  const canvas = renderPass(bitmap, pass);
  // Per-word data is only produced when "blocks" output is requested (tesseract.js >= 6); words are nested.
  const { data } = await worker.recognize(canvas, {}, { text: true, blocks: true });
  const flat = data.words ?? (data.blocks ?? []).flatMap((b) => b.paragraphs.flatMap((p) => p.lines.flatMap((l) => l.words)));
  return flat
    .map((w) => ({ text: String(w.text ?? '').trim(), confidence: (w.confidence ?? 0) / 100, bbox: w.bbox, pass }))
    // single letters are almost always noise, but a single DIGIT is part of many model names (TIGGO 8, MAZDA 3)
    .filter((w) => cleanText(w.text) >= OCR.minWordLength || /^\d$/.test(w.text));
}

/**
 * @param {ImageBitmap} bitmap decoded photo
 * @param {{x1,y1,x2,y2}} box region to read (the selected vehicle)
 * @param {(m:{status:string, progress:number|null})=>void} [onProgress] progress (pass n of m)
 * @param {{shouldStop?: (words:object[]) => boolean, maxPasses?: number}} [options]
 *   shouldStop: called after each pass with the merged words so far; return true to skip the remaining passes
 *   (e.g. the database already identified a vehicle from a full alias).
 * @returns {Promise<{words: {text:string, confidence:number, passes:string[]}[], passes: {name:string, ms:number, words:number}[]}>}
 */
export async function readText(bitmap, box, onProgress, { shouldStop = null, maxPasses = OCR.maxPasses } = {}) {
  const report = (status, progress) => onProgress?.({ status, progress });
  progressListener = (m) => (m.status && !/recogniz/i.test(m.status) ? report(m.status, typeof m.progress === 'number' ? m.progress : null) : null);
  try {
    const worker = await getWorker();
    progressListener = null;
    const image = { width: bitmap.width, height: bitmap.height };
    const queue = planPasses(box, image, { maxSide: OCR.maxSide, maxUpscale: OCR.maxUpscale, zoneUpscale: OCR.zoneUpscale });
    const results = [];
    const log = [];
    const low = [];
    let refined = false;
    let merged = [];
    for (let i = 0; i < queue.length && i < maxPasses; i++) {
      const pass = queue[i];
      report(`reading badges (attempt ${i + 1} of ${Math.min(maxPasses, queue.length)})`, i / Math.min(maxPasses, queue.length));
      const t = performance.now();
      const words = await recognisePass(worker, bitmap, pass);
      log.push({ name: pass.name, ms: Math.round(performance.now() - t), words: words.filter((w) => w.confidence >= OCR.minWordConfidence).length });
      results.push({ pass: pass.name, words: words.filter((w) => w.confidence >= OCR.minWordConfidence * 0.75) });
      for (const w of words) if (w.confidence >= OCR.refineMinConfidence && w.confidence < OCR.minWordConfidence && cleanText(w.text) >= 2) low.push(w);
      merged = mergePassWords(results, { minConfidence: OCR.minWordConfidence, minLength: OCR.minWordLength });
      if (shouldStop?.(merged)) break;
      if (!refined && i === queue.length - 1 && low.length) {
        refined = true;
        queue.push(...refinePasses(low, image, { maxRegions: OCR.maxRefineRegions }));
      }
    }
    return { words: merged.map(({ text, confidence, passes }) => ({ text, confidence, passes })), passes: log };
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
