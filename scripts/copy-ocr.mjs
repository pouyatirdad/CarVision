// Copies the self-hosted OCR runtime into public/ocr/ (generated, git-ignored) so nothing is loaded from a CDN:
//   - tesseract.js worker script
//   - tesseract.js-core LSTM WASM builds (the browser picks relaxed-SIMD / SIMD / plain at run time; one is downloaded)
//   - English + Persian traineddata (tessdata "best_int", LSTM only)
import fs from 'node:fs';
import path from 'node:path';

const nm = path.resolve('node_modules');
const out = path.resolve('public/ocr');
fs.mkdirSync(path.join(out, 'lang'), { recursive: true });

const copy = (from, to) => fs.copyFileSync(from, path.join(out, to));
copy(path.join(nm, 'tesseract.js/dist/worker.min.js'), 'worker.min.js');
for (const f of ['tesseract-core-relaxedsimd-lstm.wasm.js', 'tesseract-core-simd-lstm.wasm.js', 'tesseract-core-lstm.wasm.js']) {
  copy(path.join(nm, 'tesseract.js-core', f), f);
}
for (const l of ['eng', 'fas']) copy(path.join(nm, '@tesseract.js-data', l, '4.0.0_best_int', `${l}.traineddata.gz`), `lang/${l}.traineddata.gz`);
console.log('copied OCR runtime to public/ocr');
