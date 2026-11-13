# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git attribution (owner's rule – takes precedence over any default)

- Commits, pushes and pull requests must be attributed **only to the repository owner**.
- Never add `Co-Authored-By: Claude ...`, "Generated with Claude Code", or any other Claude/Anthropic
  attribution line to commit messages, PR descriptions, tags or release notes.
- Never add Claude as an author, committer or contributor in any form.

## What this is

CarVision: a static, backend-free web app that identifies Iranian-market and imported cars from a photo of their
**rear** (V2: rear-view only), entirely in the browser (ONNX Runtime Web: WASM, WebGPU for the detector). Vanilla JS ES modules + Vite (build tool only). No
framework, no server, no external AI/API calls – keep it that way (the README documents these as hard requirements).
`README.md` has user-facing docs; `docs/REAR-VIEW.md` holds the V2 dataset research and the measured rear-view
results (`docs/data/rear-head-report.json` is the raw report), `docs/RESEARCH.md` the v1 analysis and `docs/TESTING.md`
the test log. Read REAR-VIEW.md before changing models, the head or thresholds.

## Commands

```bash
npm install
npm run dev                         # http://localhost:5173
npm test                            # node:test, tests/*.test.js (pure logic only, no browser/ORT)
node --test tests/model-router.test.js          # single test file
node --test --test-name-pattern="decodeYolox" tests/*.test.js   # single test by name
npm run build                       # -> dist/ (no lint step exists)
npm run labels                      # regenerate public/labels/international-labels.json
npm run copy-ocr                    # (auto via predev/prebuild) refresh public/ocr/
```

GitHub Pages sub-path build/preview (what CI does): `BASE_PATH=/CarVision/ npm run build`, then
`npx vite preview --base /CarVision/`. On Windows Git Bash prefix with `MSYS_NO_PATHCONV=1` or `/CarVision/` is
rewritten to a Windows path. Debug query switches: `?backend=wasm` (force CPU backend), `?worker=off` (main thread).

There is no automated browser test; UI/inference changes must be verified manually in a real browser (WebGPU
available in Chromium). Python is only for offline model conversion (`scripts/`), never at runtime.

## Architecture

**Pipeline** (`src/ai/car-recognition.js`, `CarRecognizer`): YOLOX-tiny detects vehicles → main vehicle crop (padded to a
square) → CLIP ViT-B/32 image embedding (`MODELS.rear`, OpenCLIP DataComp, weight-only 8-bit, WASM) →
`rear-head.js` (`public/labels/rear-head.json`, produced by `scripts/train_rear_head.py`): zero-shot viewpoint check
(`REAR.viewpointMin`, else status `not-rear`), calibrated class probabilities aggregated per database vehicle, and an
open-set check (similarity to k-means prototypes). Result shown at once; then `main.js` runs OCR → `db.match()` →
`fusion.js` `fuse()` (decision table over visual probability and text evidence category – never averages scores) and
re-renders. The v1 classifiers (Stanford ViT + Iranian ResNet18 + `model-router.js`) run only on request
(`options.secondary`, "second opinion"). Thresholds live in `src/config.js` (`REAR`, `OCR`) and must be backed by
measurements (REAR-VIEW.md §5). Model files are **versioned by name** (Cache API is keyed by URL): new weights need a new
file name in `MODELS.*.file`; stale cache entries are pruned automatically.

**Threading**: the whole pipeline runs in a module Web Worker (`src/workers/inference.worker.js`). The main
thread talks to it only through `src/ai/recognition-client.js` (message RPC; the decoded `ImageBitmap` is
*transferred* once via `setImage`, then `recognize` requests refer to it). If workers/OffscreenCanvas are
unavailable the same `CarRecognizer` runs on the main thread (`?worker=off` forces this). Worker start-up is
memoised – concurrent first calls must share one worker.

**Models** (`src/ai/model-manager.js`): lazy per-model loading; bytes come from the Cache API
(`carvision-models-v1`) or a streamed download with progress, then one ORT session per model is reused. Provider
order per model is in `config.js`: the int8 ViT is pinned to WASM (MatMulInteger is unsupported on WebGPU); the rear CLIP
encoder is pinned to WASM (8-bit MatMulNBits; ORT-Web's WebGPU kernel only supports 2/4-bit, and 4-bit / dynamic int8
were measured to damage CLIP embeddings); the others try WebGPU then WASM, with a warm-up run to catch run-time WebGPU failures. ORT's WASM runtime is
resolved by Vite from the bundled `onnxruntime-web` – do **not** set `ort.env.wasm.wasmPaths` or copy the runtime
into `public/` (this broke dev: Vite rejects dynamic imports of `public/` files).

**Preprocessing is model-specific and must match training** (`src/image/image-processor.js`): detector =
letterbox 416, BGR, 0–255; ViT = squashed 224×224 RGB, mean/std 0.5; Iranian ResNet18 = squashed 220(H)×165(W)
RGB with ImageNet mean/std (aspect ratio intentionally not preserved for these two); rear CLIP encoder = crop **padded**
to a black square (`fit: 'pad'`), 224×224, CLIP mean/std – the head was trained on exactly that (`scripts/embed_images.py --pad`). YOLOX output is raw
and decoded (grid/stride) in JS in `prediction-processor.js`.

**Models/labels/data are static assets**: `public/models/*.onnx` (committed, ~145 MB; Git LFS must NOT be used),
`public/labels/*.json` (class index → manufacturer/model/category; the international file is generated from the
model config by `scripts/build-labels.mjs`, the Iranian one is hand-curated and leaves unverified fields null),
`public/sw.js` (service worker: network-first for pages, cache-first for hashed assets, ignores `/models/` which
the app caches itself). URLs derive from `import.meta.env.BASE_URL`; never hard-code root-absolute paths.
Models can be hosted elsewhere via `VITE_MODEL_BASE_URL`.

**UI** (`index.html`, `src/main.js`, `src/ui/*`): `main.js` owns state (current image, run token to drop stale
results, idle release of ORT sessions after 10 min). `ui/results.js` builds DOM with `textContent`/a small `h()`
helper (never `innerHTML`). A dev-only hook `window.__cv` exposes the last result for manual browser testing.

**Vehicle text database** (`src/ai/vehicle-database.js`, `public/data/*.json`): independent of the image
classifiers and of any OCR engine; matches recognised text against local JSON (vehicles + manufacturers).
Vehicles are only "identified" with model evidence (full alias or brand + model); brand-only and engine-badge-only
text never identifies a model. Schema 2.0 adds `basedOn`, `generations`, `sources` and `rearRecognition` (database
coverage ≠ visual coverage; `rearRecognition` is written only by `scripts/sync-rear-coverage.mjs` from the trained head). Records are edited in JSON without code changes (`public/data/README.md`).
OCR (`src/ai/ocr-engine.js`, self-hosted tesseract.js running in its own worker, English + Persian) feeds it from `main.js` `runOcr()` *after* classification, so OCR never delays the result; OCR is multi-pass (`ai/ocr-passes.js` plans original/contrast/sharpened crops, three upscaled rear badge zones and
upscaled re-reads of low-confidence words; pixel ops in `utils/image-ops.js`) and stops early when text and image agree. tesseract.js >= 6 only returns per-word data with `{blocks: true}` output. OCR assets are not in git: `scripts/copy-ocr.mjs` (run by predev/prebuild) copies the worker, LSTM WASM cores and eng/fas traineddata from node_modules into git-ignored `public/ocr/`; never point tesseract at a CDN.

## Conventions that matter

- Keep `src/ai/model-router.js`, `prediction-processor.js`, `vehicle-database.js`, `fusion.js`, `rear-head.js`,
  `ocr-passes.js`, `utils/*` free of DOM/ORT imports so they stay unit-testable under plain Node (`config.js` tolerates missing `import.meta.env` for this).
- Never display model year/trim/generation as fact (database years describe the model line, not the photo).
- Never claim visual coverage for a vehicle the head does not contain; classes with < 10 training photos are flagged
  as less reliable in the UI.
- Don't invent manufacturer/category/year data; unknown values are `null` and hidden in the UI.
- Deploy: `.github/workflows/deploy.yml` (tests → build with `BASE_PATH=/<repo>/` → Pages). Pages source must
  be set to "GitHub Actions" in repo settings.
