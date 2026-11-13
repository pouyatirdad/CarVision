# CarVision – rear-view car recognition in your browser

Identify **Iranian-market and imported cars from a photo of their rear**. Everything – vehicle detection, the
rear-view check, rear-view recognition, badge OCR and database matching – runs **locally in the browser** with ONNX
Runtime Web and tesseract.js. There is no backend, no API, no account and no upload.

| Home | Result |
|---|---|
| ![Home](docs/screenshots/home.jpg) | ![Result](docs/screenshots/result.jpg) |

<sub>Result screenshot (v1 UI): photo "Tara" by Saeed Sajjadi, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), via Wikimedia Commons.</sub>

> **Read this first.** Users upload **one rear photo** (straight or slightly angled). The recognition is real and
> measured – see [docs/REAR-VIEW.md](docs/REAR-VIEW.md) – but it has clear limits: it knows 44 vehicles visually
> (38 trained on rear photos), newer Chinese models are **not** reliably recognised from the rear yet, and about half
> of the photos of a vehicle it does not know still get a confident wrong name. The v1 multi-view research is in
> [docs/RESEARCH.md](docs/RESEARCH.md).

## Features

* "Upload a photo of the rear of your car." – drag & drop, file picker, camera; JPG / PNG / WEBP up to 15 MB; real
  validation (magic bytes, pixel-count guard, corrupted files); EXIF orientation; an example of correct framing.
* **Rear-view check**: front, side and interior photos get "Please upload a photograph of the rear of the vehicle."
  (with an "analyse anyway" override for borderline angles).
* **Rear-view recognition** from tail lights, trunk, bumper and proportions – works without readable badges.
* **Badge OCR** (English + Persian) with several attempts per photo: original, contrast-stretched, sharpened, three
  enlarged rear badge zones, and enlarged re-reads of low-confidence words.
* **Local JSON vehicle database** (90 vehicles, sourced) – the central list of names, aliases, badge texts and
  which vehicles the image model actually covers (`rearRecognition`).
* **Prediction fusion** of image and text evidence by explicit rules (no averaging of unrelated scores), with
  identified / likely / uncertain verdicts and the reason ("rear appearance and badge text agree", …).
* Optional second opinion from the v1 general-view classifiers (downloaded only on request).
* Real progress, dark / light theme, responsive, keyboard accessible, works **offline after the first visit**.

## Technology stack

HTML5 · CSS3 · vanilla JavaScript (ES modules) · [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 1.31
(WASM; WebGPU for the detector) · [tesseract.js](https://github.com/naptha/tesseract.js) (self-hosted) · Canvas /
OffscreenCanvas · Web Workers · Vite (build only) · GitHub Actions + GitHub Pages. No framework, no server, no external
AI/OCR API. Python is used **only offline** (`scripts/`) to convert models and train the rear-view head.

## Models (what is actually implemented)

| Role | Model | Source / licence | Size | Backend |
|---|---|---|---|---|
| Vehicle detector (car / bus / truck) | YOLOX-tiny, COCO | [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) 0.1.1rc0 – Apache-2.0 | 20.2 MB | WebGPU → WASM |
| **Rear-view encoder** | CLIP ViT-B/32 image tower, weight-only 8-bit (`MatMulNBits`) | [OpenCLIP](https://github.com/mlfoundations/open_clip) `datacomp_xl_s13b_b90k` ([HF](https://huggingface.co/laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K)) – MIT | 98.2 MB | WASM¹ |
| **Rear-view head** + zero-shot rear check | linear head (46 classes → 44 vehicles) initialised from CLIP text prompts, trained on rear crops; k-means prototypes for the unknown-vehicle check | trained for CarVision (`scripts/train_rear_head.py`) – see licensing note below | 1 MB JSON | JS |
| Badge OCR | tesseract.js 7, `eng` + `fas` | Apache-2.0 | ≈7 MB | WASM (own worker) |
| *Optional second opinion* | v1 Stanford-Cars ViT-B/16 int8 (196 classes) + Iranian ResNet18 (27 classes) | Apache-2.0 / MIT | 87 + 45 MB | WASM / WebGPU |

¹ ORT-Web's WebGPU `MatMulNBits` kernel only supports 2/4-bit weights; 4-bit (and dynamic int8) were measured to
damage this encoder (cosine to fp32 0.96–0.97 vs 0.9998 for 8-bit), so it runs on WASM.

**Measured** ([docs/REAR-VIEW.md §5](docs/REAR-VIEW.md)): on 3,005 held-out SIVD rear photos (29 Iranian classes)
**87.9 % top-1 / 97.3 % top-3**; on independent Wikimedia Commons rear photos (cross-fitted, n=429)
**83.2 % / 94.9 %**. At the "identified" threshold the app answers 88 % of SIVD photos with 94 % precision. The v1
pipeline on the same kind of photos: 25.5 % correct and 27.2 % confidently wrong (SIVD), 2.7 % correct / 54.1 % wrong
(Commons). Rear-view check: 97.9 % of rear photos accepted, 0 / 141 front photos accepted.

**Licensing note:** the rear-view head is trained on the SIVD dataset (scraped Iranian ad photos, **no licence
stated**) plus freely licensed Wikimedia Commons photos. No image is redistributed, but check
[docs/REAR-VIEW.md §6](docs/REAR-VIEW.md) before any commercial use.

### How a result is produced

1. Detect the main vehicle (YOLOX-tiny) and crop it (padded to a square so both tail lights stay visible).
2. CLIP embedding → zero-shot **rear-view check** (`REAR.viewpointMin`) → "please upload a rear photo" if it fails.
3. **Rear-view head**: calibrated probability per vehicle + an unknown-vehicle check (similarity to the class's
   training prototypes). Shown immediately.
4. **OCR** reads the badges (multi-pass), the text is matched against `public/data/vehicles.json` (a vehicle is only
   named from text with model evidence – full alias or brand + model; brand-only or engine badges never name a model).
5. **Fusion** (`src/ai/fusion.js`): text naming a vehicle in the visual top-3 → identified (high); confident visual
   contradicted by identified text → uncertain; identified text alone → identified from text; confident visual →
   identified (high if a badge brand/model supports it, uncertain if the badge shows another brand); medium visual →
   "likely"; otherwise uncertain with candidates.

Thresholds live in [`src/config.js`](src/config.js) (`REAR`, `OCR`) and are backed by the measurements above.

### Coverage: database vs. image model

The database lists **90** vehicles (Iranian, Chinese, Korean, Japanese, European models sold in Iran, each with
`sources`). The image model covers **44** of them – the `rearRecognition` field says which, and whether each is
`trained` (≥ 10 rear photos) or has few/no photos. The other 46 are identified only from readable badge text. The page
shows this split under "What it can recognise". Edit vehicles without code changes: [public/data/README.md](public/data/README.md).

## Run locally

Requires Node.js 20+.

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # unit tests (fusion, rear head, OCR passes, database, router, post-processing, validation)
npm run build      # -> dist/
npm run preview
```

Debug switches (query string): `?backend=wasm` forces the CPU/WebAssembly back-end, `?worker=off` runs inference on
the main thread (the automatic fallback for browsers without module workers / OffscreenCanvas).

## Model files and conversion

The `.onnx` files are committed in `public/models/` (largest is the 98.2 MB rear-view encoder – under GitHub's 100 MiB
limit, expect a size warning above 50 MB; Git LFS is **not** needed and would break Pages serving). Python is needed
only to rebuild them:

```bash
# Rear-view encoder (OpenCLIP ViT-B/32 DataComp, MIT) -> 8-bit MatMulNBits ONNX.
# Copy to public/models/ under a NEW versioned file name and update MODELS.rear.file (the Cache API is keyed by URL).
python scripts/convert_rear_encoder.py ./out ViT-B-32 datacomp_xl_s13b_b90k

# Rear-view head: embed rear training/test images (zip of <split>/<class>/<file>), then train + evaluate.
python scripts/embed_images.py train.zip train.npz public/models/vehicle-detector.onnx ViT-B-32 datacomp_xl_s13b_b90k --pad
python scripts/train_rear_head.py scripts/data/rear-head-config.json ./out   # -> rear-head.json + report.json
cp out/rear-head.json public/labels/ && node scripts/sync-rear-coverage.mjs out/report.json

# Optional: freely licensed rear/front test photos from Wikimedia Commons (manifest keeps licence + author per file)
node scripts/collect-commons.mjs scripts/data/commons-classes.json ./commons

# v1 models (optional second opinion)
python scripts/convert_iranian.py car_name_model.pth public/models/iranian-cars.onnx
python scripts/convert_international.py ./stanford-car-vit-patch16 ./out
# Detector: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx -> public/models/vehicle-detector.onnx
npm run labels
```

Python deps: `torch torchvision open_clip_torch onnx onnxruntime pillow numpy` (+ `transformers` for the v1 ViT).
A CPU is enough: embedding runs at ≈6 images/s on the development laptop, the head trains in seconds.

To host the models elsewhere (CORS enabled), build with `VITE_MODEL_BASE_URL=https://your.host/path/`.
No secrets or API keys exist in the frontend.

## Deploy to GitHub Pages

1. Create a repository (e.g. `carvision`), push this project to `main`.
2. *Settings → Pages → Build and deployment → Source: **GitHub Actions***.
3. The workflow [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) runs tests, builds with
   `BASE_PATH=/<repository-name>/` and deploys `dist/`. The site appears at `https://<user>.github.io/<repo>/`.

Nothing assumes the site is at the domain root: `vite.config.js` reads `BASE_PATH`, and all model/label/worker/WASM
URLs are derived from the base path. For a local check of the sub-path build:

```bash
BASE_PATH=/carvision/ npm run build && npx vite preview --base /carvision/
```

(On Windows Git Bash prefix with `MSYS_NO_PATHCONV=1`, otherwise it rewrites `/carvision/` into a Windows path.)

## Browser compatibility

Needs ES modules, WebAssembly, Canvas 2D. WebGPU is optional (only the detector uses it). Chrome / Edge – tested;
Firefox and Safari should work via WASM but are **untested**. Web Workers with OffscreenCanvas preferred, automatic
main-thread fallback otherwise. See [docs/TESTING.md](docs/TESTING.md) for what was **actually** tested.

## Performance

First use downloads ≈120 MB (detector 20 MB + rear-view encoder 98 MB + head 1 MB) plus the WASM runtime, and ≈7 MB
of OCR data when badge reading is on; afterwards everything comes from cache. Measured on the development laptop
(Chromium, WASM): rear-view encoder ≈0.6–0.7 s per photo after warm-up, detection ≈0.1 s; OCR adds up to 6–10 reading
attempts in its own worker after the result is already shown. Model sessions are released after 10 minutes idle and on
page hide.

## Privacy

Images are decoded in your browser, passed to a Web Worker and never sent anywhere – the only network requests are
for the app's own static files (page, WASM runtime, models, label and data files, OCR data). Nothing is stored
permanently: the only persisted data are the model files and app shell (browser caches), the theme and the OCR switch.

## Known limitations

* **Coverage is limited to 44 vehicles visually** (38 trained on rear photos). Everything else in the 90-vehicle
  database is identified only from readable badge text.
* **Newer Chinese models are not reliably recognised from the rear** (Fownix FX, Tiggo 7/8 Pro, KMC J7, Lamari, …):
  there were only 0–5 licence-clean rear photos per model. Collecting own rear photos is the way to fix this.
* **Unknown vehicles:** about half of the photos of a vehicle the model was not trained on still get a confident wrong
  name (measured: 46.8 % rejected). A readable badge that names another vehicle turns the result into "uncertain".
* Generations/facelifts are not distinguished, and the model year is never determined.
* Test accuracy on SIVD is in-domain (same ad source as training); the independent Commons estimate is 83 % top-1.
* The rear-view check lets ≈10 % of side views (mostly pickups) through and rejects ≈2 % of rear photos (mostly very
  oblique ones) – the user can override.
* OCR: small or chrome badges are often unreadable; Persian badge text is not read reliably.
* The detector (YOLOX-tiny) sometimes misses a car or picks a bystander car (click another box).

## Licence & attribution

Application code: MIT ([LICENSE](LICENSE)). Model weights keep their own licences, listed above (Apache-2.0, MIT).
ONNX Runtime Web: MIT. tesseract.js: Apache-2.0. **The rear-view head is trained on the SIVD dataset (licence not
stated by its authors) and on Wikimedia Commons photos (CC BY / BY-SA / PD)** – no dataset images are redistributed;
see [docs/REAR-VIEW.md §6](docs/REAR-VIEW.md) before commercial use. The Stanford Cars dataset
([Krause et al., 2013](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)) is used by the optional v1 model.
