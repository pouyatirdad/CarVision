# Test log (what was actually run)

Environment: Windows 11, Chromium 156 embedded in the Claude desktop app (WebGPU available), Vite dev server and
`vite preview` serving a build with base path `/carvision/`. Photos: Wikimedia Commons (CC-licensed) and random images
from the official Stanford Cars *test* split. **These are sanity checks on a handful of images – not an accuracy
benchmark.** The only accuracy number I stand behind is the ViT's 88 % top-1 on 400 Stanford images
(`scripts/eval_international.py`).

Automated: `npm test` – 12 unit tests (router decisions incl. "raw Iranian confidence must not override", YOLOX
decoding, NMS, softmax/top-k, file validation) – all pass.

## Scenario results

| # | Scenario | Result |
|---|---|---|
| 1 | Clear Iranian photos (8 Commons photos) | Tara >99 % ✔, Saipa Quick 96 % ✔, Peugeot 207i >99 % ✔, Saipa Shahin >99 % ✔ (accepted and correct). Dena: top-1 correct at 79 % → "uncertain" (below 0.80 threshold), best guess listed. Tiba: top-1 correct at 54 % → "uncertain". **Samand → "Audi 100 Sedan 1994" 87 % ✘ (confident error, see README limitations).** Renault L90: detector found no vehicle → "No car found". |
| 2 | Clear international photos (7 random Stanford test images run in the browser) | Audi R8 98 % ✔, Infiniti G Coupe IPL >99 % ✔, VW Golf >99 % ✔, Audi S6 99 % ✔, Toyota 4Runner >99 % ✔; Audi S4 (first run with the old 0.65 Iranian threshold: "Saipa Shahin 67 %" ✘ → threshold raised to 0.80, now "uncertain"); Honda Odyssey → "uncertain". |
| 3 | Blurry photo | Gaussian blur 4 px on a 4Runner: still ✔. Blur 10 px on an Audi R8: "No car found". |
| 4 | Several cars | Photo with a car + truck: two boxes, main car selected, hint shown; clicking the second box re-runs classification on it. |
| 5 | No car in the image | Cat photo and mountain photo: "No car found in this image" (with an "analyse whole image anyway" button; that button was not exercised). |
| 6 | Unsupported / invalid files | GIF → rejected; text file renamed `.png` → rejected (magic bytes); random bytes with a JPEG header → "could not be decoded"; 16 MB file → rejected with size message; the 120-megapixel guard is covered by a unit test. The error stays visible while another image is open. |
| 7 | Very large image | 6000×4000 JPEG (≈1 MB file): processed (down-scaled once on decode), result correct, 1.5–3.6 s. |
| 8 | Chrome | Chromium 156 ✔ (all of the above). |
| 9 | Firefox | **Not tested.** |
| 10 | Mobile browsers | **Not tested on a device.** 375 px viewport emulation (light theme) checked visually: layout OK. |
| 11 | WebGPU | ✔ detector and Iranian ResNet18 ran on WebGPU (ViT is pinned to WASM, see README). First recognition ≈13–15 s (local download + session creation + shader compilation), then ≈1.5–2 s. |
| 12 | WASM fallback | `?backend=wasm`: all three models on WASM, same predictions, ≈1.7–2 s per image after warm-up. Main-thread fallback (`?worker=off`) also verified. |
| 13 | Offline after caching | With the preview server **stopped**, page reload (served by the service worker) and two recognitions (models from the Cache API) worked. One hiccup: the very first reload right after stopping the server timed out starting the worker (and the main-thread fallback then also failed offline); subsequent offline reloads were fine and I could not reproduce it again. |
| 14 | Privacy | Network log during use lists only same-origin requests; `grep` finds no external URL in `src/`. |
| 15 | GitHub Pages | **Not deployed** (no repository/remote in this environment). The sub-path build (`/carvision/`) was served locally and verified instead; the workflow is written but unrun. |

## Bugs found by this testing and fixed

* ONNX Runtime's dynamic import of its WASM loader failed under Vite dev when served from `public/` → use the
  bundled runtime instead of a hand-copied one.
* Concurrent first calls created two Web Workers (an early `configure` message went to the discarded one).
* Results panel rendered nested arrays / the text "null" (DOM helper bugs).
* Upload errors were invisible while an image was already open.
* Iranian acceptance threshold too permissive for foreign cars (0.65 → 0.80, see docs/RESEARCH.md).

## OCR + vehicle database (added later)

- `npm test`: 42 tests (25 for the matcher incl. the 10 required scenarios, 5 for OCR/classifier comparison).
- Browser: synthetic badge text (Latin, 22–70 px) -> tesseract.js -> database: TOYOTA COROLLA 1.8, LAMARI NEO 1.5, PEUGEOT 206, SAIPA SHAHIN, MERCEDES C200, DENA+, TOYOTA COROLLA @22px, LAMARI NEO @22px all identified correctly (≈150–750 ms each after warm-up).
- Persian text (دنا پلاس, پژو 206, لاماری نئو, سمند), drawn with Tahoma: not recognised (only the digits of "پژو 206"), with either the combined or a Persian-only pass.
- Real photos (960 px Commons/Stanford images): essentially no usable badge words; low-confidence noise is filtered. A real dealer plate frame ("toyota.com") was read.
- Full UI path verified: photo + synthetic "IKCO TARA" strip -> classifier "Iran Khodro Tara" -> OCR panel "matches database ... agrees with the image models".

## Rear-view V2 (2026-11-13)

Measured model results (accuracy, viewpoint check, unknown-vehicle rejection, baseline comparison) are in
[REAR-VIEW.md §5](REAR-VIEW.md). This section covers the app itself.

- `npm test`: 59 tests – fusion decision table (10), rear head (5, incl. open-set prototypes), OCR pass planning/merging
  and pixel operations (7), database schema 2.0 + matcher, router, post-processing, validation. `npm run build` passes.
- **Browser (Chromium, Vite dev server, in-app browser pane):**
  - Rear Samand photo (Commons, CC BY-SA): rear check passed (0.36 ≥ 0.2), **Iran Khodro Samand 97 %**, prototype
    similarity 0.85 ≥ 0.80 → "Identified". Not part of SIVD; it *is* part of the Commons set the final head was trained
    on, so this verifies the pipeline, not accuracy.
  - Front Kia/Saipa Pride photo: rear check 2 % → "Please upload a photograph of the rear of the vehicle." The override
    button re-runs it; the head says Pride 99 % but the prototype similarity rejects it (front ≠ any rear prototype) →
    "Uncertain – may be a vehicle the model does not know". Correct behaviour for a non-rear photo.
  - Rear Toyota Corolla photo: Toyota Corolla 99 %, identified (rear appearance only).
  - Warm timings on this laptop (CPU busy with training jobs at the same time): detection ≈ 0.1 s, rear-view encoder
    ≈ 0.6–0.7 s (WASM), first use ≈ 2 s + download (98 MB, once).
  - Backends: detector WebGPU, rear encoder WASM (WebGPU MatMulNBits supports only 2/4-bit – verified error message).
- **Bugs found and fixed while testing:**
  - Overlapping runs on one ORT-Web session ("Session already started") when a new photo is uploaded while the previous
    one is still being processed → runs are now queued per session.
  - Replacing model weights under the same file name kept serving the **old bytes from the Cache API** (the browser
    gave a meaningless answer with prototype similarity 0.06) → model files are now versioned
    (`rear-encoder-dc-b32-v1.onnx`) and cache entries of files no longer in `MODELS` are deleted.
  - Training-data subset bug (stride + truncation dropped the alphabetically last classes Tiba/Xantia) → fixed before
    the reported numbers.
- **Not tested:** Firefox, Safari, real mobile devices, GitHub Pages deployment of V2 (the sub-path build is unchanged
  in mechanism), offline mode with the new model.
- **OCR on rear badges (synthetic, in the browser).** Dark badge text on a light plate drawn into a real 773 px rear
  photo at x≈62 %, y≈58 % (the rear-right badge zone), 10 cases (4 names × 10/14/20 px), matched against the database:
  **single original pass identified 4 / 10, the multi-pass reader 8 / 10** (LAMARI NEO 10/14/20 px, DENA PLUS 10/14,
  HYUNDAI TUCSON 10, TIGGO 8 PRO 10/20; missed: TIGGO 8 PRO 14 px – plate noise words got between "TIGGO" and "8" –
  and HYUNDAI TUCSON 14 px, read as "TUC!"). The TIGGO and Tucson-14 cases were re-run after the two fixes below; the
  others are from the run just before the reading-order fix. Light outlined letters directly on paint were not read at
  any size or pass. Time: ≈1.5 s for one pass, ≈14 s for all 10 passes in a background tab (the app shows the image
  result first and stops OCR early once text and image agree).
  - Fixed: single digits were dropped as noise, so "TIGGO 8", "MAZDA 3", "ARRIZO 6" could never match.
  - Fixed: merged words were sorted by confidence ("8 PRO TIGGO"), breaking multi-word names; they now keep reading
    order.
  - Real Commons rear photo (Corolla, 800 px): 10 attempts, no database name found (badge too small) – recognition
    came from the rear appearance only, as intended.
