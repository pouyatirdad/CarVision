# CarVision V2 – rear-view-only recognition: research, design and measured results

Date: 2026-11-13. Everything marked **verified** was downloaded/opened/run during this work; everything marked
*(claim)* is what the source says and could not be checked here. Numbers in §5 are measured, not estimated.

## 0. Summary

* **No public rear-view vehicle classifier exists** (searched GitHub, Hugging Face, papers). Every candidate is either a
  general multi-view classifier, a viewpoint/orientation model, or a dataset without weights.
* **No licence-clean rear-view dataset covers the Iranian market.** The only large Iranian source that could be
  downloaded (SIVD) is web-scraped ad photos with **no licence and no viewpoint labels**; the rear views had to be
  extracted automatically (§2, §5.1). The big international sets with viewpoint labels (CompCars, DVM-CAR) are
  **non-commercial only**; BoxCars116k needs a paid licence for commercial use.
* Chosen architecture (§3): **CLIP ViT-B/32 image encoder (OpenCLIP, LAION-2B weights, MIT) + a small rear-view head**
  trained offline on rear crops. The same embedding gives a **zero-shot rear-view check** (no extra model) and allows
  **text-prompt ("zero-shot") classes** for vehicles that have no rear training photos – these are labelled as such in
  the UI and in the database, never presented as trained coverage.
* Database coverage (90 vehicles) ≠ visual coverage (see §5.5, `rearRecognition` in `public/data/vehicles.json`).

## 1. Requirements recap

Rear photographs only (straight or slightly angled, any distance/light, badges optional), one photo is enough,
Iranian-market vehicles 2010+, browser-only inference (ONNX Runtime Web + tesseract.js), static hosting, no external
inference/OCR API, honest coverage and no fabricated results.

## 2. Dataset research (Phase 1)

| Source | Verified? | Size / classes | Viewpoint info | Rear images | Years | Licence / use | Weights |
|---|---|---|---|---|---|---|---|
| **SIVD** – [Farbod-Siahkali/Iranian-Vehicle-Tracking-and-Recognition](https://github.com/Farbod-Siahkali/Iranian-Vehicle-Tracking-and-Recognition) | **verified**: `train.zip` (954 MB) + `test.zip` (239 MB, 7,342 images) downloaded from the linked [Google Drive](https://drive.google.com/drive/folders/1UaYEfc2DZJ11psUp9iQm0n-mcNiTZmSA) | 29 classes, 36,705 images *(paper)* | **none** – scraped ad photos, all angles | measured with the zero-shot check, see §5.1 | mixed, many pre-2010 models (Paykan, Xantia, Pride) | **no licence** in repo or on Drive; images are third-party ad photos | ResNet50 `.pth` checkpoints on the same Drive (classification over all views) |
| **Iranian vehicle detection & classification** – [amirmgh1375/…](https://github.com/amirmgh1375/iranian-vehicle-detection-and-classification) | repo verified (README + 1 notebook only) | 148,414 images, 95–115 classes *(README is inconsistent)* | none stated | unknown | unknown | dataset **only by e-mail request**; no licence | Keras CNN on Google Drive *(claim)*, not tested |
| **CompCars** – [project page](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) | page + `instruction.txt` verified (download password is published there); data **not downloaded** (23-part archive) | web-nature: 136,727 whole-car images, 163 makes, 1,716 models | **yes, per image**: F 18,431 · **R 13,513** · S 23,551 · FS 49,301 · **RS 31,150** *(paper, Table 1)*; surveillance part (44,481) is front-only | **44,663** rear + rear-side *(paper)* | "recent ten years" before 2015; Chinese web sources → many Chinese brands | **non-commercial research only**, no redistribution of images *or derived data* | none official |
| **Stanford Cars** | used by CarVision v1 (see RESEARCH.md); [official page](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) unreachable during this research | 16,185 images, 196 classes | none | not labelled; not measured | 1991–2012, US market | research use (web images) | v1 ViT already shipped |
| **BoxCars116k** – [arXiv 1703.00686](https://arxiv.org/abs/1703.00686) | download host `medusa.fit.vutbr.cz` timed out from here | 116,286 images of 27,496 vehicles, 693 fine-grained classes, 45 makes | surveillance cameras, **front and rear** views from above | not counted | Czech traffic, European market | listed as non-commercial; a [Czech public contract](https://smlouvy.gov.cz/smlouva/soubor/6823572/ANON%20180212_Database%20License%20Intel%20updates%2016%20Feb%202018%20_19_2_18%20schvzmenydocx.pdf) shows a **commercial licence sold for €18,000** | none official |
| **VMMRdb** – [faezetta/VMMRdb](https://github.com/faezetta/VMMRdb) | repo + Dropbox link alive (verified) ; data not downloaded | 291,752 images, 9,170 classes | multiple angles, **unlabelled** | unknown | 1950–2016, US (Craigslist) | repo MIT (code); images are scraped ads, licence unclear | ResNet-50/VGG (Torch7 `.t7`, 3,036 classes) |
| **DVM-CAR 2.0** – [deepvisualmarketing.github.io](https://deepvisualmarketing.github.io/) | user manual + paper read (verified) | 1,451,784 images, 899 models, UK market ~2000–2020 | **8 viewpoints** (45° steps) predicted by a CNN, stored in the image table | not counted | 2000–2020 | **CC BY-NC** (non-commercial) | none |
| **Car-1000** – [toggle1995/Car-1000](https://github.com/toggle1995/Car-1000), [arXiv 2503.12385](https://arxiv.org/abs/2503.12385) | repo verified | 140,267 images, 1,000 models, 166 makers; 450 models after 2020 *(paper)*; selected by DongCheDi (Chinese) popularity | none | unknown | 1960s–2020s | **no licence** in repo | none; best benchmark 89.45 % (CAL) *(paper)* |
| Rear-lamp MMR paper – [Springer 2023](https://link.springer.com/article/10.1007/s11042-023-15081-x) | paper only | 44,663 rear/rear-side images, 100 classes *(paper)* | rear-focused | – | – | dataset **not found** publicly | none |
| **Wikimedia Commons** | **verified** (API) | per-model categories | file/category names (“rear”, “Heck”…) | very few for Iranian/Chinese models (often 0–2 per model) | any | **free licences** (CC BY/BY-SA/PD), attribution per file | – |
| [rdo1/euro-dashcam-vehicle-classifier](https://huggingface.co/rdo1/euro-dashcam-vehicle-classifier) | model card verified | 646 European classes, ConvNeXt-Tiny, 113 MB | dashcam crops (mostly rear) | – | European | MIT (model); training-data provenance "see dataset card" | 57.2 % top-1 on its own dashcam holdout *(claim)* |
| Orientation models: [fort-cyber/Car-orientation-image](https://huggingface.co/fort-cyber/Car-orientation-image) (ConvNeXt regression, Apache-2.0), [ArrayDice/car_orientation_classification](https://huggingface.co/ArrayDice/car_orientation_classification) (ViT, 8 directions, 69 % acc, Apache-2.0) | model cards verified | – | viewpoint only | – | – | Apache-2.0 | yes (not needed: CLIP zero-shot is used instead, §5.2) |

**Iranian-market model list** used to expand the database: Wikipedia pages for
[Iran Khodro](https://en.wikipedia.org/wiki/Iran_Khodro), [SAIPA](https://en.wikipedia.org/wiki/SAIPA),
[MVM/Fownix](https://en.wikipedia.org/wiki/Modiran_Vehicle_Manufacturing_Company), [Bahman Group](https://en.wikipedia.org/wiki/Bahman_Group),
[JAC Jiayue X7 (KMC K7)](https://en.wikipedia.org/wiki/JAC_Jiayue_X7), and Persian sources for Lamari/KMC
(cited per record in `sources`). Corrections found on the way: Bahman *Fidelity* is a Jetour X70 (the previous
"BAIC" was wrong); MVM X22 is based on the Chery Tiggo 3x. Note that the IKCO Wikipedia page is itself flagged
for possibly unsourced content.

**Licensing consequence (important).** Every large rear-capable dataset is either unlicensed (SIVD, VMMRdb, Car-1000)
or non-commercial (CompCars, DVM-CAR, BoxCars116k free tier). CarVision is MIT-licensed. The head shipped in this
repository is trained on SIVD images, whose licence is unknown – see §6 for what that means and what to do before
any commercial use.

## 3. Architecture choice (Phase 1 → 2)

Constraints: browser inference (WASM everywhere, WebGPU where available), ≤ ~100 MB per model, ONNX, no GPU for
training on the development machine (GeForce MX110, 2 GB – not usable for fine-tuning), and very little licensed
rear data.

| Option | For | Against |
|---|---|---|
| Fine-tune EfficientNet-B0/B2, MobileNetV3, ResNet50, ConvNeXt-T, Swin-T on rear crops | best accuracy *if* many labelled rear images per class; small (MobileNetV3 ≈ 20 MB, EfficientNet-B0 ≈ 20 MB, ConvNeXt-T ≈ 110 MB fp32) | needs a GPU and thousands of licensed rear images per class set; every new vehicle = full retrain; no zero-shot rear check; no way to cover vehicles without photos |
| ViT-B/16 (v1 approach) | strong fine-grained features | 86 MB int8 already, same data problem |
| **CLIP image encoder (frozen) + trained linear head** | zero-shot **rear-view validation** from text prompts (no extra model); **text-prompt classes** for vehicles without photos; head re-trains on a CPU in seconds; new vehicles = new head rows; open-set behaviour measurable | general (not car-specific) features, so lower ceiling than a fully fine-tuned network; ~90–106 MB |

Chosen: **OpenCLIP ViT-B-32 `laion2b_s34b_b79k`** (MIT weights). ViT-B/32 needs 4× fewer tokens than B/16 (faster
on WASM). The head is a linear layer **initialised from CLIP text embeddings** ("a photo of the rear of a
*Saipa Shahin*, a type of car") and fine-tuned on rear crops with an L2 pull towards that initialisation, so classes
with photos learn and classes without photos keep their text-prompt behaviour; temperature-calibrated on a
validation split.

**Quantisation (measured on 20 real rear photos, cosine to fp32 embedding):** dynamic int8 (`quantize_dynamic`,
activations quantised) mean 0.969 / min 0.930 – unusable, CLIP activations have outliers; per-channel dynamic int8
0.719; weight-only `MatMulNBits` 4-bit 0.959; **weight-only 8-bit `MatMulNBits` 0.9999** → shipped.

**Preprocessing:** the detector crop is **padded to a square** (black), not centre-cropped, so both tail lights stay in
view (§5.3 compares the two).

## 4. Pipeline as implemented

```
rear photo -> validation (type/size/magic bytes) -> YOLOX-tiny vehicle detection -> main vehicle crop
  -> CLIP ViT-B/32 embedding (public/models/rear-encoder-dc-b32-v1.onnx)
       -> zero-shot viewpoint check: P(rear)+P(rear 3/4) < 0.5  => "Please upload a photograph of the rear of the vehicle."
                                                                   (user can override)
       -> rear-view head: calibrated probabilities, aggregated per database vehicle
  -> result shown immediately (visual only)
  -> OCR (tesseract.js, eng+fas, multi-pass: original, contrast, sharpened, 3 upscaled rear badge zones,
     upscaled re-reads of low-confidence words; early stop when text and image agree)
  -> local JSON database match (aliases, brand+model, badges; engine badges never identify)
  -> fusion (decision table, no score averaging) -> final result
  optional: older general-view classifiers (Stanford ViT, Iranian ResNet18) on request, shown separately
```

Fusion rules (`src/ai/fusion.js`): text that fully names a vehicle + same vehicle in the visual top-3 → identified
(high); confident visual (≥ `REAR.visualAccept`) + contradicting identified text → uncertain; identified text
otherwise → identified (medium, "based on badge text"); visual ≥ accept → identified (high if the badge brand/model
supports it, uncertain if the badge shows another brand); visual between `visualLikely` and `visualAccept` → "likely";
else uncertain with candidates.

Open-set check: the top class's embedding similarity to its nearest training prototype (k-means, ≤ 8 per class)
must reach `openSetMinSimilarity` (set so 90 % of validation images pass); otherwise the visual answer is not
trusted ("may be a vehicle the model does not know").

## 5. Measured results

All numbers come from `scripts/train_rear_head.py` (full report: [docs/data/rear-head-report.json](data/rear-head-report.json))
and the scripts named in each section. Images are the detector crop of the main vehicle, padded to a square, embedded
with the shipped encoder's fp32 twin (cosine to the shipped 8-bit model 0.9998).

### 5.1 Test sets

| Set | What | Rear images | Labels | Licence | Role |
|---|---|---|---|---|---|
| **SIVD test split** | 7,342 ad photos, 29 classes | **3,005** after the rear check and after removing 31 near-duplicates (cosine ≥ 0.95) of training images. 39.9 % of all SIVD test photos are rear / rear-¾ (LAION check at 0.3) | dataset folders | unknown (ad photos with Sheypoor watermarks) | main test |
| **Wikimedia Commons** | 583 photos of 30 models, rear/front by file/category name (`scripts/collect-commons.mjs`) | 438 rear (4 files removed: no target car visible), 141 front | Commons categories (human-curated, a few errors) | CC BY / BY-SA / PD | independent test via **2-fold cross-fitting grouped by uploader** (120 groups, so a photo series of one car never sits in both halves) + viewpoint test |
| SIVD side / front-¾ sample | 600 SIVD photos the LAION check called side or front-¾ | – | viewpoint only | unknown | false-accept rate of the rear check |

Visual spot checks: 24 / 24 random SIVD images accepted as rear were rear or rear-¾ views; 12 / 12 rejected ones were
front/side views.

Training: SIVD **train** split (rear photos of a 9,000-image evenly spaced subset + all photos of Shahin, Tiba, Xantia)
and the Commons rear photos – 5,211 images in the final head. SIVD's own train/test split was kept.

### 5.2 Rear-view validation (zero-shot, DataComp encoder, threshold `REAR.viewpointMin`)

| Threshold | Commons rear accepted (n=438) | Commons front accepted (n=141) | SIVD front-¾ accepted (n=300) | SIVD side accepted (n=300) |
|---|---|---|---|---|
| 0.15 | 99.3 % | 0.0 % | 4.0 % | 16.3 % |
| **0.20 (shipped)** | **97.9 %** | **0.0 %** | **1.3 %** | **10.0 %** |
| 0.30 | 93.4 % | 0.0 % | 0.3 % | 2.7 % |

Most side-view false accepts are **pickups** (the open bed reads as "the back of a car"). Rejected rear photos are
mostly near-side views (≥ 60° from straight behind). The user can override ("It is a rear photo – analyse it anyway").

### 5.3 Encoder choice

5-fold cross-validation on the 3,198 SIVD rear test images (same folds, same head training):

| Encoder (ViT-B/32, MIT) | zero-shot top-1 | head top-1 | head top-3 |
|---|---|---|---|
| OpenCLIP `laion2b_s34b_b79k` | 26.7 % | 84.8 % | 95.9 % |
| **OpenCLIP `datacomp_xl_s13b_b90k` (shipped)** | **38.3 %** | **88.4 %** | **97.1 %** |

The DataComp model card, like the LAION one, calls the weights a research output and advises caution for products;
the licence itself is MIT. (ViT-B/16 and SigLIP were downloaded but not evaluated: 4× the tokens on WASM.)

### 5.4 Rear-view classification (shipped head: 46 classes → 44 database vehicles)

| Test | n | top-1 | top-3 |
|---|---|---|---|
| SIVD test, rear (in-domain: same ad source as training) | 3,005 | **87.9 %** (vehicle level 88.9 %) | **97.3 %** |
| – text prompts only (zero-shot), same images | 3,005 | 38.3 % | 62.6 % |
| Commons, vehicles also trained on SIVD (cross-fit) | 104 | **86.5 %** | 99.0 % |
| Commons, vehicles trained only on Commons photos (cross-fit) | 325 | **82.2 %** | 93.5 % |
| – text prompts only (zero-shot), same images | 325 | 61.5 % | 75.7 % |
| **v1 pipeline** (Stanford ViT + Iranian ResNet18 + routing) on the Commons rear photos | 438 | correct 2.7 %, **wrong 54.1 %**, uncertain 43.2 % | |
| v1 ViT on Commons rear photos of its own classes | 190 | 11.6 % | |
| v1 ResNet18 on Commons rear photos of its own classes | 111 | 23.4 % | |

Same 600 random SIVD rear test photos, each system with its own decision rule (`scripts/eval_baseline_rear.py`):

| | identified correctly | identified wrongly | "likely" | uncertain |
|---|---|---|---|---|
| v1 | 25.5 % | 27.2 % | – | 47.3 % |
| **v2 (rear-view)** | **81.3 %** | **3.5 %** | 6.3 % | 8.8 % |

(v1's ResNet18 alone: 57.4 % top-1 on the 418 of these photos that belong to its classes.)

**Accuracy vs confidence** (calibrated probability of the top vehicle; `REAR.visualAccept` = 0.6, `visualLikely` = 0.35):

| threshold | SIVD: answered | SIVD: accuracy of answered | Commons: answered | Commons: accuracy of answered |
|---|---|---|---|---|
| 0.35 | 98.4 % | 88.9 % | 98.8 % | 84.2 % |
| 0.5 | 93.5 % | 91.7 % | 94.6 % | 86.9 % |
| **0.6** | **88.1 %** | **94.1 %** | **90.9 %** | **87.9 %** |
| 0.8 | 75.2 % | 96.9 % | 78.8 % | 92.3 % |
| 0.9 | 63.4 % | 98.3 % | 70.6 % | 95.4 % |

**Per class, SIVD test (precision / recall %):** 206 97/92 · 207i 81/90 · 405 86/88 · Pars 87/80 · Arisun 65/73 ·
Dena 97/90 · Runna 77/84 · Samand 93/89 · Soren 85/90 · Paykan 95/99 · Paykan pickup 89/73 · L90 96/94 ·
Pride 131 86/84 · Pride 111/132 85/82 · Pride 141 88/88 · Saipa 151 78/91 · Quik 90/94 · Saina 85/88 · Shahin 74/96 ·
Tiba 87/86 · Rio 72/83 · Xantia 84/90 · Renault PK 92/99 · Brilliance H330 Cross 95/90 · MVM 315 88/85 · MVM X22 89/95 ·
JAC S5 96/96 · Cara pickup 91/85 · Zamyad Nissan pickup 76/95.

**Per vehicle, Commons cross-fit (top-1 / top-3 %, n):** Toyota Corolla 92/100 (39) · RAV4 87/100 (38) · Camry 96/96 (26) ·
Hyundai Elantra 88/100 (34) · Tucson 79/92 (24) · Kia Sportage 72/83 (36) · Cerato 76/94 (17) · BMW 3 Series 92/95 (40) ·
Mercedes C-Class 96/96 (24) · Peugeot 2008 83/94 (18) · Haval H6 (Cheval 6) 56/88 (16) · Peugeot 206 70/100 (33) ·
207 93/100 (27) · 405 100/100 (14) · Tondar 90 92/100 (13) · Pride 89/89 (9) · Samand 100 (5) ·
**Fownix FX 0/60 (5) · Tiggo 7 Pro 33/67 (3) · Tiggo 8 Pro 0/100 (2) · KMC J7 0/50 (2) · Tara 0 (1)** · Dena, Soren,
Shahin: 1 photo each (correct). Generations are pooled (e.g. Corolla E170 + E210, Tucson ix35/TL/NX4).

**Not measured:** accuracy by generation/facelift (no per-generation rear labels), and day vs night / blur /
low-resolution breakdowns (the test photos carry no such labels). The test sets do contain such photos, unlabelled.

### 5.5 Unknown-vehicle rejection

Leave-classes-out: 7 SIVD classes (405, Arisun, Paykan pickup, L90, Quik, Rio, Nissan pickup) were removed from training;
their test photos should then *not* be identified.

| Rule | unknown vehicles rejected | known vehicles still answered |
|---|---|---|
| probability ≥ 0.6 | 15.7 % | 93.1 % |
| probability ≥ 0.9 | 51.4 % | 78.3 % |
| prototype similarity ≥ threshold (0.80) | 39.1 % | 91.4 % |
| **similarity + probability ≥ 0.6 (shipped)** | **46.8 %** | **86.6 %** |

So **about half of the photos of a vehicle the model does not know still get a confident wrong name.** This is the
main limitation of a closed-set model; the UI says so next to uncertain results.

### 5.6 Coverage actually claimed

`public/data/vehicles.json` (`rearRecognition`, written by `scripts/sync-rear-coverage.mjs`): **90 vehicles** in the
database; **44** can be output by the rear-view model – **38 trained** on ≥ 10 rear photos, **6 with few or none**
(Fownix FX, Tiggo 7 Pro, Tiggo 8 Pro, KMC J7, IKCO Tara: 1–5 photos each; Saipa Atlas: text prompt only); the other
**46** can only be identified from readable badge text.

### 5.7 OCR

Synthetic rear badges drawn into a real rear photo (10 cases, 10–20 px): single pass identified 4 / 10, the multi-pass
reader 8 / 10 (details and the two OCR bugs found by this test in docs/TESTING.md, "Rear-view V2"). Accuracy *with vs without OCR* on real rear photos is **not measured**: no labelled
set of real rear photos with legible badges was available (Commons thumbnails are 800 px and badges are rarely legible).

## 6. Licensing status of the shipped head (read before any commercial use)

* Encoder weights: MIT (OpenCLIP, DataComp-1B). App code: MIT.
* The head (`public/labels/rear-head.json`, ≈1 MB: class weights + k-means prototypes, i.e. averaged embeddings) is
  **derived from SIVD images whose licence is not stated** (scraped ad photos) and from Wikimedia Commons images
  (CC BY / BY-SA; per-file attribution in the collection manifest). No image is redistributed.
* Before commercial use: get permission from the SIVD authors (contact in their README) or retrain the head on licensed
  rear photos (own photos, licensed stock, or contributions with consent). The pipeline is ready for that: run
  `scripts/embed_images.py` + `scripts/train_rear_head.py` on any `<split>/<class>/<file>` zip – a CPU is enough (head
  training takes seconds; embedding ≈ 6 images/s on the development laptop).

## 7. Not done / next steps

1. **Newer Chinese vehicles** (Fownix/Chery, KMC/JAC, Lamari, Haval, Jetour-based Fidelity, MVM X-series) – the stated
   top priority – are **not reliably recognised from the rear**: Commons has 0–5 freely licensed rear photos per model
   and no other licence-clean source was found (CompCars has 44,663 rear/rear-¾ images incl. many Chinese brands up to
   ≈2015, but is non-commercial only). They are identified when a badge is readable. The fastest real improvement is
   **collecting ~50–100 own rear photos per model** – the training pipeline is ready for it.
2. Generation/facelift classes need per-generation labels.
3. A rear-badge text detector (instead of fixed zones) would help OCR on small badges.
4. Real-device testing (mobile, Firefox, Safari) is still open – see docs/TESTING.md.
