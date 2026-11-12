# CarVision – Technical research report (Phase 1)

Date of research: 2026-11-12. Everything below was verified by downloading and running the artifacts,
not taken from README claims, unless explicitly marked *(upstream claim)*.

## 1. Iranian vehicles

| Repository | Weights actually in the repo? | Classes | Licence | Verdict |
|---|---|---|---|---|
| [RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System) | **Yes** – `weights/car_name_model.pth` (44.8 MB), `car_det_model.pt` (5.6 MB) + 4 others | **27** | **MIT** | **Selected** |
| [SoheilSaya/Iranian-Vehicle-Model-Detect](https://github.com/SoheilSaya/Iranian-Vehicle-Model-Detect) | Yes (`model_checkpoint.pth`, `accepted.pth`), architecture undocumented | 13 | GPL-3.0 | Rejected: fewer classes, undocumented, copyleft |
| [NahidEbrahimian/Cars-Classification-tensorflow](https://github.com/NahidEbrahimian/Cars-Classification-tensorflow) | Only via Google Drive links (not in repo) | 5 (≈2,100 images) | none stated | Rejected: 5 classes, ≈72 % accuracy *(upstream claim)*, no licence |

### Selected: `car_name_model.pth` (Reza)
* Architecture: torchvision **ResNet18**, `fc` replaced by `Linear(512, 27)`. Verified: the checkpoint loads
  with `strict=True` into exactly that architecture (loaded with `weights_only=True`, i.e. it is a plain
  state-dict, no pickled code).
* Preprocessing (from the upstream inference script and the training notebook):
  `Resize((220, 165))` (height 220, width 165 – the aspect ratio is deliberately *not* preserved),
  RGB, `/255`, ImageNet mean/std.
* Classes (order = output index): Arisan, Atlas, Dena, L90, Mazda vanet, Megan, Neissan, Pars, 206, 206 SD,
  207, 405, Peykan, Pride, Pride vanet, Pride 111, Quik, Rana, Rio, Saina, Samand, Shahin, Soren, Tara,
  Tiba, Tiba 2, Zantia.
* Upstream claims 97.0 % accuracy / 0.903 macro-F1 on 19,256 samples. The test split is not
  published, the data were scraped from classified-ad sites, and the model is trained on *cropped* cars
  (the upstream pipeline runs a detector first). I could not reproduce that figure (no access to the data);
  see §4 for what I did measure.
* ONNX conversion: works with the legacy `torch.onnx.export` (opset 17) → `iranian-cars.onnx`, 44.8 MB, fp32.
  See `scripts/convert_iranian.py`.
* Some classes (Arisan, Atlas, Rana, Zantia, Neissan) could not be tied to an exact model with confidence from
  public sources; the app shows the dataset label and leaves manufacturer/category blank for them.

### Licence caveat
The MIT licence covers Reza's repository. Its detector (`car_det_model.pt`) is a YOLO11 model; Ultralytics
publishes YOLO11 under **AGPL-3.0**, which would impose AGPL terms on a public deployment. I therefore did
**not** use it. The Iranian dataset's own images/redistribution terms are not stated by upstream; only the
trained weights are shipped, never the images.

## 2. International vehicles

Candidates checked on Hugging Face (Stanford Cars, 196 classes):

| Model | Size | Licence | Notes |
|---|---|---|---|
| [therealcyberlord/stanford-car-vit-patch16](https://huggingface.co/therealcyberlord/stanford-car-vit-patch16) | 344 MB fp32 | Apache-2.0 | ViT-B/16, **label names in config**, claims ≈86 % *(upstream claim)* – **Selected** |
| anonauthors/stanford_cars-timm-resnet50 | 96 MB | Apache-2.0 | Labels are bare integers, no documented accuracy → cannot be trusted/used |
| anonauthors/stanford_cars-ConvNeXt-base | 351 MB | none | Too large, no licence |
| tanganke/clip-vit-base-patch32_stanford-cars | – | none | CLIP fine-tune, not distributed as a plain ONNX-friendly classifier |

I found no ready-made ONNX export of any Stanford-Cars model, so I exported it (`scripts/convert_international.py`)
and applied dynamic int8 weight quantization to bring the download to **86 MB** (target 20–100 MB).

**Measured accuracy** (own evaluation, `scripts/eval_international.py`, 400 random images of the official
Stanford Cars *test* split, plain 224×224 bilinear resize, full images):

| Variant | Size | Top-1 | Top-5 |
|---|---|---|---|
| fp32 ONNX | 343 MB | 87.75 % | 96.50 % |
| **int8 ONNX (shipped)** | **86 MB** | **88.00 %** | **96.25 %** |

(±≈1.6 pp 95 % CI for n=400.) Quantization cost nothing measurable.
On the same images cropped to the YOLOX-detected vehicle (300 images): top-1 89 %.

**Hard limit of this model:** Stanford Cars contains 196 classes of mostly **US-market cars from 1991–2012**.
It cannot know newer models (e.g. a 2020 Toyota Corolla), most European/Asian-market-only models or any
Iranian model. It will confidently name the *closest* known class. This is exposed to the user as
"reference year" and an explicit low-confidence warning, never as a model-year claim.

## 3. Vehicle detector

`yolox_tiny.onnx` from the official YOLOX 0.1.1rc0 release (Megvii, **Apache-2.0**), 20.2 MB, COCO-trained.
Input 416×416 BGR 0–255 letter-boxed (pad 114, top-left), raw output `[1, 3549, 85]` (grid decoding done in JS).
We keep COCO classes car / bus / truck. Verified on Wikimedia Commons photos: finds the main car in most
street scenes, plus bystander cars (hence the "multiple vehicles" warning and click-to-select).

## 4. Can the two classifiers be compared? (routing analysis)

No. Both are closed-set softmax classifiers with unrelated class sets. Measured on detector crops:

| Max-softmax ≥ t | Iranian model on **foreign** (Stanford) cars | Intl model on Stanford | Intl model on Commons "Iranian car" photos |
|---|---|---|---|
| 0.65 | **39 %** pass | 90 % | 20 % |
| 0.80 | 24 % | 87 % | 12 % |
| 0.90 | 13 % | 82 % | 7 % |

The Iranian ResNet18 is badly over-confident on cars it has never seen, while the 196-way ViT is
comparatively well-behaved on cars outside its classes. Picking "highest raw confidence" would therefore be wrong.

**Routing rule implemented (ViT-first):**
1. Run the international ViT. If its top-1 ≥ **0.80** → international result (the Iranian model is not even downloaded).
2. Else run the Iranian model. If its top-1 ≥ **0.80** → Iranian result. (The upstream project uses 0.65, but its pipeline never sees foreign cars; at 0.65 the Iranian model claimed 13 of 300 Stanford cars, at 0.80 only 5. A real Audi S4 was named "Shahin 67 %" in the first browser test, which is what motivated raising it. Rejected Iranian candidates are still shown under "uncertain".)
3. Else → "uncertain": show the best candidate of each model, labelled by domain, with a warning.

On the 300 Stanford + 147 Commons images, this rule (0.80 / 0.80) routed 260/300 Stanford images to "international" (245 correct,
94 %), 5/300 to Iranian (wrong), and 35 to "uncertain"; of the 147 Commons photos 17 went to international, 49 to Iranian and 81 to uncertain. Caveat: the 147 Commons images are scene photos
fetched by category name (they include buses, bystander cars, wrong categories) and are **not a benchmark**; no
Iranian accuracy is claimed beyond the upstream's own unreproduced figure. The thresholds are hyper-parameters
chosen on that small set (sanity level, not a tuned benchmark) and live in `src/config.js`.

### 4.1 Confidence instability on out-of-domain cars (measured)
For a real Samand (Iranian car, unknown to the ViT) the ViT's top-1 was "Audi 100 Sedan 1994" with confidence
0.39 for one detector crop and 0.79 for a crop shifted by only 3 px (same image, same model), and 0.31–0.92
depending on the resize filter. So the international score is *not* a stable probability when the car is outside the
196 classes, and a fraction of Iranian cars (≈10–14 % of the Commons sample) will be mis-named as an old
look-alike with high confidence. I tested test-time augmentation (average of 4 views: original, mirrored, ×0.92, ×1.08
crops, 294+133 images): top-1 on Stanford 88.8 → 90.1 %, Iranian photos routed to "international" at the 0.8 threshold
13 → 10 %. That is a real but small gain for 4× the ViT cost (≈6 s on WASM), so it is **not** enabled.
This is the main known weakness of the system and is documented as such in the README.

## 5. Not available / not done

* No public held-out Iranian test set was available offline, so Iranian accuracy is **unmeasured** here.
* No model exists for modern international cars (post-2012) in a browser-friendly form with a usable licence;
  those will mostly come out "uncertain" or as a wrong older look-alike.
* Model year / trim are never claimed.
