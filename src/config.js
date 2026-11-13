// Central configuration: model locations, preprocessing parameters and routing thresholds.
// Everything here is static - no secrets, no API keys.

const ENV = import.meta.env || {}; // undefined when imported by plain Node (unit tests)
const BASE = ENV.BASE_URL || '/'; // e.g. "/carvision/" on GitHub Pages, "/" in dev

// Models can be hosted elsewhere (e.g. a release asset / CDN with CORS) by setting
// VITE_MODEL_BASE_URL at build time, e.g. VITE_MODEL_BASE_URL=https://example.org/carvision-models/
export const MODEL_BASE_URL = (ENV.VITE_MODEL_BASE_URL || `${BASE}models/`).replace(/\/?$/, '/');
export const LABEL_BASE_URL = `${BASE}labels/`;

export const CACHE_NAME = 'carvision-models-v1';

export const LIMITS = {
  maxFileBytes: 15 * 1024 * 1024,
  minSidePx: 48,
  maxPixels: 120_000_000, // refuse absurd images before decoding them fully
  maxDecodeSidePx: 2048, // large photos are down-scaled once on decode to save memory
  acceptedTypes: ['image/jpeg', 'image/png', 'image/webp'],
  acceptedExtensions: ['jpg', 'jpeg', 'png', 'webp'],
};

export const MODELS = {
  // Primary model: CLIP image encoder + rear-view head (public/labels/rear-head.json). See docs/REAR-VIEW.md.
  rear: {
    key: 'rear',
    // The file name carries the model version: the Cache API is keyed by URL, so replacing weights under the same name
    // would keep serving the old bytes to returning visitors.
    file: 'rear-encoder-dc-b32-v1.onnx',
    head: 'rear-head.json',
    title: 'CLIP ViT-B/32 rear-view recogniser (8-bit weights)',
    license: 'MIT (OpenCLIP ViT-B-32 datacomp_xl_s13b_b90k); head trained offline, see docs/REAR-VIEW.md',
    approxBytes: 98_211_387,
    inputSize: 224,
    fit: 'pad', // keep the whole rear (both tail lights) instead of a centre crop
    // Weight-only 8-bit MatMulNBits: ORT-Web's WebGPU kernel only supports 2/4-bit ("Only 2b and 4b quantization is
    // supported", verified in Chromium) and 4-bit is too lossy for this encoder (docs/REAR-VIEW.md), so WASM only.
    providers: ['wasm'],
  },
  detector: {
    key: 'detector',
    file: 'vehicle-detector.onnx',
    title: 'YOLOX-tiny vehicle detector',
    license: 'Apache-2.0 (Megvii YOLOX)',
    approxBytes: 20_219_662,
    inputSize: 416,
    // Small fp32 CNN: try WebGPU first, WASM fallback.
    providers: ['webgpu', 'wasm'],
  },
  international: {
    key: 'international',
    file: 'international-cars.onnx',
    labels: 'international-labels.json',
    title: 'ViT-B/16 Stanford Cars classifier (int8)',
    license: 'Apache-2.0 (therealcyberlord/stanford-car-vit-patch16)',
    approxBytes: 86_992_044,
    inputSize: 224,
    // Dynamic-int8 weights use MatMulInteger, which the WebGPU EP does not implement
    // (it would silently fall back to CPU per-node), so this model runs on WASM.
    providers: ['wasm'],
  },
  iranian: {
    key: 'iranian',
    file: 'iranian-cars.onnx',
    labels: 'iranian-labels.json',
    title: 'ResNet18 Iranian car classifier',
    license: 'MIT (RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System)',
    approxBytes: 44_751_950,
    inputHeight: 220,
    inputWidth: 165,
    providers: ['webgpu', 'wasm'],
  },
};

// Routing (see docs/RESEARCH.md section 4). The two classifiers are NOT score-comparable, so the
// decision is gated by per-model thresholds in a fixed order instead of "highest raw confidence".
export const ROUTING = {
  internationalAccept: 0.8, // ViT top-1 at/above this => international result
  iranianAccept: 0.8, // ResNet18 top-1 at/above this => Iranian result (upstream uses 0.65, but it never sees foreign cars; see docs/RESEARCH.md)
  lowConfidenceWarn: 0.5, // accepted results below this still get a visible warning
};

// Rear-view recognition (see docs/REAR-VIEW.md for how these were measured).
export const REAR = {
  // P(rear) + P(rear three-quarter) from the zero-shot viewpoint check; below => "please upload a rear photo".
  // Measured with the DataComp encoder (docs/REAR-VIEW.md §5.2): at 0.2, 97.9 % of 438 rear Commons photos pass,
  // 0 / 141 front photos, 1.3 % of front-three-quarter and 10 % of side photos (mostly pickups) pass.
  viewpointMin: 0.2,
  visualAccept: 0.6, // rear-view model probability at/above this (and open-set check passed) => identified
  visualLikely: 0.35, // between visualLikely and visualAccept => "likely", shown with a visible warning
};

export const DETECTION = {
  scoreThreshold: 0.35,
  nmsIou: 0.45,
  minBoxSidePx: 32,
  // COCO class ids kept as "vehicles": car, bus, truck
  vehicleClasses: { 2: 'car', 5: 'bus', 7: 'truck' },
  maxVehicles: 6,
};

// Self-hosted OCR (tesseract.js). Files are copied to public/ocr/ by scripts/copy-ocr.mjs - no CDN is used.
export const OCR = {
  baseUrl: `${BASE}ocr/`,
  languages: ['eng', 'fas'],
  minWordConfidence: 0.6, // tesseract word confidence (0-1) below this is dropped
  minWordLength: 2, // single characters are almost always noise
  maxUpscale: 2,
  maxSide: 1800, // longest side of the crop sent to OCR (px); badges are small, so keep resolution high
  zoneUpscale: 3, // rear badge zones (left / centre / right of the trunk lid) are enlarged up to 3x
  maxPasses: 10, // upper bound on OCR attempts per photo (3 whole-vehicle variants + 3 zones + up to 4 word zooms)
  refineMinConfidence: 0.3, // words between this and minWordConfidence are re-read from an upscaled crop
  maxRefineRegions: 4,
  storageKey: 'carvision-ocr', // localStorage flag: '0' = OCR disabled by the user
};
