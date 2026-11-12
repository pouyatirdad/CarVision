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
  storageKey: 'carvision-ocr', // localStorage flag: '0' = OCR disabled by the user
};
