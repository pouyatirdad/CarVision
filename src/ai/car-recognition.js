// The recognition pipeline (rear-view only, see docs/REAR-VIEW.md):
//   detect vehicles (YOLOX) -> crop the main vehicle -> CLIP image embedding (rear encoder)
//   -> rear-view check (zero-shot viewpoint) -> rear-view head (calibrated class probabilities)
// The older general-view classifiers (Stanford-Cars ViT, Iranian ResNet18) are a SECONDARY fallback that runs
// only on request (options.secondary), because they were not trained for rear photos.
// OCR + database matching + fusion happen on the main thread afterwards (main.js), so OCR never delays this.
// Runs inside a Web Worker (or on the main thread as a fallback) - ImageBitmap + OffscreenCanvas only, no DOM.

import { DETECTION, LABEL_BASE_URL, MODELS, REAR, ROUTING } from '../config.js';
import { CLIP, HALF, IMAGENET, cropToRGBTensorData, letterboxBGR } from '../image/image-processor.js';
import { ModelManager } from './model-manager.js';
import { decide, needsIranianModel } from './model-router.js';
import { decodeYolox, softmax, toCandidates, topK } from './prediction-processor.js';
import { RearHead } from './rear-head.js';

export class CarRecognizer {
  /** @param {(e:object)=>void} [onEvent] progress callback: {type:'status'|'model', ...} */
  constructor(onEvent) {
    this.onEvent = onEvent || (() => {});
    this.manager = new ModelManager((e) => this.onEvent({ type: 'model', ...e }));
    this.labels = {}; // key -> Promise<array>
    this.head = null; // Promise<RearHead>
    this.usedProviders = {};
  }

  capabilities() {
    return this.manager.capabilities();
  }

  configure(opts) {
    this.manager.configure(opts);
  }

  #fetchJson(file, what) {
    return fetch(LABEL_BASE_URL + file)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .catch((e) => {
        throw new Error(`Could not load the ${what} (${e.message})`);
      });
  }

  #labels(key) {
    if (!this.labels[key]) {
      this.labels[key] = this.#fetchJson(MODELS[key].labels, `${key} label file`).catch((e) => {
        delete this.labels[key];
        throw e;
      });
    }
    return this.labels[key];
  }

  #rearHead() {
    if (!this.head) {
      this.head = this.#fetchJson(MODELS.rear.head, 'rear-view model head').then((j) => new RearHead(j)).catch((e) => {
        this.head = null;
        throw e;
      });
    }
    return this.head;
  }

  #status(phase, extra = {}) {
    this.onEvent({ type: 'status', phase, ...extra });
  }

  async detect(bitmap) {
    const cfg = MODELS.detector;
    this.#status('detect');
    const { data, scale } = letterboxBGR(bitmap, cfg.inputSize);
    const { output, provider } = await this.manager.run('detector', data, [1, 3, cfg.inputSize, cfg.inputSize]);
    this.usedProviders.detector = provider;
    return decodeYolox(output, { inputSize: cfg.inputSize, scale, imageWidth: bitmap.width, imageHeight: bitmap.height });
  }

  async #rear(bitmap, box) {
    const cfg = MODELS.rear;
    this.#status('rear');
    const [head, tensor] = [await this.#rearHead(), cropToRGBTensorData(bitmap, box, { width: cfg.inputSize, height: cfg.inputSize, fit: cfg.fit, ...CLIP })];
    const { output, provider } = await this.manager.run('rear', tensor, [1, 3, cfg.inputSize, cfg.inputSize]);
    this.usedProviders.rear = provider;
    const embedding = Float32Array.from(output);
    const candidates = head.classify(embedding, 5);
    const openSet = candidates.length ? head.openSet(embedding, candidates[0].variants[0].key) : { checked: false, ok: true, similarity: null };
    return { head, viewpoint: head.viewpoint(embedding), candidates, openSet };
  }

  async #classifyInternational(bitmap, box) {
    const cfg = MODELS.international;
    const tensor = cropToRGBTensorData(bitmap, box, { width: cfg.inputSize, height: cfg.inputSize, ...HALF });
    const { output, provider } = await this.manager.run('international', tensor, [1, 3, cfg.inputSize, cfg.inputSize]);
    this.usedProviders.international = provider;
    return toCandidates(topK(softmax(output), 3), await this.#labels('international'), 'international');
  }

  async #classifyIranian(bitmap, box) {
    const cfg = MODELS.iranian;
    const tensor = cropToRGBTensorData(bitmap, box, { width: cfg.inputWidth, height: cfg.inputHeight, ...IMAGENET });
    const { output, provider } = await this.manager.run('iranian', tensor, [1, 3, cfg.inputHeight, cfg.inputWidth]);
    this.usedProviders.iranian = provider;
    return toCandidates(topK(softmax(output), 3), await this.#labels('iranian'), 'iranian');
  }

  /** The legacy general-view classifiers and their routing rule (docs/RESEARCH.md). Secondary evidence only. */
  async #secondary(bitmap, box) {
    this.#status('classify-international');
    const international = await this.#classifyInternational(bitmap, box);
    let iranian = null;
    if (needsIranianModel(international[0].probability)) {
      this.#status('classify-iranian');
      iranian = await this.#classifyIranian(bitmap, box);
    }
    return decide({ international, iranian });
  }

  /**
   * @param {ImageBitmap} bitmap
   * @param {{box?: object, detections?: object[], wholeImage?: boolean, ignoreViewpoint?: boolean, secondary?: boolean}} [options]
   *   box: classify this region (user picked another vehicle); wholeImage: classify the entire image even if no
   *   vehicle was found; detections: reuse a previous detection result; ignoreViewpoint: the user confirmed it is
   *   a rear photo although the check disagreed; secondary: also run the older general-view classifiers.
   */
  async recognize(bitmap, options = {}) {
    const t0 = performance.now();
    const timings = {};
    const warnings = [];
    const full = { x1: 0, y1: 0, x2: bitmap.width, y2: bitmap.height };

    let detections = options.detections;
    if (!detections) {
      const t = performance.now();
      detections = await this.detect(bitmap);
      timings.detect = performance.now() - t;
    }

    let box = options.box;
    let selected = box ? detections.findIndex((d) => d.x1 === box.x1 && d.y1 === box.y1) : 0;
    if (!box) {
      if (detections.length === 0) {
        if (!options.wholeImage) return this.#result({ status: 'no-vehicle', detections, bitmap, timings, t0, warnings: ['no-vehicle'] });
        box = full;
        selected = -1;
        warnings.push('no-vehicle-whole-image');
      } else {
        box = detections[0];
        selected = 0;
      }
    }
    if (detections.length > 1 && selected >= 0) warnings.push('multiple-vehicles');
    const boxArea = ((box.x2 - box.x1) * (box.y2 - box.y1)) / (bitmap.width * bitmap.height);
    if (box !== full && boxArea < 0.04) warnings.push('small-vehicle');

    const tc = performance.now();
    const rear = await this.#rear(bitmap, box);
    timings.classify = performance.now() - tc;
    const isRear = rear.viewpoint.rearProbability >= REAR.viewpointMin;
    if (!isRear && !options.ignoreViewpoint) {
      return this.#result({ status: 'not-rear', detections, selected, box, bitmap, timings, t0, warnings, rear });
    }
    if (!isRear) warnings.push('viewpoint-overridden');

    let secondary = null;
    if (options.secondary) {
      const ts = performance.now();
      secondary = await this.#secondary(bitmap, box);
      timings.secondary = performance.now() - ts;
    }
    return this.#result({ status: 'ok', detections, selected, box, bitmap, timings, t0, warnings, rear, secondary });
  }

  #result({ status, detections, selected = -1, box = null, bitmap, timings, t0, warnings, rear = null, secondary = null }) {
    timings.total = performance.now() - t0;
    const models = ['detector', 'rear', 'international', 'iranian']
      .filter((k) => this.usedProviders[k] && (k === 'detector' || k === 'rear' || (secondary && (k === 'international' || (k === 'iranian' && secondary.reason !== 'international-confident')))))
      .map((k) => ({ key: k, title: MODELS[k].title, provider: this.usedProviders[k] }));
    return {
      status,
      image: { width: bitmap.width, height: bitmap.height },
      detections: detections.map((d) => ({ x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2, score: d.score, label: d.label })),
      selected,
      box,
      viewpoint: rear?.viewpoint ?? null,
      visual: rear ? { modelId: rear.head.modelId, candidates: rear.candidates, openSet: rear.openSet } : null,
      secondary: secondary
        ? { verdict: secondary.verdict, primary: secondary.primary, alternatives: secondary.alternatives, thresholds: { international: ROUTING.internationalAccept, iranian: ROUTING.iranianAccept } }
        : null,
      warnings,
      timings,
      models,
      detectionThreshold: DETECTION.scoreThreshold,
    };
  }

  releaseAll() {
    return this.manager.releaseAll();
  }
}
