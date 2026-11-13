// The rear-view classification head that runs on top of the CLIP image embedding (src/ai/rear-recognition.js
// produces the embedding with ONNX Runtime). Pure logic, no DOM/ORT, unit-tested under Node.
//
// public/labels/rear-head.json is produced offline by scripts/train_rear_head.py and contains:
//   viewpoint: CLIP text embeddings for rear / rear-side / front / front-side / side / other  -> rear-view check
//   classes:   one entry per visual class (key, vehicleId, variant, kind 'trained' | 'zero-shot')
//   weights/bias/temperature: linear head (initialised from CLIP text embeddings, fine-tuned on rear images),
//              temperature-calibrated on a validation split.

/** base64 little-endian float32 -> Float32Array (works in browsers, workers and Node). */
export function decodeFloat32(b64) {
  const bin = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function softmaxInPlace(z) {
  let m = -Infinity;
  for (const v of z) if (v > m) m = v;
  let s = 0;
  for (let i = 0; i < z.length; i++) s += z[i] = Math.exp(z[i] - m);
  for (let i = 0; i < z.length; i++) z[i] /= s;
  return z;
}

function normalise(v) {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

export class RearHead {
  /** @param {object} json parsed rear-head.json */
  constructor(json) {
    this.modelId = json.modelId;
    this.dim = json.encoder.dim;
    this.temperature = json.temperature;
    this.classes = json.classes;
    this.W = decodeFloat32(json.weights);
    this.b = decodeFloat32(json.bias);
    this.view = { labels: json.viewpoint.labels, rear: new Set(json.viewpoint.rear), scale: json.viewpoint.logitScale, E: decodeFloat32(json.viewpoint.embeddings) };
    if (this.W.length !== this.classes.length * this.dim || this.b.length !== this.classes.length) throw new Error('rear-head.json: weight shape does not match the class list');
    if (this.view.E.length !== this.view.labels.length * this.dim) throw new Error('rear-head.json: viewpoint embedding shape mismatch');
    // Optional open-set check: k-means prototypes of each class's training embeddings.
    this.openSetMin = json.openSetMinSimilarity ?? null;
    this.protoOffset = new Map(); // class key -> [first row, count]
    if (json.prototypes) {
      this.P = decodeFloat32(json.prototypes.vectors);
      let row = 0;
      json.prototypes.counts.forEach((n, c) => {
        if (n) this.protoOffset.set(this.classes[c].key, [row, n]);
        row += n;
      });
      if (row * this.dim !== this.P.length) throw new Error('rear-head.json: prototype shape mismatch');
    }
  }

  /**
   * Open-set check for the predicted class: cosine similarity to its nearest training prototype. Images of vehicles
   * the model was never trained on tend to be further away. Classes without prototypes (text-prompt classes)
   * cannot be checked -> {checked:false, ok:true}.
   */
  openSet(embedding, classKey) {
    const o = this.protoOffset.get(classKey);
    if (!o || this.openSetMin == null) return { checked: false, ok: true, similarity: null };
    const x = normalise(embedding);
    let best = -1;
    for (let r = o[0]; r < o[0] + o[1]; r++) best = Math.max(best, this.#dot(this.P, r, x));
    return { checked: true, ok: best >= this.openSetMin, similarity: best, threshold: this.openSetMin };
  }

  #dot(M, row, x) {
    let s = 0;
    const o = row * this.dim;
    for (let i = 0; i < this.dim; i++) s += M[o + i] * x[i];
    return s;
  }

  /**
   * Zero-shot viewpoint estimate.
   * @returns {{rearProbability:number, label:string, scores:Record<string, number>}}
   */
  viewpoint(embedding) {
    const x = normalise(embedding);
    const z = new Float64Array(this.view.labels.length);
    for (let k = 0; k < z.length; k++) z[k] = this.view.scale * this.#dot(this.view.E, k, x);
    softmaxInPlace(z);
    const scores = Object.fromEntries(this.view.labels.map((l, k) => [l, z[k]]));
    let best = 0;
    for (let k = 1; k < z.length; k++) if (z[k] > z[best]) best = k;
    const rearProbability = this.view.labels.reduce((s, l, k) => s + (this.view.rear.has(l) ? z[k] : 0), 0);
    return { rearProbability, label: this.view.labels[best], scores };
  }

  /**
   * Calibrated class probabilities, aggregated per database vehicle (several visual classes such as
   * "Pride 111" and "Pride 131" can belong to one database record).
   * @returns {Array<{vehicleId:string, probability:number, variants:Array<{key:string, variant:string|null, probability:number, kind:string}>}>}
   */
  classify(embedding, k = 5) {
    const x = normalise(embedding);
    const z = new Float64Array(this.classes.length);
    for (let c = 0; c < z.length; c++) z[c] = (this.#dot(this.W, c, x) + this.b[c]) / this.temperature;
    softmaxInPlace(z);
    const byVehicle = new Map();
    this.classes.forEach((cls, c) => {
      const e = byVehicle.get(cls.vehicleId) ?? { vehicleId: cls.vehicleId, probability: 0, variants: [] };
      e.probability += z[c];
      e.variants.push({ key: cls.key, variant: cls.variant ?? null, probability: z[c], kind: cls.kind });
      byVehicle.set(cls.vehicleId, e);
    });
    const list = [...byVehicle.values()];
    for (const e of list) e.variants.sort((a, b) => b.probability - a.probability);
    list.sort((a, b) => b.probability - a.probability || a.vehicleId.localeCompare(b.vehicleId));
    return list.slice(0, k);
  }
}
