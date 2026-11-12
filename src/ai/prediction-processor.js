// Pure post-processing helpers (no DOM, no ORT) - unit-testable under plain Node.

import { DETECTION } from '../config.js';

export function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

/** Top-k [{index, probability}] sorted descending. */
export function topK(probs, k = 3) {
  const idx = Array.from(probs, (p, i) => i);
  idx.sort((a, b) => probs[b] - probs[a]);
  return idx.slice(0, k).map((index) => ({ index, probability: probs[index] }));
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter || 1);
}

export function nms(dets, iouThreshold = DETECTION.nmsIou) {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of sorted) {
    if (keep.every((k) => iou(k, d) < iouThreshold)) keep.push(d);
  }
  return keep;
}

/**
 * Decode raw YOLOX output [N, 85] (cx, cy, w, h as grid offsets / log-sizes, objectness, 80 class scores)
 * for strides 8/16/32 on an `inputSize` square input. Returns vehicle detections in ORIGINAL image
 * coordinates: [{x1,y1,x2,y2,score,label}] sorted by score*area (the "main" vehicle first).
 */
export function decodeYolox(output, { inputSize, scale, imageWidth, imageHeight }, opts = DETECTION) {
  const nCls = 80;
  const stride = 5 + nCls;
  const levels = [8, 16, 32].map((s) => ({ s, n: Math.floor(inputSize / s) }));
  const candidates = [];
  let anchor = 0;
  for (const { s, n } of levels) {
    for (let gy = 0; gy < n; gy++) {
      for (let gx = 0; gx < n; gx++, anchor++) {
        const o = anchor * stride;
        const obj = output[o + 4];
        if (obj < 0.05) continue;
        let bestScore = 0;
        let bestCls = -1;
        for (const clsId of Object.keys(opts.vehicleClasses)) {
          const sc = obj * output[o + 5 + Number(clsId)];
          if (sc > bestScore) {
            bestScore = sc;
            bestCls = Number(clsId);
          }
        }
        if (bestScore < opts.scoreThreshold) continue;
        const cx = (output[o] + gx) * s;
        const cy = (output[o + 1] + gy) * s;
        const w = Math.exp(output[o + 2]) * s;
        const h = Math.exp(output[o + 3]) * s;
        candidates.push({
          x1: Math.max(0, (cx - w / 2) / scale),
          y1: Math.max(0, (cy - h / 2) / scale),
          x2: Math.min(imageWidth, (cx + w / 2) / scale),
          y2: Math.min(imageHeight, (cy + h / 2) / scale),
          score: bestScore,
          label: opts.vehicleClasses[bestCls],
        });
      }
    }
  }
  return nms(candidates, opts.nmsIou)
    .filter((d) => d.x2 - d.x1 >= opts.minBoxSidePx && d.y2 - d.y1 >= opts.minBoxSidePx)
    .sort((a, b) => b.score * area(b) - a.score * area(a))
    .slice(0, opts.maxVehicles);
}

function area(d) {
  return (d.x2 - d.x1) * (d.y2 - d.y1);
}

/** Turn a top-k list into display candidates using a labels table. */
export function toCandidates(top, labels, domain) {
  return top.map(({ index, probability }) => ({ domain, classId: index, probability, label: labels[index] }));
}
