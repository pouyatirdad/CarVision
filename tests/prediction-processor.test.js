import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeYolox, nms, softmax, topK } from '../src/ai/prediction-processor.js';

test('softmax sums to 1 and is order-preserving', () => {
  const p = softmax(new Float32Array([1, 2, 3, 1000]));
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-6);
  assert.ok(p[3] > 0.99);
});

test('topK returns best indices first', () => {
  const top = topK(new Float32Array([0.1, 0.5, 0.2, 0.2]), 2);
  assert.equal(top[0].index, 1);
  assert.equal(top.length, 2);
});

test('nms removes overlapping lower-score boxes', () => {
  const keep = nms([
    { x1: 0, y1: 0, x2: 100, y2: 100, score: 0.9 },
    { x1: 5, y1: 5, x2: 105, y2: 105, score: 0.8 },
    { x1: 300, y1: 300, x2: 400, y2: 400, score: 0.7 },
  ]);
  assert.equal(keep.length, 2);
  assert.equal(keep[0].score, 0.9);
});

test('decodeYolox finds a car at the right place in original-image coordinates', () => {
  const inputSize = 416;
  const n = 3549;
  const out = new Float32Array(n * 85); // zeros: objectness 0 => nothing detected
  // Put a "car" (COCO id 2) on the stride-32 level: grid cell (6,6) => centre (6.5*32)... use offsets directly.
  const level32Start = 52 * 52 + 26 * 26;
  const gx = 6;
  const gy = 7;
  const o = (level32Start + gy * 13 + gx) * 85;
  out[o] = 0.5; // cx offset within cell
  out[o + 1] = 0.5;
  out[o + 2] = Math.log(200 / 32); // w = 200 px in letterbox space
  out[o + 3] = Math.log(120 / 32); // h = 120 px
  out[o + 4] = 0.95; // objectness
  out[o + 5 + 2] = 0.9; // class 'car'
  const scale = 0.5; // original image is 2x larger than the letterboxed one
  const dets = decodeYolox(out, { inputSize, scale, imageWidth: 1000, imageHeight: 800 });
  assert.equal(dets.length, 1);
  const d = dets[0];
  assert.equal(d.label, 'car');
  const cx = (0.5 + gx) * 32;
  const cy = (0.5 + gy) * 32;
  assert.ok(Math.abs((d.x1 + d.x2) / 2 - cx / scale) < 1);
  assert.ok(Math.abs((d.y1 + d.y2) / 2 - cy / scale) < 1);
  assert.ok(Math.abs(d.x2 - d.x1 - 200 / scale) < 1);
});

test('decodeYolox ignores non-vehicle classes (person)', () => {
  const out = new Float32Array(3549 * 85);
  const o = 10 * 85;
  out[o + 2] = Math.log(5);
  out[o + 3] = Math.log(5);
  out[o + 4] = 0.99;
  out[o + 5 + 0] = 0.99; // person
  assert.equal(decodeYolox(out, { inputSize: 416, scale: 1, imageWidth: 416, imageHeight: 416 }).length, 0);
});
