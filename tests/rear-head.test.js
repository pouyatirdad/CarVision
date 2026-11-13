import test from 'node:test';
import assert from 'node:assert/strict';
import { RearHead, decodeFloat32 } from '../src/ai/rear-head.js';

const b64 = (arr) => Buffer.from(new Float32Array(arr).buffer).toString('base64');

// 4-dim toy embedding space: axis 0 = "rear", axis 1 = "front"; axes 2/3 separate the classes.
const head = new RearHead({
  modelId: 'toy',
  encoder: { dim: 4 },
  temperature: 1,
  viewpoint: { labels: ['rear', 'rear-side', 'front'], rear: ['rear', 'rear-side'], logitScale: 10, embeddings: b64([1, 0, 0, 0, 0.7, 0, 0.7, 0, 0, 1, 0, 0]) },
  classes: [
    { key: 'Pride131', vehicleId: 'saipa-pride', variant: 'sedan', kind: 'trained' },
    { key: 'Pride111', vehicleId: 'saipa-pride', variant: 'hatchback', kind: 'trained' },
    { key: 'Tara', vehicleId: 'ikco-tara', variant: null, kind: 'zero-shot' },
  ],
  weights: b64([0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 5]),
  bias: b64([0, 0, 0]),
});

test('decodeFloat32 round-trips little-endian float32', () => {
  assert.deepEqual([...decodeFloat32(b64([1.5, -2, 0]))], [1.5, -2, 0]);
});

test('viewpoint: rear vs front, embedding scale does not matter', () => {
  const r = head.viewpoint(new Float32Array([3, 0, 0, 0]));
  assert.equal(r.label, 'rear');
  assert.ok(r.rearProbability > 0.9);
  const f = head.viewpoint(new Float32Array([0, 2, 0, 0]));
  assert.equal(f.label, 'front');
  assert.ok(f.rearProbability < 0.1);
});

test('classify aggregates variants of one database vehicle and sorts by probability', () => {
  const out = head.classify(new Float32Array([0, 0, 1, 0.1]));
  assert.equal(out[0].vehicleId, 'saipa-pride');
  assert.equal(out[0].variants.length, 2);
  const sum = out.reduce((s, c) => s + c.probability, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9);
  assert.ok(Math.abs(out[0].probability - out[0].variants.reduce((s, v) => s + v.probability, 0)) < 1e-12);
  const tara = head.classify(new Float32Array([0, 0, 0, 1]))[0];
  assert.equal(tara.vehicleId, 'ikco-tara');
  assert.equal(tara.variants[0].kind, 'zero-shot');
});

test('open-set check: similarity to the predicted class prototypes', () => {
  const h2 = new RearHead({
    modelId: 'toy', encoder: { dim: 4 }, temperature: 1,
    viewpoint: { labels: ['rear'], rear: ['rear'], logitScale: 1, embeddings: b64([1, 0, 0, 0]) },
    classes: [{ key: 'a', vehicleId: 'a' }, { key: 'z', vehicleId: 'z', kind: 'zero-shot' }],
    weights: b64([0, 0, 1, 0, 0, 0, 0, 1]), bias: b64([0, 0]),
    openSetMinSimilarity: 0.9,
    prototypes: { counts: [1, 0], vectors: b64([0, 0, 1, 0]) },
  });
  assert.equal(h2.openSet(new Float32Array([0, 0, 2, 0]), 'a').ok, true);
  const far = h2.openSet(new Float32Array([0, 1, 1, 0]), 'a');
  assert.equal(far.checked, true);
  assert.equal(far.ok, false);
  assert.ok(Math.abs(far.similarity - Math.SQRT1_2) < 1e-6);
  assert.deepEqual(h2.openSet(new Float32Array([0, 0, 0, 1]), 'z'), { checked: false, ok: true, similarity: null });
});

test('shape mismatch is rejected', () => {
  assert.throws(() => new RearHead({ modelId: 'x', encoder: { dim: 4 }, temperature: 1, viewpoint: { labels: ['rear'], rear: ['rear'], logitScale: 1, embeddings: b64([1, 0, 0, 0]) }, classes: [{ key: 'a', vehicleId: 'a' }], weights: b64([1, 2]), bias: b64([0]) }), /shape/);
});
