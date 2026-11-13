import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { VehicleDatabase } from '../src/ai/vehicle-database.js';
import { fuse } from '../src/ai/fusion.js';

const read = (f) => JSON.parse(readFileSync(new URL(`../public/data/${f}`, import.meta.url), 'utf8'));
const db = VehicleDatabase.fromData(read('vehicles.json'), read('manufacturers.json'));
const cfg = { visualAccept: 0.6, visualLikely: 0.35 };
const getVehicle = (id) => db.getVehicle(id);
const run = (visual, words, extra = {}) => fuse({ visual, text: words ? db.match(words) : null, getVehicle, ...extra }, cfg);

test('text and visual agree -> identified, high, both bases', () => {
  const r = run([{ vehicleId: 'saipa-shahin', probability: 0.4 }, { vehicleId: 'ikco-tara', probability: 0.3 }], ['SAIPA', 'SHAHIN']);
  assert.equal(r.verdict, 'identified');
  assert.equal(r.vehicleId, 'saipa-shahin');
  assert.equal(r.level, 'high');
  assert.deepEqual(r.basis, ['visual', 'text']);
});

test('text names a vehicle the visual model only has in its top-3 -> still agreement', () => {
  const r = run([{ vehicleId: 'ikco-tara', probability: 0.5 }, { vehicleId: 'ikco-dena', probability: 0.2 }, { vehicleId: 'saipa-shahin', probability: 0.1 }], ['SAIPA', 'SHAHIN']);
  assert.equal(r.verdict, 'identified');
  assert.equal(r.vehicleId, 'saipa-shahin');
});

test('confident visual and identified text disagree -> uncertain, both kept as candidates', () => {
  const r = run([{ vehicleId: 'ikco-tara', probability: 0.9 }], ['TOYOTA', 'COROLLA']);
  assert.equal(r.verdict, 'uncertain');
  assert.equal(r.reason, 'text-and-visual-conflict');
  assert.ok(r.candidates.some((c) => c.vehicleId === 'ikco-tara'));
  assert.ok(r.candidates.some((c) => c.vehicleId === 'toyota-corolla' && c.text));
});

test('identified text with an unsure visual model -> identified from text, medium', () => {
  const r = run([{ vehicleId: 'ikco-tara', probability: 0.3 }], ['LAMARI', 'NEO']);
  assert.equal(r.verdict, 'identified');
  assert.equal(r.vehicleId, 'lamari-neo');
  assert.equal(r.level, 'medium');
  assert.deepEqual(r.basis, ['text']);
});

test('confident visual, no text -> identified medium (visual only)', () => {
  const r = run([{ vehicleId: 'ikco-dena', probability: 0.8 }], []);
  assert.equal(r.verdict, 'identified');
  assert.equal(r.level, 'medium');
  assert.deepEqual(r.basis, ['visual']);
});

test('confident visual + matching brand text -> high', () => {
  const r = run([{ vehicleId: 'ikco-dena', probability: 0.8 }], ['IKCO']);
  assert.equal(r.verdict, 'identified');
  assert.equal(r.level, 'high');
});

test('visual says one brand, badge text says another brand -> uncertain', () => {
  const r = run([{ vehicleId: 'ikco-dena', probability: 0.8 }], ['TOYOTA']);
  assert.equal(r.verdict, 'uncertain');
  assert.equal(r.reason, 'text-brand-contradicts-visual');
});

test('medium visual probability -> likely (never identified)', () => {
  const r = run([{ vehicleId: 'saipa-tiba', probability: 0.45 }], null);
  assert.equal(r.verdict, 'likely');
  assert.equal(r.level, 'low');
});

test('open-set rejection: a high probability is not trusted when the image is far from every class', () => {
  const r = run([{ vehicleId: 'saipa-tiba', probability: 0.95 }], null, { visualOpenSetOk: false });
  assert.equal(r.verdict, 'uncertain');
  assert.equal(r.reason, 'unknown-vehicle');
});

test('engine badge alone never identifies; nothing at all -> unknown', () => {
  assert.equal(run([], ['1.5']).verdict, 'unknown');
  assert.equal(run([], null).verdict, 'unknown');
});
