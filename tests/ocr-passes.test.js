import test from 'node:test';
import assert from 'node:assert/strict';
import { grayToRGBA, sharpen, stretchContrast, toGray } from '../src/utils/image-ops.js';
import { REAR_BADGE_ZONES, VARIANTS, mergePassWords, planPasses, refinePasses } from '../src/ai/ocr-passes.js';

test('toGray uses BT.601 luma and grayToRGBA round-trips', () => {
  const g = toGray(new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 255]));
  assert.deepEqual([...g], [255, 0, 76]);
  assert.deepEqual([...grayToRGBA(g)].slice(0, 8), [255, 255, 255, 255, 0, 0, 0, 255]);
});

test('stretchContrast maps a narrow range to the full range and ignores outliers', () => {
  const gray = new Uint8Array(1000).map((_, i) => 100 + (i % 21)); // 100..120
  gray[0] = 255; // one chrome reflection
  const out = stretchContrast(gray, 0.01, 0.99);
  assert.ok(Math.min(...out) === 0);
  assert.ok(Math.max(...out.slice(1)) === 255);
  // flat image stays unchanged
  assert.deepEqual([...stretchContrast(new Uint8Array([7, 7, 7]))], [7, 7, 7]);
});

test('sharpen keeps flat regions and boosts an edge', () => {
  const w = 5;
  const h = 3;
  const flat = new Uint8Array(w * h).fill(100);
  assert.deepEqual([...sharpen(flat, w, h, 1)], [...flat]);
  const edge = new Uint8Array(w * h).map((_, i) => (i % w < 2 ? 50 : 200));
  const s = sharpen(edge, w, h, 1);
  assert.ok(s[1 * w + 1] < 50, 'dark side of the edge gets darker');
  assert.ok(s[1 * w + 2] > 200, 'bright side of the edge gets brighter');
});

test('planPasses: whole-vehicle variants then the three rear badge zones, inside the image', () => {
  const plan = planPasses({ x1: 100, y1: 50, x2: 700, y2: 450 }, { width: 800, height: 500 }, { maxSide: 1800, maxUpscale: 2, zoneUpscale: 3 });
  assert.deepEqual(plan.slice(0, 3).map((p) => p.variant), VARIANTS);
  assert.equal(plan.length, 3 + REAR_BADGE_ZONES.length);
  for (const p of plan) {
    assert.ok(p.region.x1 >= 0 && p.region.y1 >= 0 && p.region.x2 <= 800 && p.region.y2 <= 500);
    assert.ok(Math.max(p.region.x2 - p.region.x1, p.region.y2 - p.region.y1) * p.scale <= 1800 + 1);
  }
  assert.equal(plan[0].scale, 2); // 600 px wide box -> upscaled 2x (cap)
  const centre = plan.find((p) => p.name === 'rear-centre');
  assert.ok(centre.scale === 3 && centre.region.y1 > 50 && centre.region.y2 < 450);
});

test('planPasses clamps a box that leaves the image and skips tiny zones', () => {
  const plan = planPasses({ x1: -20, y1: -10, x2: 30, y2: 20 }, { width: 100, height: 100 });
  assert.deepEqual(plan[0].region, { x1: 0, y1: 0, x2: 30, y2: 20 });
  assert.equal(plan.length, 3); // zones would be < 16 px
});

test('refinePasses maps word boxes back to image coordinates, pads, upscales and de-duplicates', () => {
  const pass = { region: { x1: 100, y1: 100, x2: 500, y2: 400 }, scale: 2 };
  const words = [
    { text: 'TIGG0', confidence: 0.5, bbox: { x0: 200, y0: 100, x1: 300, y1: 120 }, pass },
    { text: 'TIGG', confidence: 0.4, bbox: { x0: 202, y0: 101, x1: 298, y1: 121 }, pass }, // same place
    { text: 'X', confidence: 0.45, bbox: { x0: 0, y0: 0, x1: 0, y1: 0 }, pass }, // degenerate
  ];
  const r = refinePasses(words, { width: 1000, height: 1000 });
  assert.equal(r.length, 1);
  const g = r[0].region;
  assert.ok(g.x1 < 200 && g.x2 > 250 && g.y1 < 150 && g.y2 > 160); // word at x 200..250, y 150..160 in the image
  assert.equal(r[0].scale, 4); // 10 px high word -> capped upscale
});

test('mergePassWords keeps the best reading and rewards agreement between passes', () => {
  const merged = mergePassWords([
    { pass: 'a', words: [{ text: 'TIGGO', confidence: 0.55 }, { text: 'xx', confidence: 0.9 }] },
    { pass: 'b', words: [{ text: 'tiggo', confidence: 0.58 }, { text: '7', confidence: 0.99 }, { text: 'Q', confidence: 0.99 }] },
    { pass: 'c', words: [{ text: 'TIGGO.', confidence: 0.5 }, { text: 'PRO', confidence: 0.7 }] },
  ], { minConfidence: 0.6, minLength: 2 });
  const tiggo = merged.find((w) => w.text.toUpperCase().startsWith('TIGGO'));
  assert.ok(tiggo, 'seen by 3 passes below the threshold, accepted through agreement');
  assert.equal(tiggo.passes.length, 3);
  assert.ok(Math.abs(tiggo.confidence - 0.68) < 1e-9);
  assert.ok(!merged.some((w) => w.text === 'Q'), 'single letters dropped');
  assert.ok(merged.some((w) => w.text === '7'), 'single digits kept (TIGGO 7, MAZDA 3)');
  assert.ok(merged.findIndex((w) => w.text === 'xx') < merged.findIndex((w) => w.text === 'PRO'));
});

test('plate digits alone never suggest a vehicle whose model name is a number (MAZDA 3)', async () => {
  const { readFileSync } = await import('node:fs');
  const { VehicleDatabase } = await import('../src/ai/vehicle-database.js');
  const read = (f) => JSON.parse(readFileSync(new URL(`../public/data/${f}`, import.meta.url), 'utf8'));
  const db = VehicleDatabase.fromData(read('vehicles.json'), read('manufacturers.json'));
  const m = db.match(['54', '3', '12']);
  assert.equal(m.candidates.length, 0);
  assert.equal(m.best, null);
  assert.equal(db.match(['MAZDA', '3']).best?.vehicle.id, 'bahman-mazda-3');
  assert.equal(db.match(['TIGGO', '8', 'PRO']).best?.vehicle.id, 'fownix-tiggo-8-pro');
});

test('mergePassWords keeps reading order so multi-word names stay adjacent', () => {
  const merged = mergePassWords([
    { pass: 'a', words: [{ text: 'TIGGO', confidence: 0.9 }, { text: '8', confidence: 0.99 }, { text: 'PRO', confidence: 0.95 }] },
    { pass: 'b', words: [{ text: '54', confidence: 0.8 }, { text: 'PRO', confidence: 0.99 }] },
  ]);
  assert.deepEqual(merged.map((w) => w.text), ['TIGGO', '8', 'PRO', '54']);
});
