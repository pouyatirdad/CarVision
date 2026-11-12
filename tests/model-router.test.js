import test from 'node:test';
import assert from 'node:assert/strict';
import { decide, needsIranianModel } from '../src/ai/model-router.js';
import { ROUTING } from '../src/config.js';

const c = (domain, p, raw = 'x') => ({ domain, classId: 0, probability: p, label: { raw } });

test('confident international result is accepted without consulting the Iranian model', () => {
  assert.equal(needsIranianModel(0.95), false);
  const r = decide({ international: [c('international', 0.95, 'BMW'), c('international', 0.02), c('international', 0.01)] });
  assert.equal(r.verdict, 'international');
  assert.equal(r.primary.label.raw, 'BMW');
  assert.equal(r.alternatives.length, 2);
});

test('falls through to the Iranian model when the international model is unsure', () => {
  assert.equal(needsIranianModel(ROUTING.internationalAccept - 0.01), true);
  const r = decide({
    international: [c('international', 0.4), c('international', 0.1)],
    iranian: [c('iranian', 0.9, 'Dena'), c('iranian', 0.05)],
  });
  assert.equal(r.verdict, 'iranian');
  assert.equal(r.primary.label.raw, 'Dena');
});

test('high raw Iranian confidence does NOT override a confident international result', () => {
  const r = decide({
    international: [c('international', 0.9, 'Audi'), c('international', 0.02)],
    iranian: [c('iranian', 0.99, 'Shahin'), c('iranian', 0.01)],
  });
  assert.equal(r.verdict, 'international');
});

test('neither model confident -> uncertain, mixed candidates, no primary', () => {
  const r = decide({
    international: [c('international', 0.5), c('international', 0.2)],
    iranian: [c('iranian', 0.5), c('iranian', 0.3)],
  });
  assert.equal(r.verdict, 'uncertain');
  assert.equal(r.primary, null);
  assert.equal(r.alternatives.length, 3);
  assert.ok(new Set(r.alternatives.map((a) => a.domain)).size === 2);
});
