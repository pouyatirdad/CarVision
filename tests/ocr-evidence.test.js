import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { VehicleDatabase } from '../src/ai/vehicle-database.js';
import { classifierVehicleId, compareEvidence } from '../src/ai/ocr-evidence.js';

const read = (f) => JSON.parse(readFileSync(new URL(`../public/data/${f}`, import.meta.url), 'utf8'));
const db = VehicleDatabase.fromData(read('vehicles.json'), read('manufacturers.json'));

test('classifier labels map to database vehicles', () => {
  assert.equal(classifierVehicleId(db, 'Iran Khodro Tara'), 'ikco-tara');
  assert.equal(classifierVehicleId(db, 'Toyota Camry Sedan'), 'toyota-camry');
  assert.equal(classifierVehicleId(db, 'Peugeot 206'), 'ikco-peugeot-206');
  assert.equal(classifierVehicleId(db, 'Renault L90'), null); // not in the database -> no comparison
});

test('OCR text agrees with the classifier', () => {
  const r = compareEvidence(db, db.match(['TARA']), 'Iran Khodro Tara');
  assert.equal(r.status, 'agrees');
});

test('OCR identifies a different vehicle than the classifier -> differs', () => {
  const r = compareEvidence(db, db.match(['TOYOTA', 'COROLLA']), 'Toyota Camry Sedan');
  assert.equal(r.status, 'differs');
});

test('only a weak OCR guess never contradicts the classifier', () => {
  const r = compareEvidence(db, db.match(['TARA']), 'Toyota Camry Sedan');
  assert.equal(r.status, 'ocr-only');
});

test('no usable text -> none; classifier uncertain -> ocr-only', () => {
  assert.equal(compareEvidence(db, db.match(['1.5']), 'Iran Khodro Tara').status, 'none');
  assert.equal(compareEvidence(db, db.match(['LAMARI', 'NEO']), null).status, 'ocr-only');
});
