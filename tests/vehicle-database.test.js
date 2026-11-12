import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { VehicleDatabase, loadVehicleDatabase, normalizeText, validateVehicles } from '../src/ai/vehicle-database.js';

const read = (f) => JSON.parse(readFileSync(new URL(`../public/data/${f}`, import.meta.url), 'utf8'));
const vehiclesJson = read('vehicles.json');
const db = VehicleDatabase.fromData(vehiclesJson, read('manufacturers.json'));

const ids = (r) => r.candidates.map((c) => c.vehicle.id);

test('shipped database validates and ids are unique', () => {
  const { errors } = validateVehicles(vehiclesJson);
  assert.deepEqual(errors, []);
});

// ---- the 10 required scenarios --------------------------------------------------------------------------

test('1. LAMARI + NEO (+1.5) -> Lamari Neo, identified', () => {
  const r = db.match(['LAMARI', 'NEO', '1.5']);
  assert.equal(r.best?.vehicle.id, 'lamari-neo');
  assert.equal(r.best.vehicle.brand, 'Lamari');
  assert.equal(r.best.vehicle.model, 'Neo');
  assert.ok(r.best.engineBadgeMatched);
  assert.equal(ids(r).includes('lamari-eama'), false);
});

test('2. DENA PLUS -> Dena Plus (not plain Dena)', () => {
  const r = db.match('DENA PLUS');
  assert.equal(r.best?.vehicle.id, 'ikco-dena-plus');
  const dena = r.candidates.find((c) => c.vehicle.id === 'ikco-dena');
  assert.ok(!dena || dena.status !== 'identified');
  assert.equal(db.match('DENA+').best?.vehicle.id, 'ikco-dena-plus');
});

test('3. PEUGEOT + 206 -> Peugeot 206', () => {
  const r = db.match(['PEUGEOT', '206']);
  assert.equal(r.best?.vehicle.id, 'ikco-peugeot-206');
  assert.equal(r.candidates.filter((c) => c.status === 'identified').length, 1);
});

test('4. SAIPA + SHAHIN -> Saipa Shahin', () => {
  assert.equal(db.match(['SAIPA', 'SHAHIN']).best?.vehicle.id, 'saipa-shahin');
});

test('5. TOYOTA + COROLLA (+1.8) -> Toyota Corolla', () => {
  const r = db.match(['TOYOTA', 'COROLLA', '1.8']);
  assert.equal(r.best?.vehicle.id, 'toyota-corolla');
  assert.ok(r.best.engineBadgeMatched);
});

test('6. BMW + 320I -> BMW 3 Series', () => {
  assert.equal(db.match(['BMW', '320I']).best?.vehicle.id, 'bmw-3-series');
});

test('7. MERCEDES + C200 -> Mercedes-Benz C-Class', () => {
  assert.equal(db.match(['MERCEDES', 'C200']).best?.vehicle.id, 'mercedes-c-class');
  assert.equal(db.match(['MERCEDES BENZ', 'C200']).best?.vehicle.id, 'mercedes-c-class');
});

test('8. unknown model -> no identification, brand reported only', () => {
  const r = db.match(['HYUNDAI', 'SOLARIS', '1.6']);
  assert.equal(r.best, null);
  assert.deepEqual(ids(r), []);
  assert.equal(r.brandHints[0].brand, 'Hyundai');
  const none = db.match(['ZXQW', 'PLORB']);
  assert.deepEqual([ids(none), none.weak, none.brandHints], [[], [], []]);
});

test('9. manufacturer name only never names a model', () => {
  for (const text of ['TOYOTA', 'LAMARI', 'PEUGEOT', 'SAIPA', 'IKCO', 'KIA']) {
    const r = db.match(text);
    assert.equal(r.best, null, text);
    assert.deepEqual(ids(r), [], text);
    assert.equal(r.brandHints.length, 1, text);
  }
  const toyota = db.match('TOYOTA').brandHints[0];
  assert.ok(toyota.vehicleIds.includes('toyota-corolla') && toyota.vehicleIds.length === 3);
});

test('10. engine badge only never identifies a vehicle', () => {
  for (const text of ['1.5', '2.0', '1.6T', '1.5T', '1.8', 'EF7', 'TU5', 'XU7', '2.0 1.6']) {
    const r = db.match(text);
    assert.equal(r.best, null, text);
    assert.deepEqual(ids(r), [], text);
  }
  const r = db.match(['1.5', '1.8']);
  assert.deepEqual(r.engineBadgesSeen, ['1.5', '1.8']);
});

// ---- ambiguity / false-positive guards --------------------------------------------------------------------

test('brand + engine only (TOYOTA 1.8) does not claim Corolla', () => {
  const r = db.match(['TOYOTA', '1.8']);
  assert.equal(r.best, null);
  assert.deepEqual(ids(r), []);
});

test('Samand vs Samand Soren vs Soren Plus, 405 vs 405 Pars, Tiba vs Tiba 2', () => {
  assert.equal(db.match('SAMAND SOREN').best?.vehicle.id, 'ikco-soren');
  assert.equal(db.match('SAMAND').best?.vehicle.id ?? null, null); // bare model = "possible", not identified
  assert.equal(db.match('SAMAND').candidates[0].vehicle.id, 'ikco-samand');
  assert.equal(db.match('405 PARS').best?.vehicle.id, 'ikco-peugeot-pars');
  assert.equal(db.match('TIBA 2').best?.vehicle.id, 'saipa-tiba2');
  assert.equal(db.match('TIBA2').best?.vehicle.id, 'saipa-tiba2');
  assert.equal(db.match('TIBA').candidates[0].vehicle.id, 'saipa-tiba');
});

test('similar numbers do not fuzzy-match (206 vs 207 vs 405)', () => {
  const r = db.match('206');
  assert.deepEqual(ids(r), ['ikco-peugeot-206']);
  assert.equal(r.candidates[0].status, 'possible');
});

test('brand contradicting the model lowers confidence', () => {
  const r = db.match(['TOYOTA', '206']);
  assert.equal(r.best, null);
});

test('short generic badges are never more than weak hints', () => {
  const r = db.match('FX');
  assert.equal(r.best, null);
  assert.deepEqual(ids(r), []);
  assert.equal(db.match(['FOWNIX', 'FX']).best?.vehicle.id, 'fownix-fx');
});

test('typo tolerance only for long letter-only words', () => {
  const r = db.match(['TOYOTA', 'COROLA']);
  assert.equal(r.candidates[0].vehicle.id, 'toyota-corolla');
  assert.equal(r.candidates[0].matchType, 'fuzzy');
});

// ---- normalisation ----------------------------------------------------------------------------------------

test('normalisation: examples from the spec', () => {
  assert.equal(db.normalize('LAMAR1 NEO'), 'LAMARI NEO');
  assert.equal(db.normalize('DENA+'), 'DENA PLUS');
  assert.equal(db.normalize('RAV 4'), 'RAV4');
  assert.equal(db.normalize('  lamari   neo!! '), 'LAMARI NEO');
});

test('normalisation does not damage valid model names', () => {
  for (const t of ['RAV4', 'X22', '206', 'J7', 'T8', 'C200', '320I', 'EF7', '1.5T']) assert.equal(db.normalize(t), t);
  assert.equal(db.normalize('PEUGEOT 206'), 'PEUGEOT 206');
  assert.equal(db.match('NE0').best?.vehicle.id ?? db.match('NE0').candidates[0].vehicle.id, 'lamari-neo');
  assert.equal(db.normalize('2O6'), '206');
});

test('Persian / Arabic text is normalised and matched', () => {
  assert.equal(normalizeText('ي'), 'ی');
  assert.equal(normalizeText('ك'), 'ک');
  assert.equal(normalizeText('پژو ۲۰۶'), 'پژو 206');
  assert.equal(normalizeText('پژو ٢٠٦'), 'پژو 206');
  assert.equal(db.match('لاماری نئو').best?.vehicle.id, 'lamari-neo');
  assert.equal(db.match('پژو ۲۰۶').best?.vehicle.id, 'ikco-peugeot-206');
  assert.equal(db.match('دنا+').best?.vehicle.id, 'ikco-dena-plus');
  assert.equal(db.match('ایران خودرو').best, null);
});

test('low-confidence OCR words are ignored', () => {
  const r = db.match([{ text: 'TOYOTA', confidence: 0.9 }, { text: 'COROLLA', confidence: 0.1 }]);
  assert.equal(r.best, null);
});

test('RAV 4 and RAV4 both match', () => {
  assert.equal(db.match(['TOYOTA', 'RAV', '4']).best?.vehicle.id, 'toyota-rav4');
  assert.equal(db.match('RAV4').candidates[0].vehicle.id, 'toyota-rav4');
});

// ---- data-driven (no code change needed) ------------------------------------------------------------------

test('a new record added to the JSON is matched without any code change', () => {
  const extra = structuredClone(vehiclesJson);
  extra.vehicles.push({ id: 'test-zeta', brand: 'Testco', model: 'Zeta', aliases: ['TESTCO ZETA', 'ZETA'], badgeTexts: ['ZETA'], engineBadges: ['1.2'] });
  const d2 = VehicleDatabase.fromData(extra);
  assert.equal(d2.match(['TESTCO', 'ZETA']).best?.vehicle.id, 'test-zeta');
});

test('validation reports duplicate ids and bad fields', () => {
  const bad = { vehicles: [{ id: 'a', brand: 'X', model: 'Y' }, { id: 'a', brand: 'X', model: '', aliases: 'nope', yearFrom: 2020, yearTo: 2010 }] };
  const { errors } = validateVehicles(bad);
  assert.ok(errors.some((e) => /duplicate id/.test(e)));
  assert.ok(errors.some((e) => /"model"/.test(e)));
  assert.ok(errors.some((e) => /"aliases"/.test(e)));
  assert.ok(errors.some((e) => /yearFrom/.test(e)));
  assert.throws(() => VehicleDatabase.fromData(bad), /Invalid vehicle database/);
});

test('loadVehicleDatabase fetches from the BASE_URL sub-path and caches', async () => {
  const urls = [];
  const fetchFn = async (url) => {
    urls.push(url);
    const file = url.split('/data/')[1];
    return { ok: true, json: async () => read(file) };
  };
  const a = await loadVehicleDatabase({ baseUrl: '/CarVision/', fetchFn, force: true });
  const b = await loadVehicleDatabase({ baseUrl: '/CarVision/', fetchFn });
  assert.equal(a, b);
  assert.deepEqual(urls, ['/CarVision/data/vehicles.json', '/CarVision/data/manufacturers.json']);
  await assert.rejects(loadVehicleDatabase({ fetchFn: async () => ({ ok: false }), force: true }), /Failed to load vehicle database/);
});
