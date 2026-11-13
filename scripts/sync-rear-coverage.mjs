// Write the rear-view model's ACTUAL coverage into public/data/vehicles.json (rearRecognition field), from
// public/labels/rear-head.json (which classes the model can output) and the training report (which were tested).
//   node scripts/sync-rear-coverage.mjs [report.json]
// Database coverage and visual coverage are kept separate on purpose; this script is the only writer.
import fs from 'node:fs';

const root = new URL('../', import.meta.url);
const headPath = new URL('public/labels/rear-head.json', root);
const dbPath = new URL('public/data/vehicles.json', root);
const reportPath = process.argv[2] ?? null;

const head = JSON.parse(fs.readFileSync(headPath, 'utf8'));
const db = JSON.parse(fs.readFileSync(dbPath, 'utf8'));
const report = reportPath ? JSON.parse(fs.readFileSync(reportPath, 'utf8')) : null;

const byVehicle = new Map();
for (const c of head.classes) {
  const e = byVehicle.get(c.vehicleId) ?? { kind: 'zero-shot', trainImages: 0, keys: [] };
  if (c.kind === 'trained') e.kind = 'trained';
  e.trainImages += c.trainImages ?? 0;
  e.keys.push(c.key);
  byVehicle.set(c.vehicleId, e);
}

const missing = [...byVehicle.keys()].filter((id) => !db.vehicles.some((v) => v.id === id));
if (missing.length) throw new Error(`rear-head.json references vehicles that are not in the database: ${missing.join(', ')}`);

for (const v of db.vehicles) {
  const e = byVehicle.get(v.id);
  if (!e) {
    v.rearRecognition = { supported: false, tested: false, modelId: null };
    continue;
  }
  // tested = held-out rear test images exist (SIVD test split per class key, or the extra Commons set per vehicle)
  const perClass = report?.per_class ?? {};
  const nSivd = e.keys.reduce((s, k) => s + (perClass[k]?.n_test ?? 0), 0);
  const extra = report?.extra_per_vehicle?.[v.id] ?? null;
  const testImages = nSivd + (extra?.n ?? 0);
  v.rearRecognition = {
    supported: true,
    tested: testImages > 0,
    modelId: head.modelId,
    kind: e.kind,
    trainImages: e.trainImages,
    testImages,
  };
}

fs.writeFileSync(dbPath, `{\n  "version": "${db.version}",\n  "vehicles": [\n${db.vehicles.map((r) => '    ' + JSON.stringify(r)).join(',\n')}\n  ]\n}\n`);
const s = db.vehicles.filter((v) => v.rearRecognition.supported);
console.log(`database: ${db.vehicles.length} vehicles; visual (rear) model: ${s.length} supported, ${s.filter((v) => v.rearRecognition.kind === 'trained').length} trained, ${s.filter((v) => v.rearRecognition.tested).length} tested`);
