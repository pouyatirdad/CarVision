// Prediction fusion: combines the rear-view visual model's candidates with the OCR/database evidence.
// Pure logic (no DOM/ORT) so it can be unit-tested.
//
// Principles (see docs/REAR-VIEW.md):
//  * scores from unrelated models are never averaged or compared directly. The visual model gives a calibrated
//    probability over its own classes; the text matcher gives an evidence category (alias / brand+model / ...).
//    Fusion is a small decision table over those categories.
//  * text that fully names a vehicle (an "identified" database match) is strong evidence, but text can also come
//    from another car, a plate frame or a sticker. It wins on its own only when the image does not confidently
//    say something else.
//  * agreement between independent evidence raises the confidence LEVEL; it never inflates a number.

import { REAR } from '../config.js';

/**
 * @param {object} input
 * @param {Array<{vehicleId:string, probability:number}>} input.visual rear-view model candidates, best first
 *        (probability = calibrated softmax over the model's classes); [] when the model did not run.
 * @param {boolean} [input.visualOpenSetOk=true] false when the image is too far from every known class (open-set
 *        rejection), i.e. the top candidate must not be trusted as an identification.
 * @param {object|null} input.text result of VehicleDatabase.match() on the OCR words, or null when OCR did not run.
 * @param {(id:string)=>object|null} input.getVehicle database lookup (for brand comparisons)
 * @returns {{
 *   verdict: 'identified'|'likely'|'uncertain'|'unknown',
 *   vehicleId: string|null,
 *   level: 'high'|'medium'|'low'|null,
 *   basis: Array<'visual'|'text'>,
 *   reason: string,
 *   candidates: Array<{vehicleId:string, visual:number|null, text:string|null}>
 * }}
 */
export function fuse({ visual = [], visualOpenSetOk = true, text = null, getVehicle = () => null }, cfg = REAR) {
  const v1 = visual[0] ?? null;
  const visualSure = !!v1 && visualOpenSetOk && v1.probability >= cfg.visualAccept;
  const visualLikely = !!v1 && visualOpenSetOk && v1.probability >= cfg.visualLikely;
  const inVisualTop = (id, k = 3) => visual.slice(0, k).some((c) => c.vehicleId === id);
  const visualProb = (id) => visual.find((c) => c.vehicleId === id)?.probability ?? null;

  const textBest = text?.best ?? null; // a single "identified" vehicle (full alias or brand + model)
  const textPossible = text?.candidates?.filter((c) => c.status === 'possible') ?? [];
  const brandOf = (id) => getVehicle(id)?.brand ?? null;
  const textBrands = new Set([
    ...(text?.brandHints ?? []).map((b) => b.brand),
    ...(text?.candidates ?? []).filter((c) => c.brandMatched).map((c) => c.vehicle.brand),
  ]);

  const candidates = [];
  const addCand = (id, textLabel = null) => {
    if (!id) return;
    const prev = candidates.find((c) => c.vehicleId === id);
    if (prev) prev.text ||= textLabel;
    else candidates.push({ vehicleId: id, visual: visualProb(id), text: textLabel });
  };
  const textLabel = (c) => (c.status === 'identified' ? 'badge text names this vehicle' : 'badge text partly matches');

  // 1) Text identifies a vehicle.
  if (textBest) {
    const id = textBest.vehicle.id;
    addCand(id, textLabel(textBest));
    visual.slice(0, 3).forEach((c) => addCand(c.vehicleId));
    if (inVisualTop(id)) {
      return { verdict: 'identified', vehicleId: id, level: 'high', basis: ['visual', 'text'], reason: 'text-and-visual-agree', candidates };
    }
    if (visualSure && v1.vehicleId !== id) {
      // Strong, contradicting evidence on both sides: do not pick one.
      return { verdict: 'uncertain', vehicleId: null, level: null, basis: ['visual', 'text'], reason: 'text-and-visual-conflict', candidates };
    }
    const visualCovers = !!getVehicle(id)?.rearRecognition?.supported;
    return { verdict: 'identified', vehicleId: id, level: 'medium', basis: ['text'], reason: visualCovers ? 'text-only-visual-unsure' : 'text-only-not-in-visual-model', candidates };
  }

  // 2) No text identification: visual model decides, text can support or contradict.
  for (const c of visual.slice(0, 3)) addCand(c.vehicleId);
  for (const c of textPossible.slice(0, 2)) addCand(c.vehicle.id, textLabel(c));

  if (visualLikely) {
    const id = v1.vehicleId;
    const textSupports = textPossible.some((c) => c.vehicle.id === id) || (textBrands.size > 0 && textBrands.has(brandOf(id)));
    const textContradicts = !textSupports && textBrands.size > 0; // badge text names a different brand
    if (textContradicts) {
      return { verdict: 'uncertain', vehicleId: null, level: null, basis: ['visual', 'text'], reason: 'text-brand-contradicts-visual', candidates };
    }
    if (visualSure) {
      return { verdict: 'identified', vehicleId: id, level: textSupports ? 'high' : 'medium', basis: textSupports ? ['visual', 'text'] : ['visual'], reason: textSupports ? 'visual-with-text-support' : 'visual-only', candidates };
    }
    return { verdict: 'likely', vehicleId: id, level: textSupports ? 'medium' : 'low', basis: textSupports ? ['visual', 'text'] : ['visual'], reason: textSupports ? 'visual-likely-text-support' : 'visual-likely', candidates };
  }

  if (candidates.length) return { verdict: 'uncertain', vehicleId: null, level: null, basis: [], reason: visualOpenSetOk ? 'no-confident-evidence' : 'unknown-vehicle', candidates };
  return { verdict: 'unknown', vehicleId: null, level: null, basis: [], reason: 'no-evidence', candidates };
}
