// Compares what the vehicle database found in the OCR text with the image classifiers' answer.
// Display-only: it never changes the classifier result (OCR and classification stay independent).
// Pure logic, no DOM.

/** Database id of the vehicle named by a classifier label such as "Iran Khodro Tara", or null if not in the database. */
export function classifierVehicleId(db, title) {
  return db.match(title).best?.vehicle.id ?? null;
}

/**
 * @param {import('./vehicle-database.js').VehicleDatabase} db
 * @param {object} ocrMatch result of db.match(ocrWords)
 * @param {string|null} classifierTitle display title of the classifier's primary result (null when uncertain)
 * @returns {{status:'agrees'|'differs'|'ocr-only'|'none', ocr: object|null, classifierId: string|null}}
 */
export function compareEvidence(db, ocrMatch, classifierTitle) {
  const ocr = ocrMatch.best ?? ocrMatch.candidates[0] ?? null;
  const classifierId = classifierTitle ? classifierVehicleId(db, classifierTitle) : null;
  if (!ocr) return { status: 'none', ocr: null, classifierId };
  if (classifierId && ocr.vehicle.id === classifierId) return { status: 'agrees', ocr, classifierId };
  if (classifierId && ocr.status === 'identified') return { status: 'differs', ocr, classifierId };
  return { status: 'ocr-only', ocr, classifierId };
}
