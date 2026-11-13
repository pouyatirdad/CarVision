// Planning and merging of the multiple OCR attempts made on a rear-view photo. Pure logic (no DOM, no
// tesseract) so it can be unit-tested; ocr-engine.js executes the plan with canvas + tesseract.js.
//
// Rear badges are small: brand logo text in the middle of the trunk lid, the model name on one side, an engine
// / trim badge on the other side. Reading the whole vehicle crop once misses most of them, so we read:
//   1. the original crop,
//   2. a grayscale + contrast-stretched version,
//   3. a sharpened version,
//   4. upscaled "badge zones" of the rear (left / centre / right of the trunk-lid band),
//   5. upscaled boxes around words that an earlier pass saw with too little confidence.
// The words of all passes are merged; a word seen by several passes gets a small confidence bonus.

/** Image variants applied to a region before OCR. */
export const VARIANTS = ['original', 'contrast', 'sharpen'];

/**
 * Badge zones relative to the vehicle box (fractions of its width/height). On a rear photo the trunk lid / tailgate
 * band sits roughly between 30 % and 85 % of the box height.
 */
export const REAR_BADGE_ZONES = [
  { name: 'rear-left', x: 0.0, y: 0.3, w: 0.45, h: 0.55 },
  { name: 'rear-centre', x: 0.25, y: 0.25, w: 0.5, h: 0.55 },
  { name: 'rear-right', x: 0.55, y: 0.3, w: 0.45, h: 0.55 },
];

/**
 * @param {{x1:number,y1:number,x2:number,y2:number}} box vehicle box in image pixels
 * @param {{width:number,height:number}} image
 * @param {{maxSide?:number, maxUpscale?:number, zoneUpscale?:number}} [opts]
 * @returns {Array<{name:string, region:{x1,y1,x2,y2}, variant:string, scale:number}>} ordered plan
 */
export function planPasses(box, image, { maxSide = 1800, maxUpscale = 2, zoneUpscale = 3 } = {}) {
  const clampBox = (b) => ({
    x1: Math.max(0, Math.floor(b.x1)),
    y1: Math.max(0, Math.floor(b.y1)),
    x2: Math.min(image.width, Math.ceil(b.x2)),
    y2: Math.min(image.height, Math.ceil(b.y2)),
  });
  const full = clampBox(box);
  const fw = full.x2 - full.x1;
  const fh = full.y2 - full.y1;
  const scaleFor = (w, h, up) => Math.max(0.1, Math.min(up, maxSide / Math.max(1, w, h)));
  const plan = VARIANTS.map((variant) => ({ name: `vehicle-${variant}`, region: full, variant, scale: scaleFor(fw, fh, maxUpscale) }));
  for (const z of REAR_BADGE_ZONES) {
    const region = clampBox({ x1: full.x1 + z.x * fw, y1: full.y1 + z.y * fh, x2: full.x1 + (z.x + z.w) * fw, y2: full.y1 + (z.y + z.h) * fh });
    if (region.x2 - region.x1 < 16 || region.y2 - region.y1 < 16) continue;
    plan.push({ name: z.name, region, variant: 'sharpen', scale: scaleFor(region.x2 - region.x1, region.y2 - region.y1, zoneUpscale) });
  }
  return plan;
}

/**
 * Regions around words that were read with low confidence, to be re-read upscaled. Word boxes are in the
 * coordinates of the pass image; `pass` gives its region and scale so they can be mapped back.
 * @returns {Array<{name, region, variant, scale}>}
 */
export function refinePasses(lowWords, image, { maxRegions = 4, padding = 0.6, minHeightPx = 48, maxUpscale = 4 } = {}) {
  const out = [];
  const seen = [];
  for (const w of [...lowWords].sort((a, b) => b.confidence - a.confidence)) {
    if (out.length >= maxRegions || !w.bbox || !w.pass) continue;
    const { region, scale } = w.pass;
    const bx = { x1: region.x1 + w.bbox.x0 / scale, y1: region.y1 + w.bbox.y0 / scale, x2: region.x1 + w.bbox.x1 / scale, y2: region.y1 + w.bbox.y1 / scale };
    const bw = bx.x2 - bx.x1;
    const bh = bx.y2 - bx.y1;
    if (bw <= 1 || bh <= 1) continue;
    const pad = Math.max(bw, bh) * padding;
    const r = {
      x1: Math.max(0, Math.floor(bx.x1 - pad)),
      y1: Math.max(0, Math.floor(bx.y1 - pad)),
      x2: Math.min(image.width, Math.ceil(bx.x2 + pad)),
      y2: Math.min(image.height, Math.ceil(bx.y2 + pad)),
    };
    if (seen.some((s) => iou(s, r) > 0.5)) continue;
    seen.push(r);
    const s = Math.min(maxUpscale, Math.max(1, minHeightPx / bh));
    out.push({ name: 'word-zoom', region: r, variant: 'sharpen', scale: s });
  }
  return out;
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

// Same text from different passes -> same key; '.' is kept only inside numbers such as "1.5".
const key = (t) => t.toUpperCase().replace(/[^\p{L}\p{N}.]/gu, '').replace(/(?<!\d)\.|\.(?!\d)/g, '');

/**
 * Merge words from several passes: one entry per distinct text, best confidence kept, +0.05 per additional pass
 * that saw the same text (capped at 0.99), so text confirmed by independent variants ranks higher.
 * @param {Array<{pass:string, words:Array<{text:string, confidence:number}>}>} results
 * @param {{minConfidence?:number, minLength?:number}} [opts]
 * @returns {Array<{text:string, confidence:number, passes:string[]}>} in READING ORDER (order of first appearance; passes run whole-vehicle first)
 */
export function mergePassWords(results, { minConfidence = 0.6, minLength = 2 } = {}) {
  const byKey = new Map();
  for (const r of results) {
    for (const w of r.words) {
      const k = key(w.text);
      if ([...k].length < minLength && !/^\d$/.test(k)) continue; // a single digit can be a model name part (TIGGO 8)
      const e = byKey.get(k);
      if (!e) byKey.set(k, { text: w.text.trim(), confidence: w.confidence, passes: [r.pass] });
      else {
        if (w.confidence > e.confidence) {
          e.confidence = w.confidence;
          e.text = w.text.trim();
        }
        if (!e.passes.includes(r.pass)) e.passes.push(r.pass);
      }
    }
  }
  return [...byKey.values()]
    .map((e) => ({ ...e, confidence: Math.min(0.99, e.confidence + 0.05 * (e.passes.length - 1)) }))
    .filter((e) => e.confidence >= minConfidence); // never re-sort: multi-word names ("TIGGO 8 PRO") must stay adjacent
}
