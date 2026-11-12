import { formatPercent } from '../utils/formatting.js';
import { h, put } from './results.js';

const AGREEMENT = {
  agrees: ['info', 'The text on the car agrees with the image models.'],
  differs: ['warn', 'The text on the car points to a different vehicle than the image models. Please check the photo yourself.'],
  'ocr-only': ['info', 'This comes from the text only; the image models did not name the same vehicle.'],
};

function vehicleLine(c) {
  const v = c.vehicle;
  const bits = [v.category, v.country].filter(Boolean).join(' · ');
  return h('li', {}, h('strong', {}, `${v.brand} ${v.model}`), bits ? ` – ${bits}` : '', ' ', h('span', { class: `badge ${c.status === 'identified' ? 'iran' : 'warn'}` }, c.status === 'identified' ? 'matches database' : 'possible'));
}

/** state: {phase:'loading', label, progress?} | {phase:'done', words, match, evidence} | {phase:'error', message} */
export function renderOcr(root, state) {
  root.hidden = false;
  root.replaceChildren();
  if (state.phase === 'loading') {
    put(root, h('p', { class: 'muted' }, state.label), h('progress', { max: '100', ...(state.progress != null ? { value: String(Math.round(state.progress * 100)) } : {}) }));
    return;
  }
  if (state.phase === 'error') {
    put(root, h('div', { class: 'alert warn', role: 'note' }, state.message));
    return;
  }

  const { words, match, evidence } = state;
  if (!words.length) {
    put(root, h('p', { class: 'muted' }, 'No readable text was found on the car. Badges are often too small, at an angle, or reflect light – a closer photo of the badge helps.'));
    return;
  }
  put(
    root,
    match.candidates.length
      ? h('ul', { class: 'alts' }, match.candidates.slice(0, 3).map(vehicleLine))
      : h('p', { class: 'muted' }, match.brandHints.length ? `Brand seen: ${match.brandHints.map((b) => b.brand).join(', ')} – but no model text, so no model is claimed.` : 'No vehicle name or badge from the local database was recognised in the text.'),
    match.ambiguous ? h('div', { class: 'alert warn', role: 'note' }, 'The text fits more than one vehicle equally well.') : null,
    match.engineBadgesSeen.length ? h('p', { class: 'muted small' }, `Engine labels seen (not used to identify the car): ${match.engineBadgesSeen.join(', ')}`) : null,
    AGREEMENT[evidence.status] ? h('div', { class: `alert ${AGREEMENT[evidence.status][0]}`, role: 'note' }, AGREEMENT[evidence.status][1]) : null,
    h('details', { class: 'muted small' }, h('summary', {}, `All recognised words (${words.length})`), h('ul', { class: 'chips' }, words.map((w) => h('li', { title: `OCR confidence ${formatPercent(w.confidence)}` }, w.text)))),
    h('p', { class: 'muted small' }, 'Text recognition (OCR) is a clue, not a verdict: it can misread letters, and words on the car may belong to something else (stickers, plates, shops).'),
  );
}
