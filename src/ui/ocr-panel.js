import { formatPercent } from '../utils/formatting.js';
import { h, put } from './results.js';

function vehicleLine(c) {
  const v = c.vehicle;
  const bits = [v.category, v.country].filter(Boolean).join(' · ');
  return h('li', {}, h('strong', {}, `${v.brand} ${v.model}`), bits ? ` – ${bits}` : '', ' ', h('span', { class: `badge ${c.status === 'identified' ? 'iran' : 'warn'}` }, c.status === 'identified' ? 'matches database' : 'possible'));
}

/** How the badge text relates to the final (fused) answer. */
function relation(fusion) {
  if (!fusion) return null;
  if (fusion.reason === 'text-and-visual-agree' || fusion.reason === 'visual-with-text-support' || fusion.reason === 'visual-likely-text-support') {
    return ['info', 'The badge text supports what the rear-view model sees.'];
  }
  if (fusion.reason === 'text-and-visual-conflict' || fusion.reason === 'text-brand-contradicts-visual') {
    return ['warn', 'The badge text points to a different vehicle than the rear-view model. Please check the photo yourself.'];
  }
  if (fusion.basis.length === 1 && fusion.basis[0] === 'text') return ['info', 'The result above comes from this text; the rear-view model was not sure.'];
  return null;
}

/** state: {phase:'loading', label, progress?} | {phase:'done', words, match, fusion, passes} | {phase:'error', message} */
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

  const { words, match, fusion, passes = [] } = state;
  const attempts = passes.length ? h('p', { class: 'muted small' }, `${passes.length} reading attempts (original, contrast, sharpened, enlarged badge zones) in ${(passes.reduce((s, p) => s + p.ms, 0) / 1000).toFixed(1)} s.`) : null;
  if (!words.length) {
    put(root, h('p', { class: 'muted' }, 'No readable badge text was found. Rear badges are often small, chrome or removed – recognition then relies on the rear appearance only.'), attempts);
    return;
  }
  const rel = relation(fusion);
  put(
    root,
    match.candidates.length
      ? h('ul', { class: 'alts' }, match.candidates.slice(0, 3).map(vehicleLine))
      : h('p', { class: 'muted' }, match.brandHints.length ? `Brand seen: ${match.brandHints.map((b) => b.brand).join(', ')} – but no model text, so no model is claimed.` : 'No vehicle name or badge from the local database was recognised in the text.'),
    match.ambiguous ? h('div', { class: 'alert warn', role: 'note' }, 'The text fits more than one vehicle equally well.') : null,
    match.engineBadgesSeen.length ? h('p', { class: 'muted small' }, `Engine labels seen (not used to identify the car): ${match.engineBadgesSeen.join(', ')}`) : null,
    rel ? h('div', { class: `alert ${rel[0]}`, role: 'note' }, rel[1]) : null,
    h('details', { class: 'muted small' }, h('summary', {}, `All recognised words (${words.length})`), h('ul', { class: 'chips' }, words.map((w) => h('li', { title: `OCR confidence ${formatPercent(w.confidence)}${w.passes ? ` · seen in ${w.passes.length} attempt(s)` : ''}` }, w.text)))),
    attempts,
    h('p', { class: 'muted small' }, 'Text recognition (OCR) is a clue, not a verdict: it can misread letters, and words on the car may belong to something else (stickers, plates, dealer frames).'),
  );
}
