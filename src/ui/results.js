import { formatMs, formatPercent } from '../utils/formatting.js';

/** Tiny DOM helper: h('div', {class:'x'}, 'text', child, ...) - always uses textContent, never innerHTML. */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}

/** append() that skips null/false (append(null) would insert the text "null"). */
export function put(parent, ...nodes) {
  parent.append(...nodes.flat(Infinity).filter((n) => n != null && n !== false));
}

/** Display fields for a legacy (secondary) classifier candidate, hiding anything the label table does not know. */
export function describe(c) {
  const l = c.label;
  if (c.domain === 'iranian') {
    return { title: l.title || l.raw, manufacturer: l.manufacturer, model: l.model, category: l.category, country: l.country };
  }
  return { title: `${l.manufacturer} ${l.model}${l.body ? ' ' + l.body : ''}`, manufacturer: l.manufacturer, model: l.model, category: l.category, country: l.country, referenceYear: l.referenceYear };
}

const vehicleTitle = (v) => (v ? `${v.brand} ${v.model}` : 'Unknown vehicle');

function meter(p, low) {
  return h('div', { class: `meter${low ? ' low' : ''}`, role: 'meter', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': String(Math.round(p * 100)) }, h('span', { style: `width:${Math.max(1, Math.round(p * 100))}%` }));
}

const WARNINGS = {
  'multiple-vehicles': 'More than one vehicle was found. Click a box on the photo to identify a different one.',
  'small-vehicle': 'The vehicle is small in the photo, so details are hard to see. Try a closer photo of the rear.',
  'no-vehicle-whole-image': 'No vehicle was detected, so the whole image was analysed. This result is very likely unreliable.',
  'viewpoint-overridden': 'The rear-view check did not recognise this as a rear photo; you chose to analyse it anyway, so the result may be unreliable.',
};

function alerts(codes) {
  return codes.filter((c) => WARNINGS[c]).map((c) => h('div', { class: `alert ${c === 'multiple-vehicles' ? 'info' : 'warn'}`, role: 'note' }, WARNINGS[c]));
}

const VERDICT = {
  identified: { high: ['ok', 'Identified · high confidence'], medium: ['iran', 'Identified'] },
  likely: { medium: ['warn', 'Likely'], low: ['warn', 'Likely – low confidence'] },
};

const BASIS = {
  'visual,text': 'Based on the rear appearance and the badge text, which agree.',
  visual: 'Based on the rear appearance only (no readable badge text confirmed it).',
  text: 'Based on the badge text only – the rear-view model was not sure or does not know this vehicle.',
};

const REASON = {
  'text-and-visual-conflict': 'The badge text names a different vehicle than the rear appearance suggests. Check the photo yourself.',
  'text-brand-contradicts-visual': 'The badge text shows a different brand than the rear appearance suggests.',
  'unknown-vehicle': 'This rear does not resemble any vehicle the model knows well. It may not be in the model yet.',
  'no-confident-evidence': 'Neither the rear appearance nor the badge text is conclusive. The candidates below are only guesses.',
  'no-evidence': 'Nothing usable was found (no known rear design, no readable badge).',
};

function candidateList(fusion, visual, db) {
  if (!fusion.candidates.length) return null;
  return h(
    'ul',
    { class: 'alts' },
    fusion.candidates.slice(0, 4).map((c) => {
      const v = db?.getVehicle(c.vehicleId);
      const vis = visual?.candidates.find((x) => x.vehicleId === c.vehicleId);
      const zeroShot = vis && vis.variants.every((x) => x.kind !== 'trained');
      return h(
        'li',
        {},
        h('span', {}, vehicleTitle(v), ' ', c.text ? h('span', { class: 'badge iran' }, 'badge text') : null, zeroShot ? h('span', { class: 'badge' }, 'few training photos') : null),
        h('strong', {}, c.visual != null ? formatPercent(c.visual) : '–'),
        c.visual != null ? meter(c.visual, true) : h('span', {}),
      );
    }),
  );
}

function footer(result) {
  const t = result.timings;
  const parts = [`Processed in ${formatMs(t.total)}`, t.detect != null ? `(detection ${formatMs(t.detect)}, rear-view model ${formatMs(t.classify)})` : `(rear-view model ${formatMs(t.classify)})`];
  const models = result.models.map((m) => `${m.title} [${m.provider === 'webgpu' ? 'WebGPU' : 'WASM'}]`).join(' · ');
  return h('p', { class: 'models-used' }, parts.join(' '), h('br'), 'Models: ', models);
}

function secondaryBlock(result, actions) {
  const s = result.secondary;
  if (!s) {
    return h(
      'div',
      { class: 'secondary-block' },
      h('button', { class: 'btn ghost', type: 'button', onclick: actions.onSecondary }, 'Also ask the older general-view models'),
      h('p', { class: 'muted small' }, 'Optional second opinion from two classifiers trained on mixed-angle photos (Stanford Cars 1991–2012, 27 Iranian models). They were not trained for rear photos and download ≈130 MB on first use.'),
    );
  }
  const list = s.primary ? [s.primary, ...s.alternatives] : s.alternatives;
  return h(
    'div',
    { class: 'secondary-block' },
    h('h3', {}, 'Second opinion: general-view models'),
    h('p', { class: 'muted small' }, s.verdict === 'uncertain' ? 'Neither general-view model was confident.' : 'Not trained for rear photos – treat as a hint only.'),
    h('ul', { class: 'alts' }, list.slice(0, 3).map((c) => h('li', {}, h('span', {}, describe(c).title, ' ', h('span', { class: `badge ${c.domain === 'iranian' ? 'iran' : 'intl'}` }, c.domain === 'iranian' ? 'Iranian model' : 'Stanford Cars')), h('strong', {}, formatPercent(c.probability)), meter(c.probability, true)))),
  );
}

/**
 * @param {HTMLElement} root
 * @param {object} result pipeline result (car-recognition.js)
 * @param {{fusion: object|null, db: object|null, textPending?: boolean}} ctx fused verdict (fusion.js) and database
 * @param {{onWholeImage: ()=>void, onIgnoreViewpoint: ()=>void, onSecondary: ()=>void}} actions
 */
export function renderResult(root, result, ctx, actions) {
  root.replaceChildren();
  root.hidden = false;
  const { fusion, db } = ctx;

  if (result.status === 'no-vehicle') {
    put(root,
      h('h2', { class: 'uncertain-title' }, 'No car found in this image'),
      h('p', { class: 'muted' }, 'The vehicle detector did not find a car, bus or truck. Please upload a photograph of the rear of the vehicle.'),
      h('button', { class: 'btn', type: 'button', onclick: actions.onWholeImage }, 'Analyse the whole image anyway'),
    );
    return;
  }

  if (result.status === 'not-rear') {
    const seen = { front: 'the front of a car', 'front-side': 'the front of a car (at an angle)', side: 'the side of a car', other: 'not the outside of a car' }[result.viewpoint.label] ?? 'not the rear of a car';
    put(root,
      h('div', { class: 'verdict' }, h('span', { class: 'badge warn' }, 'Not a rear photo')),
      h('h2', { class: 'uncertain-title' }, 'Please upload a photograph of the rear of the vehicle.'),
      h('p', { class: 'muted' }, `This photo looks like ${seen} (rear-view check: ${formatPercent(result.viewpoint.rearProbability)} rear). CarVision only recognises vehicles from the rear.`),
      h('button', { class: 'btn', type: 'button', onclick: actions.onIgnoreViewpoint }, 'It is a rear photo – analyse it anyway'),
      footer(result),
    );
    return;
  }

  const v = fusion?.vehicleId ? db?.getVehicle(fusion.vehicleId) : null;
  const visualTop = fusion?.vehicleId ? result.visual?.candidates.find((c) => c.vehicleId === fusion.vehicleId) : null;
  const variant = visualTop?.variants[0]?.variant;
  const zeroShot = visualTop && visualTop.variants.every((x) => x.kind !== 'trained');

  if (!v) {
    put(root,
      h('div', { class: 'verdict' }, h('span', { class: 'badge warn' }, 'Uncertain')),
      h('h2', { class: 'uncertain-title' }, 'I can’t identify this car with confidence'),
      h('div', { class: 'alert warn', role: 'note' }, REASON[fusion?.reason] ?? REASON['no-confident-evidence']),
      ctx.textPending ? h('p', { class: 'muted small' }, 'Still reading badge text…') : null,
      ...alerts(result.warnings),
      fusion?.candidates.length ? h('h3', {}, 'Closest guesses') : null,
      candidateList(fusion, result.visual, db),
      h('p', { class: 'muted small' }, 'Percentages are the rear-view model’s calibrated probabilities over the vehicles it knows; a vehicle missing from the model cannot score.'),
      secondaryBlock(result, actions),
      footer(result),
    );
    return;
  }

  const [cls, label] = VERDICT[fusion.verdict]?.[fusion.level] ?? ['warn', fusion.verdict];
  const years = v.yearFrom || v.yearTo ? `${v.yearFrom ?? '…'}–${v.yearTo ?? ''}` : null;
  const facts = [
    ['Brand', v.brand],
    ['Model', v.model + (variant ? ` (${variant})` : '')],
    ['Manufacturer / assembler', v.manufacturer !== v.brand ? v.manufacturer : null],
    ['Based on', v.basedOn],
    ['Category', v.category],
    ['Brand origin', v.country],
    ['Produced / introduced (database)', years],
  ].filter(([, x]) => x);

  put(root,
    h('div', { class: 'verdict' }, h('span', { class: `badge ${cls}` }, label), v.category ? h('span', { class: 'badge' }, v.category) : null),
    h('h2', { class: 'result-title' }, vehicleTitle(v)),
    h('p', { class: 'basis' }, BASIS[fusion.basis.join(',')] ?? ''),
    h('dl', { class: 'facts' }, facts.map(([k, x]) => [h('dt', {}, k), h('dd', {}, x)])),
    visualTop ? h('div', { class: 'conf-row' }, h('span', {}, 'Rear-view model probability'), h('span', {}, formatPercent(visualTop.probability))) : null,
    visualTop ? meter(visualTop.probability, fusion.level !== 'high') : null,
    zeroShot ? h('div', { class: 'alert warn', role: 'note' }, 'The rear-view model has seen few or no rear photos of this vehicle (it relies mostly on a text description of it), so this result is less reliable.') : null,
    fusion.verdict === 'likely' ? h('div', { class: 'alert warn', role: 'note' }, 'The rear-view model is leaning towards this vehicle but is not sure. A sharper photo of the whole rear, or a readable badge, helps.') : null,
    ctx.textPending ? h('p', { class: 'muted small' }, 'Still reading badge text…') : null,
    years ? h('p', { class: 'muted small' }, 'Years come from the local database for the model line, not from this photo; the model year of this car is not determined.') : null,
    ...alerts(result.warnings),
    fusion.candidates.length > 1 ? h('h3', {}, 'Other possibilities') : null,
    fusion.candidates.length > 1 ? candidateList({ candidates: fusion.candidates.filter((c) => c.vehicleId !== v.id) }, result.visual, db) : null,
    secondaryBlock(result, actions),
    footer(result),
  );
}

export function renderError(root, message, onRetry) {
  root.replaceChildren(
    h('div', { class: 'alert error', role: 'alert' }, message),
    onRetry ? h('button', { class: 'btn', type: 'button', onclick: onRetry }, 'Try again') : null,
  );
  root.hidden = false;
}
