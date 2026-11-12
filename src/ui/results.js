import { formatMs, formatPercent } from '../utils/formatting.js';

/** Tiny DOM helper: h('div', {class:'x'}, 'text', child, ...) - always uses textContent, never innerHTML. */
function h(tag, attrs = {}, ...children) {
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
function put(parent, ...nodes) {
  parent.append(...nodes.flat(Infinity).filter((n) => n != null && n !== false));
}

const DOMAIN_NAME = { iranian: 'Iranian-market model', international: 'International model' };

/** Display fields for a candidate, hiding anything the label table does not actually know. */
export function describe(c) {
  const l = c.label;
  if (c.domain === 'iranian') {
    return {
      title: l.title || l.raw,
      manufacturer: l.manufacturer,
      model: l.model,
      category: l.category,
      country: l.country,
      note: l.verified ? null : 'Label taken from the training dataset; manufacturer not verified.',
    };
  }
  return {
    title: `${l.manufacturer} ${l.model}${l.body ? ' ' + l.body : ''}`,
    manufacturer: l.manufacturer,
    model: l.model,
    category: l.category,
    country: l.country,
    referenceYear: l.referenceYear,
  };
}

function meter(p, low) {
  return h('div', { class: `meter${low ? ' low' : ''}`, role: 'meter', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': String(Math.round(p * 100)) }, h('span', { style: `width:${Math.max(1, Math.round(p * 100))}%` }));
}

const WARNINGS = {
  'low-confidence': 'The model is not very sure about this result. Treat it as a guess – a clearer, side-on photo of the whole car usually helps.',
  uncertain:
    'Neither model is confident enough to name this vehicle. It may be a model they were not trained on (for example a newer car), a heavily obstructed view, or not a car. The candidates below are only guesses.',
  'multiple-vehicles': 'More than one vehicle was found. Click a box on the photo to identify a different one.',
  'small-vehicle': 'The vehicle is small in the photo, so details are hard to see. Try a closer photo.',
  'no-vehicle-whole-image': 'No vehicle was detected, so the whole image was analysed. This result is very likely unreliable.',
};

function alerts(codes) {
  return codes.filter((c) => WARNINGS[c]).map((c) => h('div', { class: `alert ${c === 'multiple-vehicles' ? 'info' : 'warn'}`, role: 'note' }, WARNINGS[c]));
}

function candidateRow(c, thresholds) {
  const d = describe(c);
  const t = c.domain === 'iranian' ? thresholds.iranian : thresholds.international;
  return h(
    'li',
    {},
    h('span', {}, d.title, ' ', h('span', { class: `badge ${c.domain === 'iranian' ? 'iran' : 'intl'}` }, c.domain === 'iranian' ? 'Iranian model' : 'International model')),
    h('strong', {}, formatPercent(c.probability)),
    meter(c.probability, c.probability < t),
  );
}

function footer(result) {
  const parts = [
    `Processed in ${formatMs(result.timings.total)}`,
    result.timings.detect != null ? `(detection ${formatMs(result.timings.detect)}, classification ${formatMs(result.timings.classify)})` : `(classification ${formatMs(result.timings.classify)})`,
  ];
  const models = result.models.map((m) => `${m.title} [${m.provider === 'webgpu' ? 'WebGPU' : 'WASM'}]`).join(' · ');
  return h('p', { class: 'models-used' }, parts.join(' '), h('br'), 'Models: ', models);
}

/**
 * @param {HTMLElement} root
 * @param {object} result pipeline result
 * @param {{onWholeImage: ()=>void}} actions
 */
export function renderResult(root, result, actions) {
  root.replaceChildren();
  root.hidden = false;

  if (result.status === 'no-vehicle') {
    put(root, 
      h('h2', { class: 'uncertain-title' }, 'No car found in this image'),
      h('p', { class: 'muted' }, 'The vehicle detector did not find a car, bus or truck. Try a photo where the car is clearly visible.'),
      h('button', { class: 'btn', type: 'button', onclick: actions.onWholeImage }, 'Analyse the whole image anyway'),
    );
    return;
  }

  const { primary, verdict } = result;
  const t = result.thresholds;

  if (verdict === 'uncertain') {
    put(root, 
      h('div', { class: 'verdict' }, h('span', { class: 'badge warn' }, 'Uncertain')),
      h('h2', { class: 'uncertain-title' }, 'I can’t identify this car with confidence'),
      ...alerts(result.warnings),
      h('h3', {}, 'Closest guesses'),
      h('ul', { class: 'alts' }, result.alternatives.map((c) => candidateRow(c, t))),
      h('p', { class: 'muted small' }, 'Scores come from two different models and are shown relative to each model’s own acceptance threshold; they are not directly comparable.'),
      footer(result),
    );
    return;
  }

  const d = describe(primary);
  const low = primary.probability < t[primary.domain];
  const facts = [
    ['Manufacturer', d.manufacturer],
    ['Model', d.model],
    ['Category', d.category],
    ['Country', d.country],
  ].filter(([, v]) => v);

  put(root, 
    h(
      'div',
      { class: 'verdict' },
      h('span', { class: `badge ${primary.domain === 'iranian' ? 'iran' : 'intl'}` }, primary.domain === 'iranian' ? 'Iranian-market vehicle' : 'International vehicle'),
      d.category ? h('span', { class: 'badge' }, d.category) : null,
    ),
    h('h2', { class: 'result-title' }, d.title),
    h('p', { class: 'result-sub' }, DOMAIN_NAME[primary.domain], d.referenceYear ? ` · reference class year ${d.referenceYear}` : ''),
    h('dl', { class: 'facts' }, facts.map(([k, v]) => [h('dt', {}, k), h('dd', {}, v)])),
    h('div', { class: 'conf-row' }, h('span', {}, 'Confidence'), h('span', {}, formatPercent(primary.probability))),
    meter(primary.probability, low),
    d.note ? h('p', { class: 'muted small' }, d.note) : null,
    d.referenceYear
      ? h('p', { class: 'muted small' }, 'The year is part of the training-class name and is not a reliable model-year or trim reading; nearby years look almost identical.')
      : null,
    ...alerts(result.warnings),
    result.alternatives.length ? h('h3', {}, 'Other possibilities') : null,
    result.alternatives.length ? h('ul', { class: 'alts' }, result.alternatives.map((c) => candidateRow(c, t))) : null,
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
