// Local JSON vehicle database + OCR text matcher.
//
// - The only data source is public/data/vehicles.json (+ optional manufacturers.json), loaded with fetch from
//   our own origin. No remote lookup, no backend. Adding/editing vehicles never needs a code change.
// - Independent of the image classifiers and of any particular OCR engine: it takes recognised text
//   (strings, or {text, confidence} objects) and returns ranked vehicle candidates.
// - Pure ES module without DOM access, so it can be unit-tested under plain Node.
//
// Matching philosophy (see public/data/README.md): a vehicle is only "identified" when there is MODEL evidence
// (a full alias, or brand + model). A brand alone never names a model, and engine labels (1.5, 2.0, 1.6T...)
// never identify anything by themselves.

// ---------------------------------------------------------------------------------------------------------
// Text normalisation
// ---------------------------------------------------------------------------------------------------------

/** Letters that OCR engines commonly confuse with digits (and vice versa). Applied only context-aware, see below. */
const OCR_CONFUSIONS = {
  0: ['O'], O: ['0'], 1: ['I', 'L'], I: ['1'], L: ['1'], 5: ['S'], S: ['5'], 8: ['B'], B: ['8'], 2: ['Z'], Z: ['2'], 6: ['G'], G: ['6'],
};

const ENGINE_LIKE = /^\d+(\.\d+)?T?$/; // 1.5, 2.0, 1.6T, 206 ... (numeric tokens are never "OCR-corrected")
const PURE_LETTERS = /^[A-Z؀-ۿ]+$/;

/**
 * Normalise recognised text: Latin + Persian + Arabic variants, digits, punctuation, spacing.
 * With `vocabulary` (built from the database) it also applies CONTEXT-AWARE fixes:
 *   - OCR confusions (0/O, 1/I/L, 5/S, 8/B, 2/Z, 6/G) are only applied to a token that is NOT a known token and
 *     only if the corrected token IS one - so valid names like "RAV4", "X22" or "206" are never damaged.
 *   - adjacent tokens are merged when the merged form is a known single-token name ("RAV 4" -> "RAV4").
 * Static rules: "DENA+" -> "DENA PLUS".
 */
export function normalizeText(input, { vocabulary = null } = {}) {
  let s = String(input ?? '');
  // Arabic/Persian letter variants -> canonical Persian forms (before NFKC so ligatures/hamza forms are handled)
  s = s.replace(/[يى]/g, 'ی').replace(/ك/g, 'ک').replace(/[ةۀ]/g, 'ه').replace(/[أإٱآ]/g, 'ا');
  s = s.normalize('NFKC');
  s = s.replace(/[ً-ٰٟـ​‍‎‏﻿]/g, '').replace(/ە/g, 'ه'); // diacritics, tatweel, marks
  s = s.replace(/‌/g, ' '); // zero-width non-joiner separates Persian words
  s = s.replace(/[٠-٩]/g, (d) => String(d.charCodeAt(0) - 0x0660)).replace(/[۰-۹]/g, (d) => String(d.charCodeAt(0) - 0x06f0));
  s = s.toUpperCase();
  s = s.replace(/([؀-ۿ])\s*\+/g, '$1 پلاس ').replace(/\s*\+\s*/g, ' PLUS ');
  s = s.replace(/[،؛؟٪-٭۔]/g, ' ');
  s = s.replace(/[^A-Z0-9؀-ۿ.]+/g, ' ');
  s = s.replace(/(?<!\d)\.|\.(?!\d)/g, ' '); // keep '.' only inside numbers like 1.5
  let tokens = s.split(/\s+/).filter(Boolean);
  if (vocabulary) tokens = mergeTokens(tokens.map((t) => correctToken(t, vocabulary)), vocabulary);
  return tokens.join(' ');
}

function correctToken(token, vocab) {
  if (vocab.tokens.has(token) || ENGINE_LIKE.test(token) || token.length < 3 || !/^[A-Z0-9]+$/.test(token)) return token;
  const positions = [...token].map((c, i) => (OCR_CONFUSIONS[c] ? i : -1)).filter((i) => i >= 0).slice(0, 8);
  // Try 1, then 2, then 3 substitutions; first variant that is a known token wins (deterministic).
  for (let n = 1; n <= Math.min(3, positions.length); n++) {
    const hit = substitute(token, positions, n, vocab.tokens);
    if (hit) return hit;
  }
  return token;
}

function substitute(token, positions, n, known) {
  const chars = [...token];
  const walk = (start, left) => {
    if (left === 0) {
      const w = chars.join('');
      return known.has(w) ? w : null;
    }
    for (let k = start; k < positions.length; k++) {
      const p = positions[k];
      const orig = chars[p];
      for (const alt of OCR_CONFUSIONS[orig]) {
        chars[p] = alt;
        const r = walk(k + 1, left - 1);
        chars[p] = orig;
        if (r) return r;
      }
    }
    return null;
  };
  return walk(0, n);
}

function mergeTokens(tokens, vocab) {
  const out = [];
  for (let i = 0; i < tokens.length; i++) {
    const merged = i + 1 < tokens.length ? tokens[i] + tokens[i + 1] : null;
    if (merged && vocab.singles.has(merged)) {
      out.push(merged);
      i++;
    } else out.push(tokens[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------
// Edit distance (Damerau-Levenshtein, optimal string alignment) with early exit
// ---------------------------------------------------------------------------------------------------------

function editDistance(a, b, max) {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...new Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
    }
  }
  return d[a.length][b.length];
}

// ---------------------------------------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------------------------------------

/** Check the database shape. Returns {errors: string[], warnings: string[]}; unknown values may be null/omitted. */
export function validateVehicles(data) {
  const errors = [];
  const warnings = [];
  if (!data || !Array.isArray(data.vehicles)) return { errors: ['"vehicles" must be an array'], warnings };
  const ids = new Set();
  const strArr = (v, name, id) => {
    if (v === undefined || v === null) return;
    if (!Array.isArray(v) || v.some((x) => typeof x !== 'string' || !x.trim())) errors.push(`${id}: "${name}" must be an array of non-empty strings`);
  };
  data.vehicles.forEach((v, i) => {
    const id = v?.id ?? `#${i}`;
    if (typeof v?.id !== 'string' || !v.id.trim()) errors.push(`${id}: missing "id"`);
    else if (ids.has(v.id)) errors.push(`${id}: duplicate id`);
    else ids.add(v.id);
    for (const f of ['brand', 'model']) if (typeof v?.[f] !== 'string' || !v[f].trim()) errors.push(`${id}: missing "${f}"`);
    for (const f of ['aliases', 'badgeTexts', 'engineBadges', 'market', 'sources']) strArr(v?.[f], f, id);
    const years = (o, where) => {
      for (const f of ['yearFrom', 'yearTo']) {
        if (o?.[f] != null && !Number.isInteger(o[f])) errors.push(`${where}: "${f}" must be an integer or null`);
      }
      if (o?.yearFrom != null && o?.yearTo != null && o.yearFrom > o.yearTo) errors.push(`${where}: yearFrom is after yearTo`);
    };
    years(v, id);
    for (const f of ['basedOn', 'country', 'category', 'manufacturer']) {
      if (v?.[f] != null && typeof v[f] !== 'string') errors.push(`${id}: "${f}" must be a string or null`);
    }
    if (v?.generations != null) {
      if (!Array.isArray(v.generations)) errors.push(`${id}: "generations" must be an array`);
      else v.generations.forEach((g, k) => {
        if (typeof g?.name !== 'string' || !g.name.trim()) errors.push(`${id}: generations[${k}] needs a "name"`);
        years(g, `${id}: generations[${k}]`);
      });
    }
    const rr = v?.rearRecognition;
    if (rr != null) {
      if (typeof rr !== 'object' || typeof rr.supported !== 'boolean' || typeof rr.tested !== 'boolean') {
        errors.push(`${id}: "rearRecognition" must be {supported: boolean, tested: boolean, modelId: string|null}`);
      } else if (rr.supported && typeof rr.modelId !== 'string') errors.push(`${id}: rearRecognition.supported needs a modelId`);
    }
    if (!v?.aliases?.length) warnings.push(`${id}: no aliases (only the model name will be matched)`);
  });
  return { errors, warnings };
}

// ---------------------------------------------------------------------------------------------------------
// The database
// ---------------------------------------------------------------------------------------------------------

const THRESHOLD = { identified: 0.85, possible: 0.5, weak: 0.25 };
const compactOf = (phrase) => phrase.replace(/ /g, '');

export class VehicleDatabase {
  /** @param {object} vehiclesData parsed vehicles.json  @param {object|null} manufacturersData parsed manufacturers.json */
  static fromData(vehiclesData, manufacturersData = null) {
    const { errors, warnings } = validateVehicles(vehiclesData);
    if (errors.length) throw new Error('Invalid vehicle database:\n - ' + errors.join('\n - '));
    return new VehicleDatabase(vehiclesData, manufacturersData, warnings);
  }

  constructor(vehiclesData, manufacturersData, warnings = []) {
    this.version = vehiclesData.version ?? null;
    this.warnings = warnings;
    this.vehicles = vehiclesData.vehicles.map((v) => ({
      aliases: [], badgeTexts: [], engineBadges: [], market: [], country: null, category: null, basedOn: null, generations: [], sources: [],
      yearFrom: null, yearTo: null, rearRecognition: { supported: false, tested: false, modelId: null }, ...v,
    }));
    this.byId = new Map(this.vehicles.map((v) => [v.id, v]));
    this.#buildBrands(manufacturersData);
    this.#buildPhrases();
    this.vocabulary = this.#buildVocabulary();
  }

  // ---- index construction ----
  #buildBrands(manufacturersData) {
    /** brand name -> {name, country, phrases:[{text, compact}]} */
    this.brands = new Map();
    const add = (name, aliases, country) => {
      const key = normalizeText(name);
      const b = this.brands.get(key) ?? { name, country: country ?? null, phrases: new Map() };
      for (const a of [name, ...aliases]) {
        const text = normalizeText(a);
        if (text) b.phrases.set(compactOf(text), text);
      }
      if (country && !b.country) b.country = country;
      this.brands.set(key, b);
    };
    for (const m of manufacturersData?.manufacturers ?? []) add(m.name, m.aliases ?? [], m.country);
    for (const v of this.vehicles) add(v.brand, [], null); // every brand is at least known by its own name
    this.brandPhraseSet = new Set();
    for (const b of this.brands.values()) for (const c of b.phrases.keys()) this.brandPhraseSet.add(c);
  }

  #buildPhrases() {
    // Phrase kinds: 'model' (model field + aliases that are not brand names) > 'badge' (badgeTexts only).
    const df = new Map(); // compact phrase -> Set(vehicle ids)
    for (const v of this.vehicles) {
      const brand = this.brands.get(normalizeText(v.brand));
      const brandPhrases = [...brand.phrases.values()].sort((a, b) => b.length - a.length);
      const phrases = new Map(); // compact -> phrase
      const put = (text, kind, strongHint) => {
        const c = compactOf(text);
        if (!c || this.brandPhraseSet.has(c)) return; // a bare brand is brand evidence, never model evidence
        const prev = phrases.get(c);
        if (prev && (prev.kind === 'model' || kind === 'badge')) {
          prev.strongHint ||= strongHint;
          return;
        }
        phrases.set(c, { text, compact: c, kind, tokens: text.split(' ').length, strongHint: !!strongHint, pureLetters: PURE_LETTERS.test(c) });
      };
      put(normalizeText(v.model), 'model');
      for (const alias of v.aliases) {
        const a = normalizeText(alias);
        if (!a) continue;
        put(a, 'model', a.includes(' '));
        // "BRAND MODEL" -> also index the model part on its own
        const prefix = brandPhrases.find((p) => a.startsWith(p + ' '));
        if (prefix) put(a.slice(prefix.length + 1), 'model');
      }
      for (const badge of v.badgeTexts) put(normalizeText(badge), 'badge');
      v.phrases = [...phrases.values()];
      v.brandKey = normalizeText(v.brand);
      v.engineSet = new Set(v.engineBadges.map((e) => normalizeText(e)));
      for (const p of v.phrases) if (p.kind === 'model') (df.get(p.compact) ?? df.set(p.compact, new Set()).get(p.compact)).add(v.id);
    }
    // Words that are themselves a known brand/model name must never be fuzzy/partial-matched to another name
    // (e.g. SAIPA vs SAINA).
    this.knownCompact = new Set(this.brandPhraseSet);
    for (const v of this.vehicles) for (const p of v.phrases) this.knownCompact.add(p.compact);
    for (const v of this.vehicles) {
      for (const p of v.phrases) {
        p.df = df.get(p.compact)?.size ?? 1;
        // Full alias (multi-token, unique to this vehicle) = strongest evidence, no brand needed.
        p.strong = p.kind === 'model' && p.tokens >= 2 && p.df === 1 && p.compact.length >= 4;
      }
    }
  }

  #buildVocabulary() {
    const tokens = new Set();
    const singles = new Set();
    for (const v of this.vehicles) {
      for (const p of v.phrases) {
        for (const t of p.text.split(' ')) if (!/^\d+\.\d+T?$/.test(t)) tokens.add(t);
        if (p.tokens === 1) singles.add(p.text);
      }
    }
    for (const b of this.brands.values()) for (const t of b.phrases.values()) for (const w of t.split(' ')) tokens.add(w);
    return { tokens, singles };
  }

  // ---- public helpers ----
  normalize(text) {
    return normalizeText(text, { vocabulary: this.vocabulary });
  }

  getVehicle(id) {
    return this.byId.get(id) ?? null;
  }

  /** Database coverage vs. actual visual (rear-view model) coverage - these are deliberately different numbers. */
  coverage() {
    const total = this.vehicles.length;
    const supported = this.vehicles.filter((v) => v.rearRecognition?.supported).length;
    const tested = this.vehicles.filter((v) => v.rearRecognition?.supported && v.rearRecognition?.tested).length;
    return { total, rearSupported: supported, rearTested: tested, textOnly: total - supported };
  }

  /** Strip internal index fields from a vehicle record. */
  #publicVehicle(v) {
    const { phrases, brandKey, engineSet, ...rest } = v;
    return rest;
  }

  // ---- matching ----
  /**
   * @param {string|Array<string|{text:string, confidence?:number}>} input recognised text (any OCR engine)
   * @param {{minConfidence?: number}} [options] words with a lower confidence are ignored
   * @returns {{
   *   normalized: string,
   *   best: object|null,           // set only when exactly one candidate is clearly "identified"
   *   ambiguous: boolean,          // several identified candidates too close to call
   *   candidates: object[],        // status 'identified' | 'possible', best first
   *   weak: object[],              // low-evidence hints (never an identification)
   *   brandHints: object[],        // brands seen without usable model evidence (never claims a model)
   *   engineBadgesSeen: string[]   // engine labels seen (informational only)
   * }}
   */
  match(input, { minConfidence = 0.3 } = {}) {
    const words = (Array.isArray(input) ? input : [input])
      .map((w) => (typeof w === 'string' ? { text: w } : w))
      .filter((w) => w && typeof w.text === 'string' && (w.confidence == null || w.confidence >= minConfidence));
    const normalized = this.normalize(words.map((w) => w.text).join(' '));
    const tokens = normalized ? normalized.split(' ') : [];
    const empty = { normalized, best: null, ambiguous: false, candidates: [], weak: [], brandHints: [], engineBadgesSeen: [] };
    if (!tokens.length) return empty;

    // span index: compact string of 1..4 consecutive tokens -> [[start,end]]
    const spans = new Map();
    const spanList = [];
    for (let i = 0; i < tokens.length; i++) {
      let acc = '';
      for (let len = 1; len <= 4 && i + len <= tokens.length; len++) {
        acc += tokens[i + len - 1];
        (spans.get(acc) ?? spans.set(acc, []).get(acc)).push([i, i + len - 1]);
        spanList.push({ compact: acc, start: i, end: i + len - 1 });
      }
    }
    const longest = (list) => list.reduce((a, b) => (b[1] - b[0] > a[1] - a[0] ? b : a));

    // brand evidence: brand name -> strength (1 exact, 0.8 fuzzy)
    const brandHits = new Map();
    for (const [key, b] of this.brands) {
      let strength = 0;
      for (const c of b.phrases.keys()) {
        if (spans.has(c)) strength = 1;
        else if (!strength && c.length >= 5 && PURE_LETTERS.test(c)) {
          const maxD = c.length <= 7 ? 1 : 2;
          if (spanList.some((s) => s.compact.length >= 5 && PURE_LETTERS.test(s.compact) && !this.knownCompact.has(s.compact) && editDistance(s.compact, c, maxD) <= maxD)) strength = 0.8;
        }
      }
      if (strength) brandHits.set(key, strength);
    }

    const engineBadgesSeen = tokens.filter((t) => ENGINE_LIKE.test(t) && /\./.test(t));

    const raw = [];
    for (const v of this.vehicles) {
      const ev = this.#bestEvidence(v, spans, spanList, tokens, longest);
      if (!ev) continue;
      const b = brandHits.get(v.brandKey) ?? 0;
      const otherBrand = [...brandHits.keys()].some((k) => k !== v.brandKey);
      let score;
      if (ev.type === 'strong') score = 1;
      else if (ev.type === 'exact-model') score = (ev.p.df > 1 ? 0.65 : 0.7) + 0.25 * b;
      else if (ev.type === 'exact-badge') score = 0.4 + 0.4 * b;
      else if (ev.type === 'fuzzy') score = Math.min(0.95, ev.m + 0.15 * b);
      else score = 0.3 + 0.1 * b; // partial
      if (ev.type !== 'strong' && ev.p.pureLetters && ev.p.compact.length <= 2) score = Math.min(score, 0.45); // 1-2 letter bare badge
      // A bare 1-2 digit number ("3" of MAZDA 3) is everywhere on a rear photo (number plates): without the brand it is
      // not evidence at all.
      if (ev.type !== 'strong' && /^\d{1,2}$/.test(ev.p.compact) && !b) score = Math.min(score, 0.2);
      if (b === 0 && otherBrand && ev.type !== 'strong') score *= 0.6; // contradicts a different brand in the text
      const engineHit = [...v.engineSet].some((e) => tokens.includes(e) || spans.has(compactOf(e)));
      if (engineHit && score >= THRESHOLD.possible) score = Math.min(1, score + 0.03); // supporting evidence only
      raw.push({ v, ev, score, brandStrength: b, engineHit });
    }

    // A match fully contained in a longer match of another vehicle is a sub-string artefact
    // ("DENA" inside "DENA PLUS", "SAMAND" inside "SAMAND SOREN", "405" inside "405 PARS").
    for (const a of raw) {
      for (const c of raw) {
        if (a === c || !a.ev.span || !c.ev.span) continue;
        const [as, ae] = a.ev.span;
        const [cs, ce] = c.ev.span;
        if (cs <= as && ae <= ce && ce - cs > ae - as && c.score >= a.score) {
          a.score *= 0.5;
          a.subsumed = true;
          break;
        }
      }
    }

    const toCandidate = (r) => ({
      vehicle: this.#publicVehicle(r.v),
      score: Math.round(r.score * 1000) / 1000,
      status: r.score >= THRESHOLD.identified ? 'identified' : r.score >= THRESHOLD.possible ? 'possible' : 'weak',
      matchType: r.ev.type === 'strong' ? 'alias' : r.ev.type === 'exact-model' ? (r.brandStrength ? 'brand+model' : 'model') : r.ev.type === 'exact-badge' ? (r.brandStrength ? 'brand+badge' : 'badge') : r.ev.type,
      matchedTerm: r.ev.p.text,
      brandMatched: r.brandStrength > 0,
      engineBadgeMatched: r.engineHit,
      subsumed: !!r.subsumed,
    });
    const sorted = raw.filter((r) => r.score >= THRESHOLD.weak).map(toCandidate).sort((x, y) => y.score - x.score || x.vehicle.id.localeCompare(y.vehicle.id));
    const candidates = sorted.filter((c) => c.status !== 'weak');
    const weak = sorted.filter((c) => c.status === 'weak');

    const identified = candidates.filter((c) => c.status === 'identified');
    const ambiguous = identified.length > 1 && identified[0].score - identified[1].score < 0.05;
    const best = identified.length && !ambiguous ? identified[0] : null;

    // Brands seen without any identified vehicle of that brand: report the brand, never a model.
    const identifiedBrands = new Set(identified.map((c) => normalizeText(c.vehicle.brand)));
    const brandHints = [...brandHits.keys()]
      .filter((k) => !identifiedBrands.has(k))
      .map((k) => {
        const b = this.brands.get(k);
        return { brand: b.name, country: b.country, vehicleIds: this.vehicles.filter((v) => v.brandKey === k).map((v) => v.id) };
      });

    return { normalized, best, ambiguous, candidates, weak, brandHints, engineBadgesSeen };
  }

  #bestEvidence(v, spans, spanList, tokens, longest) {
    let best = null;
    const consider = (ev) => {
      if (!best || ev.rank > best.rank || (ev.rank === best.rank && (ev.span?.[1] - ev.span?.[0] || 0) > (best.span?.[1] - best.span?.[0] || 0))) best = ev;
    };
    for (const p of v.phrases) {
      const exact = spans.get(p.compact);
      if (exact) {
        const span = longest(exact);
        if (p.strong) consider({ type: 'strong', p, span, rank: 5 });
        else if (p.kind === 'model') consider({ type: 'exact-model', p, span, rank: 4 });
        else consider({ type: 'exact-badge', p, span, rank: 2 });
        continue;
      }
      if (p.kind !== 'model' || !p.pureLetters || p.compact.length < 5) continue;
      const maxD = p.compact.length <= 7 ? 1 : 2;
      for (const s of spanList) {
        if (!PURE_LETTERS.test(s.compact) || this.knownCompact.has(s.compact) || Math.abs(s.compact.length - p.compact.length) > maxD) continue;
        const d = editDistance(s.compact, p.compact, maxD);
        if (d <= maxD && d > 0) {
          const sim = 1 - d / Math.max(s.compact.length, p.compact.length);
          consider({ type: 'fuzzy', p, span: [s.start, s.end], m: 0.55 + 0.25 * Math.max(0, (sim - 0.75) / 0.25), rank: 3 });
        }
      }
      // partial: a long word that merely contains / starts like the phrase
      for (const t of tokens) {
        if (t.length >= 5 && p.compact.length >= 5 && t !== p.compact && !this.knownCompact.has(t) && (t.includes(p.compact) || p.compact.includes(t))) {
          consider({ type: 'partial', p, span: null, rank: 1 });
        }
      }
    }
    return best;
  }
}

// ---------------------------------------------------------------------------------------------------------
// Loading (browser): fetch from our own origin, cache the parsed database in memory
// ---------------------------------------------------------------------------------------------------------

let cached = null;

/**
 * Load public/data/vehicles.json (required) and manufacturers.json (optional). Works under any GitHub Pages
 * sub-path because it is based on Vite's BASE_URL. The parsed database is cached in memory.
 */
export function loadVehicleDatabase({ baseUrl = import.meta.env?.BASE_URL ?? '/', fetchFn = globalThis.fetch?.bind(globalThis), force = false } = {}) {
  if (cached && !force) return cached;
  cached = (async () => {
    const root = `${baseUrl}data/`;
    const response = await fetchFn(`${root}vehicles.json`);
    if (!response.ok) throw new Error('Failed to load vehicle database');
    const vehicles = await response.json();
    let manufacturers = null;
    try {
      const m = await fetchFn(`${root}manufacturers.json`);
      if (m.ok) manufacturers = await m.json();
    } catch {
      /* optional file */
    }
    return VehicleDatabase.fromData(vehicles, manufacturers);
  })().catch((e) => {
    cached = null; // allow a retry
    throw e;
  });
  return cached;
}
