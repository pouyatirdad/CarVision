// Model-selection strategy. Pure logic (no ORT/DOM) so it can be unit-tested.
//
// The Iranian (27-class ResNet18) and international (196-class ViT) classifiers are closed-set softmax
// models whose confidences are NOT comparable: the ResNet18 is frequently >65% confident on foreign cars
// it has never seen (measured in docs/RESEARCH.md). So we never pick "the highest raw confidence".
// Instead: ask the international model first; only if it is not confident, ask the Iranian model; each
// must clear its own acceptance threshold; otherwise report "uncertain".

import { ROUTING } from '../config.js';

/** Does the pipeline still need the Iranian model after the international one has answered? */
export function needsIranianModel(internationalTop1, routing = ROUTING) {
  return internationalTop1 < routing.internationalAccept;
}

/**
 * @param {{international: Array, iranian?: Array|null}} input  each an array of top-k candidates
 *        {domain, classId, probability, label}, best first. `iranian` is null when it was not evaluated.
 * @returns {{verdict:'international'|'iranian'|'uncertain', primary:object|null, alternatives:object[], reason:string}}
 */
export function decide({ international, iranian = null }, routing = ROUTING) {
  const i1 = international[0];
  const r1 = iranian?.[0] ?? null;

  if (i1.probability >= routing.internationalAccept) {
    return {
      verdict: 'international',
      primary: i1,
      alternatives: international.slice(1, 3),
      reason: 'international-confident',
    };
  }
  if (r1 && r1.probability >= routing.iranianAccept) {
    return {
      verdict: 'iranian',
      primary: r1,
      alternatives: iranian.slice(1, 3),
      reason: 'iranian-confident',
    };
  }

  // Uncertain: show the best few candidates from both models, labelled by domain. They are ordered by
  // "how close to its own acceptance threshold" so that neither model's raw overconfidence dominates.
  const pool = [
    ...international.slice(0, 2).map((c) => ({ c, rel: c.probability / routing.internationalAccept })),
    ...(iranian ? iranian.slice(0, 2).map((c) => ({ c, rel: c.probability / routing.iranianAccept })) : []),
  ];
  pool.sort((a, b) => b.rel - a.rel);
  return {
    verdict: 'uncertain',
    primary: null,
    alternatives: pool.slice(0, 3).map((p) => p.c),
    reason: iranian ? 'neither-confident' : 'international-not-confident',
  };
}
