// GKG crawler — GDELT Global Knowledge Graph themed retrieval for oil markets.
// Two passes, both via the DOC 2.0 API (reusing crawlGdelt):
//   1) theme/keyword topics: OPEC+, conflict, sanctions   (source = "GKG")
//   2) per-country focus for OPEC / OPEC+ members         (source = "GKG-Country")
// Coverage ~2017+ (DOC article index). Rate-limited ~1 req / 5s by crawlGdelt.

import { crawlGdelt } from './gdelt.mjs';
import { GKG_TOPICS, OPEC_COUNTRIES } from '../lib/gkg_topics.mjs';

export async function crawlGkg(opts = {}) {
  const {
    from, to, chunkDays = 7, perDay = 25, domainChunkDays = 30,
    onRow, seenUrls, checkpoint, log = console.error,
  } = opts;
  let written = 0, calls = 0;

  // --- Pass 1: GKG theme/keyword topics ---
  log('[GKG] theme topics: ' + Object.keys(GKG_TOPICS).join(', '));
  const r1 = await crawlGdelt({
    topics: GKG_TOPICS, from, to, chunkDays, perDay,
    sourceLabel: 'GKG', idPrefix: 'gkg',
    onRow, seenUrls, checkpoint, log,
  });
  written += r1.written; calls += r1.calls;

  // --- Pass 2: OPEC / OPEC+ country focus (monthly windows to limit volume) ---
  log('[GKG] country focus: ' + OPEC_COUNTRIES.length + ' OPEC/OPEC+ countries');
  const countryTopics = {};
  for (const c of OPEC_COUNTRIES) {
    countryTopics[c.key] = { label: c.name, gdeltQuery: c.query };
  }
  const r2 = await crawlGdelt({
    topics: countryTopics, from, to, chunkDays: domainChunkDays, perDay,
    sourceLabel: 'GKG-Country', idPrefix: 'gkgc',
    onRow, seenUrls, checkpoint, log,
  });
  written += r2.written; calls += r2.calls;

  return { written, calls };
}
