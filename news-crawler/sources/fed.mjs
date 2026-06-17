// Federal Reserve crawler — direct, deep history (2006 -> present).
//
// The Fed publishes machine-readable JSON feeds of every press release and
// testimony. Each item: { d: "M/D/YYYY h:mm:ss AM/PM", t: title, l: "/path.htm", pt: category }
// These reach back to January 2006, so they genuinely extend the dataset to
// the start of the requested range. Fed items are macro/monetary -> tagged
// political_economy (plus war/disaster if the title clearly matches).
//
// Note: Fed timestamps are US-Eastern wall-clock. We keep the wall-clock value
// (the `date` column is exact); intraday UTC offset is not adjusted.

import { safeFetch, sleep } from '../lib/fetch.mjs';
import { classify } from '../lib/topics.mjs';

const FEEDS = [
  { url: 'https://www.federalreserve.gov/json/ne-press.json', kind: 'press' },
  { url: 'https://www.federalreserve.gov/json/ne-testimony.json', kind: 'testimony' },
];

// "6/11/2026 11:00:00 AM" or "1/3/2006" -> Date (wall-clock, emitted as ISO)
function parseFedDate(s) {
  const m = /^(\d{1,2})\/(\d{1,2})\/(\d{4})(?:\s+(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM))?/i.exec(s || '');
  if (!m) return null;
  let [, mo, d, y, h, mi, se, ap] = m;
  h = h ? +h : 0;
  if (ap) {
    const up = ap.toUpperCase();
    if (up === 'PM' && h !== 12) h += 12;
    if (up === 'AM' && h === 12) h = 0;
  }
  return new Date(Date.UTC(+y, +mo - 1, +d, h, mi ? +mi : 0, se ? +se : 0));
}

export async function crawlFed({
  from,
  to,
  topics,          // object of selected topics
  onRow,
  seenUrls,
  checkpoint,
  log = console.error,
} = {}) {
  const wanted = new Set(Object.keys(topics));
  let written = 0;
  let calls = 0;

  for (const feed of FEEDS) {
    const id = `fed:${feed.kind}`;
    if (checkpoint?.has(id)) { continue; }

    const r = await safeFetch(feed.url, { parse: 'text', timeout: 30000, retries: 2 });
    calls++;
    if (!r.ok) { log(`  ! Fed ${feed.kind}: ${r.error}`); continue; }

    let items;
    try { items = JSON.parse(r.text.replace(/^﻿/, '')); }
    catch (e) { log(`  ! Fed ${feed.kind} parse: ${e.message}`); continue; }

    let kept = 0;
    for (const it of items) {
      const dt = parseFedDate(it.d);
      if (!dt || dt < from || dt > to) continue;
      const headline = (it.t || '').replace(/\s+/g, ' ').trim();
      if (!headline) continue;
      const url = it.l?.startsWith('http') ? it.l : `https://www.federalreserve.gov${it.l || ''}`;

      // Fed news is inherently political_economy; add war/disaster only if the title says so.
      const tset = new Set(['political_economy', ...classify(headline)]);
      const emitTopics = [...tset].filter((t) => wanted.has(t));
      for (const topic of emitTopics) {
        const key = `${url}::${topic}`;
        if (seenUrls?.has(key)) continue;
        seenUrls?.add(key);
        onRow?.({
          datetime: dt.toISOString(),
          date: dt.toISOString().slice(0, 10),
          source: 'Federal Reserve',
          topic,
          headline,
          url,
          domain: `federalreserve.gov/${feed.kind}${it.pt ? ' · ' + it.pt : ''}`,
          country: 'United States',
        });
        written++;
        kept++;
      }
    }
    checkpoint?.add(id);
    log(`  Fed ${feed.kind}: +${kept} (of ${items.length})  [total ${written}]`);
    await sleep(500);
  }
  return { written, calls };
}
