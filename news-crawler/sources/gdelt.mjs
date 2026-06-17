// GDELT DOC 2.0 crawler. Headlines + timestamps per topic across a date range,
// chunked by time, capping the densest days. Optional single-outlet `domain`.
// Coverage ~2017+. Cap 250 records/call. Rate limit ~1 req / 5s.

import { safeFetch, sleep } from '../lib/fetch.mjs';

const BASE = 'https://api.gdeltproject.org/api/v2/doc/doc';
const RATE_MS = 5200;

function parseSeen(seendate) {
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/.exec(seendate || '');
  if (!m) return null;
  const a = m;
  return new Date(Date.UTC(+a[1], +a[2] - 1, +a[3], +a[4], +a[5], +a[6]));
}

function fmtStamp(date) {
  return date.toISOString().slice(0, 19).replace(/[-:T]/g, '');
}

function* windows(from, to, days) {
  let cur = new Date(from);
  while (cur < to) {
    const next = new Date(cur);
    next.setUTCDate(next.getUTCDate() + days);
    yield [cur, next > to ? to : next];
    cur = next;
  }
}

async function fetchWindow(query, start, end) {
  const params = new URLSearchParams({
    query, mode: 'ArtList', format: 'json', maxrecords: '250', sort: 'datedesc',
    startdatetime: fmtStamp(start), enddatetime: fmtStamp(end),
  });
  const r = await safeFetch(BASE + '?' + params, { timeout: 30000, retries: 2, backoffMs: 6000 });
  if (!r.ok) return { error: r.error };
  return r.data || {};
}

function capPerDay(rows, perDay) {
  const counts = new Map();
  const out = [];
  for (const row of rows) {
    const n = counts.get(row.date) || 0;
    if (n >= perDay) continue;
    counts.set(row.date, n + 1);
    out.push(row);
  }
  return out;
}

export async function crawlGdelt(opts = {}) {
  const {
    topics, from, to, chunkDays = 7, perDay = 25,
    domain = null, sourceLabel = 'GDELT', idPrefix = 'gdelt',
    onRow, seenUrls, checkpoint, log = console.error,
  } = opts;
  let written = 0, calls = 0;

  for (const [key, def] of Object.entries(topics)) {
    const query = domain ? 'domain:' + domain + ' ' + def.gdeltQuery : def.gdeltQuery;
    for (const [wStart, wEnd] of windows(from, to, chunkDays)) {
      const id = idPrefix + ':' + key + ':' + fmtStamp(wStart);
      if (checkpoint && checkpoint.has(id)) continue;

      const data = await fetchWindow(query, wStart, wEnd);
      calls++;
      const articles = (data && data.articles) || [];
      if (data && data.error) {
        log('  ! ' + sourceLabel + ' ' + key + ' ' + wStart.toISOString().slice(0, 10) + ': ' + data.error);
      }

      const rows = [];
      for (const a of articles) {
        const dt = parseSeen(a.seendate);
        if (!dt) continue;
        const url = a.url;
        if (!url || (seenUrls && seenUrls.has(url + '::' + key))) continue;
        rows.push({
          datetime: dt.toISOString(),
          date: dt.toISOString().slice(0, 10),
          source: sourceLabel,
          topic: key,
          headline: (a.title || '').replace(/\s+/g, ' ').trim(),
          url,
          domain: a.domain || '',
          country: a.sourcecountry || '',
        });
      }
      rows.sort((x, y) => (x.datetime < y.datetime ? 1 : -1));
      const capped = capPerDay(rows, perDay);
      for (const row of capped) {
        const k = row.url + '::' + key;
        if (seenUrls && seenUrls.has(k)) continue;
        if (seenUrls) seenUrls.add(k);
        if (onRow) onRow(row);
        written++;
      }

      if (checkpoint) checkpoint.add(id);
      const ws = wStart.toISOString().slice(0, 10);
      const we = wEnd.toISOString().slice(0, 10);
      log('  ' + sourceLabel + ' ' + key + ' ' + ws + '..' + we + '  +' + capped.length + ' (raw ' + articles.length + ')  [total ' + written + ']');
      await sleep(RATE_MS);
    }
  }
  return { written, calls };
}
