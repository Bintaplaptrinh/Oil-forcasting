// oilprice.com crawler via the public XML sitemaps.
//
// oilprice exposes one article sitemap per month: sitemap_articles_YYYY_M.xml
// Each <url> entry carries a <loc> (the article URL, whose slug == the
// headline) and a precise <lastmod> timestamp. Coverage runs 2009 -> present.
// We derive the headline from the slug and keep only entries whose headline
// or category matches one of the target topics.

import { safeFetch, sleep } from '../lib/fetch.mjs';
import { classify, TOPICS } from '../lib/topics.mjs';

const SITEMAP = (y, m) => `https://oilprice.com/sitemap_articles_${y}_${m}.xml`;
const RATE_MS = 1200; // be polite

// Pull <loc>/<lastmod> pairs out of a sitemap XML blob.
function parseSitemap(xml) {
  const out = [];
  const re = /<url>\s*<loc>([^<]+)<\/loc>\s*<lastmod>([^<]+)<\/lastmod>/g;
  let m;
  while ((m = re.exec(xml)) !== null) {
    out.push({ loc: m[1].trim(), lastmod: m[2].trim() });
  }
  return out;
}

// https://oilprice.com/Energy/Crude-Oil/Some-Headline-Here.html
function deriveHeadline(loc) {
  try {
    const path = new URL(loc).pathname; // /Energy/Crude-Oil/Some-Headline-Here.html
    const segs = path.replace(/^\/+|\/+$/g, '').split('/');
    const last = segs.pop() || '';
    const slug = last.replace(/\.html?$/i, '');
    const headline = decodeURIComponent(slug)
      .replace(/-/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    const category = segs.join('/');
    return { headline, category };
  } catch {
    return { headline: '', category: '' };
  }
}

function* monthsInRange(from, to) {
  let y = from.getUTCFullYear();
  let m = from.getUTCMonth() + 1;
  const endY = to.getUTCFullYear();
  const endM = to.getUTCMonth() + 1;
  while (y < endY || (y === endY && m <= endM)) {
    yield [y, m];
    m++;
    if (m > 12) { m = 1; y++; }
  }
}

export async function crawlOilprice({
  from,
  to,
  topics = TOPICS,
  onRow,
  seenUrls,         // Set of `${url}::${topic}` keys
  checkpoint,
  log = console.error,
} = {}) {
  const wantTopics = new Set(Object.keys(topics));
  let written = 0;
  let calls = 0;

  for (const [y, m] of monthsInRange(from, to)) {
    if (y < 2009) continue; // oilprice archive starts 2009
    const id = `oilprice:${y}-${m}`;
    if (checkpoint?.has(id)) continue;

    const r = await safeFetch(SITEMAP(y, m), { parse: 'text', timeout: 25000, retries: 2 });
    calls++;
    if (!r.ok) {
      log(`  ! oilprice ${y}-${m}: ${r.error}`);
      checkpoint?.add(id);
      await sleep(RATE_MS);
      continue;
    }

    const entries = parseSitemap(r.text);
    let kept = 0;
    for (const e of entries) {
      const dt = new Date(e.lastmod);
      if (isNaN(dt)) continue;
      if (dt < from || dt > to) continue;
      const { headline, category } = deriveHeadline(e.loc);
      if (!headline) continue;
      const hits = classify(`${headline} ${category}`).filter((t) => wantTopics.has(t));
      if (hits.length === 0) continue;
      for (const topic of hits) {
        const key = `${e.loc}::${topic}`;
        if (seenUrls?.has(key)) continue;
        seenUrls?.add(key);
        onRow?.({
          datetime: dt.toISOString(),
          date: dt.toISOString().slice(0, 10),
          source: 'oilprice.com',
          topic,
          headline,
          url: e.loc,
          domain: category || 'oilprice.com',
          country: '',
        });
        written++;
        kept++;
      }
    }
    checkpoint?.add(id);
    log(`  oilprice ${y}-${String(m).padStart(2, '0')}  +${kept} (of ${entries.length} articles)  [total ${written}]`);
    await sleep(RATE_MS);
  }
  return { written, calls };
}
