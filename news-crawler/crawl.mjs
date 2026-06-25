#!/usr/bin/env node
// News crawler — datetime + headlines for War / Political Economy / Natural Disaster.
// Sources: Federal Reserve (2006+), oilprice.com (2009+), GDELT (2017+),
//          plus CNBC & OPEC via GDELT's domain filter (2017+). Output: CSV.
//
// Usage:
//   node crawl.mjs                         # full default run, all sources
//   node crawl.mjs --source=fed --from=2008-05-01
//   node crawl.mjs --source=oilprice
//   node crawl.mjs --source=gdelt --per-day=50
//   node crawl.mjs --source=cnbc           # CNBC headlines via GDELT
//   node crawl.mjs --source=opec           # OPEC headlines via GDELT
//   node crawl.mjs --topics=war,natural_disaster
//   node crawl.mjs --merge-only            # rebuild combined CSV only
//
// Flags: --source (all|fed|oilprice|gdelt|cnbc|opec|gkg) --from --to --per-day
//        --chunk-days --domain-chunk-days --topics --out --merge-only

import { mkdirSync, existsSync, readFileSync, writeFileSync, createReadStream, readdirSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { TOPICS, TOPIC_KEYS } from './lib/topics.mjs';
import { CsvAppender, COLUMNS, rowToLine } from './lib/csv.mjs';
import { crawlGdelt } from './sources/gdelt.mjs';
import { crawlOilprice } from './sources/oilprice.mjs';
import { crawlFed } from './sources/fed.mjs';
import { crawlGkg } from './sources/gkg.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, v] = a.replace(/^--/, '').split('=');
    return [k, v === undefined ? true : v];
  })
);

const SOURCE = args.source || 'all';
const want = (s) => SOURCE === 'all' || SOURCE === 'both' || SOURCE === s;
const FROM = new Date((args.from || '2008-05-01') + 'T00:00:00Z');
const TO = new Date((args.to || '2026-05-08') + 'T23:59:59Z');
const PER_DAY = parseInt(args['per-day'] || '25', 10);
const CHUNK_DAYS = parseInt(args['chunk-days'] || '7', 10);
const DOMAIN_CHUNK = parseInt(args['domain-chunk-days'] || '30', 10);
const OUT = args.out ? String(args.out) : join(__dirname, 'data');
const SELECTED = (args.topics ? String(args.topics).split(',') : TOPIC_KEYS).filter((t) => TOPIC_KEYS.includes(t));
const topics = Object.fromEntries(SELECTED.map((k) => [k, TOPICS[k]]));

mkdirSync(OUT, { recursive: true });
const FED_CSV = join(OUT, 'fed_news.csv');
const OILPRICE_CSV = join(OUT, 'oilprice_news.csv');
const GDELT_CSV = join(OUT, 'gdelt_news.csv');
const CNBC_CSV = join(OUT, 'cnbc_gdelt_news.csv');
const OPEC_CSV = join(OUT, 'opec_gdelt_news.csv');
const GKG_CSV = join(OUT, 'gkg_news.csv');
const COMBINED_CSV = join(OUT, 'combined_news.csv');
const CKPT = join(OUT, 'checkpoint.json');

function loadCkpt() {
  try { return new Set(JSON.parse(readFileSync(CKPT, 'utf8'))); }
  catch { return new Set(); }
}
const done = loadCkpt();
let dirty = 0;
const checkpoint = {
  has: (id) => done.has(id),
  add: (id) => { done.add(id); if (++dirty % 5 === 0) writeFileSync(CKPT, JSON.stringify([...done])); },
};
const seenUrls = new Set();

function banner(msg) { console.error('\n=== ' + msg + ' ==='); }

function parseCsvLine(line) {
  const vals = [];
  let cur = '', q = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (q) {
      if (c === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (c === '"') q = false;
      else cur += c;
    } else if (c === '"') q = true;
    else if (c === ',') { vals.push(cur); cur = ''; }
    else cur += c;
  }
  vals.push(cur);
  const obj = {};
  COLUMNS.forEach((col, i) => (obj[col] = vals[i] ?? ''));
  return obj;
}

async function readCsv(path) {
  if (!existsSync(path)) return [];
  const rows = [];
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  let header = true;
  for await (const line of rl) {
    if (header) { header = false; continue; }
    if (!line.trim()) continue;
    rows.push(parseCsvLine(line));
  }
  return rows;
}

async function buildCombined() {
  banner('Merging into combined_news.csv');
  const files = readdirSync(OUT).filter((f) => f.endsWith('_news.csv') && f !== 'combined_news.csv');
  let all = [];
  for (const f of files) all = all.concat(await readCsv(join(OUT, f)));
  const seen = new Set();
  const deduped = [];
  for (const r of all) {
    const k = r.url + '::' + r.topic;
    if (seen.has(k)) continue;
    seen.add(k);
    deduped.push(r);
  }
  deduped.sort((a, b) => (a.datetime < b.datetime ? -1 : a.datetime > b.datetime ? 1 : 0));
  let out = COLUMNS.join(',') + '\n';
  for (const r of deduped) out += rowToLine(r);
  writeFileSync(COMBINED_CSV, out);
  console.error('Combined rows: ' + deduped.length + ' (from ' + files.join(', ') + ') -> ' + COMBINED_CSV);
}

async function main() {
  console.error('News crawler | source=' + SOURCE + ' | ' + FROM.toISOString().slice(0, 10) + '..' + TO.toISOString().slice(0, 10) + ' | topics=' + SELECTED.join(','));
  if (args['merge-only']) { await buildCombined(); return; }

  const gdeltFrom = FROM < new Date('2017-01-01T00:00:00Z') ? new Date('2017-01-01T00:00:00Z') : FROM;

  if (want('fed')) {
    banner('Federal Reserve (JSON feeds, 2006+)');
    const csv = new CsvAppender(FED_CSV);
    const res = await crawlFed({ from: FROM, to: TO, topics, seenUrls, checkpoint, onRow: (r) => csv.write(r) });
    await csv.close();
    console.error('Federal Reserve done: ' + res.written + ' rows, ' + res.calls + ' requests');
  }

  if (want('oilprice')) {
    banner('oilprice.com (sitemaps, 2009+)');
    const csv = new CsvAppender(OILPRICE_CSV);
    const res = await crawlOilprice({ from: FROM, to: TO, topics, seenUrls, checkpoint, onRow: (r) => csv.write(r) });
    await csv.close();
    console.error('oilprice.com done: ' + res.written + ' rows, ' + res.calls + ' requests');
  }

  if (want('gdelt')) {
    banner('GDELT (DOC 2.0, ' + gdeltFrom.toISOString().slice(0, 10) + '+)');
    const csv = new CsvAppender(GDELT_CSV);
    const res = await crawlGdelt({ topics, from: gdeltFrom, to: TO, chunkDays: CHUNK_DAYS, perDay: PER_DAY, seenUrls, checkpoint, onRow: (r) => csv.write(r) });
    await csv.close();
    console.error('GDELT done: ' + res.written + ' rows, ' + res.calls + ' requests');
  }

  if (want('cnbc')) {
    banner('CNBC via GDELT (domain:cnbc.com, 2017+)');
    const csv = new CsvAppender(CNBC_CSV);
    const res = await crawlGdelt({ topics, from: gdeltFrom, to: TO, chunkDays: DOMAIN_CHUNK, perDay: PER_DAY, domain: 'cnbc.com', sourceLabel: 'CNBC (GDELT)', idPrefix: 'cnbc', seenUrls, checkpoint, onRow: (r) => csv.write(r) });
    await csv.close();
    console.error('CNBC done: ' + res.written + ' rows, ' + res.calls + ' requests');
  }

  if (want('opec')) {
    banner('OPEC via GDELT (domain:opec.org, 2017+)');
    const csv = new CsvAppender(OPEC_CSV);
    const res = await crawlGdelt({ topics, from: gdeltFrom, to: TO, chunkDays: DOMAIN_CHUNK, perDay: PER_DAY, domain: 'opec.org', sourceLabel: 'OPEC (GDELT)', idPrefix: 'opec', seenUrls, checkpoint, onRow: (r) => csv.write(r) });
    await csv.close();
    console.error('OPEC done: ' + res.written + ' rows, ' + res.calls + ' requests');
  }

  if (want('gkg')) {
    banner('GKG (GDELT Global Knowledge Graph: OPEC+/conflict/sanctions + OPEC countries, ' + gdeltFrom.toISOString().slice(0, 10) + '+)');
    const csv = new CsvAppender(GKG_CSV);
    const res = await crawlGkg({ from: gdeltFrom, to: TO, chunkDays: CHUNK_DAYS, perDay: PER_DAY, domainChunkDays: DOMAIN_CHUNK, seenUrls, checkpoint, onRow: (r) => csv.write(r) });
    await csv.close();
    console.error('GKG done: ' + res.written + ' rows, ' + res.calls + ' requests');
  }

  writeFileSync(CKPT, JSON.stringify([...done]));
  await buildCombined();
  console.error('\nAll done.');
}

main().catch((e) => {
  console.error('FATAL:', e);
  try { writeFileSync(CKPT, JSON.stringify([...done])); } catch {}
  process.exit(1);
});
