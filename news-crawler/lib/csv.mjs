// Minimal, dependency-free CSV writer with RFC-4180 escaping
// and incremental append support (so long crawls survive interruption).

import { createWriteStream, existsSync, statSync } from 'node:fs';

export const COLUMNS = [
  'datetime',  // ISO-8601 UTC timestamp
  'date',      // YYYY-MM-DD
  'source',    // GDELT | oilprice.com
  'topic',     // war | political_economy | natural_disaster
  'headline',
  'url',
  'domain',    // publisher domain (GDELT) or oilprice category
  'country',   // source country (GDELT) when available
];

function esc(v) {
  if (v == null) return '';
  const s = String(v);
  if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

export function rowToLine(row) {
  return COLUMNS.map((c) => esc(row[c])).join(',') + '\n';
}

export class CsvAppender {
  constructor(path) {
    this.path = path;
    const fresh = !existsSync(path) || statSync(path).size === 0;
    this.stream = createWriteStream(path, { flags: 'a' });
    if (fresh) this.stream.write(COLUMNS.join(',') + '\n');
  }
  write(row) {
    this.stream.write(rowToLine(row));
  }
  async close() {
    await new Promise((res) => this.stream.end(res));
  }
}
