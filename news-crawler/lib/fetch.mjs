// Shared fetch utility: timeout, retries, polite back-off.
// Returns parsed JSON when possible, otherwise the raw text body.

export async function safeFetch(url, opts = {}) {
  const {
    timeout = 20000,
    retries = 2,
    headers = {},
    parse = 'auto', // 'json' | 'text' | 'auto'
    backoffMs = 2000,
  } = opts;

  let lastError;
  for (let i = 0; i <= retries; i++) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeout);
      const res = await fetch(url, {
        signal: controller.signal,
        headers: {
          'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) NewsCrawler/1.0',
          ...headers,
        },
      });
      clearTimeout(timer);
      if (!res.ok) {
        const body = await res.text().catch(() => '');
        throw new Error(`HTTP ${res.status}: ${body.slice(0, 160)}`);
      }
      const text = await res.text();
      if (parse === 'text') return { ok: true, text };
      if (parse === 'json') return { ok: true, data: JSON.parse(text) };
      // auto
      try {
        return { ok: true, data: JSON.parse(text) };
      } catch {
        return { ok: true, text };
      }
    } catch (e) {
      lastError = e;
      if (i < retries) await sleep(backoffMs * (i + 1));
    }
  }
  return { ok: false, error: lastError?.message || 'Unknown error', url };
}

export function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
