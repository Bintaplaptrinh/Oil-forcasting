#!/usr/bin/env python3
"""
Sentiment scorer for crawled news headlines, using MiniMax-M3 via tokenrouter.com.

Adds a `sentiment` column in [-1, 1] to each row:
  sign      = direction (negative/bad-for-markets ... positive/good-for-markets)
  magnitude = importance / severity (trivial news -> near 0)

Large-CSV friendly: batched API calls, per-headline cache (resume), thread pool,
incremental cache writes (an interrupted run loses nothing).

Usage:
  export TOKENROUTER_API_KEY=sk-...
  python sentiment.py --in data/combined_news.csv --out data/combined_news_scored.csv
  python sentiment.py --in data/combined_news.csv --limit 50      # quick test
  python sentiment.py --in data/combined_news.csv --batch 12 --workers 4

Key resolution order: --api-key, $TOKENROUTER_API_KEY, $OPENAI_API_KEY, or .env file.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

SYSTEM_PROMPT = (
    "You are a financial-news sentiment rater for an oil/energy price forecasting model. "
    "For each numbered headline, output a single sentiment score in the range -1.0 to 1.0:\n"
    "  * SIGN encodes direction of impact on the global economy / energy markets: "
    "negative/risk-off/escalation/disaster < 0, positive/calming/easing/growth > 0.\n"
    "  * MAGNITUDE encodes how important / market-moving the event is: a routine or "
    "trivial item is near 0.0; a major, market-moving event is near +/-1.0.\n"
    "Neutral or irrelevant headlines score 0.0. Respond with ONLY a compact JSON array "
    'like [{"id":1,"score":-0.7},{"id":2,"score":0.2}] and nothing else.'
)


def load_api_key(cli_key):
    if cli_key:
        return cli_key
    for var in ("TOKENROUTER_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if k.strip() in ("TOKENROUTER_API_KEY", "OPENAI_API_KEY") and v:
                return v
    sys.exit("No API key. Set TOKENROUTER_API_KEY, pass --api-key, or add it to .env")


def h(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.map = {}
        self.lock = threading.Lock()
        if self.path.exists():
            for line in self.path.open(encoding="utf-8"):
                try:
                    o = json.loads(line)
                    self.map[o["h"]] = o["s"]
                except Exception:
                    pass
        self.fh = self.path.open("a", encoding="utf-8")

    def get(self, key):
        return self.map.get(key)

    def put(self, key, score):
        with self.lock:
            if key in self.map:
                return
            self.map[key] = score
            self.fh.write(json.dumps({"h": key, "s": score}) + "\n")
            self.fh.flush()

    def close(self):
        self.fh.close()


def parse_scores(text, n):
    text = THINK_RE.sub("", text or "")
    m = ARRAY_RE.search(text)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for item in arr:
        try:
            i = int(item["id"])
            s = float(item["score"])
            out[i] = max(-1.0, min(1.0, s))
        except Exception:
            continue
    return out


def score_batch(client, model, headlines, retries=2):
    numbered = "\n".join(f"{i+1}. {hl}" for i, hl in enumerate(headlines))
    budget = 80 * len(headlines) + 600
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Headlines:\n" + numbered},
                ],
                temperature=0,
                max_tokens=budget,
            )
            scores = parse_scores(r.choices[0].message.content, len(headlines))
            if scores:
                return [scores.get(i + 1, 0.0) for i in range(len(headlines))]
            last_err = "unparseable response"
        except Exception as e:
            last_err = str(e)[:160]
    sys.stderr.write(f"  ! batch failed ({last_err}); assigning 0.0\n")
    return [0.0] * len(headlines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/combined_news.csv")
    ap.add_argument("--out", dest="out", default=None)
    ap.add_argument("--text-col", default="headline")
    ap.add_argument("--model", default="MiniMax-M3")
    ap.add_argument("--base-url", default="https://api.tokenrouter.com/v1")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="score only first N rows (testing)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        sys.exit(f"Input not found: {inp}")
    out = Path(args.out) if args.out else inp.with_name(inp.stem + "_scored.csv")
    cache_path = args.cache or inp.with_name(inp.stem + ".sentiment_cache.jsonl")

    with inp.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if args.text_col not in fieldnames:
        sys.exit(f"Column '{args.text_col}' not in {inp}. Found: {fieldnames}")
    if not rows:
        sys.exit(
            f"No data rows in {inp} (only a header). Crawl some news first, e.g.:\n"
            f"  node crawl.mjs --source=fed       # fast, ~4,300 rows back to 2008\n"
            f"  node crawl.mjs --source=oilprice\n"
            f"  node crawl.mjs --merge-only        # rebuild combined_news.csv\n"
            f"then re-run this script."
        )
    if args.limit:
        rows = rows[: args.limit]

    client = OpenAI(base_url=args.base_url, api_key=load_api_key(args.api_key))
    cache = Cache(cache_path)

    todo = []
    seen = set()
    for r in rows:
        text = (r.get(args.text_col) or "").strip()
        key = h(text)
        if text and cache.get(key) is None and key not in seen:
            seen.add(key)
            todo.append(text)
    print(f"{len(rows)} rows | {len(todo)} unique headlines to score "
          f"| batch={args.batch} workers={args.workers}", file=sys.stderr)

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(score_batch, client, args.model, b): b for b in batches}
        for fut in as_completed(futs):
            b = futs[fut]
            scores = fut.result()
            for text, s in zip(b, scores):
                cache.put(h(text), s)
            done += 1
            if done % 5 == 0 or done == len(batches):
                print(f"  scored {done}/{len(batches)} batches", file=sys.stderr)

    fields = list(rows[0].keys())
    if "sentiment" not in fields:
        fields.append("sentiment")
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            text = (r.get(args.text_col) or "").strip()
            val = cache.get(h(text))
            r["sentiment"] = "" if not text else f"{(val if val is not None else 0.0):.3f}"
            w.writerow(r)
    cache.close()
    print(f"Wrote {len(rows)} rows -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
