#!/usr/bin/env python3
"""
Roll scored headlines up into DAILY features for the forecast model.

Input : a scored CSV (from sentiment.py) with columns date, topic, sentiment.
Output: daily_features.csv with one row per calendar day and, for each topic
        (war, political_economy, natural_disaster) plus an "all" bucket:

  {topic}_n          number of news items that day            (volume)
  {topic}_sent_mean  average sentiment                        (mood)
  {topic}_sent_sum   sum of sentiment  (importance-weighted net signal)
  {topic}_intensity  sum of |sentiment| (total attention/severity, unsigned)

This is the table you join to WTI/Brent daily prices on `date`.

Usage:
  python aggregate.py --in data/combined_news_scored.csv --out data/daily_features.csv
  python aggregate.py --in data/combined_news_scored.csv --fill-gaps
"""

import argparse
import csv
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

TOPICS = ["war", "political_economy", "natural_disaster"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/combined_news_scored.csv")
    ap.add_argument("--out", dest="out", default=None)
    ap.add_argument("--fill-gaps", action="store_true",
                    help="emit a row for every calendar day in range, even with no news")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")
    out = Path(args.out) if args.out else inp.with_name("daily_features.csv")

    # bucket[(day, topic)] = [scores...]
    bucket = defaultdict(list)
    days = set()
    with inp.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            d = (r.get("date") or "").strip()
            t = (r.get("topic") or "").strip()
            s = r.get("sentiment", "")
            if not d or t not in TOPICS:
                continue
            try:
                score = float(s)
            except (TypeError, ValueError):
                continue
            bucket[(d, t)].append(score)
            bucket[(d, "all")].append(score)
            days.add(d)

    def stats(scores):
        n = len(scores)
        if n == 0:
            return 0, 0.0, 0.0, 0.0
        ssum = sum(scores)
        intensity = sum(abs(x) for x in scores)
        return n, ssum / n, ssum, intensity

    day_list = sorted(days)
    if args.fill_gaps and day_list:
        start = date.fromisoformat(day_list[0])
        end = date.fromisoformat(day_list[-1])
        day_list = [(start + timedelta(days=i)).isoformat()
                    for i in range((end - start).days + 1)]

    cols = ["date"]
    for t in TOPICS + ["all"]:
        cols += [f"{t}_n", f"{t}_sent_mean", f"{t}_sent_sum", f"{t}_intensity"]

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for d in day_list:
            row = [d]
            for t in TOPICS + ["all"]:
                n, mean, ssum, inten = stats(bucket.get((d, t), []))
                row += [n, f"{mean:.4f}", f"{ssum:.4f}", f"{inten:.4f}"]
            w.writerow(row)

    print(f"Wrote {len(day_list)} daily rows -> {out}")


if __name__ == "__main__":
    main()
