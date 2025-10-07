#!/usr/bin/env python3
# select_top_ppl.py

import os
import json
import argparse
from glob import glob

def select_and_save(path, proportions):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = [s for s in data if "ppl" in s]

    data_sorted = sorted(data, key=lambda x: x["ppl"], reverse=True)
    n = len(data_sorted)

    base, ext = os.path.splitext(path)
    for prop in proportions:
        k = int(n * prop)
        if prop > 0 and k == 0:
            k = 1
        top_k = data_sorted[:k]

        pct = int(prop * 100)
        out_path = f"{base}_top_{pct}pct{ext}"
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(top_k, fout, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "-i","--inputs", required=True, nargs="+",
        help=""
    )
    parser.add_argument(
        "-p","--proportions", required=True, nargs="+", type=float,
        help="0.1 0.2 0.3 0.4"
    )
    args = parser.parse_args()

    files = sorted({p for pattern in args.inputs for p in glob(pattern)})
    for path in files:
        select_and_save(path, args.proportions)

if __name__ == "__main__":
    main()
