#!/usr/bin/env python3
"""Expand validation-selected journal configs to a fixed seed list."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def replace_seed(config_id: object, seed: int) -> str:
    text = str(config_id)
    updated = re.sub(r"_s\d+(?=$|__)", f"_s{seed}", text)
    if updated == text and not text.endswith(f"_s{seed}"):
        updated = f"{text}_s{seed}"
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand selected configs to multiple final-test seeds.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="27,42,100,101,102,103,104,105,106,107")
    parser.add_argument("--exclude_methods", type=str, default="")
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    exclude = {item.strip() for item in args.exclude_methods.split(",") if item.strip()}

    frames = []
    for path in args.inputs:
        df = pd.read_csv(path, low_memory=False)
        df["seed_expansion_source"] = str(path)
        frames.append(df)
    selected = pd.concat(frames, ignore_index=True, sort=False)
    if exclude:
        selected = selected[~selected["method"].astype(str).isin(exclude)].copy()

    expanded = []
    for _, row in selected.iterrows():
        for seed in seeds:
            item = row.copy()
            item["seed"] = seed
            item["config_id"] = replace_seed(item["config_id"], seed)
            expanded.append(item)

    out = pd.DataFrame(expanded).drop_duplicates(subset=["dataset", "method", "config_id", "seed"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"Saved {args.out} ({len(out)} rows)")
    print("Datasets:", out["dataset"].nunique())
    print("Methods:", ",".join(sorted(out["method"].astype(str).unique())))
    print("Seeds:", ",".join(str(seed) for seed in sorted(out["seed"].astype(int).unique())))
    print("Rows by method:")
    print(out.groupby("method").size().sort_index().to_string())


if __name__ == "__main__":
    main()
