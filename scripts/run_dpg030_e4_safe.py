#!/usr/bin/env python3
"""Run each remaining S5 cell in an isolated subprocess."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

TASKS = [
    ("RandomForestClassifier", 10, 1),
    ("RandomForestClassifier", 25, 0),
    ("RandomForestClassifier", 25, 1),
    ("ExtraTreesClassifier", 10, 0),
    ("ExtraTreesClassifier", 10, 1),
    ("ExtraTreesClassifier", 25, 0),
    ("ExtraTreesClassifier", 25, 1),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/dpg_0_3_0/results/e4_s5_safe.csv"))
    parser.add_argument("--workdir", type=Path, default=Path("experiments/dpg_0_3_0/results/e4_s5_cells"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from scripts.notify_telegram import send_message

    fields = ["scenario", "model", "n_samples", "n_features", "n_estimators", "seed", "status", "error",
              "k1_seconds", "kauto_seconds", "k1_nodes", "kauto_nodes", "node_ratio", "kauto",
              "kauto_violations", "git_commit"]
    send_message(f"DPG E4 safe S5 retry started: {len(TASKS)} isolated cells, sequential execution.")
    completed = ok = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (model, learners, seed) in enumerate(TASKS, start=1):
            cell_output = args.workdir / f"{model}_n{learners}_s{seed}.csv"
            command = [sys.executable, "-u", str(root / "scripts/run_dpg030_e4.py"),
                       "--scenarios", "S5_100k_x50", "--models", model,
                       "--learners", str(learners), "--seeds", str(seed),
                       "--workers", "1", "--output", str(cell_output), "--notify-every", "1"]
            process = subprocess.run(command, cwd=root, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            rows = list(csv.DictReader(cell_output.open(encoding="utf-8"))) if cell_output.exists() else []
            if rows:
                for row in rows:
                    writer.writerow(row)
                    ok += row["status"] == "ok"
                    completed += 1
            else:
                writer.writerow({"scenario": "S5_100k_x50", "model": model, "n_samples": 100000,
                                 "n_features": 50, "n_estimators": learners, "seed": seed,
                                 "status": "error", "error": f"isolated process exit code {process.returncode}",
                                 "k1_seconds": 0.0, "kauto_seconds": 0.0, "k1_nodes": 0, "kauto_nodes": 0,
                                 "node_ratio": "", "kauto": "", "kauto_violations": "", "git_commit": ""})
                completed += 1
            handle.flush()
            message = f"DPG E4 safe S5 progress: {index}/{len(TASKS)} cells; {ok} ok; output={args.output}"
            print(message, flush=True)
            send_message(message)
    message = f"DPG E4 safe S5 finished: {ok}/{completed} ok; output={args.output}"
    print(message, flush=True)
    send_message(message)


if __name__ == "__main__":
    main()
