#!/usr/bin/env python3
"""
Standalone analysis script for WandB run yv2ab6ib.

Fetches the run from WandB, reconstructs the iris dataset + RF model, and
verifies whether each counterfactual actually changed its predicted class.

Usage:
    python counterfactual/scripts/analyze_run_yv2ab6ib.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── path setup ──────────────────────────────────────────────────────────────
_cf_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_repo_root = os.path.abspath(os.path.join(_cf_root, ".."))
for _p in [_cf_root, _repo_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils.dataset_loader import load_dataset
from utils.config_manager import load_config
from counterfactual_visualizer import plot_pca_with_counterfactuals_clean

# ── constants ────────────────────────────────────────────────────────────────
RUN_ID  = "yv2ab6ib"
ENTITY  = "mllab-ts-universit-di-trieste"
PROJECT = "CounterFactualDPG"

# ── 1. fetch run ─────────────────────────────────────────────────────────────
print(f"Fetching run {RUN_ID} from WandB …")
api = wandb.Api(timeout=60)
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

config  = run.config
summary = run.summary._json_dict

dataset_name  = (config.get("data", {}).get("dataset_name")
                 or config.get("data", {}).get("dataset", "iris"))
feature_names = config.get("feature_names", [])

# ── extract sample ────────────────────────────────────────────────────────────
raw_sample = (config.get("sample")
              or summary.get("sample")
              or summary.get("original_sample"))
if isinstance(raw_sample, list) and feature_names:
    sample = dict(zip(feature_names, raw_sample))
else:
    sample = raw_sample

original_class_wandb = (config.get("sample_class")
                        or summary.get("original_class")
                        or summary.get("sample_class"))

# ── extract counterfactuals ───────────────────────────────────────────────────
cfs_raw = summary.get("final_counterfactuals", [])
if isinstance(cfs_raw, str):
    cfs_raw = json.loads(cfs_raw)
if cfs_raw and isinstance(cfs_raw[0], list) and feature_names:
    cfs = [dict(zip(feature_names, cf)) for cf in cfs_raw]
else:
    cfs = cfs_raw

print(f"  dataset       : {dataset_name}")
print(f"  run name      : {run.name}")
print(f"  sample        : {sample}")
print(f"  original class (wandb): {original_class_wandb}")
print(f"  #counterfactuals      : {len(cfs)}")
print()

# ── 2. load dataset & train model ────────────────────────────────────────────
print("Loading dataset and training model …")
config_path = os.path.join(_cf_root, "configs", dataset_name, "config.yaml")
cfg  = load_config(config_path, repo_root=_repo_root)
seed = getattr(cfg.experiment_params, "seed", 42)
np.random.seed(seed)

ds          = load_dataset(cfg, repo_root=_repo_root)
features_df = ds["features_df"]
labels      = ds["labels"]

X_train, X_test, y_train, y_test = train_test_split(
    features_df, labels, test_size=0.2, random_state=seed, stratify=labels
)
model = RandomForestClassifier(n_estimators=100, random_state=seed)
model.fit(X_train, y_train)
print(f"  model test accuracy: {model.score(X_test, y_test):.4f}")
print()

# ── 3. predict classes ────────────────────────────────────────────────────────
sample_df = pd.DataFrame([sample])
original_class = model.predict(sample_df)[0]

cf_df = pd.DataFrame(cfs)
cf_predicted_classes = model.predict(cf_df)

# Determine target class: the majority predicted class among CFs (or first CF)
from collections import Counter
predicted_counts = Counter(cf_predicted_classes)
target_class = predicted_counts.most_common(1)[0][0]

# ── 4. report ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  CLASS CHANGE ANALYSIS")
print("=" * 60)
print(f"  Original sample class : {original_class}")
print(f"  Expected target class : {target_class}")
print()

failed = []
for idx, (cf, pred) in enumerate(zip(cfs, cf_predicted_classes)):
    changed = pred != original_class
    marker  = "✓" if changed else "✗  <─── DID NOT CHANGE CLASS"
    print(f"  CF {idx + 1}  predicted={pred}  changed={changed}  {marker}")
    if not changed:
        failed.append(idx)

print()
if failed:
    print(f"  ⚠  {len(failed)} counterfactual(s) failed to change class: {[i + 1 for i in failed]}")
    print()
    for idx in failed:
        cf = cfs[idx]
        print(f"  CF {idx + 1} values:")
        for feat, val in cf.items():
            orig_val = sample.get(feat, "?")
            delta    = val - orig_val if isinstance(orig_val, (int, float)) else "?"
            delta_s  = f"{delta:+.5f}" if isinstance(delta, float) else delta
            print(f"    {feat:30s}  original={orig_val:.5f}  cf={val:.5f}  Δ={delta_s}")
    print()
    print("  These CFs are still predicted as the ORIGINAL class by the RF model.")
    print("  This explains the same-colour points in the PCA plot.")
else:
    print("  All counterfactuals successfully changed class.")
print("=" * 60)

# ── 5. detailed CF table ──────────────────────────────────────────────────────
print()
print("Detailed counterfactual table:")
cf_df_display = cf_df.copy()
cf_df_display.index = [f"CF {i + 1}" for i in range(len(cfs))]
cf_df_display["predicted_class"] = cf_predicted_classes
cf_df_display["class_changed"]   = cf_predicted_classes != original_class
print(cf_df_display.to_string())
print()

# ── 6. PCA plot ───────────────────────────────────────────────────────────────
print("Generating PCA plot …")
fig = plot_pca_with_counterfactuals_clean(
    model=model,
    dataset=features_df,
    target=labels,
    sample=sample,
    counterfactuals_df=cf_df,
    cf_predicted_classes=cf_predicted_classes,
)
if fig:
    out_path = os.path.join(_cf_root, "scripts", "outputs",
                            f"pca_analysis_{RUN_ID}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  PCA plot saved to: {out_path}")
    plt.show()
    plt.close(fig)
