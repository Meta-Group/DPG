#!/usr/bin/env python3
"""Analyze local explanation stability over nearest-neighbour test samples.

The analysis reuses the locked journal per-sample outputs. For each dataset, it
recovers the final test feature matrix, finds k nearest neighbours among test
samples, and measures whether explanation labels and diagnostic quantities vary
smoothly for similar inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]

PREFERRED_NUMERIC_COLUMNS = [
    "explanation_confidence",
    "support_margin",
    "competitor_exposure",
    "model_vote_agreement",
    "path_purity",
    "score_margin_pred_vs_competitor",
    "num_active_nodes",
    "runtime_ms",
]


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_split_registry(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_test_matrix(data_dir: Path, dataset: str, registry: dict[str, Any] | None) -> np.ndarray:
    dataset_dir = data_dir / dataset
    x_test = np.load(dataset_dir / "X_test.npy")
    if registry is None:
        return np.asarray(x_test, dtype=float)
    split = registry.get("datasets", {}).get(dataset)
    if split is None:
        raise KeyError(f"Dataset '{dataset}' is missing from split registry.")
    idx = np.asarray(split["test_indices"], dtype=int)
    return np.asarray(x_test[idx], dtype=float)


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    if text.strip().lower() in {"", "nan", "none", "null"}:
        return None
    return text


def _explained_label(row: pd.Series) -> str | None:
    for col in ("y_support_pred", "y_local_pred", "y_model_pred"):
        if col in row:
            value = _as_text(row[col])
            if value is not None:
                return value
    return None


def _top_feature(row: pd.Series) -> str | None:
    for col in ("top_feature_idx", "ice_feature_idx"):
        if col in row:
            value = _as_text(row[col])
            if value is not None:
                try:
                    return str(int(float(value)))
                except ValueError:
                    return value
    rule = _as_text(row.get("anchor_rule"))
    if rule:
        return rule
    lore = _as_text(row.get("lore_rule"))
    if lore:
        return lore
    return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _method_key(row: pd.Series) -> str:
    method = str(row["method"])
    mode = _as_text(row.get("graph_construction_mode"))
    if method.startswith("dpg") and mode:
        return f"{method}:{mode}"
    return method


def _nearest_indices(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= 1:
        return np.empty((X.shape[0], 0), dtype=int), np.empty((X.shape[0], 0), dtype=float)
    scaled = StandardScaler().fit_transform(X)
    n_neighbors = min(int(k) + 1, X.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(scaled)
    distances, indices = nn.kneighbors(scaled)
    return indices[:, 1:], distances[:, 1:]


def _dataset_pairs(
    df_method: pd.DataFrame,
    X_test: np.ndarray,
    k: int,
    numeric_columns: list[str],
) -> pd.DataFrame:
    rows = df_method.copy()
    rows["sample_idx"] = rows["sample_idx"].astype(int)
    rows = rows.drop_duplicates(subset=["sample_idx"], keep="first").set_index("sample_idx", drop=False)
    valid_idx = [idx for idx in rows.index.tolist() if 0 <= int(idx) < X_test.shape[0]]
    if len(valid_idx) <= 1:
        return pd.DataFrame()

    valid_idx = sorted(int(i) for i in valid_idx)
    local_X = X_test[valid_idx]
    nn_indices, nn_distances = _nearest_indices(local_X, k=k)
    index_by_position = {pos: sample_idx for pos, sample_idx in enumerate(valid_idx)}
    records: list[dict[str, Any]] = []

    for pos, sample_idx in enumerate(valid_idx):
        src = rows.loc[sample_idx]
        src_label = _explained_label(src)
        src_model = _as_text(src.get("y_model_pred"))
        src_top = _top_feature(src)
        for rank, (n_pos, dist) in enumerate(zip(nn_indices[pos], nn_distances[pos]), start=1):
            neighbour_idx = index_by_position[int(n_pos)]
            dst = rows.loc[neighbour_idx]
            dst_label = _explained_label(dst)
            dst_model = _as_text(dst.get("y_model_pred"))
            dst_top = _top_feature(dst)
            record: dict[str, Any] = {
                "dataset": src["dataset"],
                "method": src["method"],
                "method_key": _method_key(src),
                "sample_idx": int(sample_idx),
                "neighbor_sample_idx": int(neighbour_idx),
                "neighbor_rank": int(rank),
                "input_distance": float(dist),
                "explained_label_agrees": bool(src_label == dst_label) if src_label is not None and dst_label is not None else np.nan,
                "model_label_agrees": bool(src_model == dst_model) if src_model is not None and dst_model is not None else np.nan,
                "top_feature_agrees": bool(src_top == dst_top) if src_top is not None and dst_top is not None else np.nan,
            }
            for col in numeric_columns:
                a = _safe_float(src.get(col))
                b = _safe_float(dst.get(col))
                record[f"{col}_abs_delta"] = abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan
            records.append(record)
    return pd.DataFrame(records)


def _summarize_pairs(pairs: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    agg: dict[str, Any] = {
        "datasets": ("dataset", "nunique"),
        "pairs": ("sample_idx", "count"),
        "mean_input_distance": ("input_distance", "mean"),
        "explained_label_agreement": ("explained_label_agrees", "mean"),
        "model_label_agreement": ("model_label_agrees", "mean"),
        "top_feature_agreement": ("top_feature_agrees", "mean"),
    }
    for col in numeric_columns:
        delta_col = f"{col}_abs_delta"
        if delta_col in pairs.columns:
            agg[f"mean_abs_delta_{col}"] = (delta_col, "mean")
            agg[f"median_abs_delta_{col}"] = (delta_col, "median")
    return pairs.groupby("method_key", as_index=False).agg(**agg).sort_values("method_key").reset_index(drop=True)


def _summarize_by_dataset(pairs: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    agg: dict[str, Any] = {
        "pairs": ("sample_idx", "count"),
        "mean_input_distance": ("input_distance", "mean"),
        "explained_label_agreement": ("explained_label_agrees", "mean"),
        "model_label_agreement": ("model_label_agrees", "mean"),
        "top_feature_agreement": ("top_feature_agrees", "mean"),
    }
    for col in numeric_columns:
        delta_col = f"{col}_abs_delta"
        if delta_col in pairs.columns:
            agg[f"mean_abs_delta_{col}"] = (delta_col, "mean")
    return pairs.groupby(["dataset", "method_key"], as_index=False).agg(**agg).sort_values(["dataset", "method_key"]).reset_index(drop=True)


def _correlation_summary(pairs: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if pairs.empty:
        return pd.DataFrame()
    for method_key, group in pairs.groupby("method_key"):
        for col in numeric_columns:
            delta_col = f"{col}_abs_delta"
            if delta_col not in group:
                continue
            tmp = group[["input_distance", delta_col]].dropna()
            if tmp.shape[0] < 5 or tmp["input_distance"].nunique() < 2 or tmp[delta_col].nunique() < 2:
                rho = np.nan
            else:
                rho = tmp["input_distance"].corr(tmp[delta_col], method="spearman")
            records.append(
                {
                    "method_key": method_key,
                    "metric": col,
                    "pairs": int(tmp.shape[0]),
                    "spearman_input_distance_vs_abs_delta": float(rho) if rho == rho else np.nan,
                }
            )
    return pd.DataFrame(records)


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows).copy()
    headers = [str(c) for c in shown.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in shown.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.4g}")
            else:
                text = str(value)
                values.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    by_dataset: pd.DataFrame,
    correlations: pd.DataFrame,
    source: Path,
    k: int,
) -> None:
    lines: list[str] = [
        "# Local Stability Analysis",
        "",
        f"Source: `{source}`",
        "",
        f"For each dataset and method, this analysis compares every final-test sample with its {k} nearest test neighbours in standardized input space.",
        "Stability is measured by explained-label agreement, top-feature agreement when available, and absolute changes in DPG or shared diagnostic scores.",
        "",
        "## Method Summary",
        "",
        _markdown_table(summary, max_rows=40),
        "",
        "## Input-Distance Correlations",
        "",
        "Positive correlations mean that explanations change more as inputs become farther apart.",
        "",
        _markdown_table(correlations, max_rows=80),
        "",
        "## Dataset Summary",
        "",
        _markdown_table(by_dataset, max_rows=120),
        "",
        "## Writing Guidance",
        "",
        "- Use this as a robustness/stability analysis, not as another raw-fidelity comparison.",
        "- The central DPG claim should be that local diagnostic quantities vary smoothly for nearby samples when the model decision is locally stable, and expose instability when nearby samples disagree.",
        "- If DPG has lower top-feature agreement than feature-ranking baselines, that is not necessarily a failure: DPG is a graph diagnostic, and its strongest stability evidence should come from confidence, margin, competitor exposure, and vote agreement.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze kNN local stability from journal per-sample results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/per_sample_selected_test.csv"),
    )
    parser.add_argument("--data_dir", type=Path, default=Path("experiments_local_explanation/data_numeric"))
    parser.add_argument(
        "--split_registry",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/splits/split_registry.json"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/local_stability_knn"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()

    input_path = _resolve(args.input)
    data_dir = _resolve(args.data_dir)
    split_registry = _load_split_registry(_resolve(args.split_registry)) if args.split_registry else None
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    if args.methods:
        df = df[df["method"].astype(str).isin(set(args.methods))].copy()
    numeric_columns = [col for col in PREFERRED_NUMERIC_COLUMNS if col in df.columns]

    pair_frames: list[pd.DataFrame] = []
    for dataset in sorted(df["dataset"].dropna().astype(str).unique()):
        X_test = _load_test_matrix(data_dir, dataset, registry=split_registry)
        dataset_df = df[df["dataset"].astype(str) == dataset].copy()
        for method in sorted(dataset_df["method"].dropna().astype(str).unique()):
            method_df = dataset_df[dataset_df["method"].astype(str) == method].copy()
            pairs = _dataset_pairs(method_df, X_test=X_test, k=int(args.k), numeric_columns=numeric_columns)
            if not pairs.empty:
                pair_frames.append(pairs)

    pairs_all = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    summary = _summarize_pairs(pairs_all, numeric_columns=numeric_columns)
    by_dataset = _summarize_by_dataset(pairs_all, numeric_columns=numeric_columns)
    correlations = _correlation_summary(pairs_all, numeric_columns=numeric_columns)

    pairs_all.to_csv(out_dir / "local_stability_pairs.csv", index=False)
    summary.to_csv(out_dir / "local_stability_summary.csv", index=False)
    by_dataset.to_csv(out_dir / "local_stability_by_dataset.csv", index=False)
    correlations.to_csv(out_dir / "local_stability_correlations.csv", index=False)
    _write_report(out_dir, summary=summary, by_dataset=by_dataset, correlations=correlations, source=input_path, k=int(args.k))

    print(f"Saved local stability analysis to {out_dir}")
    print(f"  - {out_dir / 'summary.md'}")
    print(f"  - {out_dir / 'local_stability_summary.csv'}")
    print(f"  - {out_dir / 'local_stability_by_dataset.csv'}")
    print(f"  - {out_dir / 'local_stability_correlations.csv'}")
    print(f"  - {out_dir / 'local_stability_pairs.csv'}")


if __name__ == "__main__":
    main()
