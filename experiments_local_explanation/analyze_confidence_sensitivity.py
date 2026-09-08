#!/usr/bin/env python3
"""Analyze sensitivity of DPG confidence-score variants.

The ECML reviews questioned the equal-weight confidence formula and its
coverage gate. This script reuses locked final-test per-sample outputs and
compares reasonable confidence/risk variants on diagnostic tasks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DPG_METHODS = {"dpg", "dpg_execution_trace"}
COMPONENTS = [
    "support_margin",
    "predicted_class_concentration_top3",
    "model_vote_agreement",
]


def _read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if "method" not in df.columns:
        raise ValueError(f"{path} does not contain a method column")
    out = df[df["method"].astype(str).isin(DPG_METHODS)].copy()
    if out.empty:
        raise ValueError(f"No DPG rows found in {path}")
    for col in [
        *COMPONENTS,
        "trace_coverage_score",
        "explanation_confidence",
        "path_purity",
        "competitor_exposure",
        "disagree_with_model",
        "model_correct",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=0.0, upper=1.0)


def _confidence_variants(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    component_mean = out[COMPONENTS].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    coverage = _clip01(out["trace_coverage_score"]) if "trace_coverage_score" in out else pd.Series(np.nan, index=out.index)

    out["conf_reported"] = _clip01(out["explanation_confidence"])
    out["conf_multiplicative_gate"] = _clip01(coverage * component_mean)
    out["conf_no_gate"] = _clip01(component_mean)
    out["conf_additive_coverage"] = _clip01(
        out[[*COMPONENTS, "trace_coverage_score"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    )
    out["conf_margin_only"] = _clip01(out["support_margin"])
    out["conf_concentration_only"] = _clip01(out["predicted_class_concentration_top3"])
    out["conf_vote_agreement_only"] = _clip01(out["model_vote_agreement"])
    out["conf_path_purity"] = _clip01(out["path_purity"]) if "path_purity" in out else np.nan
    out["risk_competitor_exposure"] = _clip01(out["competitor_exposure"]) if "competitor_exposure" in out else np.nan

    confidence_cols = [
        "conf_reported",
        "conf_multiplicative_gate",
        "conf_no_gate",
        "conf_additive_coverage",
        "conf_margin_only",
        "conf_concentration_only",
        "conf_vote_agreement_only",
        "conf_path_purity",
    ]
    for col in confidence_cols:
        out[f"risk_low_{col.removeprefix('conf_')}"] = 1.0 - _clip01(out[col])
    return out


def _metric(y_true: pd.Series, score: pd.Series) -> tuple[float, float, int, float]:
    valid = pd.DataFrame({"y": y_true, "score": score}).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return np.nan, np.nan, 0, np.nan
    y = valid["y"].astype(int).to_numpy()
    s = valid["score"].astype(float).to_numpy()
    positive_rate = float(np.mean(y))
    if len(np.unique(y)) < 2 or len(np.unique(s)) < 2:
        return np.nan, np.nan, int(valid.shape[0]), positive_rate
    return (
        float(roc_auc_score(y, s)),
        float(average_precision_score(y, s)),
        int(valid.shape[0]),
        positive_rate,
    )


def _bootstrap_mean(values: Iterable[float], rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    if arr.size == 1 or n_boot <= 0:
        val = float(np.mean(arr))
        return val, val, val
    boot = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        boot[idx] = float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
    return float(np.mean(arr)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _per_dataset_metrics(df: pd.DataFrame, risk_cols: list[str]) -> pd.DataFrame:
    targets = {
        "local_disagreement": df["disagree_with_model"].astype(bool).astype(int),
        "model_error": (~df["model_correct"].astype(bool)).astype(int),
    }
    rows = []
    for method, method_df in df.groupby("method", dropna=False):
        method_idx = method_df.index
        for dataset, sub in method_df.groupby("dataset", dropna=False):
            sub_idx = sub.index
            for target_name, target in targets.items():
                y = target.loc[sub_idx]
                for score_name in risk_cols:
                    auroc, auprc, n_samples, positive_rate = _metric(y, sub[score_name])
                    rows.append(
                        {
                            "method": method,
                            "dataset": dataset,
                            "target": target_name,
                            "score": score_name,
                            "n_samples": n_samples,
                            "positive_rate": positive_rate,
                            "auroc": auroc,
                            "auprc": auprc,
                        }
                    )
        # Include a pooled row for interpretation, but keep dataset means primary.
        for target_name, target in targets.items():
            y = target.loc[method_idx]
            for score_name in risk_cols:
                auroc, auprc, n_samples, positive_rate = _metric(y, method_df[score_name])
                rows.append(
                    {
                        "method": method,
                        "dataset": "__pooled__",
                        "target": target_name,
                        "score": score_name,
                        "n_samples": n_samples,
                        "positive_rate": positive_rate,
                        "auroc": auroc,
                        "auprc": auprc,
                    }
                )
    return pd.DataFrame(rows)


def _summary(per_dataset: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    main = per_dataset[per_dataset["dataset"].astype(str).ne("__pooled__")].copy()
    for (method, target, score), sub in main.groupby(["method", "target", "score"], dropna=False):
        auroc_mean, auroc_low, auroc_high = _bootstrap_mean(sub["auroc"], rng, n_boot)
        auprc_mean, auprc_low, auprc_high = _bootstrap_mean(sub["auprc"], rng, n_boot)
        rows.append(
            {
                "method": method,
                "target": target,
                "score": score,
                "n_datasets": int(sub["dataset"].nunique()),
                "mean_positive_rate": float(sub["positive_rate"].mean()),
                "auroc_mean": auroc_mean,
                "auroc_ci95_low": auroc_low,
                "auroc_ci95_high": auroc_high,
                "auprc_mean": auprc_mean,
                "auprc_ci95_low": auprc_low,
                "auprc_ci95_high": auprc_high,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["method", "target", "auroc_mean"], ascending=[True, True, False]).reset_index(drop=True)


def _stability(df: pd.DataFrame, risk_cols: list[str]) -> pd.DataFrame:
    rows = []
    ref = "risk_low_reported"
    for method, method_df in df.groupby("method", dropna=False):
        for score in risk_cols:
            if score == ref:
                valid = method_df[[ref]].replace([np.inf, -np.inf], np.nan).dropna()
                corr = 1.0 if valid[ref].nunique() > 1 else np.nan
            else:
                valid = method_df[[ref, score]].replace([np.inf, -np.inf], np.nan).dropna()
                corr = np.nan
                if valid.shape[0] >= 4 and valid[ref].nunique() > 1 and valid[score].nunique() > 1:
                    corr = float(valid[ref].corr(valid[score], method="spearman"))
            rows.append(
                {
                    "method": method,
                    "reference_score": ref,
                    "score": score,
                    "n_samples": int(valid.shape[0]),
                    "spearman_rho": corr,
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "spearman_rho"], ascending=[True, False]).reset_index(drop=True)


def _calibration_bins(df: pd.DataFrame, risk_cols: list[str], n_bins: int) -> pd.DataFrame:
    rows = []
    target_cols = {
        "local_disagreement": df["disagree_with_model"].astype(bool).astype(int),
        "model_error": (~df["model_correct"].astype(bool)).astype(int),
    }
    for method, method_df in df.groupby("method", dropna=False):
        method_idx = method_df.index
        for target_name, target in target_cols.items():
            y = target.loc[method_idx]
            for score in risk_cols:
                valid = pd.DataFrame({"score": method_df[score], "target": y}).replace([np.inf, -np.inf], np.nan).dropna()
                if valid.shape[0] < n_bins or valid["score"].nunique() < 2:
                    continue
                valid["bin"] = pd.qcut(valid["score"], q=min(n_bins, valid["score"].nunique()), duplicates="drop")
                for bin_id, (_, sub) in enumerate(valid.groupby("bin", observed=False), start=1):
                    rows.append(
                        {
                            "method": method,
                            "target": target_name,
                            "score": score,
                            "bin_id": bin_id,
                            "n_samples": int(sub.shape[0]),
                            "score_min": float(sub["score"].min()),
                            "score_max": float(sub["score"].max()),
                            "score_mean": float(sub["score"].mean()),
                            "observed_rate": float(sub["target"].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, max_rows: int = 20, floatfmt: str = ".4g") -> str:
    if df.empty:
        return ""
    table = df.head(max_rows).copy()
    cols = list(table.columns)

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value)

    rows = [[fmt(row[col]) for col in cols] for _, row in table.iterrows()]
    widths = [max(len(str(col)), *(len(row[idx]) for row in rows)) for idx, col in enumerate(cols)]
    header = "| " + " | ".join(str(col).ljust(widths[idx]) for idx, col in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(cols))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(cols))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_report(out_dir: Path, summary: pd.DataFrame, stability: pd.DataFrame, source_path: Path) -> None:
    disagreement = summary[summary["target"].eq("local_disagreement")].copy()
    model_error = summary[summary["target"].eq("model_error")].copy()
    cols = [
        "method",
        "score",
        "n_datasets",
        "mean_positive_rate",
        "auroc_mean",
        "auroc_ci95_low",
        "auroc_ci95_high",
        "auprc_mean",
        "auprc_ci95_low",
        "auprc_ci95_high",
    ]
    lines = [
        "# Confidence Score Sensitivity",
        "",
        f"Source: `{source_path}`",
        "",
        "This analysis tests whether DPG diagnostic performance depends strongly on the original equal-weight confidence formula and multiplicative trace-coverage gate.",
        "",
        "Scores are evaluated as risk scores, so higher values indicate higher expected disagreement or error. For confidence-like variants, risk is `1 - confidence_variant`.",
        "",
        "## Local Disagreement",
        "",
        _markdown_table(disagreement[cols].sort_values(["method", "auroc_mean"], ascending=[True, False]), max_rows=30),
        "",
        "## Model Error",
        "",
        _markdown_table(model_error[cols].sort_values(["method", "auroc_mean"], ascending=[True, False]), max_rows=30),
        "",
        "## Ranking Stability Against Reported Low Confidence",
        "",
        _markdown_table(stability, max_rows=40),
        "",
        "## Writing Guidance",
        "",
        "- Treat confidence as a diagnostic ranking index, not a calibrated probability.",
        "- If several variants perform similarly, the paper can keep the simple reported formula and state that conclusions are robust to reasonable alternatives.",
        "- If margin-only or competitor-exposure scores dominate, emphasize the class-contrastive DPG diagnostics rather than the aggregate confidence formula.",
        "- Use local-disagreement results as the primary diagnostic-value evidence; model-error results are secondary because model error is affected by dataset difficulty and classifier quality.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DPG confidence-score sensitivity.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/per_sample_selected_test.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/confidence_sensitivity"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--calibration_bins", type=int, default=10)
    args = parser.parse_args()

    df = _confidence_variants(_read_input(args.input))
    risk_cols = [
        "risk_low_reported",
        "risk_low_multiplicative_gate",
        "risk_low_no_gate",
        "risk_low_additive_coverage",
        "risk_low_margin_only",
        "risk_low_concentration_only",
        "risk_low_vote_agreement_only",
        "risk_low_path_purity",
        "risk_competitor_exposure",
    ]
    risk_cols = [c for c in risk_cols if c in df.columns and df[c].notna().any()]

    per_dataset = _per_dataset_metrics(df, risk_cols)
    summary = _summary(per_dataset, n_boot=args.bootstrap, seed=args.seed)
    stability = _stability(df, risk_cols)
    calibration = _calibration_bins(df, risk_cols, n_bins=args.calibration_bins)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "confidence_variants_per_sample.csv", index=False)
    per_dataset.to_csv(args.out_dir / "confidence_sensitivity_per_dataset.csv", index=False)
    summary.to_csv(args.out_dir / "confidence_sensitivity_summary.csv", index=False)
    stability.to_csv(args.out_dir / "confidence_variant_stability.csv", index=False)
    calibration.to_csv(args.out_dir / "confidence_calibration_bins.csv", index=False)
    _write_report(args.out_dir, summary=summary, stability=stability, source_path=args.input)

    print(f"Saved confidence sensitivity analysis to {args.out_dir}")
    print(f"  - confidence_variants_per_sample.csv ({df.shape[0]} rows)")
    print("  - confidence_sensitivity_per_dataset.csv")
    print("  - confidence_sensitivity_summary.csv")
    print("  - confidence_variant_stability.csv")
    print("  - confidence_calibration_bins.csv")
    print("  - summary.md")


if __name__ == "__main__":
    main()
