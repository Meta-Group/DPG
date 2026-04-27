"""Run lightweight local DPG explanation experiments on sklearn datasets."""

from __future__ import annotations

import argparse
import itertools
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dpg import DPGExplainer


SUPPORTED_DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}

SUPPORTED_GRAPH_CONSTRUCTION_MODES = {
    "aggregated_transitions",
    "execution_trace",
}


def load_builtin_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{name}'. Supported: {sorted(SUPPORTED_DATASETS)}")
    dataset = SUPPORTED_DATASETS[name](as_frame=True)
    X = dataset.data.copy()
    y = pd.Series(dataset.target, name="target")
    target_names = np.unique(y).astype(str).tolist()
    return X, y, target_names


def build_explainer(
    model: RandomForestClassifier,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    perc_var: float,
    decimal_threshold: int,
    graph_construction_mode: str,
) -> DPGExplainer:
    return DPGExplainer(
        model=model,
        feature_names=list(feature_names),
        target_names=list(target_names),
        dpg_config={
            "dpg": {
                "default": {
                    "perc_var": perc_var,
                    "decimal_threshold": decimal_threshold,
                    "n_jobs": 1,
                },
                "graph_construction": {
                    "mode": graph_construction_mode,
                },
            }
        },
    )


def parse_csv_values(raw: str | Sequence[str], caster=str) -> List[Any]:
    if isinstance(raw, str):
        items = raw.split(",")
    else:
        items = []
        for value in raw:
            items.extend(str(value).split(","))

    parsed = []
    for item in items:
        stripped = item.strip()
        if not stripped:
            continue
        parsed.append(caster(stripped))
    if not parsed:
        raise ValueError("At least one value must be provided.")
    return parsed


def parse_optional_int_values(raw: str | Sequence[str]) -> List[int | None]:
    def _cast(value: str) -> int | None:
        if value == "None":
            return None
        return int(value)

    return parse_csv_values(raw, caster=_cast)


def parse_graph_mode_values(raw: str | Sequence[str]) -> List[str]:
    values = parse_csv_values(raw, caster=str)
    invalid = [value for value in values if value not in SUPPORTED_GRAPH_CONSTRUCTION_MODES]
    if invalid:
        raise ValueError(
            f"Unsupported graph construction mode(s): {invalid}. "
            f"Supported: {sorted(SUPPORTED_GRAPH_CONSTRUCTION_MODES)}"
        )
    return values


def iter_configs(
    datasets: Sequence[str],
    n_estimators_values: Sequence[int],
    max_depth_values: Sequence[int | None],
    perc_var_values: Sequence[float],
    decimal_threshold_values: Sequence[int],
    graph_construction_modes: Sequence[str],
    seed_values: Sequence[int],
) -> Iterable[Dict[str, Any]]:
    for (
        dataset,
        n_estimators,
        max_depth,
        perc_var,
        decimal_threshold,
        graph_construction_mode,
        seed,
    ) in itertools.product(
        datasets,
        n_estimators_values,
        max_depth_values,
        perc_var_values,
        decimal_threshold_values,
        graph_construction_modes,
        seed_values,
    ):
        yield {
            "dataset": dataset,
            "n_estimators": int(n_estimators),
            "max_depth": None if max_depth is None else int(max_depth),
            "perc_var": float(perc_var),
            "decimal_threshold": int(decimal_threshold),
            "graph_construction_mode": graph_construction_mode,
            "seed": int(seed),
        }


def run_single_dataset(
    dataset_name: str,
    n_estimators: int,
    max_depth: int | None,
    perc_var: float,
    decimal_threshold: int,
    graph_construction_mode: str,
    seed: int,
    max_test_samples: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    X, y, target_names = load_builtin_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    model_accuracy = accuracy_score(y_test, model.predict(X_test))

    explainer = build_explainer(
        model=model,
        feature_names=X.columns.tolist(),
        target_names=target_names,
        perc_var=perc_var,
        decimal_threshold=decimal_threshold,
        graph_construction_mode=graph_construction_mode,
    )
    explainer.fit(X_train.values)

    max_samples = min(max_test_samples, len(X_test))
    sample_rows: List[Dict[str, Any]] = []
    for sample_index, (idx, sample_row) in enumerate(X_test.iloc[:max_samples].iterrows()):
        true_label = str(y_test.loc[idx])
        model_pred = str(model.predict(sample_row.to_frame().T)[0])
        local = explainer.explain_local(sample=sample_row.values, sample_id=int(idx))
        local_pred = local.majority_vote
        sample_rows.append(
            {
                "dataset": dataset_name,
                "n_estimators": int(n_estimators),
                "max_depth": None if max_depth is None else int(max_depth),
                "perc_var": float(perc_var),
                "decimal_threshold": int(decimal_threshold),
                "graph_construction_mode": graph_construction_mode,
                "seed": int(seed),
                "sample_index": int(idx),
                "true_label": true_label,
                "model_pred": model_pred,
                "local_pred": local_pred,
                "local_matches_model": bool(local_pred == model_pred),
                "local_correct": bool(local_pred == true_label),
                "vote_confidence": float(local.sample_confidence.get("vote_confidence", 0.0)),
                "evidence_score_pred": _maybe_float(local.sample_confidence.get("evidence_score_pred")),
                "evidence_score_margin": _maybe_float(local.sample_confidence.get("evidence_score_margin")),
                "trace_coverage_score": float(local.sample_confidence.get("trace_coverage_score", 0.0)),
                "recombination_rate": float(local.sample_confidence.get("recombination_rate", 0.0)),
                "num_paths": int(local.sample_confidence.get("num_paths", len(local.tree_paths))),
                "num_valid_paths": int(local.sample_confidence.get("num_valid_paths", 0)),
            }
        )

    per_sample_df = pd.DataFrame(sample_rows)
    summary = {
        "dataset": dataset_name,
        "n_train": int(len(X_train)),
        "n_test_explained": int(len(per_sample_df)),
        "n_estimators": int(n_estimators),
        "max_depth": None if max_depth is None else int(max_depth),
        "perc_var": float(perc_var),
        "decimal_threshold": int(decimal_threshold),
        "graph_construction_mode": graph_construction_mode,
        "seed": int(seed),
        "model_accuracy": float(model_accuracy),
        "local_matches_model_rate": float(per_sample_df["local_matches_model"].mean()) if not per_sample_df.empty else 0.0,
        "local_accuracy": float(per_sample_df["local_correct"].mean()) if not per_sample_df.empty else 0.0,
        "avg_vote_confidence": float(per_sample_df["vote_confidence"].mean()) if not per_sample_df.empty else 0.0,
        "avg_evidence_score_pred": float(per_sample_df["evidence_score_pred"].fillna(0.0).mean()) if not per_sample_df.empty else 0.0,
        "avg_trace_coverage_score": float(per_sample_df["trace_coverage_score"].mean()) if not per_sample_df.empty else 0.0,
        "avg_recombination_rate": float(per_sample_df["recombination_rate"].mean()) if not per_sample_df.empty else 0.0,
        "avg_num_paths": float(per_sample_df["num_paths"].mean()) if not per_sample_df.empty else 0.0,
    }
    return summary, per_sample_df


def run_local_explanation_experiments(
    datasets: Iterable[str],
    out_dir: str,
    n_estimators: int | Sequence[int] = 5,
    max_depth: int | None | Sequence[int | None] = None,
    perc_var: float | Sequence[float] = 1e-9,
    decimal_threshold: int | Sequence[int] = 6,
    graph_construction_mode: str | Sequence[str] = "execution_trace",
    seed: int | Sequence[int] = 42,
    max_test_samples: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(n_estimators, (list, tuple)):
        n_estimators_values = [int(value) for value in n_estimators]
    else:
        n_estimators_values = [int(n_estimators)]
    if isinstance(max_depth, (list, tuple)):
        max_depth_values = [None if value is None else int(value) for value in max_depth]
    else:
        max_depth_values = [None if max_depth is None else int(max_depth)]
    if isinstance(perc_var, (list, tuple)):
        perc_var_values = [float(value) for value in perc_var]
    else:
        perc_var_values = [float(perc_var)]
    if isinstance(decimal_threshold, (list, tuple)):
        decimal_threshold_values = [int(value) for value in decimal_threshold]
    else:
        decimal_threshold_values = [int(decimal_threshold)]
    if isinstance(graph_construction_mode, (list, tuple)):
        graph_construction_modes = [str(value) for value in graph_construction_mode]
    else:
        graph_construction_modes = [str(graph_construction_mode)]
    if isinstance(seed, (list, tuple)):
        seed_values = [int(value) for value in seed]
    else:
        seed_values = [int(seed)]

    summaries: List[Dict[str, Any]] = []
    per_sample_frames: List[pd.DataFrame] = []
    for config in iter_configs(
        datasets=list(datasets),
        n_estimators_values=n_estimators_values,
        max_depth_values=max_depth_values,
        perc_var_values=perc_var_values,
        decimal_threshold_values=decimal_threshold_values,
        graph_construction_modes=graph_construction_modes,
        seed_values=seed_values,
    ):
        summary, per_sample_df = run_single_dataset(
            dataset_name=config["dataset"],
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            perc_var=config["perc_var"],
            decimal_threshold=config["decimal_threshold"],
            graph_construction_mode=config["graph_construction_mode"],
            seed=config["seed"],
            max_test_samples=max_test_samples,
        )
        summaries.append(summary)
        per_sample_frames.append(per_sample_df)

    summary_df = pd.DataFrame(summaries)
    per_sample_df = pd.concat(per_sample_frames, ignore_index=True) if per_sample_frames else pd.DataFrame()

    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    per_sample_df.to_csv(os.path.join(out_dir, "per_sample.csv"), index=False)
    return summary_df, per_sample_df


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight local DPG explanation experiments.")
    parser.add_argument(
        "--datasets",
        default="iris",
        help="Built-in sklearn datasets to run, comma-separated.",
    )
    parser.add_argument("--out_dir", default="experiments/local_explanation/results", help="Output directory for CSV files.")
    parser.add_argument("--n_estimators", default="5", help="RandomForest n_estimators, comma-separated.")
    parser.add_argument("--max_depth", default="None", help="RandomForest max_depth, comma-separated; supports None.")
    parser.add_argument("--perc_var", default="1e-9", help="DPG perc_var, comma-separated.")
    parser.add_argument("--decimal_threshold", default="6", help="DPG decimal_threshold, comma-separated.")
    parser.add_argument(
        "--graph_construction_mode",
        default="execution_trace",
        help="DPG graph construction mode, comma-separated.",
    )
    parser.add_argument("--seed", default="42", help="Random seed, comma-separated.")
    parser.add_argument("--max_test_samples", type=int, default=10, help="Maximum number of test samples to explain.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    datasets = parse_csv_values(args.datasets, caster=str)
    invalid_datasets = [dataset for dataset in datasets if dataset not in SUPPORTED_DATASETS]
    if invalid_datasets:
        raise ValueError(
            f"Unsupported dataset(s): {invalid_datasets}. Supported: {sorted(SUPPORTED_DATASETS)}"
        )
    run_local_explanation_experiments(
        datasets=datasets,
        out_dir=args.out_dir,
        n_estimators=parse_csv_values(args.n_estimators, caster=int),
        max_depth=parse_optional_int_values(args.max_depth),
        perc_var=parse_csv_values(args.perc_var, caster=float),
        decimal_threshold=parse_csv_values(args.decimal_threshold, caster=int),
        graph_construction_mode=parse_graph_mode_values(args.graph_construction_mode),
        seed=parse_csv_values(args.seed, caster=int),
        max_test_samples=args.max_test_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
