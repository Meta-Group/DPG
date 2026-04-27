"""
Minimal local explanation example on Iris.

This script:
1. trains a small RandomForestClassifier
2. fits DPGExplainer
3. explains one sample locally
4. prints key local outputs
5. optionally renders the local paths on the fitted DPG if Graphviz is available
"""

import os
import shutil
import sys

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from dpg import DPGExplainer


def main() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    feature_names = X.columns.tolist()
    target_names = np.unique(y).astype(str).tolist()

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    explainer = DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=target_names,
        dpg_config={
            "dpg": {
                "default": {
                    "perc_var": 1e-9,
                    "decimal_threshold": 6,
                    "n_jobs": 1,
                },
                "graph_construction": {
                    "mode": "execution_trace",
                },
            }
        },
    )
    explainer.fit(X.values)

    sample_id = 0
    local = explainer.explain_local(sample=X.iloc[sample_id].values, sample_id=sample_id)

    print("majority_vote:", local.majority_vote)
    print("class_votes:", local.class_votes)
    print("sample_confidence:")
    for key in [
        "vote_confidence",
        "evidence_score_pred",
        "trace_coverage_score",
        "recombination_rate",
    ]:
        print(f"  {key}: {local.sample_confidence.get(key)}")

    local_df = explainer.local_path_dataframe(local)
    print("\nlocal_path_dataframe head:")
    print(local_df.head().to_string(index=False))

    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    if shutil.which("dot") is None:
        print("\nGraphviz 'dot' not found. Skipping local plot rendering.")
        return

    fig = explainer.plot_local_on_dpg(
        plot_name="iris_local_sample0",
        local_explanation=local,
        true_class_label=str(y.iloc[sample_id]),
        save_dir=results_dir,
        theme="dpg",
        palette="olive",
        layout_template="vertical",
        show=False,
    )
    print("\nSaved local plot to:", os.path.join(results_dir, "iris_local_sample0.png"))
    print("Figure created:", fig is not None)


if __name__ == "__main__":
    main()
