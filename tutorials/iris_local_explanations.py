"""
Tutorial: local DPG explanations on Iris with compact path plots.

This script trains a RandomForestClassifier with exactly 7 learners and
builds local explanations for one correctly classified sample and one
misclassified sample (if available in the test split).
"""

import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Ensure imports resolve to the local repository package even if launched from tutorials/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpg import DPGExplainer


def main():
    if not hasattr(DPGExplainer, "explain_local"):
        raise RuntimeError(
            "Loaded DPGExplainer does not provide explain_local(). "
            f"Loaded module path: {sys.modules[DPGExplainer.__module__].__file__}"
        )

    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = pd.Series(iris.target)
    class_names = [str(n) for n in iris.target_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1, stratify=y
    )

    model = RandomForestClassifier(n_estimators=20, max_depth=2, random_state=27)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Iris test accuracy: {acc:.3f}")

    explainer = DPGExplainer(
        model=model,
        feature_names=X.columns.tolist(),
        target_names=class_names,
        dpg_config={"dpg": {"default": {"perc_var": 0.0, "decimal_threshold": 6, "n_jobs": 1}}},
    )
    explainer.fit(X_train.values)

    out_dir = os.path.join("tutorials", "results")
    os.makedirs(out_dir, exist_ok=True)
    y_test_arr = y_test.to_numpy()
    summary_rows = []
    local_by_pos = {}
    first_correct_pos = None
    first_wrong_pos = None

    # Check the whole test set with consistent sample identifiers:
    # sample_pos = row position in X_test (0..n_test-1), sample_id = original dataset index.
    for sample_pos, (sample_id, row) in enumerate(X_test.iterrows()):
        sample_vec = row.to_numpy()
        y_true = int(y_test_arr[sample_pos])
        y_hat = int(y_pred[sample_pos])
        y_true_name = class_names[y_true]
        y_hat_name = class_names[y_hat]
        correct = y_true == y_hat

        local = explainer.explain_local(
            sample=sample_vec,
            sample_id=int(sample_id),
            validate_graph=True,
        )
        explainer.plot_local_on_dpg(
            plot_name="iris_local_test_on_dpg",
            local_explanation=local,
            true_class_label=y_true_name,
            node_style={"fontsize": 100},
            edge_style={"fontsize": 100},
            graph_style={"fontsize": 100},
            save_dir=out_dir,
            show=False,
        )
        local_by_pos[sample_pos] = local
        if correct and first_correct_pos is None:
            first_correct_pos = sample_pos
        if (not correct) and first_wrong_pos is None:
            first_wrong_pos = sample_pos

        class_support = local.sample_confidence.get("class_support", {}) or {}
        class_scores = local.sample_confidence.get("class_scores", {}) or {}
        total_support = float(sum(class_support.values()))
        attraction_true = float(class_support.get(y_true_name, 0.0))
        attraction_pred = float(class_support.get(y_hat_name, 0.0))
        repulsion_true = max(0.0, total_support - attraction_true)
        repulsion_pred = max(0.0, total_support - attraction_pred)

        summary_rows.append(
            {
                "sample_id": int(sample_id),
                "true_class": y_true_name,
                "pred_class": y_hat_name,
                "correct": bool(correct),
                "majority_vote": local.majority_vote,
                "vote_confidence": local.sample_confidence.get("vote_confidence"),
                "score_margin": local.sample_confidence.get("score_margin"),
                "attraction_true": attraction_true,
                "repulsion_true": repulsion_true,
                "score_true": class_scores.get(y_true_name),
                "attraction_pred": attraction_pred,
                "repulsion_pred": repulsion_pred,
                "score_pred": class_scores.get(y_hat_name),
                "num_paths": int(local.sample_confidence.get("num_paths") or 0),
                "num_active_nodes": int(local.sample_confidence.get("num_active_nodes") or 0),
            }
            | {f"x_{col}": float(row[col]) for col in X_test.columns}
        )

    summary_df = pd.DataFrame(summary_rows).reset_index(drop=True)
    summary_df.to_csv(os.path.join(out_dir, "iris_local_testset_summary.csv"), index=False)
    #print("\nTest set local-explanation summary (head):")
    #print(summary_df.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
