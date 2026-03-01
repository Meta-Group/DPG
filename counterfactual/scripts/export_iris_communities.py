"""
Export DPG communities and constraints (boundaries) for the Iris dataset.

Works on both the old API (pre-Feb 2026, using extract_graph_metrics + LPA)
and the new API (post-Feb 2026, using extract_class_boundaries + extract_communities + absorbing Markov chain).
"""
import sys
import os
import json
import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from dpg.core import DecisionPredicateGraph
from metrics.nodes import NodeMetrics
from metrics.graph import GraphMetrics


def main(output_path):
    seed = 42
    n_learners = 5

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=seed
    )

    model = RandomForestClassifier(n_estimators=n_learners, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)

    target_names = np.unique(y_train).astype(str).tolist()

    dpg = DecisionPredicateGraph(
        model=model,
        feature_names=iris.feature_names,
        target_names=target_names,
    )
    dot = dpg.fit(X_train)
    dpg_model, nodes_list = dpg.to_networkx(dot)

    df_nodes = NodeMetrics.extract_node_metrics(dpg_model, nodes_list)

    results = {"api": None, "communities": None, "class_boundaries": None, "clusters": None}

    # Try new API first (extract_class_boundaries + extract_communities)
    has_new_api = hasattr(GraphMetrics, "extract_class_boundaries") and hasattr(GraphMetrics, "extract_communities")

    if has_new_api:
        results["api"] = "new (absorbing Markov chain)"

        class_boundaries = GraphMetrics.extract_class_boundaries(
            dpg_model, nodes_list, target_names=target_names
        )
        results["class_boundaries"] = class_boundaries

        communities = GraphMetrics.extract_communities(
            dpg_model, df_nodes, nodes_list, threshold_clusters=0.2
        )
        # Convert community data for serialization
        communities_serializable = {}
        for key, value in communities.items():
            if isinstance(value, dict):
                communities_serializable[key] = {str(k): v for k, v in value.items()}
            else:
                communities_serializable[key] = value
        results["clusters"] = communities_serializable

        # Also get the LPA-based metrics for comparison
        lpa_metrics = GraphMetrics.extract_graph_metrics_lpa(
            dpg_model, nodes_list, target_names=target_names
        )
        lpa_communities = []
        for comm in lpa_metrics.get("Communities", []):
            lpa_communities.append(sorted(comm))
        results["lpa_communities"] = lpa_communities
        results["lpa_class_bounds"] = lpa_metrics.get("Class Bounds", {})

    else:
        results["api"] = "old (LPA-based)"

        graph_metrics = GraphMetrics.extract_graph_metrics(
            dpg_model, nodes_list, target_names=target_names
        )
        communities_raw = graph_metrics.get("Communities", [])
        communities_serializable = []
        for comm in communities_raw:
            communities_serializable.append(sorted(comm))
        results["communities"] = communities_serializable
        results["class_boundaries"] = graph_metrics.get("Class Bounds", {})

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results written to {output_path}")
    print(f"API used: {results['api']}")
    print(f"Class Boundaries keys: {list((results.get('class_boundaries') or {}).keys())}")
    if results.get("clusters"):
        print(f"Cluster keys: {list(results['clusters'].keys())}")
    if results.get("communities"):
        print(f"Communities count: {len(results['communities'])}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        out = sys.argv[1]
    else:
        out = os.path.join("outputs", "iris_dpg_communities_current.json")
    main(out)
