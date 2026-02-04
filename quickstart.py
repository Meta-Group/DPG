"""
DPG Analysis Pipeline - Example Script
=====================================
This script demonstrates a complete workflow for:
1. Training a Random Forest model
2. Generating Decision Predicate Graphs (DPG)
3. Extracting and visualizing interpretability metrics
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from dpg.core import DecisionPredicateGraph
from dpg.visualizer import plot_dpg, plot_dpg_communities
from metrics.nodes import NodeMetrics
from metrics.graph import GraphMetrics
from metrics.edges import EdgeMetrics
import yaml

def load_config(config_path):
    # Read YAML configuration used by the DPG library (percentile, thresholds, etc.)
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file: {str(e)}")


def load_dataset(dataset_name, base_dir):
    # Load CSV dataset from datasets/ and split features/labels
    dataset_path = os.path.join(base_dir, "datasets", dataset_name)
    dataset_raw = pd.read_csv(dataset_path, index_col=0)

    features = dataset_raw.iloc[:, :-1]
    target_column = dataset_raw.columns[-1]
    feature_names = dataset_raw.columns[:-1]
    labels = dataset_raw[target_column]

    # Clean data: handle infinities and missing values
    features = features.replace([np.inf, -np.inf], np.nan).fillna(features.mean())
    features_matrix = np.round(features, 2)

    print("Size of X", features_matrix.shape)
    return features_matrix, labels, feature_names


def train_model_cv(model, features_matrix, labels, random_state):
    # Run K-Fold CV to report performance and keep the last trained split
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    accuracy_scores, f1_scores = [], []
    last_train = None

    for train_index, test_index in kf.split(features_matrix):
        X_train, X_test = features_matrix.iloc[train_index], features_matrix.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        last_train = (X_train, y_train)

        print(f"Fold - Accuracy: {accuracy}, F1-Score: {f1}")

    mean_accuracy = np.mean(accuracy_scores)
    metric_suffix = f"acc_{np.round(mean_accuracy, 2)}"
    return metric_suffix, last_train

def main():
    # =============================================================================
    # CONFIGURATION SECTION
    # =============================================================================
    # Tutorial note: adjust these values to point at your dataset and control the run.
    config = {
        "dataset_name": "custom.csv",
        "num_trees": 10,
        "run_tag": "CustomDPG",
        "random_state": 27,
        "config_path": "config.yaml",
        "results_dir": "results/",
    }

    # Load DPG defaults from config.yaml
    config_data = load_config(config["config_path"])
    perc_var = config_data["dpg"]["default"]["perc_var"]

    base_dir = os.getcwd()
    # Load and clean the dataset
    features_matrix, labels, feature_names = load_dataset(
        config["dataset_name"], base_dir
    )

    # Train a Random Forest and estimate performance via CV
    model = RandomForestClassifier(
        n_estimators=config["num_trees"], random_state=config["random_state"]
    )

    metric_suffix, last_train = train_model_cv(
        model, features_matrix, labels, random_state=42
    )
    X_train, y_train = last_train

    # Compose a shared run id for all outputs
    run_id = (
        f"{model.__class__.__name__}_{config['run_tag']}_s{features_matrix.shape[0]}"
        f"_bl{config['num_trees']}_{metric_suffix}_perc_{perc_var}"
    )

    # Build the Decision Predicate Graph from the trained model
    target_names = np.unique(labels).astype(str).tolist()
    dpg_builder = DecisionPredicateGraph(
        model=model, feature_names=feature_names, target_names=target_names
    )
    graph_dot = dpg_builder.fit(X_train.values)
    dpg_graph, dpg_nodes = dpg_builder.to_networkx(graph_dot)

    # Extract graph-level, node-level, and edge-level metrics
    class_boundaries = GraphMetrics.extract_class_boundaries(
        dpg_graph, dpg_nodes, target_names=target_names
    )
    class_boundaries_path = os.path.join(
        base_dir, f"{config['results_dir']}/{run_id}_dpg_class_boundaries.txt"
    )
    with open(class_boundaries_path, "w") as f:
        for key, value in class_boundaries.items():
            f.write(f"{key}: {value}\n")

    node_metrics = NodeMetrics.extract_node_metrics(dpg_graph, dpg_nodes)
    node_metrics_path = os.path.join(
        base_dir, f"{config['results_dir']}/{run_id}_node_metrics.csv"
    )
    node_metrics.to_csv(node_metrics_path, encoding="utf-8")

    edge_metrics = EdgeMetrics.extract_edge_metrics(dpg_graph, dpg_nodes)
    edge_metrics_path = os.path.join(
        base_dir, f"{config['results_dir']}/{run_id}_edge_metrics.csv"
    )
    edge_metrics.to_csv(edge_metrics_path, encoding="utf-8")

    # Render the full DPG visualization
    run_name = f"{run_id}_DPG"
    plot_dpg(
        run_name,
        graph_dot,
        node_metrics,
        edge_metrics,
        save_dir=config["results_dir"],
        class_flag=False,
        export_pdf=True,
    )

    # Detect graph communities and save as a text file
    communities = GraphMetrics.extract_communities(dpg_graph, node_metrics, dpg_nodes)
    communities_path = os.path.join(
        base_dir, f"{config['results_dir']}/{run_id}_dpg_communities.txt"
    )
    GraphMetrics.communities_to_csv(communities, communities_path)

    # Render a community-colored DPG visualization
    plot_dpg_communities(
        run_name,
        graph_dot,
        node_metrics,
        communities,
        save_dir=config["results_dir"],
        class_flag=True,
        export_pdf=True,
    )


if __name__ == "__main__":
    main()
