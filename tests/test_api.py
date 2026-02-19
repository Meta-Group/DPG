import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dpg import DPGExplainer


def test_high_level_explainer_global_outputs():
    base_dir = os.getcwd()
    dataset_path = os.path.join(base_dir, "datasets", "custom.csv")
    dataset_raw = pd.read_csv(dataset_path, index_col=0)

    features = dataset_raw.iloc[:, :-1]
    labels = dataset_raw.iloc[:, -1]
    features = features.replace([np.inf, -np.inf], np.nan).fillna(features.mean())
    features = np.round(features, 2)

    model = RandomForestClassifier(n_estimators=5, random_state=27)
    model.fit(features, labels)

    explainer = DPGExplainer(
        model=model,
        feature_names=features.columns,
        target_names=np.unique(labels).astype(str).tolist(),
    )
    explanation = explainer.explain_global(features.values)

    assert explanation.graph is not None
    assert len(explanation.nodes) > 0
    assert not explanation.node_metrics.empty
    assert not explanation.edge_metrics.empty
    assert len(explanation.class_boundaries) > 0
