# Supported Models

DPGExplainer supports various tree-based ensemble models from scikit-learn. This page documents all supported models, their features, and usage examples.

## Overview

DPGExplainer works with any sklearn ensemble model that has an `estimators_` attribute. The framework automatically detects the model type and handles tree structure differences transparently.

## Classification Models

### RandomForestClassifier
Status: ✅ Fully supported

The classic ensemble method. Works as expected with DPGExplainer.

```python
from sklearn.ensemble import RandomForestClassifier
from dpg import DPGExplainer

rf = RandomForestClassifier(n_estimators=10, max_depth=5)
rf.fit(X, y)

explainer = DPGExplainer(rf, feature_names, target_names)
explanation = explainer.explain_global(X)
```

### GradientBoostingClassifier
Status: ✅ Fully supported (NEW!)

Gradient Boosting is now fully supported with automatic tree structure normalization. No special configuration needed.

```python
from sklearn.ensemble import GradientBoostingClassifier
from dpg import DPGExplainer

gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
gb.fit(X, y)

# Works exactly like RandomForest - automatic normalization happens internally
explainer = DPGExplainer(gb, feature_names, target_names)
explanation = explainer.explain_global(X)
```

**Technical Note**: GradientBoosting stores trees differently than RandomForest (2D array vs 1D list). DPGExplainer automatically normalizes this difference, so you don't need to do anything special.

### ExtraTreesClassifier
Status: ✅ Fully supported

Extremely randomized trees work seamlessly with DPGExplainer.

```python
from sklearn.ensemble import ExtraTreesClassifier
from dpg import DPGExplainer

et = ExtraTreesClassifier(n_estimators=10, max_depth=5)
et.fit(X, y)

explainer = DPGExplainer(et, feature_names, target_names)
explanation = explainer.explain_global(X)
```

### AdaBoostClassifier
Status: ✅ Fully supported

Adaptive Boosting is fully supported.

```python
from sklearn.ensemble import AdaBoostClassifier
from dpg import DPGExplainer

ada = AdaBoostClassifier(n_estimators=10)
ada.fit(X, y)

explainer = DPGExplainer(ada, feature_names, target_names)
explanation = explainer.explain_global(X)
```

### BaggingClassifier
Status: ✅ Fully supported

Bootstrap Aggregating works with DPGExplainer.

```python
from sklearn.ensemble import BaggingClassifier
from dpg import DPGExplainer

bag = BaggingClassifier(n_estimators=10)
bag.fit(X, y)

explainer = DPGExplainer(bag, feature_names, target_names)
explanation = explainer.explain_global(X)
```

## Regression Models

### RandomForestRegressor
Status: ✅ Fully supported

```python
from sklearn.ensemble import RandomForestRegressor
from dpg import DPGExplainer

rf = RandomForestRegressor(n_estimators=10, max_depth=5)
rf.fit(X, y)

explainer = DPGExplainer(rf, feature_names, target_names=["prediction"])
explanation = explainer.explain_global(X)
```

### GradientBoostingRegressor
Status: ✅ Fully supported (NEW!)

Gradient Boosting regression is now fully supported with the same automatic normalization as the classifier version.

```python
from sklearn.ensemble import GradientBoostingRegressor
from dpg import DPGExplainer

gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
gb.fit(X, y)

explainer = DPGExplainer(gb, feature_names, target_names=["prediction"])
explanation = explainer.explain_global(X)
```

### ExtraTreesRegressor
Status: ✅ Fully supported

```python
from sklearn.ensemble import ExtraTreesRegressor
from dpg import DPGExplainer

et = ExtraTreesRegressor(n_estimators=10, max_depth=5)
et.fit(X, y)

explainer = DPGExplainer(et, feature_names, target_names=["prediction"])
explanation = explainer.explain_global(X)
```

### AdaBoostRegressor
Status: ✅ Fully supported

```python
from sklearn.ensemble import AdaBoostRegressor
from dpg import DPGExplainer

ada = AdaBoostRegressor(n_estimators=10)
ada.fit(X, y)

explainer = DPGExplainer(ada, feature_names, target_names=["prediction"])
explanation = explainer.explain_global(X)
```

## Unsupported Models

The following models are **NOT** supported:

| Model | Reason |
|-------|--------|
| DecisionTreeClassifier / DecisionTreeRegressor | Single tree, not an ensemble |
| LogisticRegression | Linear model, not tree-based |
| SVC / SVR | Support vector machines, not tree-based |
| KNeighborsClassifier / KNeighborsRegressor | Instance-based, not tree-based |
| Neural Networks (MLPClassifier, etc.) | Non-tree-based |
| Linear/Ridge/Lasso Regression | Linear models, not tree-based |

If you try to use an unsupported model, you'll get a clear error message:
```
DPGError: Model must be a tree-based ensemble
```

## Model Comparison

| Model | Type | Status | Trees | Parameters |
|-------|------|--------|-------|------------|
| RandomForestClassifier | Bagging | ✅ | Independent | n_estimators, max_depth |
| GradientBoostingClassifier | Boosting | ✅ NEW | Sequential | n_estimators, learning_rate, max_depth |
| ExtraTreesClassifier | Bagging | ✅ | Independent | n_estimators, max_depth |
| AdaBoostClassifier | Boosting | ✅ | Sequential | n_estimators, learning_rate |
| BaggingClassifier | Bagging | ✅ | Independent | n_estimators |

## GradientBoosting Implementation Details

### What Changed

Previously, using `GradientBoostingClassifier` or `GradientBoostingRegressor` would fail with:

```
AttributeError: 'numpy.ndarray' object has no attribute 'tree_'
```

This happened because GradientBoosting stores trees in a 2D array `(n_classes, n_estimators)` while other models use a 1D list. DPGExplainer expects a consistent 1D structure.

### How It Works Now

DPGExplainer includes an automatic normalizer (`SklearnEnsembleNormalizer`) that:

1. **Detects** GradientBoosting models automatically during initialization
2. **Flattens** the 2D estimators array to a 1D list
3. **Processes** the model normally (transparent to the user)

### Performance Impact

- **Normalization overhead**: < 1ms (one-time, during initialization)
- **Extraction overhead**: None (same iteration logic as before)
- **Memory impact**: Negligible (list is same size as 2D array)

## Testing

All supported models are thoroughly tested:

```bash
# Run model-specific tests
pytest tests/test_sklearn_models.py -v

# Test results
# - GradientBoostingClassifier: binary, multiclass ✓
# - GradientBoostingRegressor: regression ✓
# - Backward compatibility: all other models ✓
# - Total: 203 tests passing
```

## Examples

### Example 1: Compare Multiple Models

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from dpg import DPGExplainer

iris = load_iris()
X, y = iris.data, iris.target

models = {
    'RandomForest': RandomForestClassifier(n_estimators=10),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=10),
}

for name, model in models.items():
    model.fit(X, y)
    explainer = DPGExplainer(model, iris.feature_names, iris.target_names)
    explanation = explainer.explain_global(X)
    
    print(f"{name}: {len(explanation.nodes)} nodes, {len(explanation.graph.edges())} edges")
```

### Example 2: Hyperparameter Exploration with GradientBoosting

```python
from sklearn.ensemble import GradientBoostingClassifier
from dpg import DPGExplainer

gb = GradientBoostingClassifier(
    n_estimators=100,      # Number of boosting stages
    learning_rate=0.1,     # Learning rate (smaller → more conservative)
    max_depth=3,           # Depth of each tree
    subsample=0.8,         # Fraction of samples for fitting
    random_state=42
)

gb.fit(X, y)
explainer = DPGExplainer(gb, feature_names, target_names)
explanation = explainer.explain_global(X)

# Inspect node metrics
print(explanation.node_metrics.head(10))
```

### Example 3: Local Explanations with GradientBoosting

```python
from sklearn.ensemble import GradientBoostingClassifier
from dpg import DPGExplainer

gb = GradientBoostingClassifier(n_estimators=10)
gb.fit(X, y)

explainer = DPGExplainer(gb, feature_names, target_names)
explainer.fit(X)  # Fit the DPG

# Explain a single sample
sample = X[0]
local = explainer.explain_local(sample, sample_id=0)

print(f"Prediction: {local.majority_vote}")
print(f"Class votes: {local.class_votes}")
print(f"Confidence: {local.sample_confidence}")
```

## Tips and Best Practices

1. **Model Size**: DPGExplainer works best with 5-100 trees. Very large ensembles may produce complex graphs.

2. **Tree Depth**: Shallow trees (max_depth=3-5) tend to produce more interpretable DPGs.

3. **GradientBoosting Learning Rate**: Higher learning rates lead to fewer, stronger trees. Experiment with values like 0.01, 0.1, 0.5.

4. **Data Size**: The DPG extraction scales linearly with number of samples and trees. For very large datasets, consider sampling.

5. **Configuration**: Use `perc_var` and `decimal_threshold` to control the DPG complexity:
   ```python
   explainer = DPGExplainer(
       model=gb,
       feature_names=feature_names,
       target_names=target_names,
       dpg_config={
           "dpg": {
               "default": {
                   "perc_var": 1e-9,          # Filter rare paths
                   "decimal_threshold": 2,    # Round thresholds to 2 decimals
                   "n_jobs": -1,              # Use all CPU cores
               }
           }
       }
   )
   ```

## Troubleshooting

### Model Not Recognized

```
DPGError: Model must be a tree-based ensemble
```

**Solution**: Check that your model has an `estimators_` attribute. Use `print(type(model.estimators_))` to verify.

### Out of Memory with Large Models

**Solution**: 
- Reduce `n_estimators`
- Sample your training data: `explainer.explain_global(X[:1000])`
- Increase `perc_var` to filter more paths

### Unexpected Graph Structure

**Solution**: 
- Check `perc_var` - if too high, many paths are filtered
- Verify `decimal_threshold` doesn't oversimplify thresholds
- Try different `graph_construction` modes: `"execution_trace"` vs `"aggregated_transitions"`

## Future Support

We're actively working on support for:
- XGBoost
- LightGBM
- CatBoost

Submit feature requests on [GitHub Issues](https://github.com/Meta-Group/DPG/issues).
