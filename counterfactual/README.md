# CounterFactualDPG

A counterfactual explanation framework built on top of DPG. This project implements counterfactual generation methods for machine learning models, providing tools for explanation generation, evaluation, and visualization across multiple datasets.

## Project Relations

- **Parent Project**: [DPG](https://github.com/Meta-Group/DPG) - Direct Policy Gradient optimization framework
- **Sibling Project**: [DPG-augmentation](https://github.com/Meta-Group/DPG-augmentation) - Data augmentation techniques for DPG

## Overview

This repository provides a framework for generating and analyzing counterfactual explanations for machine learning models. The implementation focuses on practical counterfactual generation with proper handling of feature constraints, validation mechanisms, and evaluation metrics across multiple benchmark datasets.

### Core Components

- **CounterfactualModel**: Main implementation of counterfactual generation algorithms
- **ConstraintParser**: Framework for defining and handling feature constraints
- **ConstraintScorer** & **ConstraintValidator**: Tools for enforcing validity of generated counterfactuals
- **CounterFactualMetrics**: Evaluation metrics for assessing counterfactual quality
- **CounterFactualVisualizer**: Visualization tools for analyzing counterfactual explanations
- **HeuristicRunner**: Execution framework for running counterfactual experiments


## Installation

This module uses the root repository's virtual environment. From the repository root:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Experiments

Run counterfactual generation experiments using the `run_experiment.py` script:

```bash
# Basic usage with dataset and method
python scripts/run_experiment.py --dataset german_credit --method dpg
python scripts/run_experiment.py --dataset iris --method dice

# Specify a custom config file
python scripts/run_experiment.py --config configs/german_credit/config.yaml --method dpg

# Override configuration parameters
python scripts/run_experiment.py --config configs/iris/config.yaml \
    --set counterfactual.population_size=50 \
    --set experiment_params.seed=123

# Resume a previous WandB run
python scripts/run_experiment.py --resume <wandb_run_id>

# Run in offline mode (no WandB sync)
python scripts/run_experiment.py --config configs/iris/config.yaml --offline
```

### Exporting Results

Generate comparison tables and visualizations using the `export_comparison_results.py` script:

```bash
# Full export (fetches from WandB)
python scripts/export_comparison_results.py

# Regenerate visualizations from existing local data only
python scripts/export_comparison_results.py --local-only

# Fetch specific run IDs from a YAML file
python scripts/export_comparison_results.py --ids plans/dataset_run_ids.yaml

# Fetch multiple runs for a specific dataset and method
python scripts/export_comparison_results.py --dataset diabetes --method dpg --multiple-max 300
```

Results are exported to `outputs/_comparison_results/`, including comparison tables, winner heatmaps, radar charts, and per-dataset visualizations.

## Additional Scripts

The repository includes several helper scripts for batch processing and configuration management:

- `run_all_experiments`: Run experiments across multiple datasets in parallel
- `tune_random_forest`: Hyperparameter tuning for classification models with standard metrics and custom scoring based on DPG constraint quality
- `run_constraint_extraction`: Extract and analyze DPG constraints separately
- Additional utilities for configuration generation and data organization

Run individual scripts with `--help` for usage options.

## Optional: Clean Notebook Outputs

nbstripout is recommended to keep notebooks clean of output cells. Install globally and enable as a git hook:

```bash
pip install nbstripout
nbstripout --install
```

## Contributing

This repository and the related DPG projects are under active development. We welcome contributions, bug reports, and suggestions. Please feel free to open issues or submit PRs.