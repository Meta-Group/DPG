# dpg/__init__.py
from .core import DecisionPredicateGraph
from .explainer import DPGExplainer, DPGExplanation
from .themes import DPG_CLASS_PALETTE, DPG_COLORS, DPG_OLIVE_CLASS_PALETTE, resolve_theme_context
from .visualizer import (
    class_feature_predicate_counts,
    class_lookup_from_target_names,
    classwise_feature_bounds_from_communities,
    plot_class_feature_complexity,
    plot_dpg,
    plot_dpg_class_bounds_vs_dataset_feature_ranges,
    plot_dpg_constraints_overview,
    plot_dpg_reg,
    plot_lec_vs_rf_importance,
    plot_lrc_vs_rf_importance,
    plot_sample_using_bc_weights,
    plot_top_lrc_predicate_splits,
    sample_bc_weights,
)

__all__ = [
    "DecisionPredicateGraph",
    "DPGExplainer",
    "DPGExplanation",
    "DPG_COLORS",
    "DPG_CLASS_PALETTE",
    "DPG_OLIVE_CLASS_PALETTE",
    "resolve_theme_context",
    "plot_dpg",
    "plot_dpg_reg",
    "plot_dpg_constraints_overview",
    "plot_lrc_vs_rf_importance",
    "plot_lec_vs_rf_importance",
    "plot_top_lrc_predicate_splits",
    "sample_bc_weights",
    "plot_sample_using_bc_weights",
    "class_feature_predicate_counts",
    "classwise_feature_bounds_from_communities",
    "plot_class_feature_complexity",
    "plot_dpg_class_bounds_vs_dataset_feature_ranges",
    "class_lookup_from_target_names",
]
