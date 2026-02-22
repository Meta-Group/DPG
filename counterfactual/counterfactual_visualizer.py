import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
# Suppress matplotlib tight_layout warnings which occur with complex multi-panel figures
warnings.filterwarnings('ignore', message='.*tight_layout.*')
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

def plot_explainer_summary(explainer, original_sample, counterfactual):
    """
    Display a text summary of the counterfactual explanation inside a matplotlib figure, styled like other plots.
    Args:
        explainer: CounterFactualExplainer object with explanation details.
        original_sample: dict of original sample feature values.
        counterfactual: dict of counterfactual feature values.
    """
    # Compose summary text (customize as needed)
    summary_lines = []
    summary_lines.append("Counterfactual Explanation Summary\n")
    summary_lines.append(f"Target class: {explainer.target_class}")
    summary_lines.append("")
    summary_lines.append("Feature changes:")
    for feature in original_sample:
        orig = original_sample[feature]
        cf = counterfactual[feature]
        if orig != cf:
            summary_lines.append(f"- {feature}: {orig} → {cf} (Δ {cf - orig:+.2f})")
        else:
            summary_lines.append(f"- {feature}: {orig} (no change)")
    summary_lines.append("")
    if hasattr(explainer, 'explanation') and explainer.explanation:
        summary_lines.append("Explanation:")
        summary_lines.append(str(explainer.explanation))
    elif hasattr(explainer, 'get_summary'):
        summary_lines.append("Explanation:")
        summary_lines.append(str(explainer.get_summary()))
    summary = "\n".join(summary_lines)

    # Plot the text inside a figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    ax.set_frame_on(False)
    ax.text(0.5, 0.5, summary, fontsize=13, ha='center', va='center', wrap=True,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='whitesmoke', edgecolor='gray', alpha=0.95),
            transform=ax.transAxes)
    ax.set_title('Counterfactual Explanation Summary', fontsize=15, fontweight='bold', pad=18)
    plt.tight_layout()
    plt.close(fig)
    return fig


def set_default_style():
    """Set default matplotlib style for counterfactual visualizations."""
    plt.rcParams['font.size'] = 12  # Adjusts the default font size
    plt.rcParams['axes.labelsize'] = 16  # Font size for x and y labels
    plt.rcParams['axes.titlesize'] = 16  # Font size for the plot title
    plt.rcParams['xtick.labelsize'] = 12  # Font size for x-axis tick labels
    plt.rcParams['ytick.labelsize'] = 12  # Font size for y-axis tick labels


set_default_style()


def _normalize_constraint_dict(class_constraints):
    """Convert constraint format (list or dict) to {feature: {'min': ..., 'max': ...}} dict."""
    if isinstance(class_constraints, list):
        return {c['feature']: {'min': c.get('min'), 'max': c.get('max')} for c in class_constraints}
    elif isinstance(class_constraints, dict):
        return {k: {'min': v.get('min'), 'max': v.get('max')} for k, v in class_constraints.items()}
    return {}


def plot_constraints(constraints, overlapping=False, class_colors=None, class_colors_list=None, sample=None, sample_class=None):
    """
    Visualize feature constraints for each class using horizontal bar ranges.
    
    Parameters:
    -----------
    constraints : dict
        Dictionary of constraints per class
    overlapping : bool
        If True, show all classes overlapping on one plot.
        If False, show classes side-by-side (default).
    class_colors : dict, optional
        Dictionary mapping class names to colors
    class_colors_list : list, optional
        List of colors for classes by index
    sample : dict, optional
        Sample data to plot on the constraints graph.
        Keys should match feature names (e.g., 'sepal length (cm)')
    sample_class : int, optional
        The class of the sample. Used to color the sample point according to its class.
    """
    # Default colors if not provided
    if class_colors is None:
        class_colors = {}
    if class_colors_list is None:
        class_colors_list = ['purple', 'green', 'orange', 'red', 'blue', 'yellow', 'pink', 'cyan']
    
    # Extract unique features
    features = []
    for class_name in constraints:
        for constraint in constraints[class_name]:
            feature = constraint['feature'].replace('_', ' ')
            if feature not in features:
                features.append(feature)
    
    # Use provided class colors
    class_names = list(constraints.keys())
    colors = [class_colors.get(cn, class_colors_list[i % len(class_colors_list)]) for i, cn in enumerate(class_names)]
    
    if overlapping:
        # Single plot with overlapping bars
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        n_features = len(features)
        n_classes = len(constraints)
        
        # Calculate bar positions with offset for overlapping
        bar_height = 0.25
        y_positions = np.arange(n_features)
        
        for idx, class_name in enumerate(class_names):
            class_constraints = constraints[class_name]
            # Handle both formats:
            # 1. List format: [{'feature': 'Age', 'min': x, 'max': y}, ...]
            # 2. Dict format: {'Age': {'min': x, 'max': y}, ...}
            constraint_dict = {k.replace('_', ' '): v for k, v in _normalize_constraint_dict(class_constraints).items()}
            
            # Offset each class slightly
            y_offset = (idx - n_classes/2 + 0.5) * bar_height
            
            for i, feature in enumerate(features):
                if feature in constraint_dict:
                    c = constraint_dict[feature]
                    min_val = c['min'] if c['min'] is not None else 0
                    max_val = c['max'] if c['max'] is not None else 10
                    
                    # Determine the range
                    if c['min'] is None:
                        # Only max constraint
                        ax.barh(i + y_offset, max_val, left=0, height=bar_height, 
                               color=colors[idx], alpha=0.6, edgecolor='black', 
                               linewidth=1.2, label=class_name if i == 0 else "")
                    elif c['max'] is None:
                        # Only min constraint
                        range_width = 10 - min_val
                        ax.barh(i + y_offset, range_width, left=min_val, height=bar_height, 
                               color=colors[idx], alpha=0.6, edgecolor='black', 
                               linewidth=1.2, label=class_name if i == 0 else "")
                    else:
                        # Both min and max
                        range_width = max_val - min_val
                        ax.barh(i + y_offset, range_width, left=min_val, height=bar_height, 
                               color=colors[idx], alpha=0.6, edgecolor='black', 
                               linewidth=1.2, label=class_name if i == 0 else "")
        
        # Plot sample data if provided
        if sample is not None:
            sample_plotted = False
            # Determine sample color based on sample_class
            if sample_class is not None and 0 <= sample_class < len(class_colors_list):
                sample_color = class_colors_list[sample_class]
            else:
                sample_color = 'red'
            
            for i, feature in enumerate(features):
                # Try to find matching feature in sample (handle variations in naming)
                sample_value = None
                for key in sample.keys():
                    if feature.lower() in key.lower() or key.lower() in feature.lower():
                        sample_value = sample[key]
                        break
                
                if sample_value is not None:
                    # Plot sample as a vertical line (red) and marker (class color)
                    ax.plot([sample_value, sample_value], [i - 0.4, i + 0.4], 
                           color='red', linewidth=3, linestyle='-', zorder=10,
                           label='Sample' if not sample_plotted else "")
                    ax.scatter([sample_value], [i], color=sample_color, s=100, 
                              marker='D', edgecolor='darkred', linewidth=1.5, zorder=11)
                    sample_plotted = True
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(features, fontsize=11)
        ax.set_xlabel('Value Range', fontsize=12, fontweight='bold')
        ax.set_title('Feature Constraints per Class (Overlapping View)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, 8)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        plt.show()
        
    else:
        # Side-by-side plots (original view)
        n_classes = len(constraints)
        n_features = len(features)
        
        fig, axes = plt.subplots(1, n_classes, figsize=(16, 6), sharey=True)
        if n_classes == 1:
            axes = [axes]
        
        for idx, (class_name, ax) in enumerate(zip(constraints.keys(), axes)):
            class_constraints = constraints[class_name]
            
            # Create a dictionary for easy lookup
            # Handle both formats:
            # 1. List format: [{'feature': 'Age', 'min': x, 'max': y}, ...]
            # 2. Dict format: {'Age': {'min': x, 'max': y}, ...}
            constraint_dict = {k.replace('_', ' '): v for k, v in _normalize_constraint_dict(class_constraints).items()}
            
            y_positions = np.arange(n_features)
            
            for i, feature in enumerate(features):
                if feature in constraint_dict:
                    c = constraint_dict[feature]
                    min_val = c['min'] if c['min'] is not None else 0
                    max_val = c['max'] if c['max'] is not None else 10
                    
                    # Determine the range
                    if c['min'] is None:
                        # Only max constraint
                        ax.barh(i, max_val, left=0, height=0.6, 
                               color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.5)
                        ax.text(max_val/2, i, f'≤ {max_val:.2f}', 
                               ha='center', va='center', fontweight='bold', fontsize=9)
                    elif c['max'] is None:
                        # Only min constraint
                        range_width = 10 - min_val  # Arbitrary max for visualization
                        ax.barh(i, range_width, left=min_val, height=0.6, 
                               color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.5)
                        ax.text(min_val + range_width/2, i, f'≥ {min_val:.2f}', 
                               ha='center', va='center', fontweight='bold', fontsize=9)
                    else:
                        # Both min and max
                        range_width = max_val - min_val
                        ax.barh(i, range_width, left=min_val, height=0.6, 
                               color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.5)
                        ax.text(min_val + range_width/2, i, f'{min_val:.2f} - {max_val:.2f}', 
                               ha='center', va='center', fontweight='bold', fontsize=9)
            
            # Plot sample data if provided
            if sample is not None:
                # Determine sample color based on sample_class
                if sample_class is not None and 0 <= sample_class < len(class_colors_list):
                    sample_color = class_colors_list[sample_class]
                else:
                    sample_color = 'red'
                
                for i, feature in enumerate(features):
                    # Try to find matching feature in sample (handle variations in naming)
                    sample_value = None
                    for key in sample.keys():
                        if feature.lower() in key.lower() or key.lower() in feature.lower():
                            sample_value = sample[key]
                            break
                    
                    if sample_value is not None:
                        # Plot sample as a vertical line (red) and marker (class color)
                        ax.plot([sample_value, sample_value], [i - 0.3, i + 0.3], 
                               color='red', linewidth=3, linestyle='-', zorder=10)
                        ax.scatter([sample_value], [i], color=sample_color, s=100, 
                                  marker='D', edgecolor='darkred', linewidth=1.5, zorder=11)
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(features)
            ax.set_xlabel('Value Range', fontsize=11, fontweight='bold')
            ax.set_title(f'{class_name}', fontsize=13, fontweight='bold', pad=10)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_xlim(0, 8)
        
        axes[0].set_ylabel('Features', fontsize=11, fontweight='bold')
        
        # Add legend for sample if provided
        if sample is not None:
            # Determine sample color for legend
            if sample_class is not None and 0 <= sample_class < len(class_colors_list):
                sample_color = class_colors_list[sample_class]
            else:
                sample_color = 'red'
            
            # Create custom legend entry
            legend_elements = [Line2D([0], [0], marker='D', color='w', 
                                     markerfacecolor=sample_color, markersize=10, 
                                     markeredgecolor='darkred', markeredgewidth=1.5,
                                     label='Sample')]
            axes[-1].legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.suptitle('Feature Constraints per Class', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()


def plot_pca_loadings(X, feature_names):
    """
    Plot PCA loadings to show feature contributions to principal components.
    
    Parameters:
    -----------
    X : array-like
        The feature data
    feature_names : list
        Names of the features
    """
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    
    # Get the loadings (components)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot arrows for each feature
    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                head_width=0.05, head_length=0.05, fc='red', ec='red')
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, 
               feature, ha='center', va='center', fontsize=10)
    
    # Set plot properties
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title('PCA Loadings Plot', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_val = np.abs(loadings).max() * 1.3
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    
    # Add variance information to the plot
    variance_text = (
        f"PC1 Variance: {pca.explained_variance_ratio_[0]:.2%}\n"
        f"PC2 Variance: {pca.explained_variance_ratio_[1]:.2%}\n"
        f"Total Variance: {sum(pca.explained_variance_ratio_):.2%}"
    )
    ax.text(0.02, 0.98, variance_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
def plot_pca_with_counterfactual(model, dataset, target, sample, counterfactual):
    """
    Plot a PCA visualization of the dataset with the original sample and counterfactual.

    Args:
        model: Trained scikit-learn model used for predicting the class of the counterfactual.
        dataset: The original dataset (features) used for PCA.
        target: The target labels for the dataset.
        sample: The original sample as a dictionary of feature values.
        counterfactual: The counterfactual sample as a dictionary of feature values.
    """

    # Perform PCA on the scaled dataset
    pca = PCA(n_components=2)
    iris_pca = pca.fit_transform(dataset)

    # Transform the original sample and counterfactual using the same PCA
    original_sample_pca = pca.transform(pd.DataFrame([sample]))
    counterfactual_pca = pca.transform(pd.DataFrame([counterfactual]))

    # Predict the class of the counterfactual
    counterfactual_class = model.predict(pd.DataFrame([counterfactual]))[0]
    original_class = model.predict(pd.DataFrame([sample]))[0]

    # Plot the PCA results with class colors and 'x' marker for the counterfactual
    plt.figure(figsize=(10, 6))
    n_unique_classes = len(np.unique(target))
    default_palette = ['purple', 'green', 'orange', 'red', 'blue', 'yellow', 'pink', 'cyan']
    colors = default_palette[:n_unique_classes]  # Colors for the classes

    for class_value in np.unique(target):
        plt.scatter(
            iris_pca[target == class_value, 0],
            iris_pca[target == class_value, 1],
            label=f"Class {class_value}",
            color=colors[class_value],
            alpha=0.6
        )

    # Plot original sample filled with its class color and a thick outline matching its class
    plt.scatter(
        original_sample_pca[:, 0], original_sample_pca[:, 1],
        color=colors[original_class % len(colors)], label='Original Sample',
        marker='o', edgecolor=colors[original_class % len(colors)], linewidths=2.5, s=150, zorder=10
    )
    plt.scatter(
        counterfactual_pca[:, 0], counterfactual_pca[:, 1],
        color=colors[counterfactual_class], marker='o', s=150, label='Counterfactual',
        edgecolor=colors[counterfactual_class], linewidths=2.5, zorder=10
    )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Plot with Original Sample and Counterfactual')
    plt.legend()
    #plt.savefig("experiments/PCA_plot_CF.png")
    #plt.show()
    return(plt)


def plot_fitness(cf_model, figsize=(10, 6), title='Fitness Over Generations'):
    """
    Plot fitness curves using stored fitness history on a model instance.

    Args:
        cf_model: Object with attributes `best_fitness_list` and `average_fitness_list` (e.g., CounterFactualModel instance).
        figsize: Figure size tuple.
        title: Plot title.

    Returns:
        matplotlib.figure.Figure: The generated figure (closed to avoid display side effects).
    """
    fig, ax = plt.subplots(figsize=figsize)

    best = getattr(cf_model, 'best_fitness_list', []) or []
    avg = getattr(cf_model, 'average_fitness_list', []) or []

    # Plot best fitness and average fitness on the same graph
    ax.plot(best, label='Best Fitness', color='blue')
    ax.plot(avg, label='Average Fitness', color='green')
    ax.set_title(title)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.legend()

    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_fitness_std(cf_model, figsize=(10, 6), title='Fitness Standard Deviation Over Generations'):
    """
    Plot fitness standard deviation curve using stored fitness history on a model instance.
    Shows both best fitness and average fitness with error bars representing std deviation.

    Args:
        cf_model: Object with attributes `best_fitness_list`, `average_fitness_list`, 
                  and `std_fitness_list` (e.g., CounterFactualModel instance).
        figsize: Figure size tuple.
        title: Plot title.

    Returns:
        matplotlib.figure.Figure: The generated figure (closed to avoid display side effects).
    """
    fig, ax = plt.subplots(figsize=figsize)

    best = getattr(cf_model, 'best_fitness_list', []) or []
    avg = getattr(cf_model, 'average_fitness_list', []) or []
    std = getattr(cf_model, 'std_fitness_list', []) or []
    
    if len(std) == 0:
        # If no std data available, return empty figure with message
        ax.text(0.5, 0.5, 'No standard deviation data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        plt.tight_layout()
        plt.close(fig)
        return fig
    
    generations = np.arange(len(std))
    
    # Convert to numpy arrays for calculations
    avg_arr = np.array(avg) if len(avg) == len(std) else np.zeros(len(std))
    best_arr = np.array(best) if len(best) == len(std) else np.zeros(len(std))
    std_arr = np.array(std)
    
    # Plot best fitness with markers (similar to the reference image blue line)
    ax.plot(generations, best_arr, 'b-o', label='Best Fitness', markersize=4, linewidth=1)
    
    # Plot average fitness with error bars (similar to the reference image red line with error bars)
    ax.errorbar(generations, avg_arr, yerr=std_arr, fmt='r-s', 
                label='Average Best Fitness', markersize=3, linewidth=1,
                capsize=2, capthick=1, elinewidth=0.5, alpha=0.8)
    
    # Set log scale for y-axis (like the reference image)
    ax.set_yscale('log')
    
    ax.set_title(title)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_sample_and_counterfactual_comparison(model, sample, sample_df, counterfactual, constraints=None, class_colors_list=None, generation=None):
    """
    Enhanced visualization combining original and counterfactual samples with:
    1. Side-by-side feature comparison with arrows showing direction of change
    2. Feature changes (signed changes)
    3. Class probability comparison
    
    Args:
        model: Trained scikit-learn model
        sample: Original sample as dictionary
        sample_df: Original sample as DataFrame
        counterfactual: Counterfactual sample as dictionary
        constraints: Dictionary of constraints per class (optional)
        class_colors_list: List of colors for classes (default: ['purple', 'green', 'orange'])
        generation: Optional generation number to display in title (for per-generation visualizations)
    """
    if class_colors_list is None:
        class_colors_list = ['purple', 'green', 'orange']
    
    predicted_class = model.predict(sample_df)[0]
    counterfactual_class = model.predict(pd.DataFrame([counterfactual]))[0]
    
    # Calculate metrics
    feature_list = list(sample.keys())
    original_values = np.array(list(sample.values()))
    counterfactual_values = np.array(list(counterfactual.values()))
    changes = counterfactual_values - original_values
    l2_distance = np.linalg.norm(changes)
    l1_distance = np.sum(np.abs(changes))
    
    # Filter out features with zero change for charts 1 and 2
    # Only filter if there are more than 6 features
    if len(feature_list) <= 6:
        # Keep all features when 6 or fewer
        feature_list_filtered = feature_list
        original_values_filtered = original_values
        counterfactual_values_filtered = counterfactual_values
        changes_filtered = changes
    else:
        # Filter out unchanged features when more than 6
        non_zero_mask = np.abs(changes) > 0.001
        feature_list_filtered = [f for f, mask in zip(feature_list, non_zero_mask) if mask]
        original_values_filtered = original_values[non_zero_mask]
        counterfactual_values_filtered = counterfactual_values[non_zero_mask]
        changes_filtered = changes[non_zero_mask]
    
    # Calculate figure height proportionally based on number of features
    num_features = len(feature_list_filtered)
    base_height = 6  # Base height for 5-6 features
    if num_features > 6:
        # Scale height: add 0.8 inches for each feature beyond 6
        fig_height = base_height + 0.8 * (num_features - 6)
    else:
        fig_height = base_height
    
    # Create figure with custom layout
    fig, axes = plt.subplots(1, 3, figsize=(18, fig_height))
    
    # 1. Combined Feature Comparison with Arrows
    ax1 = axes[0]
    x_pos = np.arange(len(feature_list_filtered))
    width = 0.35
    
    # Bars for original and counterfactual
    bars1 = ax1.barh(x_pos - width/2, original_values_filtered, width, 
                     label='Original', color=class_colors_list[predicted_class], 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax1.barh(x_pos + width/2, counterfactual_values_filtered, width, 
                     label='Counterfactual', color=class_colors_list[counterfactual_class], 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add arrows showing direction of change
    for i, (orig, cf, change) in enumerate(zip(original_values_filtered, counterfactual_values_filtered, changes_filtered)):
        if abs(change) > 0.01:  # Only show arrow if change is significant
            arrow_color = 'darkgreen' if change < 0 else 'darkred'
            # Arrow always points from original to counterfactual
            ax1.annotate('', xy=(cf, i + width/2), xytext=(orig, i - width/2),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, 
                                      lw=2, alpha=0.6))
            # Add change value
            mid_point = (orig + cf) / 2
            ax1.text(mid_point, i, f'{change:+.2f}', 
                    ha='center', va='bottom', fontsize=9, 
                    fontweight='bold', color=arrow_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            width_bar = bar.get_width()
            ax1.text(width_bar, bar.get_y() + bar.get_height()/2,
                    f'{width_bar:.2f}', ha='left', va='center', 
                    fontsize=9, fontweight='bold')
    
    # Add constraint indicators if constraints are provided
    if constraints is not None:
        # Get the x-axis limits to determine the plotting range
        xlim = ax1.get_xlim()
        
        # Plot constraints for both classes
        for class_idx in [predicted_class, counterfactual_class]:
            class_name = f'Class {class_idx}'
            if class_name in constraints:
                class_constraints = constraints[class_name]
                
                # Create a dictionary for easy lookup by feature name
                # Handle both formats:
                # 1. List format: [{'feature': 'Age', 'min': x, 'max': y}, ...]
                # 2. Dict format: {'Age': {'min': x, 'max': y}, ...}
                constraint_dict = _normalize_constraint_dict(class_constraints)
                
                # Determine vertical offset based on class
                y_offset = -width/2 if class_idx == predicted_class else width/2
                
                for i, feature in enumerate(feature_list_filtered):
                    # Try to match feature name - be more flexible with matching
                    feature_key = None
                    feature_normalized = feature.replace(' (cm)', '').replace('_', ' ').strip().lower()
                    
                    for key in constraint_dict.keys():
                        key_normalized = key.replace('_', ' ').strip().lower()
                        if key_normalized == feature_normalized or key == feature or key.replace('_', ' ') == feature:
                            feature_key = key
                            break
                    
                    if feature_key and feature_key in constraint_dict:
                        c = constraint_dict[feature_key]
                        min_val = c['min'] if c['min'] is not None else xlim[0]
                        max_val = c['max'] if c['max'] is not None else xlim[1]
                        
                        # Draw constraint range as a horizontal line with markers
                        constraint_color = class_colors_list[class_idx]
                        alpha_constraint = 0.8  # Increased from 0.5
                        
                        # Draw the range line
                        if c['min'] is not None and c['max'] is not None:
                            # Both min and max
                            ax1.plot([min_val, max_val], [i + y_offset, i + y_offset], 
                                   color=constraint_color, linewidth=6, alpha=alpha_constraint, 
                                   linestyle='-', zorder=10)  # Higher zorder to draw on top
                            # Add markers at boundaries
                            ax1.plot([min_val], [i + y_offset], marker='|', markersize=15, 
                                   color=constraint_color, alpha=1.0, markeredgewidth=4, zorder=11)
                            ax1.plot([max_val], [i + y_offset], marker='|', markersize=15, 
                                   color=constraint_color, alpha=1.0, markeredgewidth=4, zorder=11)
                            # Add numerical labels for min and max
                            # Position min label to the left of the boundary marker, max label to the right
                            ax1.text(min_val, i + y_offset + 0.15, f'{min_val:.2f}', 
                                   ha='right', va='bottom', fontsize=8, 
                                   color=constraint_color, weight='bold', style='italic',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                           edgecolor=constraint_color, alpha=0.9, linewidth=1.5),
                                   zorder=12)
                            ax1.text(max_val, i + y_offset - 0.15, f'{max_val:.2f}', 
                                   ha='left', va='top', fontsize=8, 
                                   color=constraint_color, weight='bold', style='italic',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                           edgecolor=constraint_color, alpha=0.9, linewidth=1.5),
                                   zorder=12)
                        elif c['min'] is not None:
                            # Only min constraint
                            ax1.plot([min_val, xlim[1]], [i + y_offset, i + y_offset], 
                                   color=constraint_color, linewidth=6, alpha=alpha_constraint, 
                                   linestyle='--', zorder=10)
                            ax1.plot([min_val], [i + y_offset], marker='|', markersize=15, 
                                   color=constraint_color, alpha=1.0, markeredgewidth=4, zorder=11)
                            # Add numerical label for min
                            ax1.text(min_val, i + y_offset - 0.2, f'min:{min_val:.2f}', 
                                   ha='center', va='top', fontsize=8, 
                                   color=constraint_color, weight='bold', style='italic',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                           edgecolor=constraint_color, alpha=0.9, linewidth=1.5),
                                   zorder=12)
                        elif c['max'] is not None:
                            # Only max constraint
                            ax1.plot([xlim[0], max_val], [i + y_offset, i + y_offset], 
                                   color=constraint_color, linewidth=6, alpha=alpha_constraint, 
                                   linestyle='--', zorder=10)
                            ax1.plot([max_val], [i + y_offset], marker='|', markersize=15, 
                                   color=constraint_color, alpha=1.0, markeredgewidth=4, zorder=11)
                            # Add numerical label for max
                            ax1.text(max_val, i + y_offset - 0.2, f'max:{max_val:.2f}', 
                                   ha='center', va='top', fontsize=8, 
                                   color=constraint_color, weight='bold', style='italic',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                           edgecolor=constraint_color, alpha=0.9, linewidth=1.5),
                                   zorder=12)
    
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels([f.replace(' (cm)', '').replace('_', ' ') for f in feature_list_filtered])
    ax1.set_xlabel('Feature Value', fontsize=12, fontweight='bold')
    ax1.set_title(f'Feature Comparison\nClass {predicted_class} → Class {counterfactual_class}', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()
    
    # 2. Feature Changes (Signed changes like original)
    ax2 = axes[1]
    changes_values = changes_filtered  # Already calculated as counterfactual - original
    change_colors = ['green' if c < 0 else 'red' if c > 0 else 'gray' for c in changes_values]
    
    bars_change = ax2.barh(range(len(feature_list_filtered)), changes_values, 
                           color=change_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_yticks(range(len(feature_list_filtered)))
    ax2.set_yticklabels([f.replace(' (cm)', '').replace('_', ' ') for f in feature_list_filtered])
    ax2.set_xlabel('Change Value', fontsize=11, fontweight='bold')
    ax2.set_title('Feature Changes', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, change) in enumerate(zip(bars_change, changes_values)):
        if abs(change) > 0.01:
            label = f'{change:+.2f}'
            # Position text based on bar direction
            x_pos = change
            ha = 'left' if change > 0 else 'right'
            ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                    label, ha=ha, va='center', fontsize=9, fontweight='bold')
    
    # Add constraint boundaries showing feasible change ranges
    if constraints is not None:
        xlim2 = ax2.get_xlim()
        
        # For each feature, show the constraint range for the target class
        # This shows how much the feature COULD change to satisfy target class constraints
        for class_idx in [predicted_class, counterfactual_class]:
            class_name = f'Class {class_idx}'
            if class_name in constraints:
                class_constraints = constraints[class_name]
                # Handle both formats:
                # 1. List format: [{'feature': 'Age', 'min': x, 'max': y}, ...]
                # 2. Dict format: {'Age': {'min': x, 'max': y}, ...}
                constraint_dict = _normalize_constraint_dict(class_constraints)
                
                # Determine vertical offset based on class
                y_offset = -0.15 if class_idx == predicted_class else 0.15
                
                for i, feature in enumerate(feature_list_filtered):
                    # Try to match feature name - be more flexible with matching
                    feature_key = None
                    feature_normalized = feature.replace(' (cm)', '').replace('_', ' ').strip().lower()
                    
                    for key in constraint_dict.keys():
                        key_normalized = key.replace('_', ' ').strip().lower()
                        if key_normalized == feature_normalized or key == feature or key.replace('_', ' ') == feature:
                            feature_key = key
                            break
                    
                    if feature_key and feature_key in constraint_dict:
                        c = constraint_dict[feature_key]
                        orig_val = original_values_filtered[i]
                        
                        # Calculate the maximum possible change based on constraints
                        constraint_color = class_colors_list[class_idx]
                        
                        if c['min'] is not None or c['max'] is not None:
                            # Calculate change range based on constraints
                            min_change = (c['min'] - orig_val) if c['min'] is not None else xlim2[0]
                            max_change = (c['max'] - orig_val) if c['max'] is not None else xlim2[1]
                            
                            # Draw constraint change range
                            if c['min'] is not None and c['max'] is not None:
                                # Draw the feasible change range
                                ax2.plot([min_change, max_change], [i + y_offset, i + y_offset], 
                                       color=constraint_color, linewidth=5, alpha=0.7, 
                                       linestyle='-', zorder=10)
                                # Add boundary markers
                                ax2.plot([min_change], [i + y_offset], marker='|', markersize=12, 
                                       color=constraint_color, alpha=1.0, markeredgewidth=3.5, zorder=11)
                                ax2.plot([max_change], [i + y_offset], marker='|', markersize=12, 
                                       color=constraint_color, alpha=1.0, markeredgewidth=3.5, zorder=11)
                                # Add numerical labels
                                # Position min label to the right of the boundary marker (above the change range), max label to the left (below the change range)
                                ax2.text(min_change, i + y_offset + 0.4, f'{min_change:+.2f}', 
                                       ha='right', va='bottom', fontsize=7, 
                                       color=constraint_color, weight='bold', style='italic',
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                                               edgecolor=constraint_color, alpha=0.9, linewidth=1),
                                       zorder=12)
                                ax2.text(max_change, i + y_offset - 0.4, f'{max_change:+.2f}', 
                                       ha='left', va='top', fontsize=7, 
                                       color=constraint_color, weight='bold', style='italic',
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                                               edgecolor=constraint_color, alpha=0.9, linewidth=1),
                                       zorder=12)
                            elif c['min'] is not None:
                                # Only minimum constraint
                                ax2.plot([min_change, xlim2[1]], [i + y_offset, i + y_offset], 
                                       color=constraint_color, linewidth=5, alpha=0.6, 
                                       linestyle='--', zorder=10)
                                ax2.plot([min_change], [i + y_offset], marker='|', markersize=12, 
                                       color=constraint_color, alpha=1.0, markeredgewidth=3.5, zorder=11)
                                ax2.text(min_change, i + y_offset + 0.28, f'Δmin:{min_change:+.2f}', 
                                       ha='center', va='bottom', fontsize=7, 
                                       color=constraint_color, weight='bold', style='italic',
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                                               edgecolor=constraint_color, alpha=0.9, linewidth=1),
                                       zorder=12)
                            elif c['max'] is not None:
                                # Only maximum constraint
                                ax2.plot([xlim2[0], max_change], [i + y_offset, i + y_offset], 
                                       color=constraint_color, linewidth=5, alpha=0.6, 
                                       linestyle='--', zorder=10)
                                ax2.plot([max_change], [i + y_offset], marker='|', markersize=12, 
                                       color=constraint_color, alpha=1.0, markeredgewidth=3.5, zorder=11)
                                ax2.text(max_change, i + y_offset + 0.28, f'Δmax:{max_change:+.2f}', 
                                       ha='center', va='bottom', fontsize=7, 
                                       color=constraint_color, weight='bold', style='italic',
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                                               edgecolor=constraint_color, alpha=0.9, linewidth=1),
                                       zorder=12)
    
    # 3. Class Probability Comparison
    ax3 = axes[2]
    original_probs = model.predict_proba(sample_df)[0]
    counterfactual_probs = model.predict_proba(pd.DataFrame([counterfactual]))[0]
    n_classes = len(original_probs)
    class_names = [f'Class {i}' for i in range(n_classes)]
    
    x_pos_prob = np.arange(n_classes)
    width_prob = 0.35
    
    bars1 = ax3.bar(x_pos_prob - width_prob/2, original_probs, width_prob, 
                   label='Original', color=class_colors_list[predicted_class], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x_pos_prob + width_prob/2, counterfactual_probs, width_prob, 
                   label='Counterfactual', color=class_colors_list[counterfactual_class], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3.set_xticks(x_pos_prob)
    ax3.set_xticklabels(class_names)
    ax3.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax3.set_title('Class Probabilities', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Set main title with generation number if provided
    if generation is not None:
        plt.suptitle(f'Counterfactual Explanation Analysis - Generation {generation}', fontsize=16, fontweight='bold', y=0.98)
    else:
        plt.suptitle('Counterfactual Explanation Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_sample_and_counterfactual_comparison_simple(model, sample, sample_df, counterfactual, constraints=None, class_colors_list=None, generation=None):
    """
    Simplified visualization showing only the feature comparison with arrows showing direction of change.
    
    Args:
        model: Trained scikit-learn model
        sample: Original sample as dictionary
        sample_df: Original sample as DataFrame
        counterfactual: Counterfactual sample as dictionary
        constraints: Dictionary of constraints per class (optional)
        class_colors_list: List of colors for classes (default: ['purple', 'green', 'orange'])
        generation: Optional generation number to display in title (for per-generation visualizations)
    """
    if class_colors_list is None:
        class_colors_list = ['purple', 'green', 'orange']
    
    predicted_class = model.predict(sample_df)[0]
    counterfactual_class = model.predict(pd.DataFrame([counterfactual]))[0]
    
    # Calculate metrics
    feature_list = list(sample.keys())
    original_values = np.array(list(sample.values()))
    counterfactual_values = np.array(list(counterfactual.values()))
    changes = counterfactual_values - original_values
    l2_distance = np.linalg.norm(changes)
    l1_distance = np.sum(np.abs(changes))
    
    # Show all features (no filtering based on value changes)
    feature_list_filtered = feature_list
    original_values_filtered = original_values
    counterfactual_values_filtered = counterfactual_values
    changes_filtered = changes
    
    # Calculate figure height proportionally based on number of features
    num_features = len(feature_list_filtered)
    base_height = 6  # Base height for 5-6 features
    if num_features > 6:
        # Scale height: add 0.8 inches for each feature beyond 6
        fig_height = base_height + 0.8 * (num_features - 6)
    else:
        fig_height = base_height
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, fig_height))
    
    # Combined Feature Comparison with Arrows
    x_pos = np.arange(len(feature_list_filtered))
    width = 0.35
    
    # Bars for original and counterfactual
    bars1 = ax.barh(x_pos - width/2, original_values_filtered, width, 
                     label='Original', color=class_colors_list[predicted_class], 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.barh(x_pos + width/2, counterfactual_values_filtered, width, 
                     label='Counterfactual', color=class_colors_list[counterfactual_class], 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add arrows showing direction of change
    for i, (orig, cf, change) in enumerate(zip(original_values_filtered, counterfactual_values_filtered, changes_filtered)):
        if abs(change) > 0.01:  # Only show arrow if change is significant
            arrow_color = 'darkgreen' if change < 0 else 'darkred'
            # Arrow always points from original to counterfactual
            ax.annotate('', xy=(cf, i + width/2), xytext=(orig, i - width/2),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, 
                                      lw=2, alpha=0.6))
            # Add change value
            mid_point = (orig + cf) / 2
            ax.text(mid_point, i, f'{change:+.2f}', 
                    ha='center', va='bottom', fontsize=9, 
                    fontweight='bold', color=arrow_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            width_bar = bar.get_width()
            ax.text(width_bar, bar.get_y() + bar.get_height()/2,
                    f'{width_bar:.2f}', ha='left', va='center', 
                    fontsize=9, fontweight='bold')
    
    # Add constraint indicators if constraints are provided
    if constraints is not None:
        # Get the x-axis limits to determine the plotting range
        xlim = ax.get_xlim()
        
        # Plot constraints for both classes
        for class_idx in [predicted_class, counterfactual_class]:
            class_name = f'Class {class_idx}'
            if class_name in constraints:
                class_constraints = constraints[class_name]
                
                # Create a dictionary for easy lookup by feature name
                # Handle both formats:
                # 1. List format: [{'feature': 'Age', 'min': x, 'max': y}, ...]
                # 2. Dict format: {'Age': {'min': x, 'max': y}, ...}
                constraint_dict = _normalize_constraint_dict(class_constraints)
                
                # Determine vertical offset based on class
                y_offset = -width/2 if class_idx == predicted_class else width/2
                
                for i, feature in enumerate(feature_list_filtered):
                    # Try to match feature name - be more flexible with matching
                    feature_key = None
                    feature_normalized = feature.replace(' (cm)', '').replace('_', ' ').strip().lower()
                    
                    for key in constraint_dict.keys():
                        key_normalized = key.replace('_', ' ').strip().lower()
                        if key_normalized == feature_normalized or key == feature or key.replace('_', ' ') == feature:
                            feature_key = key
                            break
                    
                    if feature_key and feature_key in constraint_dict:
                        c = constraint_dict[feature_key]
                        min_val = c['min'] if c['min'] is not None else xlim[0]
                        max_val = c['max'] if c['max'] is not None else xlim[1]
                        
                        # Draw constraint range as a horizontal line with markers
                        constraint_color = class_colors_list[class_idx]
                        alpha_constraint = 0.8  # Increased from 0.5
                        
                        # Draw the range line
                        if c['min'] is not None and c['max'] is not None:
                            # Both min and max
                            ax.plot([min_val, max_val], [i + y_offset, i + y_offset], 
                                   color=constraint_color, linewidth=6, alpha=alpha_constraint, 
                                   linestyle='-', zorder=10)  # Higher zorder to draw on top
                            # Add markers at boundaries
                            ax.plot([min_val], [i + y_offset], marker='|', markersize=15, 
                                   color=constraint_color, alpha=1.0, markeredgewidth=4, zorder=11)
                            ax.plot([max_val], [i + y_offset], marker='|', markersize=15, 
                                   color=constraint_color, alpha=1.0, markeredgewidth=4, zorder=11)
                            # Add numerical labels for min and max
                            # Position min label to the left of the boundary marker, max label to the right
                            ax.text(min_val, i + y_offset + 0.15, f'{min_val:.2f}', 
                                   ha='right', va='bottom', fontsize=8, 
                                   color=constraint_color, weight='bold', style='italic',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                           edgecolor=constraint_color, alpha=0.9, linewidth=1.5),
                                   zorder=12)
                            ax.text(max_val, i + y_offset - 0.15, f'{max_val:.2f}', 
                                   ha='left', va='top', fontsize=8, 
                                   color=constraint_color, weight='bold', style='italic',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                           edgecolor=constraint_color, alpha=0.9, linewidth=1.5),
                                   zorder=12)
                        elif c['min'] is not None:
                            # Only min constraint
                            ax.plot([min_val, xlim[1]], [i + y_offset, i + y_offset], 
                                   color=constraint_color, linewidth=6, alpha=alpha_constraint, 
                                   linestyle='--', zorder=10)
                            ax.plot([min_val], [i + y_offset], marker='|', markersize=15, 
                                   color=constraint_color, alpha=1.0, markeredgewidth=4, zorder=11)
                            # Add numerical label for min
                            ax.text(min_val, i + y_offset - 0.2, f'min:{min_val:.2f}', 
                                   ha='center', va='top', fontsize=8, 
                                   color=constraint_color, weight='bold', style='italic',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                           edgecolor=constraint_color, alpha=0.9, linewidth=1.5),
                                   zorder=12)
                        elif c['max'] is not None:
                            # Only max constraint
                            ax.plot([xlim[0], max_val], [i + y_offset, i + y_offset], 
                                   color=constraint_color, linewidth=6, alpha=alpha_constraint, 
                                   linestyle='--', zorder=10)
                            ax.plot([max_val], [i + y_offset], marker='|', markersize=15, 
                                   color=constraint_color, alpha=1.0, markeredgewidth=4, zorder=11)
                            # Add numerical label for max
                            ax.text(max_val, i + y_offset - 0.2, f'max:{max_val:.2f}', 
                                   ha='center', va='top', fontsize=8, 
                                   color=constraint_color, weight='bold', style='italic',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                           edgecolor=constraint_color, alpha=0.9, linewidth=1.5),
                                   zorder=12)
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f.replace(' (cm)', '').replace('_', ' ') for f in feature_list_filtered])
    ax.set_xlabel('Feature Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Comparison\nClass {predicted_class} → Class {counterfactual_class}', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    # Set main title with generation number if provided
    if generation is not None:
        plt.suptitle(f'Counterfactual Explanation Analysis - Generation {generation}', fontsize=16, fontweight='bold', y=0.98)
    else:
        plt.suptitle('Counterfactual Explanation Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_pairwise_with_counterfactual(model, dataset, target, sample, counterfactual):
    """
    Plot a Seaborn pairplot of the dataset, highlighting the original sample and counterfactual.

    Args:
        model: Trained scikit-learn model used for predicting the class of the counterfactual.
        dataset: The original dataset (features) used for the plot.
        target: The target labels for the dataset.
        sample: The original sample as a dictionary of feature values.
        counterfactual: The counterfactual sample as a dictionary of feature values.
    """
    # Convert the dataset into a DataFrame and add the target labels
    data_df = pd.DataFrame(dataset, columns=pd.DataFrame([sample]).columns)
    data_df['label'] = 'Dataset'

    # Convert the original sample and counterfactual to DataFrames
    original_sample_df = pd.DataFrame([sample])
    counterfactual_df = pd.DataFrame([counterfactual])

    # Add labels to distinguish the original sample and counterfactual in the plot
    original_sample_df['label'] = 'Original Sample'
    counterfactual_df['label'] = 'Counterfactual'

    # Combine the original sample and counterfactual with the dataset for plotting
    combined_df = pd.concat([data_df, original_sample_df, counterfactual_df], ignore_index=True)

    # Determine the class color for the original sample
    original_class = model.predict(pd.DataFrame([sample]))[0]
    default_palette = ['purple', 'green', 'orange', 'red', 'blue', 'yellow', 'pink', 'cyan']
    colors = default_palette

    # Plot the pairplot with Seaborn using the class color for the original sample
    sns.pairplot(combined_df, hue='label', palette={'Dataset': 'gray', 'Original Sample': colors[original_class % len(colors)], 'Counterfactual': 'blue'})
    plt.suptitle('Pairwise Plot with Original Sample and Counterfactual', y=1.02)
    return(plt)    

    #plt.show()

def plot_sample_and_counterfactual_comparison_combined(model, sample, sample_df, dpg_cf, dice_cf, method_names=('DPG', 'DiCE'), constraints=None, restrictions=None, class_colors_list=None, generation=None):
    """
    Combined visualization showing original sample with counterfactuals from two different methods
    in a single bar graph with three bars per feature.
    
    Args:
        model: Trained scikit-learn model
        sample: Original sample as dictionary
        sample_df: Original sample as DataFrame
        dpg_cf: DPG counterfactual as dictionary
        dice_cf: DiCE counterfactual as dictionary
        method_names: Tuple of method names (default: ('DPG', 'DiCE'))
        constraints: Dictionary of constraints per class (optional)
        restrictions: Dictionary of actionability restrictions per feature (optional)
        class_colors_list: List of colors for classes (default: ['purple', 'green', 'orange'])
        generation: Optional generation number to display in title
    """
    if class_colors_list is None:
        class_colors_list = ['purple', 'green', 'orange']
    
    predicted_class = model.predict(sample_df)[0]
    dpg_cf_class = model.predict(pd.DataFrame([dpg_cf]))[0]
    dice_cf_class = model.predict(pd.DataFrame([dice_cf]))[0]
    
    # Get all features
    feature_list = list(sample.keys())
    
    # Filter out features with 'no_change' actionability restriction
    if restrictions is not None:
        feature_list = [f for f in feature_list if restrictions.get(f) != 'no_change']
    
    # Calculate figure height proportionally based on number of features
    num_features = len(feature_list)
    base_height = 6  # Base height for 5-6 features
    if num_features > 6:
        # Scale height: add 0.8 inches for each feature beyond 6
        fig_height = base_height + 0.8 * (num_features - 6)
    else:
        fig_height = base_height
    
    # Create single figure (wider to accommodate constraint labels)
    fig, ax = plt.subplots(1, 1, figsize=(15, fig_height))
    
    # Get values for all three series
    original_values = np.array([sample[f] for f in feature_list])
    dpg_values = np.array([dpg_cf[f] for f in feature_list])
    dice_values = np.array([dice_cf[f] for f in feature_list])
    
    # Calculate changes
    dpg_changes = dpg_values - original_values
    dice_changes = dice_values - original_values
    
    x_pos = np.arange(len(feature_list))
    width = 0.25  # Width for each bar (3 bars per feature)
    
    # Define colors: Original (class color), DPG (orange), DiCE (blue)
    original_color = class_colors_list[predicted_class]
    dpg_color = '#CC0000'  # Orange for DPG
    dice_color = '#006DAC'  # Blue for DiCE
    
    # Bars for original, DPG CF, and DiCE CF
    bars_orig = ax.barh(x_pos - width, original_values, width, 
                        label='Original', color=original_color, 
                        alpha=0.7, edgecolor='black', linewidth=1.5)
    bars_dpg = ax.barh(x_pos, dpg_values, width, 
                       label=f'{method_names[0]} CF', color=dpg_color, 
                       alpha=0.7, edgecolor='black', linewidth=1.5)
    bars_dice = ax.barh(x_pos + width, dice_values, width, 
                        label=f'{method_names[1]} CF', color=dice_color, 
                        alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars_orig, bars_dpg, bars_dice]:
        for bar in bars:
            width_bar = bar.get_width()
            ax.text(width_bar, bar.get_y() + bar.get_height()/2,
                    f'{width_bar:.2f}', ha='left', va='center', 
                    fontsize=8, fontweight='bold')
    
    # Add constraint indicators if constraints are provided
    if constraints is not None:
        # Get the x-axis limits to determine the plotting range
        xlim = ax.get_xlim()
        
        # Plot constraints for both original and CF classes
        for class_idx in [predicted_class, dpg_cf_class, dice_cf_class]:
            class_name = f'Class {class_idx}'
            if class_name in constraints:
                class_constraints = constraints[class_name]
                
                # Handle both list and dict formats
                constraint_dict = _normalize_constraint_dict(class_constraints)
                
                # Determine vertical offset based on class
                if class_idx == predicted_class:
                    y_offset = -width  # Original bar position
                elif class_idx == dpg_cf_class:
                    y_offset = 0  # DPG bar position
                else:
                    y_offset = width  # DiCE bar position
                
                for i, feature in enumerate(feature_list):
                    # Match feature name
                    feature_key = None
                    feature_normalized = feature.replace(' (cm)', '').replace('_', ' ').strip().lower()
                    
                    for key in constraint_dict.keys():
                        key_normalized = key.replace('_', ' ').strip().lower()
                        if key_normalized == feature_normalized or key == feature or key.replace('_', ' ') == feature:
                            feature_key = key
                            break
                    
                    if feature_key and feature_key in constraint_dict:
                        c = constraint_dict[feature_key]
                        min_val = c['min'] if c['min'] is not None else xlim[0]
                        max_val = c['max'] if c['max'] is not None else xlim[1]
                        
                        constraint_color = class_colors_list[class_idx]
                        alpha_constraint = 0.8
                        
                        # Draw constraint range
                        if c['min'] is not None and c['max'] is not None:
                            ax.plot([min_val, max_val], [i + y_offset, i + y_offset], 
                                   color=constraint_color, linewidth=6, alpha=alpha_constraint, 
                                   linestyle='-', zorder=10)
                            ax.plot([min_val], [i + y_offset], marker='|', markersize=15, 
                                   color=constraint_color, alpha=1.0, markeredgewidth=4, zorder=11)
                            ax.plot([max_val], [i + y_offset], marker='|', markersize=15, 
                                   color=constraint_color, alpha=1.0, markeredgewidth=4, zorder=11)
                            # Position constraint labels further from bars to avoid overlap
                            label_offset = 0.35
                            ax.text(min_val, i + y_offset + label_offset, f'{min_val:.2f}', 
                                   ha='right', va='bottom', fontsize=7, 
                                   color=constraint_color, weight='bold', style='italic',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                                           edgecolor=constraint_color, alpha=0.9, linewidth=1.2),
                                   zorder=12)
                            ax.text(max_val, i + y_offset - label_offset, f'{max_val:.2f}', 
                                   ha='left', va='top', fontsize=7, 
                                   color=constraint_color, weight='bold', style='italic',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                                           edgecolor=constraint_color, alpha=0.9, linewidth=1.2),
                                   zorder=12)
                        elif c['min'] is not None:
                            ax.plot([min_val, xlim[1]], [i + y_offset, i + y_offset], 
                                   color=constraint_color, linewidth=6, alpha=alpha_constraint, 
                                   linestyle='--', zorder=10)
                            ax.plot([min_val], [i + y_offset], marker='|', markersize=15, 
                                   color=constraint_color, alpha=1.0, markeredgewidth=4, zorder=11)
                            ax.text(min_val, i + y_offset - 0.35, f'min:{min_val:.2f}', 
                                   ha='center', va='top', fontsize=7, 
                                   color=constraint_color, weight='bold', style='italic',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                                           edgecolor=constraint_color, alpha=0.9, linewidth=1.2),
                                   zorder=12)
                        elif c['max'] is not None:
                            ax.plot([xlim[0], max_val], [i + y_offset, i + y_offset], 
                                   color=constraint_color, linewidth=6, alpha=alpha_constraint, 
                                   linestyle='--', zorder=10)
                            ax.plot([max_val], [i + y_offset], marker='|', markersize=15, 
                                   color=constraint_color, alpha=1.0, markeredgewidth=4, zorder=11)
                            ax.text(max_val, i + y_offset - 0.35, f'max:{max_val:.2f}', 
                                   ha='center', va='top', fontsize=7, 
                                   color=constraint_color, weight='bold', style='italic',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                                           edgecolor=constraint_color, alpha=0.9, linewidth=1.2),
                                   zorder=12)
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f.replace(' (cm)', '').replace('_', ' ') for f in feature_list])
    ax.set_xlabel('Feature Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Comparison: Original (Class {predicted_class}) vs {method_names[0]} CF (Class {dpg_cf_class}) vs {method_names[1]} CF (Class {dice_cf_class})', 
                 fontsize=13, fontweight='bold')
    
    # Create custom legend including constraint indicators
    
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=original_color, ec='black', alpha=0.7, linewidth=1.5, label='Original Sample'),
        Rectangle((0, 0), 1, 1, fc=dpg_color, ec='black', alpha=0.7, linewidth=1.5, label=f'{method_names[0]} CF'),
        Rectangle((0, 0), 1, 1, fc=dice_color, ec='black', alpha=0.7, linewidth=1.5, label=f'{method_names[1]} CF'),
    ]
    
    # Add constraint legend entries for each class if constraints are present
    if constraints is not None:
        # Get unique classes involved
        unique_classes = sorted(set([predicted_class, dpg_cf_class, dice_cf_class]))
        for class_idx in unique_classes:
            class_name = f'Class {class_idx}'
            if class_name in constraints:
                constraint_color = class_colors_list[class_idx]
                legend_elements.append(Line2D([0], [0], color=constraint_color, linewidth=6, 
                                             alpha=0.8, label=f'Class {class_idx} Constraints'))
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    # Set main title with generation number if provided
    if generation is not None:
        fig.suptitle(f'Counterfactual Comparison', fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'Counterfactual Comparison - All Features', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_sample_and_counterfactual_heatmap(sample, class_sample, counterfactual, class_counterfactual, restrictions, is_valid=True):
    """
    Plot the original sample, the differences, and the counterfactual as a heatmap,
    and indicate restrictions using icons.

    Args:
        sample (dict): Original sample values.
        counterfactual (dict): Counterfactual sample values.
        restrictions (dict): Restrictions applied to each feature.
        is_valid (bool): Whether the counterfactual is valid (reached target class). Default True.
    """
    # Create DataFrame from the samples
    sample_df = pd.DataFrame([sample], index=['Original Sample'])
    cf_df = pd.DataFrame([counterfactual], index=['Counterfactual'])

    # Calculate differences
    differences = (cf_df.loc['Counterfactual'] - sample_df.loc['Original Sample']).to_frame('Difference').T

    # Combine all data
    full_df = pd.concat([sample_df, differences, cf_df])

    # Map restrictions to symbols
    symbol_map = {
        'no_change': '⊝',  # Locked symbol for no change
        'non_increasing': '⬇️',  # Down arrow for non-increasing
        'non_decreasing': '⬆️'  # Up arrow for non-decreasing
    }
    restrictions_ser = pd.Series(restrictions).replace(symbol_map)

    mask = np.full_like(full_df, False, dtype=bool)  # Start with no masking
    mask[[0, -1], :] = True  # Only mask the first and last rows


    vmax = np.max(np.abs(full_df.values))
    vmin = -vmax

    # Plotting the heatmap for numeric data
    fig = plt.figure(figsize=(10, 5))
    ax = sns.heatmap(full_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, linewidths=1.2, linecolor='k',
                     vmin=vmin, vmax=vmax, mask=mask)

    # Annotate with restrictions (skip None and actionable values)
    for i, (feat, restr) in enumerate(restrictions_ser.items()):
        if restr is not None and isinstance(restr, str) and restr.lower() not in ['none', 'actionable']:
            ax.text(i + 0.5, 3.5, restr, ha='center', va='center', color='black', fontweight='bold', fontsize=14)

    annotations = full_df.round(2).copy().astype(str)
    for col in full_df.columns:
        annotations.loc['Difference', col] = f"Δ {full_df.loc['Difference', col]:.2f}"

    for (i, j), val in np.ndenumerate(full_df):
        if i == 1:
            continue
        ax.text(j + 0.5, i + 0.5, annotations.iloc[i, j],
                horizontalalignment='center', verticalalignment='center', color='black')


    plt.title(f'Original (Class {class_sample}), Counterfactual (Class {class_counterfactual}) with Restrictions')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0, va="center")

    # Add margins to prevent text cutoff
    plt.tight_layout(pad=2.0)

    # Add NON-VALID overlay if counterfactual didn't reach target class
    if not is_valid:
        fig.text(0.5, 0.5, 'NON-VALID', fontsize=60, color='red', alpha=0.3,
                ha='center', va='center', weight='bold', rotation=30,
                transform=fig.transFigure, zorder=100)
    
    # Don't close the figure here - let the caller close it after saving/logging
    # plt.close(fig)
    return fig

# Example usage
#sample = {'petal width (cm)': 6.1, 'petal length (cm)': 2.8, 'sepal length (cm)': 4.7, 'sepal width (cm)': 1.2}
#counterfactual = {'petal width (cm)': 5.78, 'petal length (cm)': 2.55, 'sepal length (cm)': 1.53, 'sepal width (cm)': 1.2}
#class_sample = model.predict(pd.DataFrame([sample]))[0]
#class_counterfactual = model.predict(pd.DataFrame([counterfactual]))[0]
#restrictions = {'petal width (cm)': 'non_decreasing', 'petal length (cm)': 'non_increasing', 'sepal length (cm)': 'non_increasing', 'sepal width (cm)': 'no_change'}
#plot_sample_and_counterfactual_heatmap(sample, class_sample,  counterfactual, class_counterfactual, restrictions)

def plot_pairwise_with_counterfactual_df(model, dataset, target, sample, counterfactual_df):
    """
    Plot pairwise plots of the dataset, highlighting the original sample and multiple counterfactuals using matplotlib.

    Args:
        model: Trained scikit-learn model used for predicting the class of the counterfactuals.
        dataset: The original dataset (features) used for the plot.
        target: The target labels for the dataset.
        sample: The original sample as a dictionary of feature values.
        counterfactual_df: DataFrame containing several counterfactual samples.
    """
    # Convert the dataset into a DataFrame
    data_df = pd.DataFrame(dataset, columns=list(sample.keys()))

    # Convert the original sample to DataFrame
    sample_df = pd.DataFrame([sample], index=['Original'])

    # Combine the dataset with marked samples
    combined_df = pd.concat([data_df, sample_df, counterfactual_df])

    # Get feature names
    features = list(sample.keys())
    num_features = len(features)

    # Create a grid of plots
    fig, axes = plt.subplots(nrows=num_features, ncols=num_features, figsize=(15, 15))

    # Determine class color for the original sample
    original_class = model.predict(pd.DataFrame([sample]))[0]
    colors = ['purple', 'green', 'orange']

    # Plot each pair of features
    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):
            ax = axes[i, j]
            if i != j:
                # Scatter plot for different features

                # Adds a regression line
                sns.regplot(data=data_df, x=feature_i, y=feature_j, ax=ax, 
                           scatter_kws={'alpha': 0.5, 'color': 'gray', 's': 30}, 
                           line_kws={'color': 'darkgray', 'linewidth': 2})

                ax.scatter(data_df[feature_i], data_df[feature_j], c='gray', label='Dataset', alpha=0.5)
                ax.scatter(sample_df[feature_i], sample_df[feature_j], c=colors[original_class % len(colors)], label='Original Sample', edgecolors='black', linewidths=2.5, s=120)
                ax.scatter(counterfactual_df[feature_i], counterfactual_df[feature_j], c='blue', label='Counterfactuals', alpha=0.6, edgecolors='k', s=50)
                ax.set_ylabel('')
                ax.set_xlabel('')
            else:
                # Histogram on the diagonal
                ax.hist(data_df[feature_i], color='gray', bins=30, alpha=0.5)
                ax.hist(sample_df[feature_i], color='red', bins=1)
                ax.hist(counterfactual_df[feature_i], color='blue', bins=30, alpha=0.6)

            # Label axes if on the perimeter
            if i == num_features - 1:
                ax.set_xlabel(feature_j)
            if j == 0:
                ax.set_ylabel(feature_i)

    # Add legends and adjust layout
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            plt.tight_layout()
        except:
            pass  # Ignore tight_layout errors for complex plots
    plt.close(fig)
    return fig

# Example usage
# Assuming you have a trained model, a dataset (as a DataFrame or array), target labels, original sample, and counterfactual_df
#plot_pairwise_with_counterfactual_df(model, X, y, sample, counterfactuals_df)

def plot_pca_with_counterfactuals_clean(model, dataset, target, sample, counterfactuals_df, cf_predicted_classes):
    """
    Plot a PCA visualization of the dataset with the original sample and multiple counterfactuals from a DataFrame.
    This is the CLEAN version without any generation history - only shows original and final counterfactuals.
    
    Args:
        model: Trained model for predictions
        dataset: Full dataset for PCA
        target: Target labels
        sample: Original sample dict
        counterfactuals_df: DataFrame of final counterfactuals
        cf_predicted_classes: Pre-computed predicted classes for counterfactuals (numpy array or list)
    """
    # Standardize the dataset
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset.select_dtypes(include=[np.number]))

    # Perform PCA on the scaled dataset
    pca = PCA(n_components=2)
    iris_pca = pca.fit_transform(dataset_scaled)

    # Ensure the sample and counterfactuals are formatted as numeric DataFrame
    sample_df = pd.DataFrame([sample]).select_dtypes(include=[np.number])
    sample_df_scaled = scaler.transform(sample_df)
    if not isinstance(counterfactuals_df, pd.DataFrame):
        raise ValueError("counterfactuals_df must be a pandas DataFrame")

    numeric_cf_df = counterfactuals_df.select_dtypes(include=[np.number])
    numeric_cf_df_scaled = scaler.transform(numeric_cf_df)

    # Transform using PCA
    original_sample_pca = pca.transform(sample_df_scaled)
    counterfactuals_pca = pca.transform(numeric_cf_df_scaled)

    # Use pre-computed predicted classes
    counterfactual_classes = cf_predicted_classes

    # Plot the PCA results
    fig = plt.figure(figsize=(10, 6))
    colors = ['purple', 'green', 'orange']
    # Determine original sample class for consistent styling
    original_class = model.predict(pd.DataFrame([sample]))[0]

    for class_value in np.unique(target):
        plt.scatter(
            iris_pca[target == class_value, 0],
            iris_pca[target == class_value, 1],
            label=f"Class {class_value}",
            color=colors[class_value % len(colors)],
            alpha=0.5
        )

    plt.scatter(
        original_sample_pca[:, 0], original_sample_pca[:, 1],
        color=colors[original_class % len(colors)], label='Original Sample',
        marker='o', edgecolor=colors[original_class % len(colors)], linewidths=2.5, s=150, zorder=10
    )
    
    # Plot final counterfactuals as circles (same style as class samples)
    for idx, cf_class in enumerate(counterfactual_classes):
        cf_color = colors[cf_class % len(colors)]
        # Plot circle marker with edge matching class color
        plt.scatter(
            counterfactuals_pca[idx, 0], counterfactuals_pca[idx, 1],
            color=cf_color, marker='o', s=150,
            edgecolor=cf_color, linewidths=2.5, alpha=1.0, zorder=10
        )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Plot with Original Sample and Counterfactuals')
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[original_class % len(colors)], markersize=10, 
               markeredgecolor=colors[original_class % len(colors)], markeredgewidth=1.5, label='Original Sample'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10,
               markeredgecolor='gray', markeredgewidth=1.5, label='Final Counterfactuals')
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.close(fig)
    return fig

def plot_pca_with_counterfactuals_comparison(
    model, dataset, target, sample,
    counterfactuals_df_1, cf_predicted_classes_1,
    counterfactuals_df_2, cf_predicted_classes_2,
    method_1_name='Method 1', method_2_name='Method 2',
    method_1_color="#CC0000", method_2_color="#006DAC"
):
    """
    Plot a PCA visualization comparing counterfactuals from two different methods.
    
    Args:
        model: Trained model for predictions
        dataset: Full dataset for PCA
        target: Target labels
        sample: Original sample dict
        counterfactuals_df_1: DataFrame of counterfactuals from method 1
        cf_predicted_classes_1: Pre-computed predicted classes for method 1 counterfactuals
        counterfactuals_df_2: DataFrame of counterfactuals from method 2
        cf_predicted_classes_2: Pre-computed predicted classes for method 2 counterfactuals
        method_1_name: Display name for method 1 (default: 'Method 1')
        method_2_name: Display name for method 2 (default: 'Method 2')
        method_1_color: Color for method 1 X markers (default: 'red')
        method_2_color: Color for method 2 X markers (default: 'blue')
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Standardize the dataset
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset.select_dtypes(include=[np.number]))

    # Perform PCA on the scaled dataset
    pca = PCA(n_components=2)
    iris_pca = pca.fit_transform(dataset_scaled)

    # Ensure the sample is formatted as numeric DataFrame
    sample_df = pd.DataFrame([sample]).select_dtypes(include=[np.number])
    sample_df_scaled = scaler.transform(sample_df)
    
    # Transform original sample using PCA
    original_sample_pca = pca.transform(sample_df_scaled)
    
    # Process counterfactuals from method 1
    if not isinstance(counterfactuals_df_1, pd.DataFrame):
        raise ValueError("counterfactuals_df_1 must be a pandas DataFrame")
    numeric_cf_df_1 = counterfactuals_df_1.select_dtypes(include=[np.number])
    numeric_cf_df_1_scaled = scaler.transform(numeric_cf_df_1)
    counterfactuals_pca_1 = pca.transform(numeric_cf_df_1_scaled)
    
    # Process counterfactuals from method 2
    if not isinstance(counterfactuals_df_2, pd.DataFrame):
        raise ValueError("counterfactuals_df_2 must be a pandas DataFrame")
    numeric_cf_df_2 = counterfactuals_df_2.select_dtypes(include=[np.number])
    numeric_cf_df_2_scaled = scaler.transform(numeric_cf_df_2)
    counterfactuals_pca_2 = pca.transform(numeric_cf_df_2_scaled)

    # Plot the PCA results
    fig = plt.figure(figsize=(10, 6))
    colors = ['purple', 'green', 'orange']
    
    # Determine original sample class for consistent styling
    original_class = model.predict(pd.DataFrame([sample]))[0]

    lower_opacity=0.8
    size=75
    linewidth=2

    # Plot dataset points by class
    for class_value in np.unique(target):
        plt.scatter(
            iris_pca[target == class_value, 0],
            iris_pca[target == class_value, 1],
            label=f"Class {class_value}",
            color=mcolors.to_rgba(colors[class_value % len(colors)], 0.4),
            edgecolor=mcolors.to_rgba(colors[class_value % len(colors)], 0.1),
        )
    # Plot original sample with edge matching its class color

    original_color = colors[original_class % len(colors)]
    plt.scatter(
        original_sample_pca[:, 0], original_sample_pca[:, 1],
        label='Original Sample',
        marker='o',
        color=mcolors.to_rgba(original_color, lower_opacity),
        edgecolor=(0, 0, 0, 1.0),
        linewidths=linewidth,
        s=size, 
        zorder=10,
    )
    
    # Plot counterfactuals from method 1 as circles with edges matching method color
    for idx, cf_class in enumerate(cf_predicted_classes_1):
        class_color = colors[cf_class % len(colors)]
        plt.scatter(
            counterfactuals_pca_1[idx, 0], counterfactuals_pca_1[idx, 1],
            color=mcolors.to_rgba(class_color, lower_opacity), 
            marker='v', 
            s=size,
            edgecolor=method_1_color,
            linewidths=linewidth, 
            zorder=8,
        )
    
    # Plot counterfactuals from method 2 as circles with edges matching method color
    for idx, cf_class in enumerate(cf_predicted_classes_2):
        class_color = colors[cf_class % len(colors)]
        plt.scatter(
            counterfactuals_pca_2[idx, 0], counterfactuals_pca_2[idx, 1],
            color=mcolors.to_rgba(class_color, lower_opacity), 
            marker='s', 
            s=size,
            edgecolor=method_2_color,
            linewidths=linewidth, 
            zorder=8,
        )

    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} explained variance)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} explained variance)')
    plt.title('PCA Scores Projection')
    
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], 
               marker='o', 
               color='w', 
               markerfacecolor=(0, 0, 0, 0), 
               markersize=10, 
               markeredgecolor='black', 
               markeredgewidth=linewidth, 
               label='Original Sample'),
        Line2D([0], [0], 
               marker='v', 
               color='w', 
               markerfacecolor=(0, 0, 0, 0), 
               markersize=10,
               markeredgecolor=method_1_color, 
               markeredgewidth=linewidth, 
               label=f'{method_1_name} CFs'),
        Line2D([0], [0], 
               marker='s', 
               color='w', 
               markerfacecolor=(0, 0, 0, 0), 
               markersize=10,
               markeredgecolor=method_2_color, 
               markeredgewidth=linewidth, 
               label=f'{method_2_name} CFs'),
        Line2D([0], [0], 
               marker='o', 
               color='w', 
               markerfacecolor=mcolors.to_rgba(colors[0], 0.5),
               markersize=10,
               markeredgecolor=(0, 0, 0, 0), 
               markeredgewidth=linewidth, 
               label='Class 0'),
        Line2D([0], [0], 
               marker='o', 
               color='w', 
               markerfacecolor=mcolors.to_rgba(colors[1], 0.5),
               markersize=10,
               markeredgecolor=(0, 0, 0, 0), 
               markeredgewidth=linewidth, 
               label='Class 1')
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.close(fig)
    return fig

def plot_pca_with_counterfactuals(model, dataset, target, sample, counterfactuals_df, cf_predicted_classes, evolution_histories=None, cf_generations_found=None):
    """
    Plot a PCA visualization of the dataset with the original sample and multiple counterfactuals from a DataFrame.
    Shows the evolutionary process of GA with opacity gradient from initial to final generation.
    
    Args:
        model: Trained model for predictions
        dataset: Full dataset for PCA
        target: Target labels
        sample: Original sample dict
        counterfactuals_df: DataFrame of final counterfactuals
        cf_predicted_classes: Pre-computed predicted classes for counterfactuals (numpy array or list)
        evolution_histories: List of evolution histories (one per replication), each a list of dicts
        cf_generations_found: List of generation numbers where each CF was found (0-indexed), or None
    """
    # Standardize the dataset
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset.select_dtypes(include=[np.number]))

    # Perform PCA on the scaled dataset
    pca = PCA(n_components=2)
    iris_pca = pca.fit_transform(dataset_scaled)

    # Ensure the sample and counterfactuals are formatted as numeric DataFrame
    sample_df = pd.DataFrame([sample]).select_dtypes(include=[np.number])
    sample_df_scaled = scaler.transform(sample_df)
    if not isinstance(counterfactuals_df, pd.DataFrame):
        raise ValueError("counterfactuals_df must be a pandas DataFrame")


    numeric_cf_df = counterfactuals_df.select_dtypes(include=[np.number])
    numeric_cf_df_scaled = scaler.transform(numeric_cf_df)

    # Transform using PCA
    original_sample_pca = pca.transform(sample_df_scaled)
    counterfactuals_pca = pca.transform(numeric_cf_df_scaled)

    # Use pre-computed predicted classes
    counterfactual_classes = cf_predicted_classes

    # Plot the PCA results
    fig = plt.figure(figsize=(10, 6))
    colors = ['purple', 'green', 'orange']
    # Determine original sample class for consistent styling
    original_class = model.predict(pd.DataFrame([sample]))[0]

    for class_value in np.unique(target):
        plt.scatter(
            iris_pca[target == class_value, 0],
            iris_pca[target == class_value, 1],
            label=f"Class {class_value}",
            color=colors[class_value % len(colors)],
            alpha=0.5
        )

    plt.scatter(
        original_sample_pca[:, 0], original_sample_pca[:, 1],
        color=colors[original_class % len(colors)], label='Original Sample',
        marker='o', edgecolor=colors[original_class % len(colors)], linewidths=2.5, s=150, zorder=10
    )

    # Plot evolution histories if provided
    if evolution_histories:
        for cf_idx, history in enumerate(evolution_histories):
            if not history:
                continue
            
            
            # Convert evolution history to DataFrame and transform
            history_df = pd.DataFrame(history)
            if history_df.empty:
                continue
            
            # Extract fitness values if available (stored with '_fitness' key)
            fitness_values = None
            if '_fitness' in history_df.columns:
                fitness_values = history_df['_fitness'].values
                history_numeric = history_df.drop(columns=['_fitness']).select_dtypes(include=[np.number])
            else:
                history_numeric = history_df.select_dtypes(include=[np.number])
            
            history_scaled = scaler.transform(history_numeric)
            history_pca = pca.transform(history_scaled)
            
            # Predict classes for each generation
            history_classes = model.predict(history_numeric)
            
            # Compute opacity based on fitness (lower fitness = better = higher opacity)
            num_generations = len(history)
            if fitness_values is not None and len(fitness_values) > 0:
                min_fit = np.min(fitness_values)
                max_fit = np.max(fitness_values)
                fit_range = max_fit - min_fit if max_fit > min_fit else 1.0
            
            for gen_idx, (coords, gen_class) in enumerate(zip(history_pca, history_classes)):
                # Opacity based on fitness: better fitness (lower) -> higher opacity
                if fitness_values is not None and len(fitness_values) > 0:
                    # Normalize: 0 = worst fitness -> 0.1 alpha, 1 = best fitness -> 1.0 alpha
                    normalized_fit = (max_fit - fitness_values[gen_idx]) / fit_range
                    alpha = 0.1 + 0.9 * normalized_fit
                else:
                    # Fallback to generation-based if no fitness available
                    alpha = 0.1 + (0.9 * gen_idx / max(1, num_generations - 1))
                # Clamp alpha to [0, 1] to avoid floating-point precision errors
                alpha = np.clip(alpha, 0.0, 1.0)
                # Size for circle outline
                size = 60 if gen_idx < num_generations - 1 else 100
                # Use class color
                color = colors[gen_class % len(colors)]

                # Plot circle outline marker
                plt.scatter(
                    coords[0], coords[1],
                    facecolors='none', edgecolors=color, marker='o', s=size*2,
                    alpha=alpha, linewidths=1.5,
                    zorder=5
                )

                # Add generation number as text offset from the point
                # Offset position diagonally up-right to avoid overlap
                text_offset_x = 0.15
                text_offset_y = 0.15
                text_fontsize = 10 if gen_idx < num_generations - 1 else 12
                plt.text(
                    coords[0] + text_offset_x, coords[1] + text_offset_y, str(gen_idx + 1),
                    ha='left', va='bottom', fontsize=text_fontsize,
                    color=color, alpha=min(alpha + 0.2, 1.0), weight='bold',
                    zorder=15,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                              edgecolor=color, alpha=0.8, linewidth=1)
                )

                # Add thicker circle around final generation
                if gen_idx == num_generations - 1:
                    plt.scatter(
                        coords[0], coords[1],
                        facecolors='none', edgecolors=color,
                        s=int(size * 1.3), linewidths=2.5, alpha=1.0,
                        zorder=6
                    )

            # Draw a line connecting original sample → generations up to CF discovery → final counterfactual
            try:
                if len(history_pca) >= 1:
                    # Get the corresponding final counterfactual coordinates
                    if cf_idx < len(counterfactuals_pca):
                        final_cf_coords = counterfactuals_pca[cf_idx]
                        final_class = counterfactual_classes[cf_idx]
                        line_color = colors[final_class % len(colors)]
                        
                        # Determine how many generations to include in the path
                        # cf_generations_found is 0-indexed (gen 0 = first generation)
                        # So if CF was found at gen 5, we include history_pca[0:5] (gens 0-4)
                        if cf_generations_found and cf_idx < len(cf_generations_found) and cf_generations_found[cf_idx] is not None:
                            gen_found = cf_generations_found[cf_idx]
                            # Include generations from 0 to gen_found-1 (the ones before CF was found)
                            # If gen_found is 0, no intermediate generations (CF found in first gen after original)
                            history_slice = history_pca[:gen_found] if gen_found > 0 else np.array([]).reshape(0, 2)
                        else:
                            # Fallback: use all history
                            history_slice = history_pca
                        
                        # Build path: original → evolution generations up to discovery → final counterfactual
                        if len(history_slice) > 0:
                            x_coords = np.concatenate([
                                [original_sample_pca[0, 0]], 
                                history_slice[:, 0],
                                [final_cf_coords[0]]
                            ])
                            y_coords = np.concatenate([
                                [original_sample_pca[0, 1]], 
                                history_slice[:, 1],
                                [final_cf_coords[1]]
                            ])
                        else:
                            # Direct connection from original to CF (no intermediate gens)
                            x_coords = np.array([original_sample_pca[0, 0], final_cf_coords[0]])
                            y_coords = np.array([original_sample_pca[0, 1], final_cf_coords[1]])
                        
                        plt.plot(x_coords, y_coords,
                                 color=line_color, linewidth=1.0, alpha=0.6, zorder=4)
            except Exception as e:
                print(f"WARNING: Failed to draw evolution line for cf_idx {cf_idx}: {e}")
    
    # Always plot final counterfactuals as circles (same style as class samples)
    for idx, cf_class in enumerate(counterfactual_classes):
        cf_color = colors[cf_class % len(colors)]
        # Plot circle marker with edge matching class color
        plt.scatter(
            counterfactuals_pca[idx, 0], counterfactuals_pca[idx, 1],
            color=cf_color, marker='o', s=150,
            edgecolor=cf_color, linewidths=2.5, alpha=1.0, zorder=10
        )
        
        # Add generation number label (same style as evolution nodes)
        # Use actual generation_found if available, otherwise fall back to history length
        if cf_generations_found and idx < len(cf_generations_found) and cf_generations_found[idx] is not None:
            # Add 1 because generation_found is 0-indexed but we display from 1
            gen_num = cf_generations_found[idx] + 1
            gen_label = str(gen_num)
        elif evolution_histories and idx < len(evolution_histories) and evolution_histories[idx]:
            # Fallback: use history length + 1 (final CF is one after last recorded generation)
            gen_num = len(evolution_histories[idx]) + 1
            gen_label = str(gen_num)
        else:
            gen_label = '?'  # No history or generation info available
            
        plt.text(
            counterfactuals_pca[idx, 0] + 0.15, counterfactuals_pca[idx, 1] + 0.15,
            gen_label,
            ha='left', va='bottom', fontsize=12,
            color=cf_color, alpha=1.0, weight='bold',
            zorder=16,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                      edgecolor=cf_color, alpha=0.8, linewidth=1)
        )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Plot with GA Evolution (opacity: initial→final)')
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[original_class % len(colors)], markersize=10, 
               markeredgecolor=colors[original_class % len(colors)], markeredgewidth=1.5, label='Original Sample'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markersize=8,
               markeredgecolor='gray', markeredgewidth=1.5, label='GA Evolution (faint→solid)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10,
               markeredgecolor='gray', markeredgewidth=1.5, label='Final Counterfactuals')
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.close(fig)
    return fig

def heatmap_techniques(sample, class_sample, cf_list_1, cf_list_2, technique_names, restrictions=None):
    """
    Plot a heatmap comparing counterfactuals from two different techniques.

    The heatmap shows the original sample in the first row, followed by counterfactuals
    from both techniques. Differences from the original are highlighted with color shading.

    Args:
        sample (dict): Original sample values.
        class_sample: Class of the original sample.
        cf_list_1 (list): List of counterfactual dictionaries from technique 1.
        cf_list_2 (list): List of counterfactual dictionaries from technique 2.
        technique_names (tuple): Names of the two techniques (e.g., ('DPG', 'DiCE')).
        restrictions (dict, optional): Restrictions applied to each feature.

    Returns:
        matplotlib.figure.Figure: The generated figure (closed to avoid display side effects).
    """
    # Create DataFrame from the original sample
    sample_df = pd.DataFrame([sample], index=['Original'])

    # Combine counterfactuals from both techniques
    all_rows = [sample_df]

    # Add counterfactuals from technique 1
    for i, cf in enumerate(cf_list_1):
        cf_df = pd.DataFrame([cf], index=[f'{technique_names[0]} CF {i+1}'])
        all_rows.append(cf_df)

    # Add counterfactuals from technique 2
    for i, cf in enumerate(cf_list_2):
        cf_df = pd.DataFrame([cf], index=[f'{technique_names[1]} CF {i+1}'])
        all_rows.append(cf_df)

    # Combine all rows
    full_df = pd.concat(all_rows)

    # Sort features alphabetically
    full_df = full_df[sorted(full_df.columns)]

    # Reorder restrictions to match sorted features
    if restrictions:
        restrictions = {feat: restrictions[feat] for feat in sorted(full_df.columns) if feat in restrictions}

    # Calculate differences from original for color highlighting
    # First row (original) = 0 for neutral color, other rows = delta from original
    diff_matrix = full_df.copy()
    diff_matrix.iloc[0] = 0  # Original row: no color (neutral)
    for i in range(1, len(full_df)):
        diff_matrix.iloc[i] = full_df.iloc[i] - full_df.iloc[0]

    # Calculate vmin/vmax for symmetric color scaling (exclude first row which is zeros)
    if len(diff_matrix) > 1:
        vmax = np.max(np.abs(diff_matrix.iloc[1:].values))
    else:
        vmax = 0
    vmin = -vmax

    if vmax == 0:
        vmax = 1
        vmin = -1

    # Create the heatmap using diff_matrix for colors, full_df values for annotations
    fig = plt.figure(figsize=(12, max(6, 2 + 0.2 * len(cf_list_1) + len(cf_list_2))))

    ax = sns.heatmap(
        diff_matrix,  # Use diff_matrix for coloring (first row = 0 = neutral)
        annot=full_df.values,  # Show actual values as annotations
        fmt=".2f",
        cmap='coolwarm',
        cbar=True,
        linewidths=1.0,
        linecolor='k',
        vmin=vmin,
        cbar_kws={"label": "Magnitude difference"},
        vmax=vmax,
    )

    # Add restriction icons below the heatmap if provided
    if restrictions:
        symbol_map = {
            'no_change': '⊝',
            'non_increasing': '⬇️',
            'non_decreasing': '⬆️'
        }
        for i, (feat, restr) in enumerate(restrictions.items()):
            if restr in symbol_map:
                ax.text(i + 0.5, len(full_df) + 0.5, symbol_map[restr],
                       ha='center', va='top', color='black',
                       fontweight='bold', fontsize=14, transform=ax.transData)

    plt.title(f'Technique Comparison - Original (Class {class_sample}) vs Counterfactuals', fontsize=14, fontweight='bold')
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0, va="center")
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_ridge_comparison(
    sample, 
    cf_list_1, 
    cf_list_2, 
    technique_names=('DPG', 'DiCE'),
    method_1_color="#CC0000", 
    method_2_color="#006DAC",
    sample_color="#333333",
    dataset_df=None,
    dataset_color="#7B68EE",
    constraints=None,
    target_class=None,
    original_class=None,
    show_original_class_constraints=False,
    showExtendedConstraints=False,
    figsize=None,
    target=None,
    show_per_class_distribution=True,
    show_overall_distribution=True,
    class_colors_list=None
):
    """
    Plot a ridge plot (joy plot) comparing the original sample with counterfactuals
    from two different techniques. Each row represents a different feature.
    
    Shows the full dataset distribution and optionally per-class distributions overlaid.
    
    Args:
        sample (dict): Original sample values.
        cf_list_1 (list): List of counterfactual dictionaries from technique 1 (e.g., DPG).
        cf_list_2 (list): List of counterfactual dictionaries from technique 2 (e.g., DiCE).
        technique_names (tuple): Names of the two techniques (default: ('DPG', 'DiCE')).
        method_1_color (str): Color for technique 1 (default: orange).
        method_2_color (str): Color for technique 2 (default: blue).
        sample_color (str): Color for the original sample marker (default: dark gray).
        dataset_df (pd.DataFrame, optional): Full dataset DataFrame to show distribution.
        dataset_color (str): Color for the dataset distribution (default: purple).
        constraints (dict, optional): DPG constraints dict with format {class: {feature: {'min': val, 'max': val}}}.
        target_class (int, optional): Target class for constraints. If None, uses first class in constraints.
        original_class (int, optional): Original sample's class. Required for show_original_class_constraints.
        show_original_class_constraints (bool, optional): Whether to show constraints from the original class. Default False.
        showExtendedConstraints (bool, optional): Whether to show orange extended constraints when original sample is outside DPG constraints. Default False.
        figsize (tuple, optional): Figure size. If None, calculated based on number of features.
        target (array-like, optional): Class labels for each sample in dataset_df. Required for per-class distributions.
        show_per_class_distribution (bool, optional): Whether to show per-class distributions. Default True.
        show_overall_distribution (bool, optional): Whether to show overall dataset distribution. Default True.
        class_colors_list (list, optional): List of colors for each class. If None, uses default palette.
    
    Returns:
        matplotlib.figure.Figure: The generated ridge plot figure.
    """
    # Get feature names sorted alphabetically
    feature_names = sorted(sample.keys())
    n_features = len(feature_names)
    
    # Default class colors if not provided
    if class_colors_list is None:
        class_colors_list = ['purple', 'green', 'orange', 'red', 'blue', 'yellow', 'pink', 'cyan']
    
    # Calculate figure size if not provided
    if figsize is None:
        figsize = (10, max(6, n_features * 0.8))
    
    # Compute min/max per feature for normalization (0-1 range)
    # Include dataset values if provided
    feature_min = {}
    feature_max = {}
    for feat in feature_names:
        all_values = [sample[feat]]
        all_values.extend([cf[feat] for cf in cf_list_1])
        all_values.extend([cf[feat] for cf in cf_list_2])
        if dataset_df is not None and feat in dataset_df.columns:
            all_values.extend(dataset_df[feat].dropna().tolist())
        feature_min[feat] = min(all_values)
        feature_max[feat] = max(all_values)
    
    def normalize(value, feat):
        """Normalize value to [0, 1] range for a given feature."""
        min_val = feature_min[feat]
        max_val = feature_max[feat]
        if max_val == min_val:
            return 0.5  # Constant feature -> center it
        return (value - min_val) / (max_val - min_val)
    
    # Prepare counterfactual data (kept for future use, not plotted currently)
    cf_data_rows = []
    
    # Add original sample (normalized)
    for feat in feature_names:
        cf_data_rows.append({
            'feature': feat,
            'value': normalize(sample[feat], feat),
            'type': 'Original',
            'technique': 'Original'
        })
    
    # Add counterfactuals from technique 1 (normalized)
    for i, cf in enumerate(cf_list_1):
        for feat in feature_names:
            cf_data_rows.append({
                'feature': feat,
                'value': normalize(cf[feat], feat),
                'type': 'CF',
                'technique': technique_names[0]
            })
    
    # Add counterfactuals from technique 2 (normalized)
    for i, cf in enumerate(cf_list_2):
        for feat in feature_names:
            cf_data_rows.append({
                'feature': feat,
                'value': normalize(cf[feat], feat),
                'type': 'CF',
                'technique': technique_names[1]
            })
    
    cf_df = pd.DataFrame(cf_data_rows)  # Kept for future use
    
    # Prepare dataset data for plotting
    if dataset_df is None:
        print("Warning: No dataset_df provided, cannot create ridge plot")
        return None
    
    # Check if we can show per-class distributions
    can_show_per_class = show_per_class_distribution and target is not None and len(target) == len(dataset_df)
    
    if show_per_class_distribution and not can_show_per_class:
        print("Warning: Cannot show per-class distributions. Either target is None or length mismatch with dataset_df.")
        print(f"  dataset_df length: {len(dataset_df)}, target length: {len(target) if target is not None else 'None'}")
    
    dataset_rows = []
    for idx, row in enumerate(dataset_df.itertuples()):
        for feat in feature_names:
            if feat in dataset_df.columns:
                val = getattr(row, feat.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_'), None)
                if val is None:
                    # Try direct column access
                    val = dataset_df.iloc[idx][feat]
                
                if pd.notna(val):
                    row_data = {
                        'feature': feat,
                        'value': normalize(val, feat),
                        'type': 'Dataset'
                    }
                    if can_show_per_class:
                        row_data['class'] = int(target[idx])
                    dataset_rows.append(row_data)
    
    df = pd.DataFrame(dataset_rows)
    
    # Set up the plot style
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Initialize the FacetGrid with features as rows
    g = sns.FacetGrid(
        df,
        row="feature", 
        aspect=12, 
        height=0.7, 
        row_order=feature_names
    )
    
    if can_show_per_class:
        # Manually plot per-class distributions on each axis
        unique_classes = sorted(df['class'].unique())
        
        for ax, feat in zip(g.axes.flat, feature_names):
            # Get data for this feature
            feat_df = df[df['feature'] == feat]
            
            # Plot each class
            for class_idx in unique_classes:
                class_color = class_colors_list[class_idx % len(class_colors_list)]
                class_data = feat_df[feat_df['class'] == class_idx]['value']
                
                if len(class_data) > 1:
                    # Draw filled KDE for this class
                    sns.kdeplot(
                        data=class_data,
                        bw_adjust=0.5,
                        clip_on=False,
                        fill=True,
                        alpha=0.4,
                        linewidth=1.5,
                        color=class_color,
                        ax=ax
                    )
                    # Draw outline
                    sns.kdeplot(
                        data=class_data,
                        bw_adjust=0.5,
                        clip_on=False,
                        color=class_color,
                        lw=1.5,
                        alpha=0.8,
                        ax=ax
                    )
            
            # Also draw the overall dataset distribution (lighter, as background) if enabled
            if show_overall_distribution:
                sns.kdeplot(
                    data=feat_df['value'],
                    bw_adjust=0.5,
                    clip_on=False,
                    fill=True,
                    alpha=0.2,
                    linewidth=0.8,
                    color=dataset_color,
                    ax=ax
                )
    else:
        # Draw only the overall densities if enabled (original behavior)
        if show_overall_distribution:
            g.map(sns.kdeplot, "value", bw_adjust=0.5, clip_on=False, fill=True, alpha=0.6, linewidth=1.5, color=dataset_color)
            g.map(sns.kdeplot, "value", bw_adjust=0.5, clip_on=False, color="w", lw=2)
    
    # Add horizontal reference line at y=0 to each axis (10% wider than data)
    for ax in g.axes.flat:
        ax.plot([-0.08, 1.08], [-0.15, -0.15], color="#7B68EE", linewidth=0.1, clip_on=False)
    
    # Compute KDE for each feature to get y-values for markers
    # Use scipy's gaussian_kde with Scott's factor adjusted by 0.5 to match seaborn's bw_adjust=0.5
    from scipy.stats import gaussian_kde
    feature_kdes = {}
    feature_kdes_per_class = {}  # Store per-class KDEs
    
    for feat in feature_names:
        feat_values = df[df['feature'] == feat]['value'].values
        if len(feat_values) > 1 and np.std(feat_values) > 0:
            kde = gaussian_kde(feat_values)
            # Adjust bandwidth to match seaborn's bw_adjust=0.5
            kde.set_bandwidth(kde.factor * 0.5)
            feature_kdes[feat] = kde
        else:
            feature_kdes[feat] = None
        
        # Compute per-class KDEs if available
        if can_show_per_class:
            feature_kdes_per_class[feat] = {}
            unique_classes = sorted(df['class'].unique())
            for class_idx in unique_classes:
                class_values = df[(df['feature'] == feat) & (df['class'] == class_idx)]['value'].values
                if len(class_values) > 1 and np.std(class_values) > 0:
                    class_kde = gaussian_kde(class_values)
                    class_kde.set_bandwidth(class_kde.factor * 0.5)
                    feature_kdes_per_class[feat][class_idx] = class_kde
                else:
                    feature_kdes_per_class[feat][class_idx] = None
    
    # Apply sample marker and counterfactual markers to each facet (using normalized values)
    for ax, feat in zip(g.axes.flat, feature_names):
        # Determine which KDE to use for positioning markers
        if not show_overall_distribution and can_show_per_class and target_class is not None:
            # Use target class KDE when overall distribution is not shown
            kde = feature_kdes_per_class.get(feat, {}).get(target_class)
        else:
            # Use overall dataset KDE
            kde = feature_kdes.get(feat)
        
        # Original sample marker
        sample_val = normalize(sample[feat], feat)
        sample_y = kde(sample_val)[0] if kde else 0
        ax.scatter([sample_val], [sample_y], marker='o', s=70, color=sample_color, edgecolor='white', linewidth=1, zorder=15, clip_on=False)
        
        # Plot counterfactuals from method 1 (DPG) - triangles
        for cf in cf_list_1:
            cf_val = normalize(cf[feat], feat)
            cf_y = kde(cf_val)[0] if kde else 0
            ax.scatter([cf_val], [cf_y + 0.7], marker='v', s=90, color=method_1_color, edgecolor='white', linewidth=1, zorder=12, clip_on=False, alpha=0.9)
        
        # Plot counterfactuals from method 2 (DiCE) - squares
        for cf in cf_list_2:
            cf_val = normalize(cf[feat], feat)
            cf_y = kde(cf_val)[0] if kde else 0
            ax.scatter([cf_val], [cf_y - 0.7], marker='^', s=90, color=method_2_color, edgecolor='white', linewidth=1, zorder=11, clip_on=False, alpha=0.8)
        
        ax.set_xlim(0, 1)  # Set consistent x-axis range (0-1 normalized)
    
    # Draw DPG constraints as shaded valid region between min and max
    constraint_color = "#228B22"  # Forest green for target class
    original_constraint_color = "#4169E1"  # Royal blue for original class
    orange_color = "#FF8C00"  # Dark orange for out-of-constraint indicator
    has_constraints = False
    has_original_constraints = False
    if constraints:
        # Determine which class constraints to use for target class
        if target_class is not None:
            class_key = f'Class {target_class}'
        else:
            # Use first available class
            class_key = list(constraints.keys())[0] if constraints else None
        
        if class_key and class_key in constraints:
            class_constraints = constraints[class_key]
            # Handle both list and dict formats
            constraint_dict = _normalize_constraint_dict(class_constraints)
            
            for ax, feat in zip(g.axes.flat, feature_names):
                if feat in constraint_dict:
                    feat_constraints = constraint_dict[feat]
                    min_val = feat_constraints.get('min')
                    max_val = feat_constraints.get('max')
                    
                    # Normalize constraint values
                    min_norm = normalize(min_val, feat) if min_val is not None else 0
                    max_norm = normalize(max_val, feat) if max_val is not None else 1
                    
                    # Draw shaded band for valid constraint region (limit height to avoid overlap)
                    ax.axvspan(min_norm, max_norm, ymin=0.1, ymax=0.7, color=constraint_color, alpha=0.15, zorder=1)
                    # Draw boundary lines at min and max (also limited in height)
                    if min_val is not None:
                        ax.axvline(x=min_norm, ymin=0.1, ymax=0.7, color=constraint_color, linewidth=1.5, linestyle='--', alpha=0.8, zorder=5)
                        # Add triangle pointing inward (to the right) from min constraint
                        # Use axis coordinates (transform) for y-position to match axvline ymin/ymax
                        ax.scatter([min_norm + 0.008], [0.4], marker='>', s=80, 
                                  color=constraint_color, alpha=1.0, zorder=6, 
                                  edgecolor='white',
                                  transform=ax.get_xaxis_transform())
                    if max_val is not None:
                        ax.axvline(x=max_norm, ymin=0.1, ymax=0.7, color=constraint_color, linewidth=1.5, linestyle='--', alpha=0.8, zorder=5)
                        # Add triangle pointing inward (to the left) from max constraint
                        ax.scatter([max_norm - 0.008], [0.4], marker='<', s=80, 
                                  color=constraint_color, alpha=1.0, zorder=6, 
                                  edgecolor='white',
                                  transform=ax.get_xaxis_transform())
                    
                    # Draw orange boundary region when original sample is outside constraints
                    if showExtendedConstraints:
                        # Get normalized value of original sample for this feature
                        sample_val_norm = normalize(sample[feat], feat)
                        
                        # Check if original value is outside constraints
                        is_outside = False
                        target_norm = None
                        
                        if min_val is not None and max_val is not None:
                            # Both constraints exist, check if outside range
                            if sample_val_norm < min_norm:
                                is_outside = True
                                target_norm = min_norm  # Point toward min
                            elif sample_val_norm > max_norm:
                                is_outside = True
                                target_norm = max_norm  # Point toward max
                        elif min_val is not None:
                            # Only min constraint, check if below
                            if sample_val_norm < min_norm:
                                is_outside = True
                                target_norm = min_norm  # Point toward min
                        elif max_val is not None:
                            # Only max constraint, check if above
                            if sample_val_norm > max_norm:
                                is_outside = True
                                target_norm = max_norm  # Point toward max
                        
                        # Draw orange shaded region and boundary lines similar to green constraints
                        if is_outside and target_norm is not None:
                            # Determine the range for the orange shaded region
                            if sample_val_norm < target_norm:
                                # Original is below, shade from original to min
                                orange_min, orange_max = sample_val_norm, target_norm
                            else:
                                # Original is above, shade from max to original
                                orange_min, orange_max = target_norm, sample_val_norm
                            
                            # Draw shaded band for extended constraint region (similar style to green)
                            ax.axvspan(orange_min, orange_max, ymin=0.1, ymax=0.7, 
                                      color=orange_color, alpha=0.15, zorder=2)
                            
                            # Draw boundary line at original sample position (solid, like constraint lines)
                            ax.axvline(x=sample_val_norm, ymin=0.1, ymax=0.7, 
                                      color=orange_color, linewidth=1.5, linestyle='--', alpha=0.8, zorder=6)

                        has_constraints = True
        
        # Draw constraints from original class if requested
        if show_original_class_constraints and original_class is not None:
            original_class_key = f'Class {original_class}'
            if original_class_key in constraints:
                original_class_constraints = constraints[original_class_key]
                # Handle both list and dict formats
                orig_constraint_dict = _normalize_constraint_dict(original_class_constraints)
                
                for ax, feat in zip(g.axes.flat, feature_names):
                    if feat in orig_constraint_dict:
                        feat_constraints = orig_constraint_dict[feat]
                        min_val = feat_constraints.get('min')
                        max_val = feat_constraints.get('max')
                        
                        # Normalize constraint values
                        min_norm = normalize(min_val, feat) if min_val is not None else 0
                        max_norm = normalize(max_val, feat) if max_val is not None else 1
                        
                        # Draw shaded band for original class constraint region (at top)
                        # Use ymin=0.75, ymax=0.95 to place it at the top, distinct from target class constraints
                        ax.axvspan(min_norm, max_norm, ymin=0.75, ymax=0.95, color=original_constraint_color, alpha=0.2, zorder=1)
                        # Draw boundary lines at min and max
                        if min_val is not None:
                            ax.axvline(x=min_norm, ymin=0.75, ymax=0.95, color=original_constraint_color, linewidth=1.5, linestyle=':', alpha=0.8, zorder=5)
                            # Add triangle pointing inward (to the right) from min constraint
                            # Use axis coordinates (transform) for y-position to match axvline ymin/ymax (midpoint = 0.85)
                            ax.scatter([min_norm + 0.008], [0.85], marker='>', s=60, 
                                      color=original_constraint_color, alpha=1.0, zorder=6, 
                                      transform=ax.get_xaxis_transform())
                        if max_val is not None:
                            ax.axvline(x=max_norm, ymin=0.75, ymax=0.95, color=original_constraint_color, linewidth=1.5, linestyle=':', alpha=0.8, zorder=5)
                            # Add triangle pointing inward (to the left) from max constraint
                            ax.scatter([max_norm - 0.008], [0.85], marker='<', s=60, 
                                      color=original_constraint_color, alpha=1.0, zorder=6, 
                                      transform=ax.get_xaxis_transform())
                        
                        has_original_constraints = True
    
    # Overlap the plots vertically for the ridge effect (reduced overlap to prevent spillover)
    # Also reduce left margin to make the overall figure narrower
    g.figure.subplots_adjust(hspace=-0.1, left=0.1)
    
    # Limit y-axis height - cap at a reasonable fixed value to prevent overflow
    for ax in g.axes.flat:
        ylim = ax.get_ylim()
        ax.set_ylim(min(ylim[0], 0), min(ylim[1], 20))
    
    # Remove axes details
    g.set_titles("")
    g.set(yticks=[], ylabel="", xticks=[], xlabel="")
    g.despine(bottom=True, left=True)
    
    
    for ax, feat in zip(g.axes.flat, feature_names):
        # Clean feature name for display
        clean_name = feat.replace(' (cm)', '').replace('_', ' ')
        ax.text(-0.10, 0.2, clean_name, fontweight="bold", color="black",
                ha="right", va="center", transform=ax.transAxes, fontsize=10, zorder=25)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=sample_color, markersize=10, 
               markeredgecolor='white', markeredgewidth=1.5, linestyle='None', 
               label='Original Sample'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor=method_1_color, markersize=10,
               markeredgecolor='white', markeredgewidth=1, linestyle='None',
               label=f'{technique_names[0]} CFs'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=method_2_color, markersize=9,
               markeredgecolor='white', markeredgewidth=1, linestyle='None',
               label=f'{technique_names[1]} CFs'),
    ]
    
    # Add dataset distribution legend entry
    if can_show_per_class:
        # Show per-class entries
        unique_classes = sorted(df['class'].unique())
        for class_idx in unique_classes:
            class_color = class_colors_list[class_idx % len(class_colors_list)]
            legend_elements.append(
                Line2D([0], [0], color=class_color, lw=4, alpha=0.6, label=f'Class {class_idx} Distribution')
            )
        # Add overall dataset as background if enabled
        if show_overall_distribution:
            legend_elements.append(
                Line2D([0], [0], color=dataset_color, lw=3, alpha=0.3, label='Overall Distribution')
            )
    else:
        # Show only overall dataset if enabled
        if show_overall_distribution:
            legend_elements.append(
                Line2D([0], [0], color=dataset_color, lw=4, alpha=0.6, label='Dataset Distribution')
            )
    
    # Add constraint legend entries if constraints were drawn
    if has_constraints:
        from matplotlib.patches import Patch
        legend_elements.append(
            Patch(facecolor=constraint_color, alpha=0.15, edgecolor=constraint_color,
                  linestyle='--', linewidth=1.5, label='Target Class Constraints')
        )
        # Add orange constraint legend for extended constraints (only if shown)
        if showExtendedConstraints:
            legend_elements.append(
                Patch(facecolor=orange_color, alpha=0.15, edgecolor=orange_color,
                      linestyle='--', linewidth=1.5, label='Extended DPG Constraints')
            )
    
    # Add original class constraints legend if shown
    if has_original_constraints:
        from matplotlib.patches import Patch
        legend_elements.append(
            Patch(facecolor=original_constraint_color, alpha=0.2, edgecolor=original_constraint_color,
                  linestyle=':', linewidth=1.5, label='Original Class Constraints')
        )
    
    # Always show DPG Constraints legend entry when constraints parameter is provided (but no constraints drawn)
    elif constraints:
        from matplotlib.patches import Patch
        legend_elements.append(
            Patch(facecolor=constraint_color, alpha=0.15, edgecolor=constraint_color,
                  linestyle='--', linewidth=1.5, label='DPG Class 1 Constraints')
        )

    legend = g.figure.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                             framealpha=1.0, facecolor='white', edgecolor='gray',
                             bbox_to_anchor=(1.02, 0.98))
    for text in legend.get_texts():
        text.set_color('black')
        
    # Add title
    g.figure.suptitle(f'Feature Distribution: {technique_names[0]} vs {technique_names[1]} Counterfactuals', 
                      fontsize=14, fontweight='bold', y=1.02)
    
    # Add background-colored rectangles on both sides to align horizontal bars
    fig_background_color = g.figure.get_facecolor()
    for ax in g.axes.flat:
        # Left alignment rectangle
        left_rect = Rectangle(
            xy=(-0.2, 0), 
            width=0.11, 
            height=0.5,
            transform=ax.transAxes,
            facecolor=fig_background_color,
            edgecolor='none',
            zorder=20,
            clip_on=False
        )
        ax.add_patch(left_rect)
        
        # Right alignment rectangle
        right_rect = Rectangle(
            xy=(1.07, 0), 
            width=0.1, 
            height=1.0,
            transform=ax.transAxes,
            facecolor=fig_background_color,
            edgecolor='none',
            zorder=20,
            clip_on=False
        )
        ax.add_patch(right_rect)
    
    fig = g.figure
    plt.close(fig)
    
    # Reset seaborn style to avoid affecting other plots
    sns.reset_defaults()
    
    return fig


