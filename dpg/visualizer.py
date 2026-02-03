import os
import re
import numpy as np
from io import BytesIO
from typing import Dict, List, Optional, TYPE_CHECKING
from graphviz import Source
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from PIL import Image
from .utils import highlight_class_node, change_node_color, delete_folder_contents

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from PIL import Image
from graphviz import Source

Image.MAX_IMAGE_PIXELS = 500000000  # Adjust based on your needs

def plot_dpg(
    plot_name,
    dot,
    df,
    df_edges,
    save_dir="results/",
    attribute=None,
    clusters=None,
    threshold_clusters=None,
    class_flag=False,
    show=True,
    export_pdf=False,
):
    """
    Plot a Decision Predicate Graph (DPG) with optional node/edge styling.

    Args:
    plot_name: Output base name for saved files (no extension).
    dot: Graphviz Digraph instance representing the DPG structure.
    df: DataFrame with node metrics; must include 'Node' and 'Label' columns.
    df_edges: DataFrame with edge metrics; must include 'Source_id', 'Target_id', and 'Weight'.
    save_dir: Directory where output images are saved. Default is "results/".
    attribute: Optional node metric column name to color nodes by (e.g., 'Degree').
    clusters: Optional mapping {cluster_label: [node_id, ...]} to color nodes by clusters.
    threshold_clusters: Optional value used only to annotate the output name.
    class_flag: If True, class nodes are highlighted in yellow before other coloring.
    show: Whether to display the image via matplotlib. Default is True.
    export_pdf: If True, also writes a PDF next to the PNG.

    Returns:
    None
    """
    print("Plotting DPG...")
    # Basic color scheme if no attribute or communities are specified
    if attribute is None and clusters is None:
        for index, row in df.iterrows():
            if 'Class' in row['Label']:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(157, 195, 230))  # Light blue for class nodes
            else:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(222, 235, 247))  # Light grey for other nodes


    # Color nodes based on a specific attribute
    elif attribute is not None and clusters is None:
        colormap = cm.Blues  # Choose a colormap
        norm = None

        # Highlight class nodes if class_flag is True
        if class_flag:
            for index, row in df.iterrows():
                if 'Class' in row['Label']:
                    change_node_color(dot, row['Node'], '#ffc000')  # Yellow for class nodes
            df = df[~df.Label.str.contains('Class')].reset_index(drop=True)  # Exclude class nodes from further processing
        
        # Normalize the attribute values if norm_flag is True
        max_score = df[attribute].max()
        norm = mcolors.Normalize(0, max_score)
        colors = colormap(norm(df[attribute]))  # Assign colors based on normalized scores
        
        for index, row in df.iterrows():
            color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255), int(colors[index][1]*255), int(colors[index][2]*255))
            change_node_color(dot, row['Node'], color)
        
        plot_name = plot_name + f"_{attribute}".replace(" ","")
    

    elif attribute is None and clusters is not None:
        colormap = cm.get_cmap('tab20')  # Choose a colormap
        
        # Highlight class nodes if class_flag is True
        if class_flag:
            for index, row in df.iterrows():
                if 'Class' in row['Label']:
                    change_node_color(dot, row['Node'], '#ffc000')  # Yellow for class nodes
            df = df[~df.Label.str.contains('Class')].reset_index(drop=True)  # Exclude class nodes from further processing
        
        node_to_cluster = {}
        
        for clabel, node_list in clusters.items():
            for node_id in node_list:
                node_to_cluster[str(node_id)] = clabel

        df['Cluster'] = df['Node'].astype(str).map(lambda n: node_to_cluster.get(n, 'ambiguous'))

        unique_clusters = sorted([c for c in df['Cluster'].unique() if c != 'ambiguous'])
        cluster_to_idx = {lab: i for i, lab in enumerate(unique_clusters)}
        ambiguous_idx = len(unique_clusters)
        cluster_to_idx['ambiguous'] = ambiguous_idx

        n_colors = max(1, len(cluster_to_idx))
        palette_rgba = [colormap(i / max(1, n_colors - 1)) for i in range(n_colors)]
        palette_hex = ["#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
                    for (r, g, b, _) in palette_rgba]

        if 'ambiguous' in cluster_to_idx:
            palette_hex[cluster_to_idx['ambiguous']] = '#bdbdbd'  # grigio chiaro

        for i, row in df.iterrows():
            idx = cluster_to_idx.get(row['Cluster'], cluster_to_idx['ambiguous'])
            color = palette_hex[idx]
            change_node_color(dot, row['Node'], color)

        plot_name = plot_name + f"_clusters_{threshold_clusters}"


    else:
        raise AttributeError("The plot can show the basic plot, clusters or a specific node-metric")


    # Highlight edges
    colormap_edge = cm.Greys  # Colormap edges
    max_edge_value = df_edges['Weight'].max()
    min_edge_value = df_edges['Weight'].min()
    norm_edge = mcolors.Normalize(vmin=min_edge_value, vmax=max_edge_value)
    for index, row in df_edges.iterrows():
        edge_value = row['Weight']
        color = colormap_edge(norm_edge(edge_value))
        color_hex = "#{:02x}{:02x}{:02x}".format(int(color[0]*255),
                                                    int(color[1]*255),
                                                    int(color[2]*255))
        penwidth = 1 + 3 * norm_edge(edge_value)

        change_edge_color(dot, row['Source_id'], row['Target_id'], new_color=color_hex, new_width = penwidth)

    # Convert to scientific notation
    # def to_sci_notation(match):
    #     num = float(match.group(1))
    #     return f'label="{num:.2e}"'
    # pattern = r'label=([0-9]+\.?[0-9]*)'
    # for i in range(len(dot.body)):
    #     dot.body[i] = re.sub(pattern, to_sci_notation, dot.body[i])
        # if "->" in dot.body[i]:
        #     dot.body[i] = re.sub(r'\s*label="[^"]*"', '', dot.body[i])
    

    # Highlight class nodes
    highlight_class_node(dot)

    # Render the graph to PNG bytes (avoid temp files)
    def _sanitize_dot_source(source: str) -> str:
        # Escape brackets and quotes in node labels to avoid DOT parse errors.
        def repl(m):
            label = m.group(1)
            label = label.replace("\\", "\\\\").replace('"', '\\"')
            label = label.replace("[", "\\[").replace("]", "\\]")
            return f'label="{label}"'

        source = re.sub(r'label="([^"]*)"', repl, source)
        source = re.sub(r'label=([^\\s\\]]+)', r'label="\\1"', source)
        return source

    try:
        graph = Source(dot.source)
        png_bytes = graph.pipe(format="png")
    except Exception as e:
        print(f"Plotting failed with {type(e).__name__}; retrying with sanitized DOT source.")
        graph = Source(_sanitize_dot_source(dot.source))
        png_bytes = graph.pipe(format="png")

    # Open and display the rendered image
    img = Image.open(BytesIO(png_bytes))
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_axis_off()
    ax.set_title(plot_name)
    ax.imshow(img)
    
    # Add a color bar if an attribute is specified
    if attribute is not None:
        # Place the colorbar just below the graph to reduce whitespace
        ax_pos = ax.get_position()
        cbar_height = 0.02
        cbar_pad = 0.02
        cbar_y = max(0.01, ax_pos.y0 - (cbar_height + cbar_pad))
        cax = fig.add_axes([ax_pos.x0, cbar_y, ax_pos.width, cbar_height])
        cbar = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=colormap),
            cax=cax,
            orientation='horizontal',
        )
        cbar.set_label(attribute)

    # Save the plot to the specified directory
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, plot_name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    if export_pdf:
        fig.savefig(
            os.path.join(save_dir, plot_name + ".pdf"),
            format="pdf",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    #plt.show()
    # No PDF output by default
    
    # Clean up temporary files
    # delete_folder_contents("temp")
    if not show:
        plt.close(fig)

def plot_dpg_communities(
    plot_name,
    dot,
    df,
    dpg_metrics,
    save_dir="results/",
    class_flag=False,
    df_edges=None,
    show=True,
    export_pdf=False,
):
    """
    Plot a DPG colored by community assignment.

    Args:
    plot_name: Output base name for saved files (no extension).
    dot: Graphviz Digraph instance representing the DPG structure.
    df: DataFrame with node metrics; must include 'Node' and 'Label' columns.
    dpg_metrics: Dict containing 'Communities' (list of sets/lists of node labels).
    save_dir: Directory where output images are saved. Default is "results/".
    class_flag: If True, class nodes are highlighted in yellow before other coloring.
    df_edges: Optional DataFrame with edge metrics to color edges by weight.
    show: Whether to display the image via matplotlib. Default is True.
    export_pdf: If True, also writes a PDF next to the PNG.

    Returns:
    None
    """
    print("Plotting DPG (communities)...")

    if dpg_metrics is None or "Communities" not in dpg_metrics:
        raise AttributeError("dpg_metrics with 'Communities' is required to plot communities.")

    colormap = cm.YlOrRd  # Choose a colormap

    # Highlight class nodes if class_flag is True
    if class_flag:
        for index, row in df.iterrows():
            if 'Class' in row['Label']:
                change_node_color(dot, row['Node'], '#ffc000')  # Yellow for class nodes
        df = df[~df.Label.str.contains('Class')].reset_index(drop=True)  # Exclude class nodes from further processing

    # Map labels to community indices
    communities = dpg_metrics.get("Communities", [])
    label_to_community = {}
    for idx, community in enumerate(communities):
        for label in community:
            label_to_community[label] = idx
    df['Community'] = df['Label'].map(label_to_community)

    max_score = df['Community'].max()
    norm = mcolors.Normalize(0, max_score)  # Normalize the community indices

    colors = colormap(norm(df['Community']))  # Assign colors based on normalized community indices

    for index, row in df.iterrows():
        color = "#{:02x}{:02x}{:02x}".format(
            int(colors[index][0] * 255),
            int(colors[index][1] * 255),
            int(colors[index][2] * 255),
        )
        change_node_color(dot, row['Node'], color)

    plot_name = plot_name + "_communities"

    # Highlight edges (optional)
    if df_edges is not None:
        colormap_edge = cm.Greys  # Colormap edges
        max_edge_value = df_edges['Weight'].max()
        min_edge_value = df_edges['Weight'].min()
        norm_edge = mcolors.Normalize(vmin=min_edge_value, vmax=max_edge_value)
        for index, row in df_edges.iterrows():
            edge_value = row['Weight']
            color = colormap_edge(norm_edge(edge_value))
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255),
            )
            penwidth = 1 + 3 * norm_edge(edge_value)

            change_edge_color(dot, row['Source_id'], row['Target_id'], new_color=color_hex, new_width=penwidth)

    # Highlight class nodes
    highlight_class_node(dot)

    # Render the graph to PNG bytes (avoid temp files)
    def _sanitize_dot_source(source: str) -> str:
        # Escape brackets and quotes in node labels to avoid DOT parse errors.
        def repl(m):
            label = m.group(1)
            label = label.replace("\\", "\\\\").replace('"', '\\"')
            label = label.replace("[", "\\[").replace("]", "\\]")
            return f'label="{label}"'

        source = re.sub(r'label="([^"]*)"', repl, source)
        source = re.sub(r'label=([^\\s\\]]+)', r'label="\\1"', source)
        return source

    try:
        graph = Source(dot.source)
        png_bytes = graph.pipe(format="png")
    except Exception as e:
        print(f"Plotting failed with {type(e).__name__}; retrying with sanitized DOT source.")
        graph = Source(_sanitize_dot_source(dot.source))
        png_bytes = graph.pipe(format="png")

    # Open and display the rendered image
    img = Image.open(BytesIO(png_bytes))
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_axis_off()
    ax.set_title(plot_name)
    ax.imshow(img)

    # Save the plot to the specified directory with tight borders
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        os.path.join(save_dir, plot_name + ".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    if export_pdf:
        fig.savefig(
            os.path.join(save_dir, plot_name + ".pdf"),
            format="pdf",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    if not show:
        plt.close(fig)
    # No PDF output by default

    # Clean up temporary files
    # delete_folder_contents("temp")

def change_node_color(dot, node_id, fillcolor):
    r, g, b = int(fillcolor[1:3], 16), int(fillcolor[3:5], 16), int(fillcolor[5:7], 16)
    brightness = (r * 299 + g * 587 + b * 114) / 1000  # fórmula perceptual
    fontcolor = "white" if brightness < 100 else "black"

    # Modifica o nó no objeto Graphviz
    dot.node(node_id, style="filled", fillcolor=fillcolor, fontcolor=fontcolor)

def normalize_data(df, attribute, colormap):
    norm = Normalize(vmin=df[attribute].min(), vmax=df[attribute].max())
    colors = [colormap(norm(value)) for value in df[attribute]]
    return {node: "#{:02x}{:02x}{:02x}".format(int(color[0]*255), int(color[1]*255), int(color[2]*255)) for node, color in zip(df['Node'], colors)}

def plot_dpg_reg(plot_name, dot, df, df_dpg, save_dir="examples/", attribute=None, communities=False, leaf_flag=False):
    print("Rendering plot...")
    
    node_colors = {}
    if attribute or communities:
        if attribute:
            df = df[~df['Label'].str.contains('Pred')] if leaf_flag else df
            node_colors = normalize_data(df, attribute, plt.cm.Blues)
            plot_name += f"_{attribute.replace(' ', '')}"
        elif communities:
            df['Community'] = df['Label'].map({label: idx for idx, s in enumerate(df_dpg['Communities']) for label in s})
            node_colors = normalize_data(df, 'Community', plt.cm.YlOrRd)
            plot_name += "_communities"
    else:
        base_color = "#9ec3e6" if 'Pred' in df['Label'] else "#dee1f7"
        node_colors = {row['Node']: base_color for index, row in df.iterrows()}

    # Apply node colors
    for node, color in node_colors.items():
        change_node_color(dot, node, color)

    graph_path = os.path.join(save_dir, f"{plot_name}_temp.gv")
    dot.render(graph_path, view=False, format='png')

    # Display and save the image
    img_path = f"{graph_path}.png"
    img = Image.open(img_path)
    plt.figure(figsize=(16, 8))
    plt.axis('off')
    plt.title(plot_name)
    plt.imshow(img)

    if attribute:
        cax = plt.axes([0.11, 0.1, 0.8, 0.025])
        norm = Normalize(vmin=df[attribute].min(), vmax=df[attribute].max())
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=plt.cm.Blues), cax=cax, orientation='horizontal')
        cbar.set_label(attribute)

    plt.savefig(os.path.join(save_dir, f"{plot_name}_REG.png"), dpi=300)
    plt.close()  # Free up memory by closing the plot


    # Clean up temporary files
    delete_folder_contents("temp")


def plot_dpg_constraints_overview(
    normalized_constraints: Dict,
    feature_names: List[str],
    class_colors_list: List[str],
    output_path: str = None,
    title: str = "DPG Constraints Overview",
    original_sample: Dict = None,
    original_class: int = None,
    target_class: int = None
) -> plt.Figure:
    """Create a horizontal bar chart showing DPG constraints for all features.

    Similar to the "Feature Changes" chart style, this shows:
    - Original sample values as markers/bars
    - Constraint boundaries (min/max) for original and target classes as colored regions

    Args:
        normalized_constraints: Dict with structure {class_name: {feature: {min, max}}}
        feature_names: List of feature names to display
        class_colors_list: List of colors for each class
        output_path: Optional path to save the figure
        title: Title for the figure
        original_sample: Optional dict of original sample feature values
        original_class: Optional original class index (for highlighting)
        target_class: Optional target class index (for highlighting)

    Returns:
        matplotlib Figure object
    """
    if not normalized_constraints:
        print("WARNING: No constraints available for visualization")
        return None

    # Get list of classes
    class_names = sorted(normalized_constraints.keys())
    n_classes = len(class_names)

    # Filter features that have constraints in at least one class
    features_with_constraints = []
    for feat in feature_names:
        has_constraint = any(
            feat in normalized_constraints.get(cname, {})
            for cname in class_names
        )
        if has_constraint:
            features_with_constraints.append(feat)

    if not features_with_constraints:
        print("WARNING: No features with constraints found")
        return None

    n_features = len(features_with_constraints)

    # Identify non-overlapping features between classes
    # Non-overlapping means the ranges are disjoint (no intersection)
    # c1_max < c2_min means c1 range ends BEFORE c2 range starts (strictly less than)
    non_overlapping_features = set()
    for feat in features_with_constraints:
        for i, c1 in enumerate(class_names):
            for c2 in class_names[i+1:]:
                c1_bounds = normalized_constraints.get(c1, {}).get(feat, {})
                c2_bounds = normalized_constraints.get(c2, {}).get(feat, {})

                c1_min = c1_bounds.get('min')
                c1_max = c1_bounds.get('max')
                c2_min = c2_bounds.get('min')
                c2_max = c2_bounds.get('max')

                # Check for non-overlap (strictly less than, not equal)
                # Equal bounds means they touch/overlap, not non-overlapping
                if c1_max is not None and c2_min is not None and c1_max < c2_min:
                    non_overlapping_features.add(feat)
                if c2_max is not None and c1_min is not None and c2_max < c1_min:
                    non_overlapping_features.add(feat)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(6, n_features * 0.5)))

    # Y positions for features
    y_positions = np.arange(n_features)
    bar_height = 0.35

    # Collect global min/max for x-axis scaling
    all_values = []
    for cname in class_names:
        for feat in features_with_constraints:
            if feat in normalized_constraints.get(cname, {}):
                bounds = normalized_constraints[cname][feat]
                if bounds.get('min') is not None:
                    all_values.append(bounds['min'])
                if bounds.get('max') is not None:
                    all_values.append(bounds['max'])

    # Include original sample values in scaling if provided
    if original_sample:
        for feat in features_with_constraints:
            if feat in original_sample:
                all_values.append(original_sample[feat])

    if not all_values:
        print("WARNING: No constraint values found")
        return None

    # Filter out NaN and Inf values
    all_values = [v for v in all_values if v is not None and np.isfinite(v)]
    if not all_values:
        print("WARNING: No valid (finite) constraint values found")
        return None

    value_range = max(all_values) - min(all_values)
    # Handle case where all values are the same (range = 0)
    if value_range == 0 or not np.isfinite(value_range):
        value_range = abs(max(all_values)) * 0.2 if max(all_values) != 0 else 1.0

    x_min = min(all_values) - 0.1 * value_range
    x_max = max(all_values) + 0.1 * value_range

    # For each feature, draw constraint regions for each class
    for feat_idx, feat in enumerate(features_with_constraints):
        y = y_positions[feat_idx]

        # Highlight non-overlapping features
        is_discriminative = feat in non_overlapping_features
        if is_discriminative:
            ax.axhspan(y - 0.45, y + 0.45, alpha=0.1, color='gold', zorder=0)

        # Draw constraints for each class
        for class_idx, cname in enumerate(class_names):
            color = class_colors_list[class_idx % len(class_colors_list)]

            if feat in normalized_constraints.get(cname, {}):
                bounds = normalized_constraints[cname][feat]
                feat_min = bounds.get('min')
                feat_max = bounds.get('max')

                # Determine y offset for this class
                y_offset = (class_idx - (n_classes - 1) / 2) * bar_height * 0.8

                # Draw constraint region as horizontal bar
                if feat_min is not None and feat_max is not None:
                    # Both bounds - draw filled rectangle
                    rect = mpatches.Rectangle(
                        (feat_min, y + y_offset - bar_height/2),
                        feat_max - feat_min,
                        bar_height,
                        linewidth=2,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.3,
                        zorder=2
                    )
                    ax.add_patch(rect)

                    # Add min/max value labels
                    ax.text(feat_min, y + y_offset, f'{feat_min:.2f}',
                           ha='right', va='center', fontsize=7, color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor=color, alpha=0.8, linewidth=0.5))
                    ax.text(feat_max, y + y_offset, f'{feat_max:.2f}',
                           ha='left', va='center', fontsize=7, color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor=color, alpha=0.8, linewidth=0.5))

                elif feat_min is not None:
                    # Only min bound - draw line with arrow pointing right
                    ax.plot([feat_min, x_max], [y + y_offset, y + y_offset],
                           color=color, linewidth=3, alpha=0.5, linestyle='--', zorder=2)
                    ax.scatter([feat_min], [y + y_offset], color=color, s=100,
                              marker='|', zorder=3, linewidths=3)
                    ax.text(feat_min, y + y_offset + bar_height/2, f'min:{feat_min:.2f}',
                           ha='center', va='bottom', fontsize=7, color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor=color, alpha=0.8, linewidth=0.5))

                elif feat_max is not None:
                    # Only max bound - draw line with arrow pointing left
                    ax.plot([x_min, feat_max], [y + y_offset, y + y_offset],
                           color=color, linewidth=3, alpha=0.5, linestyle='--', zorder=2)
                    ax.scatter([feat_max], [y + y_offset], color=color, s=100,
                              marker='|', zorder=3, linewidths=3)
                    ax.text(feat_max, y + y_offset + bar_height/2, f'max:{feat_max:.2f}',
                           ha='center', va='bottom', fontsize=7, color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor=color, alpha=0.8, linewidth=0.5))

        # Draw original sample value if provided
        if original_sample and feat in original_sample:
            sample_val = original_sample[feat]
            # Draw as a prominent marker
            ax.scatter([sample_val], [y], color='black', s=150, marker='o',
                      zorder=10, edgecolors='white', linewidths=2)
            ax.plot([sample_val, sample_val], [y - 0.4, y + 0.4],
                   color='black', linewidth=2, linestyle='-', zorder=9, alpha=0.7)
            ax.text(sample_val, y + 0.42, f'{sample_val:.2f}',
                   ha='center', va='bottom', fontsize=8, color='black', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow',
                            edgecolor='black', alpha=0.9, linewidth=1))

    # Configure axes
    ax.set_yticks(y_positions)

    # Format y-tick labels with discriminative feature highlighting
    y_labels = []
    for feat in features_with_constraints:
        if feat in non_overlapping_features:
            y_labels.append(f'★ {feat}')
        else:
            y_labels.append(feat)
    ax.set_yticklabels(y_labels, fontsize=10)

    # Color discriminative feature labels
    for tick_label, feat in zip(ax.get_yticklabels(), features_with_constraints):
        if feat in non_overlapping_features:
            tick_label.set_color('darkgreen')
            tick_label.set_weight('bold')

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Feature Value', fontsize=12, loc='left')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, zorder=1)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    # Create legend
    legend_elements = []
    for class_idx, cname in enumerate(class_names):
        color = class_colors_list[class_idx % len(class_colors_list)]
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.3,
                          linewidth=2, label=f'{cname} Constraints')
        )

    if original_sample:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markeredgecolor='white', markersize=10, label='Original Sample')
        )

    if non_overlapping_features:
        legend_elements.append(
            mpatches.Patch(facecolor='gold', alpha=0.2,
                          label=f'★ Non-overlapping ({len(non_overlapping_features)} features)')
        )

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Title with class info
    title_text = title
    if original_class is not None and target_class is not None:
        title_text += f'\nOriginal Class: {original_class} → Target Class: {target_class}'
    ax.set_title(title_text, fontsize=14, weight='bold', pad=10)

    # Add statistics subtitle
    n_non_overlap = len(non_overlapping_features)
    subtitle = f"Features: {n_features} | Non-overlapping: {n_non_overlap} | Classes: {n_classes}"
    fig.text(0.5, 0.01, subtitle, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"INFO: Saved DPG constraints overview to {output_path}")

    return fig


def change_edge_color(graph, source_id, target_id, new_color, new_width):
    """
    Changes the color and dimension (penwidth) of a specified edge in the Graphviz Digraph.

    Args:
        graph: A Graphviz Digraph object.
        source_id: The source node of the edge.
        target_id: The target node of the edge.
        new_color: The new color to be applied to the edge.
        new_width: The new penwidth (edge thickness) to be applied.

    Returns:
        None
    """
    # Look for the existing edge in the graph body
    for i, line in enumerate(graph.body):
        if f'{source_id} -> {target_id}' in line:
            # Modify the existing edge attributes to include both color and penwidth
            new_line = line.rstrip().replace(']', f' color="{new_color}" penwidth="{new_width}"]')
            graph.body[i] = new_line
            break
