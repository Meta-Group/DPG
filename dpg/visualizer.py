import os
import copy
from collections import Counter
import re
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from graphviz import Source
from graphviz.backend.execute import ExecutableNotFound
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from PIL import Image
from .utils import highlight_class_node, change_node_color, delete_folder_contents

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

Image.MAX_IMAGE_PIXELS = 500000000  # Adjust based on your needs

_PREDICATE_PATTERN = re.compile(
    r"(.+?)\s*(<=|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


_LAYOUT_TEMPLATES = {
    # Current behavior, close to Graphviz defaults for DPG usage.
    "default": {
        "graph": {"rankdir": "LR"},
        "node": {},
        "edge": {},
    },
    # Good for reducing horizontal spread and making figures more compact.
    "compact": {
        "graph": {"rankdir": "TB", "nodesep": "0.2", "ranksep": "0.25"},
        "node": {"margin": "0.03,0.02"},
        "edge": {"arrowsize": "0.6"},
    },
    # Strong vertical layout for long/wide graphs.
    "vertical": {
        "graph": {"rankdir": "TB", "nodesep": "0.25", "ranksep": "0.35"},
        "node": {},
        "edge": {},
    },
    # Extra-spaced vertical layout for dense local-path overlays.
    "vertical_spacious": {
        "graph": {"rankdir": "TB", "nodesep": "1.85", "ranksep": "3.25"},
        "node": {"margin": "0.07,0.05"},
        "edge": {"arrowsize": "0.8"},
    },
    # Explicitly wide left-to-right style.
    "wide": {
        "graph": {"rankdir": "LR", "nodesep": "0.5", "ranksep": "0.6"},
        "node": {},
        "edge": {},
    },
}


def _apply_layout_template(dot, layout_template=None, graph_style=None, node_style=None, edge_style=None):
    """Apply optional graph layout/style settings to Graphviz Digraph."""
    raw_name = str(layout_template or "default")
    template_name = raw_name.strip().strip("'\"").lower()
    template = _LAYOUT_TEMPLATES.get(template_name)
    if template is None:
        warnings.warn(
            f"Unknown layout_template='{raw_name}'. Falling back to 'default'. "
            f"Available templates: {sorted(_LAYOUT_TEMPLATES.keys())}",
            UserWarning,
            stacklevel=2,
        )
        template = _LAYOUT_TEMPLATES["default"]
    merged_graph = dict(template.get("graph", {}))
    merged_node = dict(template.get("node", {}))
    merged_edge = dict(template.get("edge", {}))

    if graph_style:
        merged_graph.update(graph_style)
    if node_style:
        merged_node.update(node_style)
    if edge_style:
        merged_edge.update(edge_style)

    if merged_graph:
        dot.attr("graph", **{str(k): str(v) for k, v in merged_graph.items()})
    if merged_node:
        dot.attr("node", **{str(k): str(v) for k, v in merged_node.items()})
    if merged_edge:
        dot.attr("edge", **{str(k): str(v) for k, v in merged_edge.items()})


def _graphviz_not_found_error() -> RuntimeError:
    message = (
        "Graphviz executable 'dot' was not found in PATH.\n"
        "Install Graphviz and ensure 'dot' is available from your terminal.\n"
        "Install examples:\n"
        "- macOS (Homebrew): brew install graphviz\n"
        "- Ubuntu/Debian: sudo apt-get install graphviz\n"
        "- Windows (winget): winget install Graphviz.Graphviz"
    )
    return RuntimeError(message)


def _pipe_graph_png_with_fallback(dot_source: str, sanitizer) -> bytes:
    try:
        return Source(dot_source).pipe(format="png")
    except ExecutableNotFound as exc:
        raise _graphviz_not_found_error() from exc
    except Exception as first_exc:
        print(f"Plotting failed with {type(first_exc).__name__}; retrying with sanitized DOT source.")
        try:
            return Source(sanitizer(dot_source)).pipe(format="png")
        except ExecutableNotFound as exc:
            raise _graphviz_not_found_error() from exc
        except Exception:
            raise first_exc

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
    layout_template="default",
    graph_style=None,
    node_style=None,
    edge_style=None,
    fig_size=(16, 8),
    dpi=300,
    pdf_dpi=600,
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
    layout_template: Optional layout preset. One of {'default','compact','vertical','wide'}.
    graph_style: Optional dict of Graphviz graph attributes to override template values.
    node_style: Optional dict of Graphviz node attributes to override template values.
    edge_style: Optional dict of Graphviz edge attributes to override template values.
    fig_size: Matplotlib figure size (width, height).
    dpi: PNG export/display resolution.
    pdf_dpi: PDF export resolution when export_pdf=True.
    show: Whether to display the image via matplotlib. Default is True.
    export_pdf: If True, also writes a PDF next to the PNG.

    Returns:
    None
    """
    print("Plotting DPG...")
    _apply_layout_template(
        dot,
        layout_template=layout_template,
        graph_style=graph_style,
        node_style=node_style,
        edge_style=edge_style,
    )
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

    png_bytes = _pipe_graph_png_with_fallback(dot.source, _sanitize_dot_source)

    # Open and display the rendered image
    img = Image.open(BytesIO(png_bytes))
    fig, ax = plt.subplots(figsize=fig_size)
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
    fig.savefig(os.path.join(save_dir, plot_name + ".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    if export_pdf:
        fig.savefig(
            os.path.join(save_dir, plot_name + ".pdf"),
            format="pdf",
            dpi=pdf_dpi,
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
    layout_template="default",
    graph_style=None,
    node_style=None,
    edge_style=None,
    fig_size=(16, 8),
    dpi=300,
    pdf_dpi=600,
    show=True,
    export_pdf=False,
):
    """
    Plot a DPG colored by community assignment.

    Args:
    plot_name: Output base name for saved files (no extension).
    dot: Graphviz Digraph instance representing the DPG structure.
    df: DataFrame with node metrics; must include 'Node' and 'Label' columns.
    dpg_metrics: Dict containing either 'Communities' (list of sets/lists of node labels)
                 or 'Clusters' (mapping cluster_label -> list of node labels).
    save_dir: Directory where output images are saved. Default is "results/".
    class_flag: If True, class nodes are highlighted in yellow before other coloring.
    df_edges: Optional DataFrame with edge metrics to color edges by weight.
    layout_template: Optional layout preset. One of {'default','compact','vertical','wide'}.
    graph_style: Optional dict of Graphviz graph attributes to override template values.
    node_style: Optional dict of Graphviz node attributes to override template values.
    edge_style: Optional dict of Graphviz edge attributes to override template values.
    fig_size: Matplotlib figure size (width, height).
    dpi: PNG export/display resolution.
    pdf_dpi: PDF export resolution when export_pdf=True.
    show: Whether to display the image via matplotlib. Default is True.
    export_pdf: If True, also writes a PDF next to the PNG.

    Returns:
    None
    """
    print("Plotting DPG (communities)...")
    _apply_layout_template(
        dot,
        layout_template=layout_template,
        graph_style=graph_style,
        node_style=node_style,
        edge_style=edge_style,
    )

    if dpg_metrics is None:
        raise AttributeError("dpg_metrics is required to plot communities.")

    colormap = cm.YlOrRd  # Choose a colormap

    # Highlight class nodes if class_flag is True
    if class_flag:
        for index, row in df.iterrows():
            if 'Class' in row['Label']:
                change_node_color(dot, row['Node'], '#ffc000')  # Yellow for class nodes
        df = df[~df.Label.str.contains('Class')].reset_index(drop=True)  # Exclude class nodes from further processing

    # Map labels to community indices
    if "Communities" in dpg_metrics:
        communities = dpg_metrics.get("Communities", [])
    elif "Clusters" in dpg_metrics:
        clusters = dpg_metrics.get("Clusters", {})
        communities = list(clusters.values())
    else:
        raise AttributeError("dpg_metrics must include 'Communities' or 'Clusters' to plot communities.")

    label_to_community = {}
    for idx, community in enumerate(communities):
        for label in community:
            label_to_community[label] = idx
    df['Community'] = df['Label'].map(label_to_community)

    if df['Community'].isna().all():
        raise AttributeError("No nodes matched communities/clusters labels.")

    max_score = df['Community'].max()
    if max_score <= 0:
        norm = mcolors.Normalize(0, 1)
    else:
        norm = mcolors.Normalize(0, max_score)  # Normalize the community indices

    colors = colormap(norm(df['Community']))  # Assign colors based on normalized community indices

    for index, row in df.iterrows():
        if pd.isna(row['Community']):
            color = "#bdbdbd"
        else:
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

    png_bytes = _pipe_graph_png_with_fallback(dot.source, _sanitize_dot_source)

    # Open and display the rendered image
    img = Image.open(BytesIO(png_bytes))
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_axis_off()
    ax.set_title(plot_name)
    ax.imshow(img)

    # Save the plot to the specified directory with tight borders
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        os.path.join(save_dir, plot_name + ".png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    if export_pdf:
        fig.savefig(
            os.path.join(save_dir, plot_name + ".pdf"),
            format="pdf",
            dpi=pdf_dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    if not show:
        plt.close(fig)
    # No PDF output by default

    # Clean up temporary files
    # delete_folder_contents("temp")


def plot_dpg_local_path(
    plot_name,
    dot,
    df,
    df_edges,
    path_node_ids,
    save_dir="results/",
    path_labels=None,
    path_confidence=None,
    sample_id=None,
    true_class_label=None,
    class_flag=True,
    layout_template="default",
    graph_style=None,
    node_style=None,
    edge_style=None,
    fig_size=(16, 8),
    dpi=300,
    pdf_dpi=600,
    show=True,
    export_pdf=False,
):
    """
    Plot DPG with a single local path highlighted.

    Args:
    plot_name: Output base name for saved files (no extension).
    dot: Graphviz Digraph instance representing the DPG structure.
    df: DataFrame with node metrics and labels.
    df_edges: Edge metrics DataFrame with source/target ids and weights.
    path_node_ids: Ordered node id sequence representing the local path.
    save_dir: Directory where output files are written.
    path_labels: Optional ordered labels for title context.
    path_confidence: Optional confidence score shown in title.
    sample_id: Optional sample identifier shown in title.
    true_class_label: Optional true class label to highlight as green class node.
    class_flag: If True, keep class nodes visually distinct.
    """
    if path_node_ids is None or len(path_node_ids) == 0:
        raise ValueError("path_node_ids must contain at least one node id.")

    work_dot = copy.deepcopy(dot)
    _apply_layout_template(
        work_dot,
        layout_template=layout_template,
        graph_style=graph_style,
        node_style=node_style,
        edge_style=edge_style,
    )

    # Base node styling
    base_node_color = "#dce3ea"
    class_node_color = "#ffe08a"
    true_class_node_color = "#93c47d"
    for _, row in df.iterrows():
        label = str(row.get("Label", ""))
        if class_flag and label.startswith("Class "):
            class_name = label.replace("Class ", "", 1)
            if true_class_label is not None and class_name == str(true_class_label):
                change_node_color(work_dot, row["Node"], true_class_node_color)
            else:
                change_node_color(work_dot, row["Node"], class_node_color)
        else:
            change_node_color(work_dot, row["Node"], base_node_color)

    # Base edge styling from edge frequency
    if df_edges is not None and len(df_edges) > 0:
        colormap_edge = cm.Greys
        max_edge_value = df_edges["Weight"].max()
        min_edge_value = df_edges["Weight"].min()
        norm_edge = mcolors.Normalize(vmin=min_edge_value, vmax=max_edge_value)
        for _, row in df_edges.iterrows():
            edge_value = row["Weight"]
            color = colormap_edge(norm_edge(edge_value))
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255),
            )
            penwidth = 1 + 2 * norm_edge(edge_value)
            change_edge_color(work_dot, row["Source_id"], row["Target_id"], new_color=color_hex, new_width=penwidth)

    # Highlight path nodes with step-aware color ramp
    cmap = plt.get_cmap("Blues")
    n = max(1, len(path_node_ids) - 1)
    for i, node_id in enumerate(path_node_ids):
        shade = cmap(0.35 + 0.60 * (i / n))
        shade_hex = "#{:02x}{:02x}{:02x}".format(
            int(shade[0] * 255),
            int(shade[1] * 255),
            int(shade[2] * 255),
        )
        change_node_color(work_dot, node_id, shade_hex)

    # Highlight path edges strongly
    for src, dst in zip(path_node_ids, path_node_ids[1:]):
        change_edge_color(work_dot, src, dst, new_color="#0b5394", new_width=4.5)

    # Render graph
    def _sanitize_dot_source(source: str) -> str:
        def repl(m):
            label = m.group(1)
            label = label.replace("\\", "\\\\").replace('"', '\\"')
            label = label.replace("[", "\\[").replace("]", "\\]")
            return f'label="{label}"'

        source = re.sub(r'label="([^"]*)"', repl, source)
        source = re.sub(r'label=([^\\s\\]]+)', r'label="\\1"', source)
        return source

    png_bytes = _pipe_graph_png_with_fallback(work_dot.source, _sanitize_dot_source)
    img = Image.open(BytesIO(png_bytes))
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_axis_off()

    leaf_label = path_labels[-1] if path_labels else "n/a"
    sample_text = f"sample_id={sample_id} | " if sample_id is not None else ""
    if path_confidence is None:
        ax.set_title(f"{sample_text}{plot_name} | leaf={leaf_label}")
    else:
        ax.set_title(f"{sample_text}{plot_name} | leaf={leaf_label} | conf={path_confidence:.3f}")
    ax.imshow(img)

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, plot_name + ".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    if export_pdf:
        fig.savefig(
            os.path.join(save_dir, plot_name + ".pdf"),
            format="pdf",
            dpi=pdf_dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    if not show:
        plt.close(fig)
    return fig


def plot_dpg_local_paths_aggregate(
    plot_name,
    dot,
    df,
    df_edges,
    paths_node_ids,
    save_dir="results/",
    path_confidences=None,
    sample_id=None,
    true_class_label=None,
    obtained_class_label=None,
    sample_metrics=None,
    class_flag=True,
    layout_template="default",
    graph_width=None,
    graph_height=None,
    graph_size_lock=False,
    graph_style=None,
    node_style=None,
    edge_style=None,
    fig_size=(16, 8),
    dpi=300,
    pdf_dpi=600,
    show=True,
    export_pdf=False,
):
    """
    Plot DPG with multiple local paths aggregated in a single graph.

    Edges are highlighted by local path frequency (number of selected paths that use each edge).
    """
    if paths_node_ids is None or len(paths_node_ids) == 0:
        raise ValueError("paths_node_ids must contain at least one path.")

    normalized_paths = []
    for p in paths_node_ids:
        p_clean = [n for n in p if n is not None]
        if len(p_clean) >= 1:
            normalized_paths.append(p_clean)
    if len(normalized_paths) == 0:
        raise ValueError("No valid node paths to aggregate.")

    work_dot = copy.deepcopy(dot)
    local_graph_style = dict(graph_style or {})
    if graph_width is not None or graph_height is not None:
        # Be permissive: if only one side is provided, infer the other from fig_size.
        if graph_width is None:
            graph_width = fig_size[0]
        if graph_height is None:
            graph_height = fig_size[1]
        if float(graph_width) <= 0 or float(graph_height) <= 0:
            raise ValueError("graph_width and graph_height must be positive.")
        size_suffix = "!" if graph_size_lock else ""
        local_graph_style["size"] = f"{float(graph_width)},{float(graph_height)}{size_suffix}"
    _apply_layout_template(
        work_dot,
        layout_template=layout_template,
        graph_style=local_graph_style,
        node_style=node_style,
        edge_style=edge_style,
    )
    node_fontsize = None
    edge_fontsize = None
    if isinstance(node_style, dict) and "fontsize" in node_style:
        node_fontsize = node_style["fontsize"]
    if isinstance(edge_style, dict) and "fontsize" in edge_style:
        edge_fontsize = edge_style["fontsize"]
    try:
        node_font = float(node_fontsize) if node_fontsize is not None else 20.0
    except (TypeError, ValueError):
        node_font = 20.0
    try:
        edge_font = float(edge_fontsize) if edge_fontsize is not None else node_font
    except (TypeError, ValueError):
        edge_font = node_font
    node_scale = max(0.6, min(3.0, node_font / 20.0))
    edge_scale = max(0.6, min(3.0, edge_font / 18.0))

    # Aggregate local path frequencies
    edge_counter = Counter()
    node_counter = Counter()
    for p in normalized_paths:
        for node_id in p:
            node_counter[node_id] += 1
        for src, dst in zip(p, p[1:]):
            edge_counter[(src, dst)] += 1

    visited_nodes = set(node_counter.keys())
    max_edge_freq = max(edge_counter.values()) if edge_counter else 1
    node_cmap = plt.get_cmap("Blues")
    edge_cmap = plt.get_cmap("YlOrRd")

    # Node styling:
    # - unvisited predicates = white
    # - visited predicates = colored by Local Reaching Centrality
    # - class leaves = green (true class) / red (other classes)
    node_df = df.copy()
    node_df["is_class"] = node_df["Label"].astype(str).str.startswith("Class ")
    lrc_map = node_df.set_index("Node")["Local reaching centrality"].to_dict()
    visited_pred_lrc = [
        float(lrc_map[n])
        for n in visited_nodes
        if n in lrc_map and n in set(node_df[~node_df["is_class"]]["Node"])
    ]
    if visited_pred_lrc:
        lrc_norm = mcolors.Normalize(vmin=min(visited_pred_lrc), vmax=max(visited_pred_lrc))
    else:
        lrc_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for _, row in node_df.iterrows():
        node_id = row["Node"]
        label = str(row["Label"])
        if class_flag and label.startswith("Class "):
            class_name = label.replace("Class ", "", 1)
            if true_class_label is not None and class_name == str(true_class_label):
                change_node_color(
                    work_dot,
                    node_id,
                    "#93c47d",
                    new_fontsize=node_fontsize,
                    border_width=1.5 * node_scale,
                )  # true class leaf
            else:
                change_node_color(
                    work_dot,
                    node_id,
                    "#e57373",
                    new_fontsize=node_fontsize,
                    border_width=1.3 * node_scale,
                )  # non-true class leaves
            continue

        if node_id in visited_nodes:
            lrc_val = float(lrc_map.get(node_id, 0.0))
            shade = node_cmap(0.35 + 0.60 * lrc_norm(lrc_val))
            shade_hex = "#{:02x}{:02x}{:02x}".format(
                int(shade[0] * 255),
                int(shade[1] * 255),
                int(shade[2] * 255),
            )
            change_node_color(
                work_dot,
                node_id,
                shade_hex,
                new_fontsize=node_fontsize,
                border_width=1.15 * node_scale,
            )
        else:
            change_node_color(
                work_dot,
                node_id,
                "#ffffff",
                new_fontsize=node_fontsize,
                border_width=0.9 * node_scale,
            )

    # Base edge style: all edges light gray and thin.
    if df_edges is not None and len(df_edges) > 0:
        for _, row in df_edges.iterrows():
            change_edge_color(
                work_dot,
                row["Source_id"],
                row["Target_id"],
                new_color="#d0d0d0",
                new_width=0.9 * edge_scale,
                new_fontsize=edge_fontsize,
            )

    # Highlight visited edges by local path frequency (yellow -> red).
    for (src, dst), freq in edge_counter.items():
        frac = freq / max_edge_freq
        color = edge_cmap(0.20 + 0.75 * frac)
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
        )
        width = 1.8 + 4.2 * frac
        change_edge_color(
            work_dot,
            src,
            dst,
            new_color=color_hex,
            new_width=width * edge_scale,
            new_fontsize=edge_fontsize,
        )

    # Render graph
    def _sanitize_dot_source(source: str) -> str:
        def repl(m):
            label = m.group(1)
            label = label.replace("\\", "\\\\").replace('"', '\\"')
            label = label.replace("[", "\\[").replace("]", "\\]")
            return f'label="{label}"'

        source = re.sub(r'label="([^"]*)"', repl, source)
        source = re.sub(r'label=([^\\s\\]]+)', r'label="\\1"', source)
        return source

    png_bytes = _pipe_graph_png_with_fallback(work_dot.source, _sanitize_dot_source)
    img = Image.open(BytesIO(png_bytes))
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_axis_off()

    sample_text = f"sample_id={sample_id} | " if sample_id is not None else ""
    expected_text = f"expected={true_class_label}" if true_class_label is not None else "expected=n/a"
    obtained_text = (
        f"obtained={obtained_class_label}" if obtained_class_label is not None else "obtained=n/a"
    )
    metric_chunks = []
    if isinstance(sample_metrics, dict):
        evidence_score_pred = sample_metrics.get("evidence_score_pred", sample_metrics.get("vote_confidence"))
        if evidence_score_pred is not None:
            metric_chunks.append(f"evidence_score_pred={float(evidence_score_pred):.3f}")
        competitor = sample_metrics.get("top_competitor_class_pred")
        if competitor is not None:
            metric_chunks.append(f"top_competitor={competitor}")
        competitor_score = sample_metrics.get("evidence_score_competitor_pred")
        if competitor_score is not None:
            metric_chunks.append(f"evidence_score_competitor={float(competitor_score):.3f}")
        competitor_margin = sample_metrics.get("evidence_margin_pred_vs_competitor")
        if competitor_margin is not None:
            metric_chunks.append(f"pred_vs_competitor={float(competitor_margin):.3f}")
    metric_chunks.append(f"paths={len(normalized_paths)}")
    metric_chunks.append(f"max_edge_freq={max_edge_freq}")
    ax.set_title(
        f"{sample_text}{plot_name} | {expected_text} | {obtained_text} | " + " | ".join(metric_chunks)
    )
    ax.imshow(img)

    # Legend and colorbars
    handles = [
        mpatches.Patch(facecolor="#ffffff", edgecolor="black", label="Unvisited predicate"),
        mpatches.Patch(facecolor="#93c47d", edgecolor="black", label="True class leaf"),
        mpatches.Patch(facecolor="#e57373", edgecolor="black", label="Other class leaf"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        framealpha=0.92,
        fontsize=8,
        title="Legend",
        title_fontsize=9,
    )

    # Add colorbars only when meaningful
    fig.subplots_adjust(bottom=0.18, right=0.80)
    if visited_pred_lrc:
        sm_lrc = ScalarMappable(norm=lrc_norm, cmap=node_cmap)
        sm_lrc.set_array([])
        cax1 = fig.add_axes([0.18, 0.08, 0.30, 0.018])
        cbar1 = fig.colorbar(sm_lrc, cax=cax1, orientation="horizontal")
        cbar1.set_label("Visited predicate LRC", fontsize=8)
        cbar1.ax.tick_params(labelsize=7)
    if edge_counter:
        freq_norm = mcolors.Normalize(vmin=1, vmax=max_edge_freq)
        sm_edge = ScalarMappable(norm=freq_norm, cmap=edge_cmap)
        sm_edge.set_array([])
        cax2 = fig.add_axes([0.56, 0.08, 0.30, 0.018])
        cbar2 = fig.colorbar(sm_edge, cax=cax2, orientation="horizontal")
        cbar2.set_label("Edge path frequency", fontsize=8)
        cbar2.ax.tick_params(labelsize=7)

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, plot_name + ".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    if export_pdf:
        fig.savefig(
            os.path.join(save_dir, plot_name + ".pdf"),
            format="pdf",
            dpi=pdf_dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    if not show:
        plt.close(fig)
    return fig

def change_node_color(
    dot,
    node_id,
    fillcolor,
    new_fontsize=None,
    border_color="#263238",
    border_width=None,
):
    r, g, b = int(fillcolor[1:3], 16), int(fillcolor[3:5], 16), int(fillcolor[5:7], 16)
    brightness = (r * 299 + g * 587 + b * 114) / 1000  # fórmula perceptual
    fontcolor = "white" if brightness < 100 else "black"

    # Modifica o nó no objeto Graphviz
    attrs = {
        "style": "filled",
        "fillcolor": fillcolor,
        "fontcolor": fontcolor,
        "color": str(border_color),
    }
    if new_fontsize is not None:
        attrs["fontsize"] = str(new_fontsize)
    if border_width is not None:
        attrs["penwidth"] = str(border_width)
    dot.node(node_id, **attrs)

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
    try:
        dot.render(graph_path, view=False, format='png')
    except ExecutableNotFound as exc:
        raise _graphviz_not_found_error() from exc

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


def parse_predicate_parts(label: str) -> Optional[Tuple[str, str, float]]:
    """Parse predicate labels like 'feature <= 1.23' or 'feature > 0.7'."""
    match = _PREDICATE_PATTERN.search(str(label))
    if not match:
        return None
    return match.group(1).strip(), match.group(2), float(match.group(3))


def parse_feature_from_predicate(label: str) -> str:
    parsed = parse_predicate_parts(label)
    return parsed[0] if parsed else str(label)


def _feature_color_map(features: List[str]) -> Dict[str, Any]:
    unique = list(dict.fromkeys(features))
    if not unique:
        return {}
    cmap = plt.cm.tab20
    if len(unique) == 1:
        return {unique[0]: cmap(0)}
    return {feature: cmap(i / (len(unique) - 1)) for i, feature in enumerate(unique)}


def lrc_predicate_scores(explanation, top_k: int = 10) -> pd.DataFrame:
    """Return top-k predicate rows ranked by Local reaching centrality."""
    nm = explanation.node_metrics.copy()
    mask = (
        nm["Label"].astype(str).str.contains("<=", regex=False, na=False)
        | nm["Label"].astype(str).str.contains(">", regex=False, na=False)
    )
    nm = nm[mask].sort_values("Local reaching centrality", ascending=False).head(top_k)

    rows = []
    for _, row in nm.iterrows():
        parsed = parse_predicate_parts(row["Label"])
        if not parsed:
            continue
        feature, operator, threshold = parsed
        rows.append(
            {
                "predicate": str(row["Label"]),
                "feature": feature,
                "operator": operator,
                "threshold": threshold,
                "lrc": float(row["Local reaching centrality"]),
            }
        )
    return pd.DataFrame(rows)


def plot_lrc_vs_rf_importance(
    explanation,
    model,
    X_df: pd.DataFrame,
    top_k: int = 10,
    dataset_name: str = "Dataset",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare top LRC predicates and top RF feature importances side-by-side.

    Returns:
        Matplotlib figure.
    """
    top_lrc = lrc_predicate_scores(explanation, top_k=top_k).copy()
    if top_lrc.empty:
        raise ValueError("No predicate labels available to compute LRC scores.")

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model must expose feature_importances_.")

    top_rf = (
        pd.DataFrame(
            {
                "feature": list(getattr(model, "feature_names_in_", X_df.columns)),
                "rf_importance": np.asarray(model.feature_importances_, dtype=float),
            }
        )
        .sort_values("rf_importance", ascending=False)
        .head(top_k)
    )

    top_lrc_plot = top_lrc.sort_values("lrc", ascending=True)
    top_rf_plot = top_rf.sort_values("rf_importance", ascending=True)
    all_features = top_lrc_plot["feature"].tolist() + top_rf_plot["feature"].tolist()
    feature_to_color = _feature_color_map(all_features)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, top_k * 0.45)))

    axes[0].barh(
        top_lrc_plot["predicate"],
        top_lrc_plot["lrc"],
        color=[feature_to_color[f] for f in top_lrc_plot["feature"]],
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].set_title(f"{dataset_name}: Top {top_k} LRC predicates")
    axes[0].set_xlabel("Local Reaching Centrality")
    axes[0].set_ylabel("Predicate")

    axes[1].barh(
        top_rf_plot["feature"],
        top_rf_plot["rf_importance"],
        color=[feature_to_color[f] for f in top_rf_plot["feature"]],
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].set_title(f"{dataset_name}: Top {top_k} RF feature importances")
    axes[1].set_xlabel("Random Forest feature importance")
    axes[1].set_ylabel("Feature")

    legend_features = list(dict.fromkeys(all_features))
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=feature,
            markerfacecolor=feature_to_color[feature],
            markeredgecolor="black",
            markersize=8,
        )
        for feature in legend_features
    ]
    fig.legend(
        handles=legend_handles,
        title="Feature colors",
        loc="lower center",
        ncol=min(4, max(1, len(legend_handles))),
        frameon=True,
    )

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_lec_vs_rf_importance(*args, **kwargs) -> plt.Figure:
    """
    Backward-compatible alias for a common typo.

    Use `plot_lrc_vs_rf_importance` instead.
    """
    warnings.warn(
        "plot_lec_vs_rf_importance is deprecated; use plot_lrc_vs_rf_importance.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_lrc_vs_rf_importance(*args, **kwargs)


def plot_top_lrc_predicate_splits(
    explanation,
    X_df: pd.DataFrame,
    y,
    top_predicates: int = 5,
    top_features: int = 2,
    dataset_name: str = "Dataset",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Scatter the top-2 LRC features and overlay top-LRC predicate split lines.

    Returns:
        Matplotlib figure, or None when top features cannot be resolved.
    """
    top_lrc = lrc_predicate_scores(explanation, top_k=max(top_predicates, 10)).copy()
    top_pred = top_lrc.sort_values("lrc", ascending=False).head(top_predicates).copy()

    feature_rank = (
        top_lrc.groupby("feature", as_index=False)["lrc"]
        .sum()
        .sort_values("lrc", ascending=False)
        .head(top_features)
    )
    selected_features = feature_rank["feature"].tolist()
    if len(selected_features) < 2:
        return None

    fx, fy = selected_features[0], selected_features[1]
    if fx not in X_df.columns or fy not in X_df.columns:
        return None

    split_rows = top_pred[top_pred["feature"].isin([fx, fy])].copy()
    fig, ax = plt.subplots(figsize=(8, 6))

    y_series = pd.Series(np.asarray(y))
    y_numeric = pd.to_numeric(y_series, errors="coerce")
    colorbar_label = "Class id"
    colorbar_ticks = None
    colorbar_ticklabels = None
    if y_numeric.notna().all():
        color_values = y_numeric.to_numpy()
    else:
        # Support string/categorical class labels (e.g., "F1", "F2") in scatter coloring.
        color_values, unique_labels = pd.factorize(y_series.astype(str), sort=True)
        colorbar_label = "Class"
        colorbar_ticks = np.arange(len(unique_labels))
        colorbar_ticklabels = [str(label) for label in unique_labels]

    scatter = ax.scatter(
        X_df[fx],
        X_df[fy],
        c=color_values,
        cmap="viridis",
        s=36,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
    )

    feature_to_color = _feature_color_map([fx, fy])
    labels_seen = set()
    for _, row in split_rows.iterrows():
        feature = row["feature"]
        operator = row["operator"]
        threshold = row["threshold"]
        score = row["lrc"]
        linestyle = "--" if operator == "<=" else "-"
        label = f"{feature} {operator} {threshold:.2f} (LRC={score:.3f})"
        if feature == fx:
            ax.axvline(
                threshold,
                color=feature_to_color[feature],
                linestyle=linestyle,
                linewidth=2,
                alpha=0.9,
                label=label if label not in labels_seen else None,
            )
            labels_seen.add(label)
        elif feature == fy:
            ax.axhline(
                threshold,
                color=feature_to_color[feature],
                linestyle=linestyle,
                linewidth=2,
                alpha=0.9,
                label=label if label not in labels_seen else None,
            )
            labels_seen.add(label)

    ax.set_title(f"{dataset_name}: Top-{top_predicates} LRC predicate splits")
    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    if colorbar_ticks is not None and colorbar_ticklabels is not None:
        cbar.set_ticks(colorbar_ticks)
        cbar.set_ticklabels(colorbar_ticklabels)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Top LRC predicate lines", loc="best", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _resolve_graph_node(graph: nx.DiGraph, candidate):
    if candidate in graph:
        return candidate
    candidate_str = str(candidate)
    for node in graph.nodes:
        if str(node) == candidate_str:
            return node
    return None


def _normalize_class_label(label: Any) -> str:
    text = str(label)
    if text.startswith("Class "):
        return text.replace("Class ", "", 1)
    return text


def _community_specs(explanation, graph: nx.DiGraph, node_df: pd.DataFrame) -> List[Dict[str, Any]]:
    communities = getattr(explanation, "communities", None)
    if not communities:
        return []

    raw_specs = []
    if isinstance(communities, dict) and "Clusters" in communities:
        for key, members in communities.get("Clusters", {}).items():
            class_name = _normalize_class_label(key)
            if class_name.lower() == "ambiguous":
                class_name = None
            raw_specs.append({"class_name": class_name, "members": members})
    elif isinstance(communities, dict) and "Communities" in communities:
        for members in communities.get("Communities", []):
            raw_specs.append({"class_name": None, "members": members})

    label_to_nodes: Dict[str, List[Any]] = {}
    for _, row in node_df.iterrows():
        label_to_nodes.setdefault(str(row["Label"]), []).append(row["Node"])

    output = []
    for idx, spec in enumerate(raw_specs):
        resolved = set()
        for item in spec["members"]:
            node = _resolve_graph_node(graph, item)
            if node is not None:
                resolved.add(node)
                continue
            for candidate in label_to_nodes.get(str(item), []):
                node_candidate = _resolve_graph_node(graph, candidate)
                if node_candidate is not None:
                    resolved.add(node_candidate)
        if resolved:
            output.append(
                {
                    "community_id": idx,
                    "class_name": spec["class_name"],
                    "nodes": resolved,
                }
            )
    return output


def _class_nodes_map(explanation) -> Dict[Any, str]:
    node_df = explanation.node_metrics.copy()
    graph = getattr(explanation, "graph", None)
    if graph is None:
        raise ValueError("explanation.graph is required")

    class_df = node_df[node_df["Label"].astype(str).str.startswith("Class ")].copy()
    class_nodes = {}
    for _, row in class_df.iterrows():
        node = _resolve_graph_node(graph, row["Node"])
        if node is not None:
            class_nodes[node] = str(row["Label"]).replace("Class ", "", 1)
    return class_nodes


def _predicate_node_lookup(explanation) -> Dict[Any, Tuple[str, str, float]]:
    node_df = explanation.node_metrics.copy()
    graph = getattr(explanation, "graph", None)
    if graph is None:
        raise ValueError("explanation.graph is required")

    pred_df = node_df.copy()
    pred_df["parsed"] = pred_df["Label"].apply(parse_predicate_parts)
    pred_df = pred_df[pred_df["parsed"].notna()].copy()

    lookup: Dict[Any, Tuple[str, str, float]] = {}
    for _, row in pred_df.iterrows():
        node = _resolve_graph_node(graph, row["Node"])
        if node is None:
            continue
        feature, operator, threshold = row["parsed"]
        lookup[node] = (str(feature), str(operator), float(threshold))
    return lookup


def class_feature_predicate_counts(explanation) -> pd.DataFrame:
    """
    Compute class-vs-feature predicate frequency table from DPG communities.

    Returns:
        DataFrame indexed by class, columns as features, values as counts.
    """
    node_df = explanation.node_metrics.copy()
    graph = getattr(explanation, "graph", None)
    if graph is None:
        raise ValueError("explanation.graph is required for class-path analysis")
    if "Node" not in node_df.columns or "Label" not in node_df.columns:
        raise ValueError("node_metrics must contain Node and Label columns")

    class_nodes = _class_nodes_map(explanation)
    pred_lookup = _predicate_node_lookup(explanation)

    comm_specs = _community_specs(explanation, graph, node_df)
    if not comm_specs:
        comm_specs = [{"community_id": 0, "class_name": None, "nodes": set(pred_lookup.keys())}]

    class_feature_counts: Dict[str, List[str]] = {}
    for spec in comm_specs:
        class_from_cluster = spec["class_name"]
        for node in spec["nodes"]:
            if node not in pred_lookup:
                continue
            feature, _, _ = pred_lookup[node]
            if class_from_cluster is not None:
                target_classes = [str(class_from_cluster)]
            else:
                descendants = nx.descendants(graph, node)
                target_classes = [class_nodes[c] for c in class_nodes if c in descendants]
            for cls in target_classes:
                class_feature_counts.setdefault(cls, []).append(feature)

    if not class_feature_counts:
        return pd.DataFrame()

    series_map = {k: pd.Series(v).value_counts() for k, v in class_feature_counts.items() if v}
    if not series_map:
        return pd.DataFrame()

    heat = pd.DataFrame(series_map).T.fillna(0).astype(int)
    heat = heat.loc[:, heat.sum(axis=0).sort_values(ascending=False).index]
    return heat


def classwise_feature_bounds_from_communities(explanation) -> pd.DataFrame:
    """Build per-class, per-community finite/unbounded feature ranges from predicates."""
    node_df = explanation.node_metrics.copy()
    graph = getattr(explanation, "graph", None)
    if graph is None:
        raise ValueError("explanation.graph is required")

    class_nodes = _class_nodes_map(explanation)
    pred_lookup = _predicate_node_lookup(explanation)

    comm_specs = _community_specs(explanation, graph, node_df)
    if not comm_specs:
        comm_specs = [{"community_id": 0, "class_name": None, "nodes": set(pred_lookup.keys())}]

    bucket: Dict[Tuple[str, int, str], Dict[str, List[float]]] = {}
    for spec in comm_specs:
        community_id = int(spec["community_id"])
        class_from_cluster = spec["class_name"]
        for node in spec["nodes"]:
            if node not in pred_lookup:
                continue
            feature, operator, threshold = pred_lookup[node]
            if class_from_cluster is not None:
                target_classes = [str(class_from_cluster)]
            else:
                descendants = nx.descendants(graph, node)
                target_classes = [class_nodes[c] for c in class_nodes if c in descendants]
            if not target_classes:
                continue
            for cls in target_classes:
                key = (cls, community_id, feature)
                bucket.setdefault(key, {"gt": [], "le": [], "all": []})
                if operator == ">":
                    bucket[key]["gt"].append(threshold)
                elif operator == "<=":
                    bucket[key]["le"].append(threshold)
                bucket[key]["all"].append(threshold)

    rows = []
    for (cls, community_id, feature), values in bucket.items():
        lower = min(values["gt"]) if values["gt"] else float("-inf")
        upper = max(values["le"]) if values["le"] else float("inf")
        if lower > upper:
            lower = min(values["all"]) if values["all"] else float("-inf")
            upper = max(values["all"]) if values["all"] else float("inf")
        width = (upper - lower) if (np.isfinite(lower) and np.isfinite(upper)) else np.nan
        rows.append(
            {
                "class_name": cls,
                "community_id": community_id,
                "feature": feature,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "range_width": float(width) if pd.notna(width) else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "class_name",
                "community_id",
                "feature",
                "lower_bound",
                "upper_bound",
                "range_width",
            ]
        )
    return pd.DataFrame(rows)


def class_feature_predicate_positions(explanation) -> pd.DataFrame:
    """
    Collect raw predicate thresholds by class/feature/operator for density overlays.
    """
    node_df = explanation.node_metrics.copy()
    graph = getattr(explanation, "graph", None)
    if graph is None:
        raise ValueError("explanation.graph is required")

    class_nodes = _class_nodes_map(explanation)
    pred_lookup = _predicate_node_lookup(explanation)
    comm_specs = _community_specs(explanation, graph, node_df)
    if not comm_specs:
        comm_specs = [{"community_id": 0, "class_name": None, "nodes": set(pred_lookup.keys())}]

    rows = []
    for spec in comm_specs:
        community_id = int(spec["community_id"])
        class_from_cluster = spec["class_name"]
        for node in spec["nodes"]:
            if node not in pred_lookup:
                continue
            feature, operator, threshold = pred_lookup[node]
            if class_from_cluster is not None:
                target_classes = [str(class_from_cluster)]
            else:
                descendants = nx.descendants(graph, node)
                target_classes = [class_nodes[c] for c in class_nodes if c in descendants]
            for cls in target_classes:
                rows.append(
                    {
                        "class_name": cls,
                        "community_id": community_id,
                        "feature": feature,
                        "operator": operator,
                        "threshold": threshold,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["class_name", "community_id", "feature", "operator", "threshold"])
    return pd.DataFrame(rows)


def _aggregate_close_positions(values, tol: float):
    vals = np.sort(np.asarray(values, dtype=float))
    if vals.size == 0:
        return []
    groups = [[vals[0]]]
    for value in vals[1:]:
        if abs(value - groups[-1][-1]) <= tol:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [(float(np.mean(group)), len(group)) for group in groups]


def class_lookup_from_target_names(target_names: Optional[List[str]]) -> Dict[str, int]:
    if target_names is None:
        return {}
    return {str(name): i for i, name in enumerate(list(target_names))}


def _class_mask(class_name: str, y, class_lookup: Optional[Dict[str, int]] = None):
    y_arr = np.asarray(y)

    # First try direct label matching; this supports string labels (e.g., "F1")
    # even when a class_lookup mapping is provided.
    direct_mask = pd.Series(y_arr).astype(str).values == str(class_name)
    if np.any(direct_mask):
        return direct_mask

    # Fallback to lookup mapping (e.g., class name -> integer id).
    if class_lookup and str(class_name) in class_lookup:
        mapped_mask = y_arr == class_lookup[str(class_name)]
        if np.any(mapped_mask):
            return mapped_mask
    try:
        as_int = int(class_name)
        return y_arr == as_int
    except Exception:
        pass
    return direct_mask


def dataset_feature_bounds_by_class(
    X_df: pd.DataFrame,
    y,
    class_names: List[str],
    class_lookup: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    rows = []
    for cls in class_names:
        mask = _class_mask(cls, y, class_lookup=class_lookup)
        class_frame = X_df.loc[mask]
        if class_frame.empty:
            continue
        for feature in X_df.columns:
            rows.append(
                {
                    "class_name": str(cls),
                    "feature": str(feature),
                    "ds_lower_bound": float(class_frame[feature].min()),
                    "ds_upper_bound": float(class_frame[feature].max()),
                }
            )
    return pd.DataFrame(rows)


def plot_dpg_class_bounds_vs_dataset_feature_ranges(
    explanation,
    X_df: pd.DataFrame,
    y,
    dataset_name: str = "Dataset",
    top_features: int = 4,
    class_lookup: Optional[Dict[str, int]] = None,
    predicate_positions: Optional[pd.DataFrame] = None,
    class_bounds: Optional[pd.DataFrame] = None,
    class_filter: Optional[List[str]] = None,
    density_tol_ratio: float = 0.03,
    predicate_alpha: float = 0.55,
    dataset_range_lw: float = 10,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot DPG class bounds against empirical dataset ranges per feature.

    Args:
        explanation: DPGExplanation instance.
        X_df: Feature dataframe.
        y: Class labels aligned with X_df.
        class_lookup: Optional class-name to class-id mapping.
        predicate_positions: Optional precomputed output from class_feature_predicate_positions.
        class_bounds: Optional precomputed output from classwise_feature_bounds_from_communities.
        class_filter: Optional class or list of classes to render.

    Returns:
        Matplotlib figure, or None when no plottable classes exist.
    """
    if class_bounds is None:
        class_bounds = classwise_feature_bounds_from_communities(explanation)
    if class_bounds.empty:
        return None
    if predicate_positions is None:
        predicate_positions = class_feature_predicate_positions(explanation)

    dpg_bounds = (
        class_bounds.groupby(["class_name", "feature"], as_index=False)
        .agg(
            lower_bound=("lower_bound", "min"),
            upper_bound=("upper_bound", "max"),
            community_support=("community_id", "nunique"),
        )
    )
    dpg_bounds["range_width"] = np.where(
        np.isfinite(dpg_bounds["lower_bound"]) & np.isfinite(dpg_bounds["upper_bound"]),
        dpg_bounds["upper_bound"] - dpg_bounds["lower_bound"],
        np.nan,
    )
    classes = sorted(dpg_bounds["class_name"].unique())
    if class_filter is not None:
        if isinstance(class_filter, (list, tuple, set, np.ndarray, pd.Series)):
            allowed = {str(value) for value in class_filter}
        else:
            allowed = {str(class_filter)}
        classes = [cls for cls in classes if str(cls) in allowed]
        if not classes:
            return None

    ds_bounds = dataset_feature_bounds_by_class(X_df, y, classes, class_lookup=class_lookup)
    if ds_bounds.empty:
        return None

    fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 5), squeeze=False)
    axes = axes[0]
    density_gt_labeled = False
    density_le_labeled = False

    for ax, cls in zip(axes, classes):
        class_dpg = dpg_bounds[dpg_bounds["class_name"] == cls].copy()
        class_dpg = class_dpg.sort_values(["community_support", "range_width"], ascending=[False, False]).head(top_features)
        class_dpg = class_dpg.sort_values("range_width", ascending=True)
        merged = class_dpg.merge(
            ds_bounds[ds_bounds["class_name"] == cls],
            on=["class_name", "feature"],
            how="left",
        )
        if merged.empty:
            continue

        y_positions = np.arange(len(merged))
        ds_min = float(merged["ds_lower_bound"].min())
        ds_max = float(merged["ds_upper_bound"].max())
        feature_global_min = float(X_df[merged["feature"]].min().min())
        feature_global_max = float(X_df[merged["feature"]].max().max())

        dpg_lo_axis = merged["lower_bound"].astype(float).to_numpy(copy=True)
        dpg_hi_axis = merged["upper_bound"].astype(float).to_numpy(copy=True)
        finite_dpg_lo = dpg_lo_axis[np.isfinite(dpg_lo_axis)]
        finite_dpg_hi = dpg_hi_axis[np.isfinite(dpg_hi_axis)]
        dpg_min = float(finite_dpg_lo.min()) if finite_dpg_lo.size else ds_min
        dpg_max = float(finite_dpg_hi.max()) if finite_dpg_hi.size else ds_max

        x_min = max(0.0, min(ds_min, feature_global_min, dpg_min))
        x_max = max(ds_max, feature_global_max, dpg_max)
        pad = max((x_max - x_min) * 0.2, 1e-6)
        left_lim = max(0.0, x_min - pad)
        right_lim = x_max + pad

        ax.hlines(
            y_positions,
            merged["ds_lower_bound"],
            merged["ds_upper_bound"],
            color="lightgray",
            linewidth=dataset_range_lw,
            alpha=0.85,
            label="dataset class range" if cls == classes[0] else None,
        )
        ax.scatter(
            merged["ds_lower_bound"],
            y_positions,
            color="dimgray",
            s=28,
            label="dataset min/max" if cls == classes[0] else None,
        )
        ax.scatter(merged["ds_upper_bound"], y_positions, color="dimgray", s=28)

        dpg_lo = merged["lower_bound"].astype(float).to_numpy(copy=True)
        dpg_hi = merged["upper_bound"].astype(float).to_numpy(copy=True)
        lo_inf = ~np.isfinite(dpg_lo)
        hi_inf = ~np.isfinite(dpg_hi)
        draw_lo = np.where(lo_inf, left_lim, dpg_lo)
        draw_hi = np.where(hi_inf, right_lim, dpg_hi)

        ax.hlines(
            y_positions,
            draw_lo,
            draw_hi,
            color="tab:blue",
            linewidth=3,
            alpha=0.95,
            label="DPG community range" if cls == classes[0] else None,
        )

        finite_lo = np.isfinite(dpg_lo)
        finite_hi = np.isfinite(dpg_hi)
        ax.scatter(
            dpg_lo[finite_lo],
            y_positions[finite_lo],
            color="tab:green",
            s=38,
            label="DPG min bound" if cls == classes[0] else None,
        )
        ax.scatter(
            dpg_hi[finite_hi],
            y_positions[finite_hi],
            color="tab:red",
            s=38,
            label="DPG max bound" if cls == classes[0] else None,
        )

        if lo_inf.any():
            ax.scatter(
                np.full(lo_inf.sum(), left_lim),
                y_positions[lo_inf],
                marker="<",
                color="tab:green",
                s=70,
                label="DPG min = -inf" if cls == classes[0] else None,
            )
        if hi_inf.any():
            ax.scatter(
                np.full(hi_inf.sum(), right_lim),
                y_positions[hi_inf],
                marker=">",
                color="tab:red",
                s=70,
                label="DPG max = +inf" if cls == classes[0] else None,
            )

        if predicate_positions is not None and not predicate_positions.empty:
            class_pred = predicate_positions[predicate_positions["class_name"] == cls]
            tol = max((right_lim - left_lim) * float(density_tol_ratio), 1e-9)
            for y_index, feat in enumerate(merged["feature"]):
                pred_feature = class_pred[class_pred["feature"] == feat]
                vals_gt = pred_feature.loc[pred_feature["operator"] == ">", "threshold"].astype(float).to_numpy()
                vals_le = pred_feature.loc[pred_feature["operator"] == "<=", "threshold"].astype(float).to_numpy()
                vals_gt = vals_gt[(vals_gt >= left_lim) & (vals_gt <= right_lim)]
                vals_le = vals_le[(vals_le >= left_lim) & (vals_le <= right_lim)]

                dense_gt = _aggregate_close_positions(vals_gt, tol) if vals_gt.size else []
                dense_le = _aggregate_close_positions(vals_le, tol) if vals_le.size else []

                if dense_gt:
                    xs = np.array([d[0] for d in dense_gt], dtype=float)
                    counts = np.array([d[1] for d in dense_gt], dtype=float)
                    sizes = 14 + 16 * np.sqrt(counts)
                    ax.scatter(
                        xs,
                        np.full_like(xs, y_index, dtype=float) + 0.18,
                        s=sizes,
                        marker="^",
                        c="tab:green",
                        alpha=predicate_alpha,
                        edgecolors="black",
                        linewidths=0.35,
                        label="predicate density (>)" if not density_gt_labeled else None,
                        zorder=4,
                    )
                    density_gt_labeled = True

                if dense_le:
                    xs = np.array([d[0] for d in dense_le], dtype=float)
                    counts = np.array([d[1] for d in dense_le], dtype=float)
                    sizes = 14 + 16 * np.sqrt(counts)
                    ax.scatter(
                        xs,
                        np.full_like(xs, y_index, dtype=float) - 0.18,
                        s=sizes,
                        marker="v",
                        c="tab:red",
                        alpha=predicate_alpha,
                        edgecolors="black",
                        linewidths=0.35,
                        label="predicate density (<=)" if not density_le_labeled else None,
                        zorder=4,
                    )
                    density_le_labeled = True

        ax.set_xlim(left_lim, right_lim)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(merged["feature"])
        ax.set_xlabel("Feature value range")
        ax.set_title(f"{dataset_name} - Class {cls}: DPG vs dataset range")
        ax.grid(axis="x", linestyle="--", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)

    plt.tight_layout(rect=(0, 0.10, 1, 1))
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def change_edge_color(graph, source_id, target_id, new_color, new_width, new_fontsize=None):
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
            new_line = line.rstrip()
            # Replace existing color/penwidth if present to avoid duplicate attrs.
            new_line = re.sub(r'\scolor="[^"]*"', "", new_line)
            new_line = re.sub(r'\spenwidth="[^"]*"', "", new_line)
            if new_fontsize is not None:
                if re.search(r'\sfontsize="[^"]*"', new_line):
                    new_line = re.sub(r'\sfontsize="[^"]*"', f' fontsize="{new_fontsize}"', new_line)
                else:
                    new_line = new_line.replace("]", f' fontsize="{new_fontsize}"]')
            # Add updated color + penwidth.
            new_line = new_line.replace("]", f' color="{new_color}" penwidth="{new_width}"]')
            graph.body[i] = new_line
            break
