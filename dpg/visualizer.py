import os
import re
from graphviz import Source
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from .utils import highlight_class_node, change_node_color, delete_folder_contents

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from PIL import Image
from graphviz import Source

Image.MAX_IMAGE_PIXELS = 500000000  # Adjust based on your needs

def plot_dpg(plot_name, dot, df, df_edges, df_dpg, save_dir="examples/", attribute=None, communities=False, clusters=None, threshold_clusters=None, class_flag=False):
    """
    Plots a Decision Predicate Graph (DPG) with various customization options.

    Args:
    plot_name: The name of the plot.
    dot: A Graphviz Digraph object representing the DPG.
    df: A pandas DataFrame containing node metrics.
    df_dpg: A pandas DataFrame containing DPG metrics.
    save_dir: Directory to save the plot image. Default is "examples/".
    attribute: A specific node attribute to visualize. Default is None.
    communities: Boolean indicating whether to visualize communities. Default is False.
    class_flag: Boolean indicating whether to highlight class nodes. Default is False.

    Returns:
    None
    """
    print("Plotting DPG...")
    # Basic color scheme if no attribute or communities are specified
    if attribute is None and not communities and clusters is None:
        for index, row in df.iterrows():
            if 'Class' in row['Label']:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(157, 195, 230))  # Light blue for class nodes
            else:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(222, 235, 247))  # Light grey for other nodes


    # Color nodes based on a specific attribute
    elif attribute is not None and not communities and clusters is None:
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
    

    # Color nodes based on community detection
    elif communities and attribute is None and clusters is None:
        colormap = cm.YlOrRd  # Choose a colormap
        
        # Highlight class nodes if class_flag is True
        if class_flag:
            for index, row in df.iterrows():
                if 'Class' in row['Label']:
                    change_node_color(dot, row['Node'], '#ffc000')  # Yellow for class nodes
            df = df[~df.Label.str.contains('Class')].reset_index(drop=True)  # Exclude class nodes from further processing

        # Map labels to community indices
        label_to_community = {label: idx for idx, s in enumerate(df_dpg['Communities']) for label in s}
        df['Community'] = df['Label'].map(label_to_community)
        
        max_score = df['Community'].max()
        norm = mcolors.Normalize(0, max_score)  # Normalize the community indices
        
        colors = colormap(norm(df['Community']))  # Assign colors based on normalized community indices

        for index, row in df.iterrows():
            color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255), int(colors[index][1]*255), int(colors[index][2]*255))
            change_node_color(dot, row['Node'], color)

        plot_name = plot_name + "_communities"
    

    elif attribute is None and not communities and clusters is not None:
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
        raise AttributeError("The plot can show the basic plot, communities or a specific node-metric")


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

    # Render the graph and display it
    dot.render("temp/" + plot_name, format="pdf")
    graph = Source(dot.source, format="png")
    graph.render("temp/" + plot_name + "_temp", view=False)

    # Open and display the rendered image
    img = Image.open("temp/" + plot_name + "_temp.png")
    plt.figure(figsize=(16, 8))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.title(plot_name)
    plt.imshow(img)
    
    # Add a color bar if an attribute is specified
    if attribute is not None:
        cax = plt.axes([0.11, 0.1, 0.8, 0.025])  # Define color bar position
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax, orientation='horizontal')
        cbar.set_label(attribute)

    # Save the plot to the specified directory
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, plot_name + ".png"), dpi=300)
    #plt.show()
    graph_pdf_path = os.path.join("temp", plot_name + "_graph.pdf")
    plt.savefig(graph_pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    
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