import os
import shutil
import yaml
from graphviz import Digraph
import networkx as nx
import pandas as pd
import numpy as np


def highlight_class_node(dot):
    """
    Highlights nodes in the Graphviz Digraph that contain "Class" in their identifiers by changing their fill color
    and adding a rounded shape.

    Args:
    dot: A Graphviz Digraph object.

    Returns:
    dot: The modified Graphviz Digraph object with the class nodes highlighted.
    """

    if not isinstance(dot, Digraph):
        raise ValueError("Input must be a Graphviz Digraph object")
    
    config_path="config.yaml"
    try:
        with open(config_path) as f:
                config = yaml.safe_load(f)
        # Get class node styling from config (with defaults if not specified)
        class_style = config.get('dpg', {}).get('visualization', {}).get('class_node', {})
        fillcolor = class_style.get('fillcolor', '#a4c2f4')  # Default light blue
        shape = class_style.get('shape', 'box')
        style = class_style.get('style', 'rounded, filled')

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file: {str(e)}")
    
    # Iterate over each line in the dot body
    for i, line in enumerate(dot.body):
        # Extract the node identifier from the line
        line_id = line.split(' ')[1].replace("\t", "")
        # Check if the node identifier contains "Class"
        if "Class" in line_id:
            new_attrs = f'fillcolor="{fillcolor}" shape={shape} style="{style}"'
            # If node already has attributes, modify them
            if '[' in line:
                parts = line.split('[')
                attrs = parts[1].rstrip(']')
                # Remove existing attributes we're replacing
                for attr in ['fillcolor', 'shape', 'style']:
                    attrs = ' '.join([a for a in attrs.split() if not a.startswith(attr)])
                # Add new attributes
                dot.body[i] = f"{parts[0]}[{attrs} {new_attrs}]"
            else:
                # Node has no attributes yet
                node_id = line.split(' ')[0]
                dot.body[i] = f'{node_id} [{new_attrs}]'
    
    # Return the modified Graphviz Digraph object
    return dot



def change_node_color(graph, node_id, new_color):
    """
    Changes the fill color of a specified node in the Graphviz Digraph.

    Args:
    graph: A Graphviz Digraph object.
    node_id: The identifier of the node whose color is to be changed.
    new_color: The new color to be applied to the node.

    Returns:
    None
    """
    if not any(node_id in line for line in graph.body):
        raise ValueError(f"Node {node_id} not found in graph")
    
    # Remove existing color attribute if present
    for i, line in enumerate(graph.body):
        if node_id in line and 'fillcolor=' in line:
            parts = line.split('fillcolor=')
            graph.body[i] = parts[0] + parts[1].split(']')[0][-1] + ']'
    

    # Append a new line to the graph body to change the fill color of the specified node
    graph.body.append(f'{node_id} [fillcolor="{new_color}"]')



def delete_folder_contents(folder_path):
    """
    Deletes all contents of the specified folder.

    Args:
    folder_path: The path to the folder whose contents are to be deleted.

    Returns:
    None
    """

    if not os.path.isdir(folder_path):
        raise ValueError(f"Path {folder_path} is not a valid directory")
    
    # Iterate over each item in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)  # Get the full path of the item
        try:
            # Check if the item is a file or a symbolic link
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove the file or link
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove the directory and its contents
        except Exception as e:
            # Print an error message if the deletion fails
            print(f'Failed to delete {item_path}. Reason: {e}')



def get_dpg_edge_metrics(dpg_model, nodes_list):
    """
    Extracts metrics from the edges of a DPG model, including:
    - Edge Load Centrality
    - Trophic Differences
    
    Args:
    dpg_model: A NetworkX graph representing the DPG.
    nodes_list: A list of nodes where each node is a tuple. The first element is the node identifier and the second is the node label.

    Returns:
    df: A pandas DataFrame containing the metrics for each edge in the DPG.
    """
    

    # Calculate edge weights (assuming edges have 'weight' attribute)
    edge_weights = nx.get_edge_attributes(dpg_model, 'weight')
    
    # Aggiungi le etichette dei nodi
    edge_data_with_labels = []
    for u, v in dpg_model.edges():
        # Ottieni le etichette per i nodi coinvolti nell'arco
        u_label = next((label for node, label in nodes_list if node == u), None)
        v_label = next((label for node, label in nodes_list if node == v), None)
        
        # Ottieni gli identificativi (ID) per i nodi coinvolti nell'arco
        u_id = next((node for node, label in nodes_list if node == u), None)
        v_id = next((node for node, label in nodes_list if node == v), None)
        
        # Aggiungi i dati per l'arco con le etichette e gli ID
        edge_data_with_labels.append([f"{u}-{v}",  
                                     edge_weights.get((u, v), 0),
                                     u_label, v_label, u_id, v_id])
    
    # Crea un DataFrame con gli archi, le etichette e gli ID
    df_edges_with_labels = pd.DataFrame(edge_data_with_labels, columns=["Edge", "Weight", 
                                                                        "Node_u_label", "Node_v_label", "Source_id", "Target_id"])
    

    # Restituisci il DataFrame risultante
    return df_edges_with_labels


def clustering(dpg_model, class_nodes, threshold = None):
    
    classes = sorted(set(class_nodes.values()))
    class_by_node = dict(class_nodes)
    class_set = set(class_by_node.keys())

    nodes = list(dpg_model.nodes())
    n = len(nodes)
    
    idx = {idx_node : node for node, idx_node in enumerate(nodes)}
    
    # P
    P = np.zeros((n, n), dtype = float)
    for node in nodes:
        i = idx[node]
        if node in class_set:
            P[i, i] = 1.0
            continue

        out_edges = list(dpg_model.out_edges(node, data=True))
        
        weight_sum = 0

        for out_node, in_node, weight in out_edges:
            weight_sum += weight.get('weight', 1)

        if weight_sum > 0:
            for out_node, in_node, weight in out_edges:
                j = idx[in_node]
                P[i, j] = weight.get('weight', 1) / weight_sum
        else:
            P[i, i] = 1.0
    
    # Order to obtain Q and R
    transient = []
    absorbing = []
    for node in nodes:
        if node not in class_set:
            transient.append(node)
        elif node in class_set:
            absorbing.append(node)

    t = len(transient)

    perm = transient + absorbing
    
    perm_idx = [idx[node] for node in perm]
    
    Pp = P[perm_idx][:, perm_idx]

    Q = Pp[:t, :t]
    R = Pp[:t, t:]

    # N
    I = np.eye(t)
    N = np.linalg.solve(I - Q, I)

    # Absorbing probability for each node
    B = N @ R

    # ----- #
    class_labels = [class_by_node[node] for node in absorbing]

    class_to_cols = {}
    for class_index in range(len(absorbing)):
        label = class_labels[class_index]
        if label not in class_to_cols:
            class_to_cols[label] = []
        class_to_cols[label].append(class_index)
    
    # Distribution for transient nodes
    node_probs = {}

    for index_row in range(len(transient)):
        node = transient[index_row]

        probs = {}
        for label in classes:
            probs[label] = 0.0
        
        # sum columns for class
        for label in classes:
            cols = class_to_cols.get(label, [])
            total = 0.0
            for index_col in cols:
                total += B[index_row, index_col]
            probs[label] = total

        node_probs[node] = probs
    
    # Distribution for absorbing nodes
    for node in absorbing:
        probs = {}
        for label in classes:
            probs[label] = 0.0
        probs[class_nodes[node]] = 1.0
        
        node_probs[node] = probs

    # Clusters
    clusters = {}
    for label in classes:
        clusters[label] = []
    
    if threshold is not None:
        clusters['Ambiguous'] = []

    confidence = {}

    for node in nodes:
        probs = node_probs[node]

        top_label = None
        top_prob = -1.0
        second_top_prob = -1.0

        # Top probability and cluster identification
        for label in classes:
            prob = probs[label]
            if prob > top_prob:
                top_prob = prob
                top_label = label

        # Second top probability
        for label in classes:
            prob = probs[label]
            if label != top_label and prob > second_top_prob:
                second_top_prob = prob

        margin = top_prob - (second_top_prob if second_top_prob >= 0.0 else 0.0)

        confidence[node] = margin

        
        if threshold is None:
            clusters[top_label].append(node)

        else:
            if top_prob > threshold:       
                clusters[top_label].append(node)     
            else:
                clusters['Ambiguous'].append(node)


    return clusters, node_probs, confidence