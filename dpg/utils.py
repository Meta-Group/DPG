import os
import re
import shutil
import yaml
from graphviz import Digraph
import networkx as nx
import pandas as pd
import numpy as np


def highlight_class_node(dot, dpg_config=None):
    """
    Highlights nodes in the Graphviz Digraph that contain "Class" in their identifiers by changing their fill color
    and adding a rounded shape.

    Args:
    dot: A Graphviz Digraph object.
    dpg_config: Optional DPG config dict (from DecisionPredicateGraph)

    Returns:
    dot: The modified Graphviz Digraph object with the class nodes highlighted.
    """

    if not isinstance(dot, Digraph):
        raise ValueError("Input must be a Graphviz Digraph object")
    
    # Get class node styling from config or use defaults
    if dpg_config is not None:
        class_style = dpg_config.get('dpg', {}).get('visualization', {}).get('class_node', {})
    else:
        # Fallback to loading from config file
        config_path="config.yaml"
        try:
            with open(config_path) as f:
                    config = yaml.safe_load(f)
            class_style = config.get('dpg', {}).get('visualization', {}).get('class_node', {})
        except FileNotFoundError:
            class_style = {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file: {str(e)}")
    
    # Get values with defaults
    fillcolor = class_style.get('fillcolor', '#a4c2f4')  # Default light blue
    shape = class_style.get('shape', 'box')
    style = class_style.get('style', 'rounded, filled')
    
    # Iterate over each line in the dot body
    for i, line in enumerate(dot.body):
        # Check for class labels in the node attributes
        if 'label="Class' in line:
            new_attrs = f'fillcolor="{fillcolor}" shape={shape} style="{style}"'
            if '[' in line:
                pre, rest = line.split('[', 1)
                attrs = rest.rsplit(']', 1)[0]
                # Remove existing attributes we're replacing (quoted or unquoted)
                attrs = re.sub(r'\b(fillcolor|shape|style)=(".*?"|[^ \]]+)', '', attrs)
                attrs = re.sub(r'\s+', ' ', attrs).strip()
                if attrs:
                    attrs = attrs + ' '
                dot.body[i] = f"{pre}[{attrs}{new_attrs}]"
            else:
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


