import os
import json
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_causal_graph(
    causal_matrix,
    df,
    FONT_SIZE_NODE_GRAPH: int = 10,
    ARROWS_SIZE_NODE_GRAPH: int = 30,
    NODE_SIZE_GRAPH: int = 1000,
) -> nx.DiGraph:
    """
    Plots a causal graph based on the causal matrix with node names from the DataFrame columns.

    Parameters:
        causal_matrix (np.array): A 2D numpy array representing causal relationships.
        df (pd.DataFrame): DataFrame whose column names are used as node labels.
        :param NODE_SIZE_GRAPH:
        :param ARROWS_SIZE_NODE_GRAPH:
        :param FONT_SIZE_NODE_GRAPH:

    """
    # Ensure the causal matrix is square and matches the number of features in the DataFrame
    assert (
        causal_matrix.shape[0] == causal_matrix.shape[1] == len(df.columns)
    ), "Causal matrix dimensions must match the number of features in the DataFrame."

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with labels based on DataFrame columns
    feature_names = df.columns
    G.add_nodes_from(feature_names)

    # Add edges based on the causal matrix
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            if causal_matrix[i, j] == 1:
                G.add_edge(feature_names[i], feature_names[j])

    # Draw the graph
    fig = plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # Layout for a visually appealing graph
    nx.draw(
        G,
        with_labels=True,
        font_size=FONT_SIZE_NODE_GRAPH,
        arrowsize=ARROWS_SIZE_NODE_GRAPH,
        arrows=True,
        edge_color="orange",
        node_size=NODE_SIZE_GRAPH,
        font_weight="bold",
        node_color="skyblue",
        pos=pos,
        # pos=nx.circular_layout(graph),
    )
    plt.title("Causal Graph")
    #plt.show()

    return G, fig


def process_data(data_start: pd.DataFrame, ignore_cols: list[str]) -> pd.DataFrame:
    #columns_to_neglect = ["timestamp", "device_type", "config", "GPU"]
    columns_to_neglect = ignore_cols
    data = data_start.drop(columns=columns_to_neglect)
    #data["success"] = data["success"].astype(int)
    data = data.convert_dtypes()
    print(f"\t\t{data.dtypes=}")
    data = data.fillna(0)
    strings = data.select_dtypes(include=["string"]).columns
    data = pd.get_dummies(data, columns=strings)

    return data


def save_graph_and_metrics(
    graph: nx.DiGraph, fig_graph: plt.Figure, metrics, dir_save: str, algo_name: str
):
    print(metrics)
    current_dir = f"results/{dir_save}"
    os.makedirs(current_dir, exist_ok=True)

    # graph
    save_digraph_as_json(graph, f"{current_dir}/{algo_name}_causal_graph.json")

    # figure
    fig_graph.savefig(f"{current_dir}/{algo_name}_causal_graph.png")

    # metrics
    if metrics is not None:
        # Convert numpy types to native Python types
        metrics_new = {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in metrics.items()
        }

        # Save to a JSON file
        with open(f"{current_dir}/{algo_name}_metrics.json", "w") as f:
            json.dump(metrics_new, f, indent=4)


def save_digraph_as_json(digraph, filename):
    """
    Save a networkx.DiGraph as a JSON file.

    Parameters:
        digraph (nx.DiGraph): The directed graph to save.
        filename (str): The path to the JSON file to save the graph.
    """
    # Convert the graph to a dictionary format suitable for JSON
    graph_dict = nx.node_link_data(
        digraph, edges="links"
    )  # Explicitly set `edges="links"`

    # Write the dictionary to a JSON file
    with open(filename, "w") as f:
        json.dump(graph_dict, f, indent=4)


def load_digraph_from_json(filename):
    """
    Load a networkx.DiGraph from a JSON file.

    Parameters:
        filename (str): The path to the JSON file with the graph data.

    Returns:
        nx.DiGraph: The loaded directed graph.
    """
    digraph = None
    # Read the JSON file
    if filename:
        with open(filename, "r") as f:
            graph_dict = json.load(f)

        # Convert the dictionary to a networkx.DiGraph
        digraph = nx.node_link_graph(
            graph_dict, directed=True, edges="links"
        )  # Specify 'links' or 'edges' as needed

    return digraph


def get_my_adjacency_matrix(digraph):
    # Get the sorted list of nodes
    nodes = sorted(digraph.nodes())
    # Create an empty adjacency matrix
    adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

    # Map each node to an index in the matrix
    node_index = {node: i for i, node in enumerate(nodes)}

    # Fill the adjacency matrix
    for edge in digraph.edges():
        source, target = edge
        adj_matrix[node_index[source]][node_index[target]] = 1

    return adj_matrix
