import json
import logging
import os

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')


def plot_causal_graph(
    causal_matrix: np.array,
    df: pd.DataFrame,
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


def label_encode_categoricals(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
        Label encodes categorical columns in the given DataFrame and saves the encodings to a JSON file.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be encoded.
            name (str): The name to be used for the encoding file.

        Returns:
            pd.DataFrame: The DataFrame with the categorical columns label encoded.
        """
    df = df.convert_dtypes()
    logging.info(f"{df.dtypes=}")
    # objects = df.select_dtypes(include=["object"]).columns
    objects = df.select_dtypes(include=["string[python]"]).columns
    logging.info(f"{objects=}")
    le = LabelEncoder()
    df[objects] = df[objects].apply(le.fit_transform)
    logging.info(f"Label-encoded data types:\n{df.dtypes}")
    encodings = {}
    for col in objects:
        labels = [int(label) for label in df[col].unique()]
        originals = le.inverse_transform(labels)
        encoding = dict(zip(labels, originals))
        logging.info(f"encoding of {col}: {encoding}")
        encodings[col] = encoding
    current_dir = f"results/{name}"
    os.makedirs(current_dir, exist_ok=True)
    with open(f"{current_dir}/{name}_encodings.json", "w") as f:
        json.dump(encodings, f,
                  indent=4)  # from https://docs.python.org/3.11/library/json.html#json.dump "Note Keys in key/value pairs of JSON are always of the type str. When a dictionary is converted into JSON, all the keys of the dictionary are coerced to strings."
    bools = df.select_dtypes(include=["bool"]).columns
    for col in bools:
        df[col] = df[col].astype(int)
    print(f"########## Final data types:\n{df.dtypes}")
    return df


def process_data(data: pd.DataFrame, ignore_cols: list[str], outpath: str) -> pd.DataFrame:
    """
        Processes the input data by dropping specified columns, filling missing values, and label encoding categorical columns.

        Args:
            data (pd.DataFrame): The input DataFrame containing the data to be processed.
            ignore_cols (list[str]): A list of column names to be ignored (dropped) from the DataFrame.
            outpath (str): The output path where the processed data and encodings will be saved.

        Returns:
            pd.DataFrame: The processed DataFrame with specified columns dropped, missing values filled, and categorical columns label encoded.
        """
    out_file_name = f"{outpath.split('/')[-1].split('.')[0]}"
    data = data.drop(columns=ignore_cols)
    #data = data.convert_dtypes()
    print(f"########## Raw data types:\n{data.dtypes}")
    data = data.fillna(0)  # TODO questionable
    df = label_encode_categoricals(data, out_file_name)
    return df


def save_graph_and_metrics(
    graph: nx.DiGraph, fig_graph: plt.Figure, metrics: dict, outpath: str, algo_name: str
):
    """
        Saves the causal graph, its figure, and the calculated metrics to the specified directory.

        Args:
            graph (nx.DiGraph): The directed graph to be saved.
            fig_graph (plt.Figure): The matplotlib figure of the graph to be saved.
            metrics (dict): The metrics calculated for the graph.
            outpath (str): The output path where the results will be saved.
            algo_name (str): The name of the algorithm used to generate the graph.

        Returns:
            None
        """
    print(metrics)
    out_file_name = f"{outpath.split('/')[-1].split('.')[0]}"
    current_dir = f"results/{out_file_name}"
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


def save_digraph_as_json(digraph: nx.DiGraph, filename: str):
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


def load_digraph_from_json(filename: str) -> nx.DiGraph:
    """
    Load a networkx.DiGraph from a JSON file.

    Parameters:
        filename (str): The path to the JSON file with the graph data.

    Returns:
        nx.DiGraph: The loaded directed graph.
    """
    digraph = None
    if filename:
        with open(filename, "r") as f:
            graph_dict = json.load(f)
        digraph = nx.node_link_graph(
            graph_dict, directed=True, edges="links"
        )  # Specify 'links' or 'edges' as needed TODO ???
    return digraph


def get_my_adjacency_matrix(digraph: nx.DiGraph) -> np.ndarray:
    """
        Generates an adjacency matrix from a given directed graph.

        Args:
            digraph (nx.DiGraph): The directed graph from which to generate the adjacency matrix.

        Returns:
            np.ndarray: A 2D numpy array representing the adjacency matrix of the graph.
        """
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
