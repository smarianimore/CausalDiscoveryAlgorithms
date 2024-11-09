import networkx as nx
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
    plt.figure(figsize=(8, 6))
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
    plt.show()

    return G


def process_data(data_start: pd.DataFrame) -> pd.DataFrame:
    columns_to_neglect = ["timestamp", "device_type", "config", "GPU"]
    data = data_start.drop(columns=columns_to_neglect)
    data["success"] = data["success"].astype(int)
    data = data.fillna(0)

    return data
