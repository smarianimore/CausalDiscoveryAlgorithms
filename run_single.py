from typing import Any, Tuple

import numpy as np
import pandas as pd
from castle import algorithms, MetricsDAG
from castle.common import Tensor

from src.utils import (
    process_data,
    plot_causal_graph,
    save_graph_and_metrics,
    load_digraph_from_json,
    get_my_adjacency_matrix,
)


def run_grandag(
    data: pd.DataFrame, algo_name: str = "PC", causal_matrix_gt: np.ndarray = None, input_dim: int = 0
) -> Tuple[Tensor, Any]:
    """
        Runs the GraNDAG algorithm on the provided data and returns the estimated causal matrix and metrics.

        Args:
            data (pd.DataFrame): The input DataFrame containing the data to be processed.
            algo_name (str): The name of the algorithm to be used. Defaults to "PC".
            causal_matrix_gt (np.ndarray, optional): The ground truth adjacency matrix of the causal graph. Defaults to None.
            input_dim (int, optional): The input dimension for the algorithm. Defaults to 0.

        Returns:
            Tuple[Tensor, Any]: A tuple containing the estimated causal matrix and the calculated metrics (if ground truth is provided).
        """
    if hasattr(algorithms, algo_name):
        algo_class = getattr(algorithms, algo_name)
        if isinstance(algo_class, type) and hasattr(algo_class, "learn"):
            #try:
                # Instantiate and run the algorithm
                if hasattr(algo_class, "device_type"):
                    cd = algo_class(input_dim, device_type="gpu")
                else:
                    cd = algo_class(input_dim)

                cd.learn(data)

                causal_matrix_est = cd.causal_matrix

                if causal_matrix_gt is not None:
                    # calculate metrics
                    print(f"DIMS: {causal_matrix_est.shape=} vs. {causal_matrix_gt.shape=}")
                    mt = MetricsDAG(causal_matrix_est, causal_matrix_gt)
                    metrics = mt.metrics
                else:
                    metrics = None

                return causal_matrix_est, metrics

            #except Exception as e:
                #print(f"{algo_name} failed with error: {e}")
        else:
            print(f"{algo_name} does not have a learn method or is not a valid class.")
    else:
        print(f"{algo_name} is not a recognized algorithm in the castle library.")

def run_algorithm(
    data: pd.DataFrame, algo_name: str = "PC", causal_matrix_gt: np.ndarray = None
) -> Tuple[Tensor, Any]:
    """
        Runs a specified causal discovery algorithm on the provided data and returns the estimated causal matrix and metrics.

        Args:
            data (pd.DataFrame): The input DataFrame containing the data to be processed.
            algo_name (str): The name of the algorithm to be used. Defaults to "PC".
            causal_matrix_gt (np.ndarray, optional): The ground truth adjacency matrix of the causal graph. Defaults to None.

        Returns:
            Tuple[Tensor, Any]: A tuple containing the estimated causal matrix and the calculated metrics (if ground truth is provided).
        """
    if hasattr(algorithms, algo_name):
        algo_class = getattr(algorithms, algo_name)
        if isinstance(algo_class, type) and hasattr(algo_class, "learn"):
            #try:
                # Instantiate and run the algorithm
                if hasattr(algo_class, "device_type"):
                    cd = algo_class(device_type="gpu")
                else:
                    cd = algo_class()

                cd.learn(data)

                causal_matrix_est = cd.causal_matrix

                if causal_matrix_gt is not None:
                    # calculate metrics
                    print(f"DIMS: {causal_matrix_est.shape=} vs. {causal_matrix_gt.shape=}")
                    mt = MetricsDAG(causal_matrix_est, causal_matrix_gt)
                    metrics = mt.metrics
                else:
                    metrics = None

                return causal_matrix_est, metrics

            #except Exception as e:
                #print(f"{algo_name} failed with error: {e}")
        else:
            print(f"{algo_name} does not have a learn method or is not a valid class.")
    else:
        print(f"{algo_name} is not a recognized algorithm in the castle library.")


if __name__ == "__main__":
    data_name = "xavier_gpu_6_20"
    algo_name = "PC"

    df_start = pd.read_csv(f"./data/{data_name}.csv")

    df = process_data(df_start)

    gt_graph = load_digraph_from_json(f"ground_truth/{data_name}.json")
    gt_array = get_my_adjacency_matrix(gt_graph) if gt_graph is not None else None

    causal_matrix_est, metrics = run_algorithm(df, algo_name, gt_array)

    graph, fig_graph = plot_causal_graph(causal_matrix_est, df)

    save_graph_and_metrics(graph, fig_graph, metrics, data_name, algo_name)
