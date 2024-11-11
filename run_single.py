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


def run_algorithm(
    data: pd.DataFrame, algo_name: str = "PC", causal_matrix_gt: np.ndarray = None
) -> Tuple[Tensor, Any]:
    if hasattr(algorithms, algo_name):
        algo_class = getattr(algorithms, algo_name)
        if isinstance(algo_class, type) and hasattr(algo_class, "learn"):
            try:
                # Instantiate and run the algorithm
                if hasattr(algo_class, "device_type"):
                    cd = algo_class(device_type="gpu")
                else:
                    cd = algo_class()

                cd.learn(data)

                causal_matrix_est = cd.causal_matrix

                if causal_matrix_gt is not None:
                    # calculate metrics
                    mt = MetricsDAG(causal_matrix_est, causal_matrix_gt)
                    metrics = mt.metrics
                else:
                    metrics = None

                return causal_matrix_est, metrics

            except Exception as e:
                print(f"{algo_name} failed with error: {e}")
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
