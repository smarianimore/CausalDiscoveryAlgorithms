import traceback

import networkx as nx
import numpy as np
import pandas as pd
from castle import algorithms

from run_single import run_algorithm
from run_single import run_grandag
from src.utils import (
    plot_causal_graph,
    process_data,
    save_graph_and_metrics,
    load_digraph_from_json,
    get_my_adjacency_matrix,
)


def run_algorithms(
    data: pd.DataFrame, dir_save: str, ground_truth_graph: np.ndarray = None
):
    # Loop through all available algorithms in castle.algorithms
    for algo_name in dir(algorithms):
        if algo_name not in ["ANMNonlinear"]:
            if algo_name == "GAE":
                data = data.to_numpy()
            print("\n********")
            print(algo_name)

            try:
                if algo_name == "GraNDAG":
                    causal_matrix_est, metrics = run_grandag(
                        data, algo_name, ground_truth_graph, input_dim=len(data.columns)
                    )
                else:
                    causal_matrix_est, metrics = run_algorithm(
                        data, algo_name, ground_truth_graph
                    )

                if algo_name == "GAE":
                    data = pd.DataFrame(data)
                graph, fig_graph = plot_causal_graph(causal_matrix_est, data)

                save_graph_and_metrics(graph, fig_graph, metrics, dir_save, algo_name)
            except Exception as e:
                with open("error_log.txt", "a") as f:
                    print(f"{algo_name} failed with error: {e}\nStack trace logged on file.")
                    datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{datetime} : {algo_name} failed with error : {e}\n\tStack trace:\n")
                    f.write(f"\t{traceback.print_exc()}")


if __name__ == "__main__":
    data_name = "../Datasets-causality/TuWien-guys/FGCS/backup_entire_data_Laptop"
    df_start = pd.read_csv(f"{data_name}.csv")

    df = process_data(df_start, ["execution_time", "timestamp", "stream_count"])

    gt_graph = load_digraph_from_json(f"ground_truth/{data_name.split('/')[-1]}.json")
    gt_array = get_my_adjacency_matrix(gt_graph) if gt_graph is not None else None

    run_algorithms(df, data_name, gt_array)
