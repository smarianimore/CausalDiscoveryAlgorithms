import os
import traceback
from datetime import datetime

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
    """
        Runs various causal discovery algorithms on the provided data and saves the results.

        Args:
            data (pd.DataFrame): The input DataFrame containing the data to be processed.
            dir_save (str): The directory path where the results will be saved.
            ground_truth_graph (np.ndarray, optional): The ground truth adjacency matrix of the causal graph. Defaults to None.

        Returns:
            None
        """
    # Loop through all available algorithms in castle.algorithms
    for algo_name in dir(algorithms):
        if algo_name not in ["ANMNonlinear"]:
            print("\n********")
            print(algo_name)

            try:
                if algo_name == "GraNDAG":
                    causal_matrix_est, metrics = run_grandag(
                        data, algo_name, ground_truth_graph, input_dim=len(data.columns)
                    )
                elif algo_name == "GAE":
                    data_np = data.to_numpy()
                    causal_matrix_est, metrics = run_algorithm(
                        data_np, algo_name, ground_truth_graph
                    )
                else:
                    causal_matrix_est, metrics = run_algorithm(
                        data, algo_name, ground_truth_graph
                    )

                graph, fig_graph = plot_causal_graph(causal_matrix_est, data)
                save_graph_and_metrics(graph, fig_graph, metrics, dir_save, algo_name)

            except Exception as e:
                out_file_name = f"{dir_save.split('/')[-1].split('.')[0]}"
                current_dir = f"results/{out_file_name}"
                os.makedirs(current_dir, exist_ok=True)
                with open(f"{current_dir}/error_log.txt", "a") as f:
                    print(f"{algo_name} failed with error: {e}\nStack trace logged on file.")
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{timestamp} : {algo_name} failed with error : {e}\n\tStack trace:\n")
                    traceback.print_exc(file=f)
                    traceback.print_stack(file=f)


if __name__ == "__main__":
    input_data_file_path = "../Datasets-causality/DigitaTwins-Fischer/multi-process-station-data/vacuum-gripper-dt_logs.csv"
    gt_graph_filepath = "../Datasets-causality/DigitaTwins-Fischer/multi-process-station-graphs/Vacuum-Gripper_Graph-reduced.json"
    data = pd.read_csv(input_data_file_path, usecols=['vacuum-gripper-motor-clockwise', 'vacuum-gripper-at-turntable',
                                                      'vacuum-gripper-motor-counterclockwise', 'OEE', 'vacuum-gripper-at-oven',
                                                      'machine-state', 'piece-count', 'vacuum-gripper-piston', 'vacuum-gripper-gripper'])
    #print(f"PRE {data.columns=}")
    df = process_data(data, [], input_data_file_path)
    #print(f"POST {data.columns=}")

    gt_graph = load_digraph_from_json(gt_graph_filepath)
    gt_array = get_my_adjacency_matrix(gt_graph) if gt_graph is not None else None
    run_algorithms(df, input_data_file_path, gt_array)

# gCastle metrics (from https://github.com/huawei-noah/trustworthyAI/blob/master/gcastle/castle/metrics/evaluation.py)
#     fdr: (reverse + FP) / (TP + FP)
#     tpr: TP/(TP + FN)
#     fpr: (reverse + FP) / (TN + FP)
#     shd: undirected extra + undirected missing + reverse
#     nnz: TP + FP
#     precision: TP/(TP + FP)
#     recall: TP/(TP + FN)
#     F1: 2*(recall*precision)/(recall+precision)
#     gscore: max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
# where
#     true positive(TP): an edge estimated with correct direction.
#     true nagative(TN): an edge that is neither in estimated graph nor in true graph.
#     false positive(FP): an edge that is in estimated graph but not in the true graph.
#     false negative(FN): an edge that is not in estimated graph but in the true graph.
#     reverse = an edge estimated with reversed direction.