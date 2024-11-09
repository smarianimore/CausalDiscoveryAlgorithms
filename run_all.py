import pandas as pd
from castle import algorithms

from run_single import run_algorithm
from src.utils import plot_causal_graph, process_data


def run_algorithms(data: pd.DataFrame, ground_truth_graph=None):
    # Loop through all available algorithms in castle.algorithms
    for algo_name in dir(algorithms):
        if algo_name != "ANMNonlinear":
            print("\n********")
            print(algo_name)
            causal_matrix = run_algorithm(data, algo_name, ground_truth_graph)

            graph = plot_causal_graph(causal_matrix, data)

            # TODO: save graph in some files and in png, compute metrics and store them


if __name__ == "__main__":
    df_start = pd.read_csv("./data/xavier_gpu_6_20.csv")

    df = process_data(df_start)

    run_algorithms(df)
