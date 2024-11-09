import pandas as pd
from castle import algorithms
from castle.common import Tensor

from src.utils import process_data, plot_causal_graph


def run_algorithm(data, algo_name="PC", ground_truth_graph=None) -> Tensor:
    if hasattr(algorithms, algo_name):
        algo_class = getattr(algorithms, algo_name)
        if isinstance(algo_class, type) and hasattr(algo_class, "learn"):
            try:
                # Instantiate and run the algorithm
                cd = algo_class()
                cd.learn(data)

                # if hasattr(cd, "causal_matrix"):
                # print(f"{algo_name} causal matrix:\n", type(cd.causal_matrix))

                return cd.causal_matrix

            except Exception as e:
                print(f"{algo_name} failed with error: {e}")
        else:
            print(f"{algo_name} does not have a learn method or is not a valid class.")
    else:
        print(f"{algo_name} is not a recognized algorithm in the castle library.")


if __name__ == "__main__":
    df_start = pd.read_csv("./data/xavier_gpu_6_20.csv")

    df = process_data(df_start)

    # Specify the algorithm name if you want to run only one
    specific_algorithm = (
        "PC"  # Example: replace with the name of the algorithm you want to run
    )
    causal_matrix = run_algorithm(df, specific_algorithm)

    plot_causal_graph(causal_matrix, df)
