import argparse
import json
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from castle import algorithms
from sklearn.preprocessing import LabelEncoder

from run_single import run_algorithm
from run_single import run_grandag
from src.utils import (
    plot_causal_graph,
    save_graph_and_metrics,
    load_digraph_from_json,
    get_my_adjacency_matrix,
)


def process_time(df: pd.DataFrame, column: str, time_unit: str = 'ms'):
    """
        Converts the specified time column to datetime format and floors the time to the nearest second.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column (str): The name of the column containing the time values.
            time_unit (str): The unit of the time values (default is 'ms').

        Returns:
            None
        """
    df[column] = pd.to_datetime(df[column], unit=time_unit)
    df[column] = df[column].dt.floor('s')


def aggregate_time(df: pd.DataFrame, column: str):
    """
        Aggregates the DataFrame by the specified time column, computing the mean for float64 columns and the mode for other columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column (str): The name of the column to group by.

        Returns:
            None
        """
    df = df.groupby(column).agg(lambda x: x.mean() if x.dtype == 'float64' else x.mode().iloc[0])
    df.reset_index(inplace=True)


def label_encode_categoricals(df: pd.DataFrame, name: str, out_path: str) -> pd.DataFrame:
    """
        Label encodes categorical columns in the given DataFrame and saves the encodings to a JSON file.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be encoded.
            name (str): The name to be used for the encoding file.
            out_path (str): The output directory where the encoding file will be saved.

        Returns:
            pd.DataFrame: The DataFrame with the categorical columns label encoded.
        """
    #df = df.convert_dtypes()
    print(f"raw data types:\n{df.dtypes}")
    objects = df.select_dtypes(include=["object"]).columns
    #objects = df.select_dtypes(include=["string[python]"]).columns
    #print(f"\tneed encoding: {objects}")
    le_map = {}
    for col in objects:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_map[col] = le
    #LOG.info(f"\tlabel-encoded data types:\n{df.dtypes}")
    encodings = {}
    for col in objects:
        labels = [int(label) for label in df[col].unique()]
        originals = le_map[col].inverse_transform(labels)
        encoding = dict(zip(labels, originals))
        print(f"\tencoding of {col}: {encoding}")
        encodings[col] = encoding
    current_dir = f"results/{name}"
    os.makedirs(current_dir, exist_ok=True)
    with open(f"{current_dir}/{name}_encodings.json", "w") as f:
        json.dump(encodings, f,
                  indent=4)  # from https://docs.python.org/3.11/library/json.html#json.dump "Note Keys in key/value pairs of JSON are always of the type str. When a dictionary is converted into JSON, all the keys of the dictionary are coerced to strings."
    bools = df.select_dtypes(include=["bool"]).columns
    for col in bools:
        df[col] = df[col].astype(int)
    print(f"processed data types:\n{df.dtypes}")
    return df


def read_csv(filepath: str, columns_in: list, columns_out: list, out_path: str, name: str) -> pd.DataFrame:
    """
        Reads a CSV file and processes it according to the specified columns to include or exclude, while also label
        encoding categorical variables (see function label_encode_categoricals()).

        Args:
            filepath (str): Path to the CSV file to read.
            columns_in (list): List of columns to include in the DataFrame.
            columns_out (list): List of columns to exclude from the DataFrame.
            out_path (str): Output directory where the encoding file will be saved.
            name (str): The name to be used for the encoding file.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
    df = pd.read_csv(filepath, usecols=columns_in if columns_in else None, index_col=False)
    if columns_out:
        df.drop(columns=columns_out, inplace=True, axis=1)
    df.dropna(inplace=True)
    df = label_encode_categoricals(df, name, out_path)
    return df


def discretize(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    """
        Discretizes the columns of the DataFrame based on the provided arguments.

        Args:
            args (argparse.Namespace): The arguments containing the binning information.
            df (pd.DataFrame): The DataFrame to be discretized.

        Returns:
            pd.DataFrame: The DataFrame with discretized columns.
        """
    bins = args.bin_dict if args.bin_dict else args.qbin_dict if args.qbin_dict else None
    bins = json.loads(bins) if type(bins) == str else bins
    if bins:
        for column, n_bins in bins.items():
            if column in df.columns:
                df[column] = pd.cut(df[column], bins=n_bins, labels=False) if args.bin_dict else pd.qcut(df[column],
                                                                                                         q=n_bins,
                                                                                                         labels=False)
    return df


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
    print(f"{os.getcwd()=}")

    mps_datapath = "../Datasets-causality/DigitaTwins-Fischer/multi-process-station-data/mps_dt_logs.csv"
    line_datapath = "../Datasets-causality/DigitaTwins-Fischer/indexed-line-data/indexed-line-dt_logs.csv"
    args_parse = argparse.ArgumentParser()
    args_parse.add_argument('-o', '--output', type=str, required=False)
    args_parse.add_argument('-b', '--bin_dict', type=str, required=False)
    args = args_parse.parse_args(['--output=merge-mps_indexed_line',
                                  '--bin_dict={"output-conveyor-oee":10, "oven-oee":10, "turntable-oee":10, "vacuum-gripper-oee":10, "conveyor-out-oee":10, "conveyor-in-oee":10, "drill-oee":10, "cutter-oee":10, "OEE_mps":10, "OEE_line":10}'])
    print(f"{args=}")
    name = "mps+indexed_line"

    df_mps = pd.read_csv(mps_datapath)
    process_time(df_mps, 'Timestamp')
    aggregate_time(df_mps, 'Timestamp')

    df_line = pd.read_csv(line_datapath)
    process_time(df_line, 'Timestamp')
    aggregate_time(df_line, 'Timestamp')

    # merge the two dataframes based on the index TODO true merge should be something like "merge rows if timestamps are 'close enough'"
    df_merged = pd.merge(df_mps, df_line, how='inner', left_index=True, right_index=True, suffixes=('_mps', '_line'))
    df_merged.drop(columns=[col for col in df_merged.columns if 'Timestamp' in col or 'Source' in col], inplace=True,
                   axis=1)
    df_merged = label_encode_categoricals(df_merged, name, f"results/{args.output}")
    df_merged = discretize(args, df_merged)

    gt_graph = load_digraph_from_json(None)
    gt_array = get_my_adjacency_matrix(gt_graph) if gt_graph is not None else None
    run_algorithms(df_merged, name, gt_array)

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