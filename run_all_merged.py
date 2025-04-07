import argparse
import json
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from run_all import run_algorithms
from src.utils import (
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

    df_merged.to_csv(f"results/{name}/{name}.csv", index=False)

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