import argparse
import os

import pandas as pd

from run_all import run_algorithms
from run_all_merged import label_encode_categoricals, discretize
from src.utils import load_digraph_from_json, get_my_adjacency_matrix


def clean_timestamp(df: pd.DataFrame, column: str='Timestamp', unit: str='ms'):
    """
        convert given column expressed in given unit to datetime

        Args:
            df (pd.DataFrame): The input DataFrame.
            column (str): The name of the column containing the timestamp values.
            unit (str): The unit of the timestamp values (default is 'ms').
        """
    df['Timestamp'] = pd.to_datetime(df[column], unit=unit)


def aggregate_rows(df, window_size=2, time_col='Timestamp'):
    aggregated_data = {col: [] for col in df.columns}

    for i in range(0, len(df)-window_size):
        # Prendi le due righe da aggregare
        rows_t = df.iloc[i: i + window_size]
        rows = df.iloc[i+1: i+1 + window_size]

        for col in df.columns:
            if col == time_col:
                aggregated_data[col].append(rows_t[col].iloc[0])
            else:
                aggregated_data[col].append(rows[col].mean() if rows[col].dtype == 'float64' else rows[col].mode().iloc[0])

    # Crea un nuovo DataFrame con i dati aggregati
    aggregated_df = pd.DataFrame(aggregated_data)

    return aggregated_df

if __name__ == "__main__":
    print(f"{os.getcwd()=}")

    mps_datapath = "../Datasets-causality/DigitaTwins-Fischer/multi-process-station-data/mps_dt_logs.csv"
    args_parse = argparse.ArgumentParser()
    args_parse.add_argument('-o', '--output', type=str, required=False)
    args_parse.add_argument('-b', '--bin_dict', type=str, required=False)
    args = args_parse.parse_args(['--output=mps_timeseries_10',
                                  '--bin_dict={"output-conveyor-oee_original":10, "oven-oee_original":10, "turntable-oee_original":10, "vacuum-gripper-oee_original":10, "OEE_original":10, "output-conveyor-oee_aggregated":10, "oven-oee_aggregated":10, "turntable-oee_aggregated":10, "vacuum-gripper-oee_aggregated":10, "OEE_aggregated":10}'])
    print(f"{args=}")
    name = "mps_timeseries_10"

    df_mps = pd.read_csv(mps_datapath)
    clean_timestamp(df_mps)

    aggregated_df = aggregate_rows(df_mps, window_size=3, time_col='Timestamp')
    df_merged = pd.merge(df_mps, aggregated_df, on='Timestamp', suffixes=('_original', '_aggregated'))

    df_merged.drop(columns=[col for col in df_merged.columns if 'Timestamp' in col or 'Source' in col], inplace=True,
                   axis=1)

    df_merged = label_encode_categoricals(df_merged, name, f"results/{args.output}")
    df_merged = discretize(args, df_merged)

    df_merged.to_csv(f"results/{args.output}/{name}.csv", index=False)

    gt_graph = load_digraph_from_json(None)
    gt_array = get_my_adjacency_matrix(gt_graph) if gt_graph is not None else None
    run_algorithms(df_merged, name, gt_array)
