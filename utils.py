import numpy as np
import pandas as pd


def read_temporal_edges(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None, names=['u', 'i', 'ts'])
    df = df.dropna(subset=['u', 'i', 'ts'])

    df = df.astype({'u': np.int32, 'i': np.int32, 'ts': np.int64})

    sources = df['u'].values
    targets = df['i'].values
    timestamps = df['ts'].values

    return sources, targets, timestamps

def read_temporal_edges_df(file_path):
    print(f"[INFO] Reading CSV: {file_path}")
    return pd.read_csv(file_path, skiprows=1, header=None, names=['u', 'i', 'ts'])
