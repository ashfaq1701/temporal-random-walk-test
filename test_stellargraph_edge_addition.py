import time
import pickle
import argparse
import pandas as pd
import numpy as np
from stellargraph.core import StellarGraph
from stellargraph.data import TemporalRandomWalk as TemporalRandomWalkStellarGraph

edge_counts = [
    1_000,      # 1K
    5_000,      # 5K
    10_000,     # 10K
    50_000,     # 50K
    100_000,    # 100K
    500_000,    # 500K
    1_000_000,  # 1M
    5_000_000,  # 5M
    10_000_000, # 10M
    50_000_000, # 50M
    100_000_000,# 100M
    200_000_000,# 200M
    256_804_235 # Full dataset
]


def load_data(data_file_path):
    df = pd.read_csv(
        data_file_path,
        sep=r'\s+',
        skiprows=1,
        header=None,
        names=['u', 'i', 'x', 'ts'])

    return df


def progressively_higher_edge_addition_test_stellargraph(data_df, n_runs):
    results = []

    sources = data_df['u'].to_numpy().astype(str)
    targets = data_df['i'].to_numpy().astype(str)
    timestamps = data_df['ts'].to_numpy()

    for edge_count in edge_counts:
        print(f"\n--- Testing with {edge_count:,} edges ---")

        current_sources = sources[:edge_count]
        current_targets = targets[:edge_count]
        current_timestamps = timestamps[:edge_count]

        # Create edges DataFrame
        edges = pd.DataFrame({
            "source": current_sources,
            "target": current_targets,
            "time": current_timestamps
        })

        # Create nodes DataFrame
        unique_nodes = np.unique(np.concatenate([current_sources, current_targets]))
        nodes = pd.DataFrame(index=unique_nodes)

        current_times = []

        for run in range(n_runs):
            start_time = time.time()
            graph = StellarGraph(
                nodes=nodes,
                edges=edges,
                edge_weight_column="time",
            )
            temporal_rw = TemporalRandomWalkStellarGraph(graph)
            run_time = time.time() - start_time
            current_times.append(run_time)

        avg_time = np.mean(current_times)
        results.append(current_times)
        print(f"[RESULT] Avg time for {edge_count:,} edges: {avg_time:.3f} seconds")

    return results


def main(data_file_path, n_runs):
    data_df = load_data(data_file_path)
    print(f'Loaded data, it has {len(data_df)} rows.')

    stellargraph_edge_addition = progressively_higher_edge_addition_test_stellargraph(data_df, n_runs)

    results = {
        'stellargraph_edge_addition': stellargraph_edge_addition,
    }

    pickle.dump(results, open(f"results/results_stellargraph_edge_additions.pickle", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StellarGraph Benchmark")

    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--data_file', type=str, required=True, help='Data filepath')

    args = parser.parse_args()
    main(args.data_file, args.n_runs)