import time
import pickle
import argparse
import pandas as pd
import numpy as np
from raphtory import Graph
from temporal_random_walk import TemporalRandomWalk

MAX_WALK_LENGTH = 100

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
    301_183_000 # Full dataset
]

def load_data(data_file_path):
    df = pd.read_csv(
        data_file_path,
        sep=r'\s+',
        skiprows=1,
        header=None,
        names=['u', 'i', 'x', 'ts'])

    return df


def progressively_higher_edge_addition_test_trw(data_df, use_gpu, populate_weights, n_runs):
    results = []

    sources = data_df['u'].to_numpy()
    targets = data_df['i'].to_numpy()
    timestamps = data_df['ts'].to_numpy()

    for edge_count in edge_counts:
        print(f"\n--- Testing with {edge_count:,} edges ---")

        current_sources = sources[:edge_count]
        current_targets = targets[:edge_count]
        current_timestamps = timestamps[:edge_count]

        current_results = []

        for _ in range(n_runs):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=use_gpu, max_time_capacity=-1,
                enable_weight_computation=populate_weights
            )

            start = time.time()
            trw.add_multiple_edges(current_sources, current_targets, current_timestamps)
            current_results.append(time.time() - start)

        avg_time = np.mean(current_results)
        weights_status = "with weights" if populate_weights else "without weights"
        print(f"[RESULT] Avg time for {edge_count:,} edges ({weights_status}): {avg_time:.3f} seconds")

        results.append(current_results)

    return results


def progressively_higher_edge_addition_test_raphtory(data_df, n_runs):
    results = []

    sources = data_df['u'].to_numpy()
    targets = data_df['i'].to_numpy()
    timestamps = data_df['ts'].to_numpy()

    for edge_count in edge_counts:
        print(f"\n--- Testing with {edge_count} edges ---")

        # Limit data to edge_count
        current_sources = sources[:edge_count]
        current_targets = targets[:edge_count]
        current_timestamps = timestamps[:edge_count]

        # Create DataFrame for this iteration
        current_df = pd.DataFrame({
            'u': current_sources,
            'i': current_targets,
            'ts': current_timestamps
        })

        current_times = []
        for run in range(n_runs):
            g = Graph()

            start_time = time.time()
            g.load_edges_from_pandas(
                df=current_df,
                time="ts",
                src="u",
                dst="i"
            )
            run_time = time.time() - start_time
            current_times.append(run_time)

        avg_time = np.mean(current_times)
        results.append(current_times)
        print(f"[RESULT] Avg time for {edge_count} edges: {avg_time:.3f} seconds")

    return results


def progressively_higher_walk_sampling_test(data_df, use_gpu, use_weights, fixed_edges_for_walk_gen, n_runs):
    all_num_walks = [
        10_000, 50_000, 100_000, 200_000, 500_000,
        1_000_000, 2_000_000, 5_000_000, 10_000_000
    ]

    results = []

    sources = data_df['u'].to_numpy()
    targets = data_df['i'].to_numpy()
    timestamps = data_df['ts'].to_numpy()

    if fixed_edges_for_walk_gen != -1:
        sources = sources[:fixed_edges_for_walk_gen]
        targets = targets[:fixed_edges_for_walk_gen]
        timestamps = timestamps[:fixed_edges_for_walk_gen]

    trw = TemporalRandomWalk(
        is_directed=True, use_gpu=use_gpu, max_time_capacity=-1,
        enable_weight_computation=True
    )
    trw.add_multiple_edges(sources, targets, timestamps)

    for num_walks in all_num_walks:
        print(f"\n--- Testing with {num_walks:,} walks ---")

        current_results = []

        for _ in range(n_runs):
            start = time.time()
            trw.get_random_walks_and_times(
                max_walk_len=MAX_WALK_LENGTH,
                walk_bias="ExponentialIndex" if not use_weights else "ExponentialWeight",
                num_walks_total=num_walks,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_results.append(time.time() - start)

        avg_time = np.mean(current_results)
        bias_type = "Weight" if use_weights else "Index"
        print(f"[RESULT] {bias_type} based walk sampling time: {avg_time:.3f} seconds")
        results.append(current_results)

    return results


def main(data_file_path, fixed_edges_for_walk_gen, n_runs):
    data_df = load_data(data_file_path)
    print(f'Loaded data, it has {len(data_df)} rows.')

    walk_sampling_gpu_weight_based = progressively_higher_walk_sampling_test(data_df, True, True, fixed_edges_for_walk_gen, n_runs)
    walk_sampling_gpu_index_based = progressively_higher_walk_sampling_test(data_df, True, False, fixed_edges_for_walk_gen, n_runs)
    walk_sampling_cpu_weight_based = progressively_higher_walk_sampling_test(data_df, False, True, fixed_edges_for_walk_gen,  n_runs)
    walk_sampling_cpu_index_based = progressively_higher_walk_sampling_test(data_df, False, False, fixed_edges_for_walk_gen, n_runs)

    trw_edge_addition_gpu_with_weights = progressively_higher_edge_addition_test_trw(data_df, True, True, n_runs)
    trw_edge_addition_gpu_without_weights = progressively_higher_edge_addition_test_trw(data_df, True, False, n_runs)
    trw_edge_addition_cpu_with_weights = progressively_higher_edge_addition_test_trw(data_df, False, True, n_runs)
    trw_edge_addition_cpu_without_weights = progressively_higher_edge_addition_test_trw(data_df, False, False, n_runs)

    raphtory_edge_addition = progressively_higher_edge_addition_test_raphtory(data_df, n_runs)

    results = {
        'trw_edge_addition_gpu_with_weights': trw_edge_addition_gpu_with_weights,
        'trw_edge_addition_gpu_without_weights': trw_edge_addition_gpu_without_weights,
        'trw_edge_addition_cpu_with_weights': trw_edge_addition_cpu_with_weights,
        'trw_edge_addition_cpu_without_weights': trw_edge_addition_cpu_without_weights,
        'walk_sampling_gpu_weight_based': walk_sampling_gpu_weight_based,
        'walk_sampling_gpu_index_based': walk_sampling_gpu_index_based,
        'walk_sampling_cpu_weight_based': walk_sampling_cpu_weight_based,
        'walk_sampling_cpu_index_based': walk_sampling_cpu_index_based,
        'raphtory_edge_addition': raphtory_edge_addition
    }

    pickle.dump(results, open(f"results/results_trw_raphtory_edge_addition.pickle", "wb"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal Walk Benchmark")

    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--data_file', type=str, required=True, help='Data filepath')
    parser.add_argument('--fixed_edges_for_walk_gen', type=int, default=-1, help='Number of edges used to generate walks')

    args = parser.parse_args()
    main(args.data_file, args.fixed_edges_for_walk_gen, args.n_runs)
