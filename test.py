import argparse
import pickle
import time
import numpy as np
import pandas as pd
from temporal_random_walk import TemporalRandomWalk

N_RUNS = 5

def read_temporal_edges(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None, names=['source', 'target', 'timestamp'])
    df = df.dropna(subset=['source', 'target', 'timestamp'])

    df = df.astype({'source': np.int32, 'target': np.int32, 'timestamp': np.int64})

    sources = df['source'].values
    targets = df['target'].values
    timestamps = df['timestamp'].values

    return sources, targets, timestamps


def progressive_higher_edge_addition_test(all_sources, all_targets, all_timestamps, use_gpu):
    edge_counts = [
        10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000,
        5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000,
        50_000_000, 60_000_000
    ]

    walk_count = 1_000_000
    max_walk_len = 100

    edge_addition_times_without_weights = []
    walk_sampling_times_index_based = []
    edge_addition_times_with_weights = []
    walk_sampling_times_weight_based = []

    for edge_count in edge_counts:
        print(f"\n--- Testing with {edge_count} edges ---")
        sources = all_sources[:edge_count]
        targets = all_targets[:edge_count]
        timestamps = all_timestamps[:edge_count]

        # without weights
        print("Without weights")
        add_times = []
        walk_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=use_gpu, max_time_capacity=-1,
                enable_weight_computation=False
            )
            start = time.time()
            trw.add_multiple_edges(sources, targets, timestamps)
            add_times.append(time.time() - start)

            start = time.time()
            trw.get_random_walks_and_times(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialIndex",
                num_walks_total=walk_count,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            walk_times.append(time.time() - start)

        edge_addition_times_without_weights.append(add_times)
        walk_sampling_times_index_based.append(walk_times)

        # With weights
        print("With weights")
        add_times = []
        walk_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=use_gpu, max_time_capacity=-1,
                enable_weight_computation=True
            )
            start = time.time()
            trw.add_multiple_edges(sources, targets, timestamps)
            add_times.append(time.time() - start)

            start = time.time()
            trw.get_random_walks_and_times(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialWeight",
                num_walks_total=walk_count,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            walk_times.append(time.time() - start)

        edge_addition_times_with_weights.append(add_times)
        walk_sampling_times_weight_based.append(walk_times)

    return {
        "edge_addition_time_without_weights": edge_addition_times_without_weights,
        "walk_sampling_time_index_based": walk_sampling_times_index_based,
        "edge_addition_time_with_weights": edge_addition_times_with_weights,
        "walk_sampling_time_weight_based": walk_sampling_times_weight_based
    }


def progressively_higher_walk_sampling_test(all_sources, all_targets, all_timestamps, use_gpu):
    num_edges = 40_000_000
    max_walk_len = 100
    walk_nums = [
        10_000, 50_000, 100_000, 200_000, 500_000,
        1_000_000, 2_000_000, 5_000_000, 10_000_000
    ]

    walk_sampling_times_index_based = []
    walk_sampling_times_weight_based = []

    sources = all_sources[:num_edges]
    targets = all_targets[:num_edges]
    timestamps = all_timestamps[:num_edges]

    for num_walks in walk_nums:
        print(f"Testing walk count: {num_walks:,}")

        # index based
        current_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=use_gpu, max_time_capacity=-1,
                enable_weight_computation=False
            )
            trw.add_multiple_edges(sources, targets, timestamps)

            start = time.time()
            trw.get_random_walks_and_times(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialIndex",
                num_walks_total=num_walks,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start)

        avg_time = np.mean(current_times)
        print(f"Index Based walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times_index_based.append(current_times)

        # weight based
        current_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=use_gpu, max_time_capacity=-1,
                enable_weight_computation=True
            )
            trw.add_multiple_edges(sources, targets, timestamps)

            start = time.time()
            trw.get_random_walks_and_times(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialWeight",
                num_walks_total=num_walks,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start)

        avg_time = np.mean(current_times)
        print(f"Weight based walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times_weight_based.append(current_times)

    return {
        "walk_sampling_time_index_based": walk_sampling_times_index_based,
        "walk_sampling_time_weight_based": walk_sampling_times_weight_based
    }


def varying_max_walk_length_test(all_sources, all_targets, all_timestamps, use_gpu):
    num_edges = 40_000_000
    walk_count = 3_000_000
    walk_lengths = list(range(10, 310, 10))

    walk_sampling_times = []

    sources = all_sources[:num_edges]
    targets = all_targets[:num_edges]
    timestamps = all_timestamps[:num_edges]

    for walk_len in walk_lengths:
        print(f"Testing walk length: {walk_len}")

        current_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True,
                use_gpu=use_gpu,
                max_time_capacity=-1,
                enable_weight_computation=False
            )
            trw.add_multiple_edges(sources, targets, timestamps)

            start_time = time.time()
            trw.get_random_walks_and_times(
                max_walk_len=walk_len,
                walk_bias="ExponentialIndex",
                num_walks_total=walk_count,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start_time)

        avg_time = np.mean(current_times)
        print(f"walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times.append(current_times)

    return {
        "walk_sampling_time": walk_sampling_times
    }


def main(use_gpu):
    all_sources, all_targets, all_timestamps = read_temporal_edges("data/sx-stackoverflow.csv")
    print(f"Loaded {len(all_timestamps):,} edges.")

    running_device = "GPU" if use_gpu else "CPU"
    print(f"---- Running on {running_device}. ----\n")

    results_edges = progressive_higher_edge_addition_test(all_sources, all_targets, all_timestamps, use_gpu)
    results_walks = progressively_higher_walk_sampling_test(all_sources, all_targets, all_timestamps, use_gpu)
    result_max_walk_lens = varying_max_walk_length_test(all_sources, all_targets, all_timestamps, use_gpu)

    print(f"Edge Addition Test ({running_device}):")
    for k, v in results_edges.items():
        print(f"{k}: {v}")

    print(f"\nWalk Sampling Test ({running_device}):")
    for k, v in results_walks.items():
        print(f"{k}: {v}")

    print(f"\nMax Walk Length Test ({running_device}):")
    for k, v in result_max_walk_lens.items():
        print(f"{k}: {v}")

    pickle.dump(results_edges, open(f"results/result_edges_{running_device}.pkl", "wb"))
    pickle.dump(results_walks, open(f"results/result_walks_{running_device}.pkl", "wb"))
    pickle.dump(result_max_walk_lens, open(f"results/result_max_walk_lens_{running_device}.pkl", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal Walk Benchmark")

    parser.add_argument('--use_gpu', action='store_true', help='Enable GPU acceleration')

    args = parser.parse_args()
    main(args.use_gpu)
