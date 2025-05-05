import argparse
import pickle
import time
import numpy as np
from temporal_random_walk import TemporalRandomWalk

N_RUNS = 3

def read_temporal_edges(file_path):
    edges = []
    with open(file_path, 'r') as f:
        idx = 0

        for line in f:
            if idx == 0:
                idx += 1
                continue

            parts = line.strip().split(',')
            if len(parts) != 3:
                continue
            src, dst, timestamp = map(int, parts)
            edges.append((src, dst, timestamp))

            idx += 1
    return edges


def progressive_higher_edge_addition_test(dataset, use_gpu):
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
        edges = dataset[:edge_count]

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
            trw.add_multiple_edges(edges)
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

        edge_addition_times_without_weights.append(np.mean(add_times))
        walk_sampling_times_index_based.append(np.mean(walk_times))

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
            trw.add_multiple_edges(edges)
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

        edge_addition_times_with_weights.append(np.mean(add_times))
        walk_sampling_times_weight_based.append(np.mean(walk_times))

    return {
        "edge_addition_time_without_weights": edge_addition_times_without_weights,
        "walk_sampling_time_index_based": walk_sampling_times_index_based,
        "edge_addition_time_with_weights": edge_addition_times_with_weights,
        "walk_sampling_time_weight_based": walk_sampling_times_weight_based
    }


def progressively_higher_walk_sampling_test(dataset, use_gpu):
    num_edges = 40_000_000
    max_walk_len = 200
    walk_nums = [
        10_000, 50_000, 100_000, 200_000, 500_000,
        1_000_000, 2_000_000, 5_000_000, 10_000_000
    ]

    walk_sampling_times_index_based = []
    walk_sampling_times_weight_based = []

    edges = dataset[:num_edges]

    for num_walks in walk_nums:
        print(f"Testing walk count: {num_walks:,}")

        # index based
        current_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=use_gpu, max_time_capacity=-1,
                enable_weight_computation=False
            )
            trw.add_multiple_edges(edges)

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
        walk_sampling_times_index_based.append(avg_time)

        # weight based
        current_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=use_gpu, max_time_capacity=-1,
                enable_weight_computation=True
            )
            trw.add_multiple_edges(edges)

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
        walk_sampling_times_weight_based.append(avg_time)

    return {
        "walk_sampling_time_index_based": walk_sampling_times_index_based,
        "walk_sampling_time_weight_based": walk_sampling_times_weight_based
    }


def varying_max_walk_length_test(dataset, use_gpu):
    num_edges = 40_000_000
    walk_count = 2_000_000
    walk_lengths = list(range(10, 310, 10))

    walk_sampling_times = []

    edges = dataset[:num_edges]

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
            trw.add_multiple_edges(edges)

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
        walk_sampling_times.append(avg_time)

    return {
        "walk_sampling_time": walk_sampling_times
    }


def incremental_edge_addition_sliding_window_test(dataset, use_gpu):
    total_edges = 60_000_000
    increment_size = 500_000
    sliding_window = 30_000_000  # time steps
    walk_count = 5_000_000
    max_walk_len = 100

    edge_addition_times = []
    walk_sampling_times = []
    total_edges_array = []
    active_edges_array = []

    # Get node count for the entire dataset we'll use
    edges_subset = dataset[:total_edges]

    # Create a single TRW instance that we'll incrementally update
    trw = TemporalRandomWalk(
        is_directed=True,
        use_gpu=use_gpu,
        max_time_capacity=sliding_window,  # Set sliding window
        enable_weight_computation=False
    )

    current_edge_count = 0

    while current_edge_count < total_edges:
        # Calculate next batch of edges to add
        start_idx = current_edge_count
        end_idx = min(current_edge_count + increment_size, total_edges)
        edges_to_add = dataset[start_idx:end_idx]

        print(f"\n--- Adding edges {start_idx:,} to {end_idx:,} ---")

        # Measure edge addition time
        start_time = time.time()
        trw.add_multiple_edges(edges_to_add)
        edge_addition_time = time.time() - start_time

        # Update current count
        current_edge_count = end_idx
        total_edges_array.append(current_edge_count)

        # Record edge addition time
        edge_addition_times.append(edge_addition_time)
        print(f"Edge addition time: {edge_addition_time:.3f} sec")

        # Now sample walks and measure time
        current_times = []
        for _ in range(N_RUNS):
            start_time = time.time()
            trw.get_random_walks_and_times(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialIndex",
                num_walks_total=walk_count,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start_time)

        avg_time = np.mean(current_times)
        walk_sampling_times.append(avg_time)
        print(f"Walk sampling time: {avg_time:.3f} sec")

        # Report current active edge count (maybe less than total due to sliding window)
        active_edge_count = trw.get_edge_count()
        active_edges_array.append(active_edge_count)
        print(f"Active edges in graph: {active_edge_count:,} (with sliding window)")

    return {
        "total_edges": total_edges_array,
        "active_edges": active_edges_array,
        "edge_addition_time": edge_addition_times,
        "walk_sampling_time": walk_sampling_times
    }


def main(use_gpu):
    dataset = read_temporal_edges("data/sx-stackoverflow.csv")
    print(f"Loaded {len(dataset):,} edges.")

    running_device = "GPU" if use_gpu else "CPU"
    print(f"---- Running on {running_device}. ----\n")

    results_edges = progressive_higher_edge_addition_test(dataset, use_gpu)
    results_walks = progressively_higher_walk_sampling_test(dataset, use_gpu)
    result_max_walk_lens = varying_max_walk_length_test(dataset, use_gpu)
    results_incremental = incremental_edge_addition_sliding_window_test(dataset, use_gpu)

    print(f"Edge Addition Test ({running_device}):")
    for k, v in results_edges.items():
        print(f"{k}: {v}")

    print(f"\nWalk Sampling Test ({running_device}):")
    for k, v in results_walks.items():
        print(f"{k}: {v}")

    print(f"\nMax Walk Length Test ({running_device}):")
    for k, v in result_max_walk_lens.items():
        print(f"{k}: {v}")

    print(f"\nIncremental Edge Addition with Sliding Window Test ({running_device}):")
    for k, v in results_incremental.items():
        print(f"{k}: {v}")

    pickle.dump(results_edges, open(f"results/result_edges_{running_device}.pkl", "wb"))
    pickle.dump(results_walks, open(f"results/result_walks_{running_device}.pkl", "wb"))
    pickle.dump(result_max_walk_lens, open(f"results/result_max_walk_lens_{running_device}.pkl", "wb"))
    pickle.dump(results_incremental, open(f"results/result_incremental_sliding_{running_device}.pkl", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal Walk Benchmark")
    parser.add_argument('--use_gpu', action='store_true', help='Enable GPU acceleration')
    args = parser.parse_args()

    main(args.use_gpu)
