import argparse
import pickle
import time

import numpy as np
from temporal_random_walk import TemporalRandomWalk

from utils import read_temporal_edges

N_RUNS = 5


def incremental_edge_addition_sliding_window_test(all_sources, all_targets, all_timestamps, use_gpu, increment_size, sliding_window, walk_count):
    total_edges = 60_000_000
    max_walk_len = 100

    edge_addition_times = []
    walk_sampling_times = []
    total_edges_array = []
    active_edges_array = []

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

        sources = all_sources[start_idx:end_idx]
        targets = all_targets[start_idx:end_idx]
        timestamps = all_timestamps[start_idx:end_idx]

        print(f"\n--- Adding edges {start_idx:,} to {end_idx:,} ---")

        # Measure edge addition time
        start_time = time.time()
        trw.add_multiple_edges(sources, targets, timestamps)
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


def main(data_file, use_gpu, increment_size, sliding_window, walk_count):
    all_sources, all_targets, all_timestamps = read_temporal_edges(data_file)
    print(f"Loaded {len(all_timestamps):,} edges.")

    running_device = "GPU" if use_gpu else "CPU"
    print(f"---- Running on {running_device}. ----\n")

    results_incremental = incremental_edge_addition_sliding_window_test(
        all_sources,
        all_targets,
        all_timestamps,
        use_gpu,
        increment_size,
        sliding_window,
        walk_count
    )

    print(f"\nIncremental Edge Addition with Sliding Window Test ({running_device}):")
    for k, v in results_incremental.items():
        print(f"{k}: {v}")

    pickle.dump(results_incremental, open(f"results/result_incremental_sliding_{running_device}.pkl", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal Walk Benchmark")

    parser.add_argument('--use_gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--data_file', type=str, default="data/sx-stackoverflow.csv", help='Data filepath')
    parser.add_argument('--increment_size', type=int, default=5_000_000,
                        help='Timestamp range for incremental edge addition (default: 5,000,000)')
    parser.add_argument('--sliding_window', type=int, default=50_000_000,
                        help='Sliding window (default: 50,000,000)')
    parser.add_argument('--walk_count', type=int, default=2_000_000,
                        help='Walk count')

    args = parser.parse_args()

    main(args.data_file, args.use_gpu, args.increment_size, args.sliding_window, args.walk_count)
