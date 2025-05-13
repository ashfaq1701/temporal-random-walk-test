import argparse
import os
import pickle
import time

import pandas as pd
from temporal_random_walk import TemporalRandomWalk

MAX_WALK_LEN = 100


def main(base_dir, minutes_pre_step, window_size, walk_count, shuffle_data, use_gpu):
    runtime_start = time.time()

    running_device = "GPU" if use_gpu else "CPU"
    print(f"---- Running on {running_device}. ----\n")

    trw = TemporalRandomWalk(
        is_directed=True,
        use_gpu=use_gpu,
        max_time_capacity=window_size,
        enable_weight_computation=False
    )

    edge_addition_times = []
    walk_times = []

    total_minutes_data_processed = 0

    total_edges_per_iteration = []
    active_edges_per_iteration = []

    total_edges_added = 0

    for i in range(0, 20160, minutes_pre_step):

        dfs = []
        for j in range(minutes_pre_step):
            dfs.append(pd.read_parquet(os.path.join(base_dir, f'data_{i + j}.parquet')))

        merged_df = pd.concat(dfs, ignore_index=True)

        if shuffle_data:
            final_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            final_df = merged_df

        total_edges_added += len(final_df)

        sources = final_df['u'].values
        targets = final_df['i'].values
        timestamps = final_df['ts'].values

        start_time = time.time()
        trw.add_multiple_edges(sources, targets, timestamps)
        edge_addition_time = time.time() - start_time

        edge_addition_times.append(edge_addition_time)

        active_edge_count = trw.get_edge_count()
        total_edges_per_iteration.append(total_edges_added)
        active_edges_per_iteration.append(active_edge_count)

        start_time = time.time()
        trw.get_random_walks_and_times(
            max_walk_len=MAX_WALK_LEN,
            walk_bias="ExponentialIndex",
            num_walks_total=walk_count,
            initial_edge_bias="Uniform",
            walk_direction="Forward_In_Time"
        )
        walk_sampling_time = time.time() - start_time
        walk_times.append(walk_sampling_time)

        total_minutes_data_processed += minutes_pre_step
        print(f'{total_minutes_data_processed} minutes data processed, edge addition time: {edge_addition_time:.3f}, walk sampling time: {walk_sampling_time:.3f}')


    print('Completed processing all data')
    results = {
        'total_runtime': time.time() - runtime_start,
        'edge_addition_time': edge_addition_times,
        'walk_sampling_time': walk_times,
        'total_edges': total_edges_per_iteration,
        'active_edges': active_edges_per_iteration
    }

    pickle.dump(results, open(f"results/result_alibaba_streaming_{running_device}.pkl", "wb"))
    print(f"\nTotal runtime: {results['total_runtime']:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raphtory vs Temporal Walk")

    # Base directory for input/output files
    parser.add_argument(
        '--base_dir', type=str, required=True,
        help='Base directory containing input files'
    )

    # Enable GPU acceleration
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Enable GPU acceleration'
    )

    # Window timestep size in milliseconds (default: 1 hour)
    parser.add_argument(
        '--window_size', type=int, default=900_000,
        help='Sliding window size in milliseconds (default: 900,000 = 15 minutes)'
    )

    parser.add_argument(
        '--minutes_pre_step', type=int, default=3,
        help='Increment size in minutes (default: 3)'
    )

    # Walk count
    parser.add_argument(
        '--walk_count', type=int, default=2_000_000,
        help='Number of walks to generate (default 2_000_000)'
    )

    # Shuffle data
    parser.add_argument(
        '--shuffle_data', action='store_true',
        help='Shuffle the data'
    )

    args = parser.parse_args()

    # Example usage
    print(f"Base dir: {args.base_dir}")
    print(f"Use GPU: {args.use_gpu}")
    print(f"Window size: {args.window_size} ms")

    main(args.base_dir, args.minutes_pre_step, args.window_size, args.walk_count, args.shuffle_data, args.use_gpu)
