import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from raphtory import Graph
from temporal_random_walk import TemporalRandomWalk

MAX_WALK_LEN = 100


def human_readable_count(n):
    if n >= 1_000_000_000:
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) / 1_000_000
        return f"{billions} billion {millions:.1f} million"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f} million"
    elif n >= 1_000:
        return f"{n / 1_000:.1f} thousand"
    else:
        return str(n)



def main(base_dir, minutes_per_step):
    runtime_start = time.time()

    g = Graph()

    edge_addition_times = []

    total_minutes_data_processed = 0

    total_edges_per_iteration = []

    total_edges_added = 0

    for i in range(0, 20160, minutes_per_step):
        dfs = [pd.read_parquet(os.path.join(base_dir, f'data_{i + j}.parquet')) for j in range(minutes_per_step)]
        merged_df = pd.concat(dfs, ignore_index=True)
        final_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

        total_edges_added += len(final_df)

        edge_addition_start_time = time.time()
        g.load_edges_from_pandas(
            df=final_df,
            time="timestamp",
            src="source",
            dst="target"
        )
        edge_addition_time = time.time() - edge_addition_start_time

        edge_addition_times.append(edge_addition_time)

        total_edges_per_iteration.append(total_edges_added)

        total_minutes_data_processed += minutes_per_step
        print(
            f"{total_minutes_data_processed} minutes data processed | "
            f"Edge addition time: {edge_addition_time:.3f}s | "
            f"Total edges: {human_readable_count(total_edges_added)} | "
        )

    print('Completed processing all data')
    results = {
        'total_runtime': time.time() - runtime_start,
        'edge_addition_time': edge_addition_times,
        'total_edges': total_edges_per_iteration
    }

    pickle.dump(results, open(f"results/result_alibaba_streaming_raphtory.pkl", "wb"))
    print(f"\nTotal runtime: {results['total_runtime']:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raphtory vs Temporal Walk")

    # Base directory for input/output files
    parser.add_argument(
        '--base_dir', type=str, required=True,
        help='Base directory containing input files'
    )

    parser.add_argument(
        '--minutes_per_step', type=int, default=3,
        help='Increment size in minutes (default: 3)'
    )

    args = parser.parse_args()

    # Example usage
    print(f"Base dir: {args.base_dir}")

    main(args.base_dir, args.minutes_per_step)
