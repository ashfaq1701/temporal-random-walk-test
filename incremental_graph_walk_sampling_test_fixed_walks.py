import argparse
import pickle
import time

import pandas as pd
from temporal_random_walk import TemporalRandomWalk

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
MAX_WALK_LENGTH = 50
NUM_WALKS_TOTAL = 10_000_000

EDGE_COUNTS = [
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    200_000_000,
    301_183_000
]

MODE_TO_BIAS = {
    "index": "ExponentialIndex",
    "weight": "ExponentialWeight",
    "tn2v": "TemporalNode2Vec",
}

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def load_data(data_file_path):
    df = pd.read_csv(
        data_file_path,
        sep=r"\s+",
        skiprows=2,
        header=None,
        names=["u", "i", "x", "ts"]
    )
    return df


# ------------------------------------------------------------
# Walk sampling vs edge count (incremental graph)
# ------------------------------------------------------------
def walk_sampling_vs_edges_incremental(
    data_df,
    use_gpu,
    mode,      # "index", "weight", "tn2v"
    n_runs
):
    results = []

    all_sources = data_df["u"].to_numpy()
    all_targets = data_df["i"].to_numpy()
    all_timestamps = data_df["ts"].to_numpy()

    print(f"\n=== {mode.upper()} | {'GPU' if use_gpu else 'CPU'} ===")

    for run_id in range(n_runs):
        print(f"\n--- Run {run_id + 1}/{n_runs} ---")

        trw = TemporalRandomWalk(
            is_directed=True,
            use_gpu=use_gpu,
            max_time_capacity=-1,
            enable_weight_computation=(mode == "weight"),
            enable_temporal_node2vec=(mode == "tn2v")
        )

        prev_edge_count = 0
        run_times = []

        for edge_count in EDGE_COUNTS:
            print(f"\nAdding edges: {prev_edge_count:,} → {edge_count:,}")

            # Incremental edge addition
            trw.add_multiple_edges(
                all_sources[prev_edge_count:edge_count],
                all_targets[prev_edge_count:edge_count],
                all_timestamps[prev_edge_count:edge_count]
            )

            print(f"Sampling {NUM_WALKS_TOTAL:,} walks")

            start = time.time()
            trw.get_random_walks_and_times(
                max_walk_len=MAX_WALK_LENGTH,
                walk_bias=MODE_TO_BIAS[mode],
                num_walks_total=NUM_WALKS_TOTAL,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            elapsed = time.time() - start

            print(f"[RESULT] Walk sampling time: {elapsed:.3f} sec")
            run_times.append(elapsed)

            prev_edge_count = edge_count

        results.append(run_times)

    return results


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(data_file, n_runs):
    data_df = load_data(data_file)
    print(f"Loaded dataset with {len(data_df):,} edges")

    results = {
        # GPU
        "walk_time_gpu_index": walk_sampling_vs_edges_incremental(
            data_df, True, "index", n_runs
        ),
        "walk_time_gpu_weight": walk_sampling_vs_edges_incremental(
            data_df, True, "weight", n_runs
        ),
        "walk_time_gpu_tn2v": walk_sampling_vs_edges_incremental(
            data_df, True, "tn2v", n_runs
        ),

        # CPU
        "walk_time_cpu_index": walk_sampling_vs_edges_incremental(
            data_df, False, "index", n_runs
        ),
        "walk_time_cpu_weight": walk_sampling_vs_edges_incremental(
            data_df, False, "weight", n_runs
        ),
        "walk_time_cpu_tn2v": walk_sampling_vs_edges_incremental(
            data_df, False, "tn2v", n_runs
        ),
    }

    out_file = "results/results_walk_time_vs_edges_10M_incremental.pkl"
    pickle.dump(results, open(out_file, "wb"))

    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk Sampling Time vs Edge Count (10M walks, incremental graph)"
    )

    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--n_runs", type=int, default=3)

    args = parser.parse_args()
    main(args.data_file, args.n_runs)
