import time
import pickle
import argparse
import pandas as pd
import numpy as np

from temporal_random_walk import TemporalRandomWalk

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
MAX_WALK_LENGTH = 50

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
# Walk sampling vs edge count (1 walk per node, simplified)
# ------------------------------------------------------------
def walk_sampling_vs_edges_simple(
    data_df,
    use_gpu,
    mode,     # "index", "weight", "tn2v"
    n_runs
):
    results = []

    all_sources = data_df["u"].to_numpy()
    all_targets = data_df["i"].to_numpy()
    all_timestamps = data_df["ts"].to_numpy()

    print(f"\n=== {mode.upper()} | {'GPU' if use_gpu else 'CPU'} ===")

    for edge_count in EDGE_COUNTS:
        print(f"\n--- {edge_count:,} edges ---")

        sources = all_sources[:edge_count]
        targets = all_targets[:edge_count]
        timestamps = all_timestamps[:edge_count]

        # Number of nodes → total walks (1 walk per node)
        num_nodes = int(max(sources.max(), targets.max()) + 1)
        print(f"Sampling {num_nodes:,} total walks (1 per node)")

        # --------------------------------------------------
        # Construct + ingest ONCE
        # --------------------------------------------------
        trw = TemporalRandomWalk(
            is_directed=True,
            use_gpu=use_gpu,
            max_time_capacity=-1,
            enable_weight_computation=(mode == "weight"),
            enable_temporal_node2vec=(mode == "tn2v")
        )

        trw.add_multiple_edges(sources, targets, timestamps)

        # --------------------------------------------------
        # Repeated sampling
        # --------------------------------------------------
        run_times = []

        for run_id in range(n_runs):
            start = time.time()
            trw.get_random_walks_and_times(
                max_walk_len=MAX_WALK_LENGTH,
                walk_bias=MODE_TO_BIAS[mode],
                num_walks_total=num_nodes,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            elapsed = time.time() - start

            print(f"  Run {run_id + 1}/{n_runs}: {elapsed:.3f} sec")
            run_times.append(elapsed)

        avg = np.mean(run_times)
        print(f"[AVG] {avg:.3f} sec")

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
        "walk_time_gpu_index": walk_sampling_vs_edges_simple(
            data_df, True, "index", n_runs
        ),
        "walk_time_gpu_weight": walk_sampling_vs_edges_simple(
            data_df, True, "weight", n_runs
        ),
        "walk_time_gpu_tn2v": walk_sampling_vs_edges_simple(
            data_df, True, "tn2v", n_runs
        ),

        # CPU
        "walk_time_cpu_index": walk_sampling_vs_edges_simple(
            data_df, False, "index", n_runs
        ),
        "walk_time_cpu_weight": walk_sampling_vs_edges_simple(
            data_df, False, "weight", n_runs
        ),
        "walk_time_cpu_tn2v": walk_sampling_vs_edges_simple(
            data_df, False, "tn2v", n_runs
        ),
    }

    out_file = "results/results_walk_time_vs_edges_all_nodes_simple.pkl"
    pickle.dump(results, open(out_file, "wb"))

    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk Sampling Time vs Edge Count (1 walk per node, simplified)"
    )

    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--n_runs", type=int, default=3)

    args = parser.parse_args()
    main(args.data_file, args.n_runs)
