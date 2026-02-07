import time
import pickle
import argparse
import pandas as pd
import numpy as np

from temporal_random_walk import TemporalRandomWalk

MAX_WALK_LENGTH = 40

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
# Walk sampling vs edge count (1 walk per node)
# ------------------------------------------------------------
def walk_sampling_vs_edges(
    data_df,
    use_gpu,
    mode,      # "index", "weight", "tn2v"
    n_runs
):
    results = []

    all_sources = data_df["u"].to_numpy()
    all_targets = data_df["i"].to_numpy()
    all_timestamps = data_df["ts"].to_numpy()

    for edge_count in EDGE_COUNTS:
        print(f"\n--- {mode.upper()} | {edge_count:,} edges ---")

        sources = all_sources[:edge_count]
        targets = all_targets[:edge_count]
        timestamps = all_timestamps[:edge_count]

        # Number of nodes = max node id + 1
        num_nodes = int(max(sources.max(), targets.max()) + 1)
        num_walks_per_node = 1

        print(f"Sampling {num_nodes:,} walks (1 per node)")

        run_times = []

        for _ in range(n_runs):
            trw = TemporalRandomWalk(
                is_directed=True,
                use_gpu=use_gpu,
                max_time_capacity=-1,
                enable_weight_computation=(mode == "weight"),
                enable_temporal_node2vec=(mode == "tn2v")
            )

            trw.add_multiple_edges(sources, targets, timestamps)

            start = time.time()
            trw.get_random_walks_and_times_for_all_nodes(
                max_walk_len=MAX_WALK_LENGTH,
                walk_bias={
                    "index": "ExponentialIndex",
                    "weight": "ExponentialWeight",
                    "tn2v": "TemporalNode2Vec"
                }[mode],
                num_walks_per_node=num_walks_per_node,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            run_times.append(time.time() - start)

        avg = np.mean(run_times)
        print(f"[RESULT] Avg walk sampling time: {avg:.3f} sec")
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
        "walk_time_gpu_index": walk_sampling_vs_edges(
            data_df, True, "index", n_runs
        ),
        "walk_time_gpu_weight": walk_sampling_vs_edges(
            data_df, True, "weight", n_runs
        ),
        "walk_time_gpu_tn2v": walk_sampling_vs_edges(
            data_df, True, "tn2v", n_runs
        ),

        # CPU
        "walk_time_cpu_index": walk_sampling_vs_edges(
            data_df, False, "index", n_runs
        ),
        "walk_time_cpu_weight": walk_sampling_vs_edges(
            data_df, False, "weight", n_runs
        ),
        "walk_time_cpu_tn2v": walk_sampling_vs_edges(
            data_df, False, "tn2v", n_runs
        ),
    }

    out_file = "results/results_walk_time_vs_edges_all_nodes.pkl"
    pickle.dump(results, open(out_file, "wb"))
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk Sampling Time vs Edge Count (1 walk per node)"
    )

    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--n_runs", type=int, default=3)

    args = parser.parse_args()
    main(args.data_file, args.n_runs)
