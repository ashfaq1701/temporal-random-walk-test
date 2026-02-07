import time
import pickle
import argparse
import pandas as pd
import numpy as np

from raphtory import Graph
from temporal_random_walk import TemporalRandomWalk

MAX_WALK_LENGTH = 100

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

WALK_COUNTS = [
    10_000, 50_000, 100_000, 200_000, 500_000,
    1_000_000, 2_000_000, 5_000_000, 10_000_000
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
# Edge ingestion benchmarks (Tempest)
# ------------------------------------------------------------
def edge_addition_test_trw(
    data_df,
    use_gpu,
    mode,              # "index", "weight", "tn2v"
    n_runs
):
    results = []

    sources = data_df["u"].to_numpy()
    targets = data_df["i"].to_numpy()
    timestamps = data_df["ts"].to_numpy()

    for edge_count in EDGE_COUNTS:
        print(f"\n--- TRW edge ingestion: {edge_count:,} edges ({mode}) ---")

        cur_src = sources[:edge_count]
        cur_dst = targets[:edge_count]
        cur_ts = timestamps[:edge_count]

        run_times = []

        for _ in range(n_runs):
            trw = TemporalRandomWalk(
                is_directed=True,
                use_gpu=use_gpu,
                max_time_capacity=-1,
                enable_weight_computation=(mode == "weight"),
                enable_temporal_node2vec=(mode == "tn2v")
            )

            start = time.time()
            trw.add_multiple_edges(cur_src, cur_dst, cur_ts)
            run_times.append(time.time() - start)

        avg = np.mean(run_times)
        print(f"[RESULT] Avg ingestion time: {avg:.3f} sec")
        results.append(run_times)

    return results


# ------------------------------------------------------------
# Edge ingestion benchmarks (Raphtory)
# ------------------------------------------------------------
def edge_addition_test_raphtory(data_df, n_runs):
    results = []

    sources = data_df["u"].to_numpy()
    targets = data_df["i"].to_numpy()
    timestamps = data_df["ts"].to_numpy()

    for edge_count in EDGE_COUNTS:
        print(f"\n--- Raphtory ingestion: {edge_count:,} edges ---")

        cur_df = pd.DataFrame({
            "u": sources[:edge_count],
            "i": targets[:edge_count],
            "ts": timestamps[:edge_count]
        })

        run_times = []

        for _ in range(n_runs):
            g = Graph()
            start = time.time()
            g.load_edges_from_pandas(df=cur_df, time="ts", src="u", dst="i")
            run_times.append(time.time() - start)

        avg = np.mean(run_times)
        print(f"[RESULT] Avg ingestion time: {avg:.3f} sec")
        results.append(run_times)

    return results


# ------------------------------------------------------------
# Walk sampling benchmarks
# ------------------------------------------------------------
def walk_sampling_test(
    data_df,
    use_gpu,
    mode,                    # "index", "weight", "tn2v"
    fixed_edges,
    n_runs
):
    results = []

    sources = data_df["u"].to_numpy()
    targets = data_df["i"].to_numpy()
    timestamps = data_df["ts"].to_numpy()

    if fixed_edges != -1:
        sources = sources[:fixed_edges]
        targets = targets[:fixed_edges]
        timestamps = timestamps[:fixed_edges]

    trw = TemporalRandomWalk(
        is_directed=True,
        use_gpu=use_gpu,
        max_time_capacity=-1,
        enable_weight_computation=(mode == "weight"),
        enable_temporal_node2vec=(mode == "tn2v")
    )

    trw.add_multiple_edges(sources, targets, timestamps)

    for num_walks in WALK_COUNTS:
        print(f"\n--- Walk sampling: {num_walks:,} walks ({mode}) ---")

        run_times = []

        for _ in range(n_runs):
            start = time.time()
            trw.get_random_walks_and_times(
                max_walk_len=MAX_WALK_LENGTH,
                walk_bias={
                    "index": "ExponentialIndex",
                    "weight": "ExponentialWeight",
                    "tn2v": "TemporalNode2Vec"
                }[mode],
                num_walks_total=num_walks,
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
def main(data_file, fixed_edges_for_walk_gen, n_runs):
    data_df = load_data(data_file)
    print(f"Loaded dataset with {len(data_df):,} edges")

    results = {
        # Walk sampling
        "walk_sampling_gpu_index": walk_sampling_test(
            data_df, True, "index", fixed_edges_for_walk_gen, n_runs
        ),
        "walk_sampling_gpu_weight": walk_sampling_test(
            data_df, True, "weight", fixed_edges_for_walk_gen, n_runs
        ),
        "walk_sampling_gpu_tn2v": walk_sampling_test(
            data_df, True, "tn2v", fixed_edges_for_walk_gen, n_runs
        ),
        "walk_sampling_cpu_index": walk_sampling_test(
            data_df, False, "index", fixed_edges_for_walk_gen, n_runs
        ),
        "walk_sampling_cpu_weight": walk_sampling_test(
            data_df, False, "weight", fixed_edges_for_walk_gen, n_runs
        ),
        "walk_sampling_cpu_tn2v": walk_sampling_test(
            data_df, False, "tn2v", fixed_edges_for_walk_gen, n_runs
        ),

        # Edge ingestion (Tempest)
        "edge_ingest_gpu_index": edge_addition_test_trw(
            data_df, True, "index", n_runs
        ),
        "edge_ingest_gpu_weight": edge_addition_test_trw(
            data_df, True, "weight", n_runs
        ),
        "edge_ingest_gpu_tn2v": edge_addition_test_trw(
            data_df, True, "tn2v", n_runs
        ),
        "edge_ingest_cpu_index": edge_addition_test_trw(
            data_df, False, "index", n_runs
        ),
        "edge_ingest_cpu_weight": edge_addition_test_trw(
            data_df, False, "weight", n_runs
        ),
        "edge_ingest_cpu_tn2v": edge_addition_test_trw(
            data_df, False, "tn2v", n_runs
        ),

        # Raphtory baseline
        "raphtory_edge_ingest": edge_addition_test_raphtory(
            data_df, n_runs
        )
    }

    out_file = "results/results_trw_raphtory_with_tn2v.pkl"
    pickle.dump(results, open(out_file, "wb"))
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Walk Benchmark (with TN2V)")

    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--fixed_edges_for_walk_gen", type=int, default=-1)

    args = parser.parse_args()
    main(args.data_file, args.fixed_edges_for_walk_gen, args.n_runs)
