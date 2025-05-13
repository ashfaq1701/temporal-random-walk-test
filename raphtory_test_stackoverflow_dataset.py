import argparse
import pickle
import time
import pandas as pd
from raphtory import Graph

N_RUNS = 5

def read_temporal_edges_df(file_path):
    print(f"[INFO] Reading CSV: {file_path}")
    return pd.read_csv(file_path, skiprows=1, header=None, names=['source', 'target', 'timestamp'])

def progressive_higher_edge_addition_test_raphtory(edges_df):
    print("\n[INFO] Starting progressive edge addition test...")
    edge_counts = [
        10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000,
        5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000,
        50_000_000, 60_000_000
    ]

    edge_addition_times = []

    for edge_count in edge_counts:
        print(f"[INFO] Loading {edge_count} edges...")

        current_edge_addition_times = []
        total_time = 0.0
        for run in range(N_RUNS):
            print(f"  [RUN {run+1}/{N_RUNS}]")

            df_with_selected_edges = edges_df.iloc[0:edge_count]
            g = Graph()

            start = time.time()
            g.load_edges_from_pandas(
                df=df_with_selected_edges,
                time="timestamp",
                src="source",
                dst="target"
            )
            run_time = time.time() - start
            print(f"    [DONE] Time: {run_time:.2f} sec")
            current_edge_addition_times.append(run_time)
            total_time += run_time

        avg_time = total_time / N_RUNS
        edge_addition_times.append(current_edge_addition_times)
        print(f"[RESULT] Avg time for {edge_count} edges: {avg_time:.2f} seconds")

    return edge_addition_times

def incremental_edge_addition_test_raphtory(edges_df, increment_size):
    print("\n[INFO] Starting incremental edge addition test...")
    total_edges = 60_000_000
    current_start = 0

    g = Graph()
    edge_addition_times = []

    while current_start < total_edges:
        current_end = current_start + increment_size
        print(f"[INFO] Adding edges {current_start} to {current_end}...")

        df_with_selected_edges = edges_df.iloc[current_start:current_end]

        start = time.time()
        g.load_edges_from_pandas(
            df=df_with_selected_edges,
            time="timestamp",
            src="source",
            dst="target"
        )
        elapsed = time.time() - start
        edge_addition_times.append(elapsed)

        print(f"[SUCCESS] Added {increment_size} edges in {elapsed:.2f} seconds.")
        current_start = current_end

    return edge_addition_times

def main(increment_size):
    print("[INFO] Starting benchmark...\n")
    edges_df = read_temporal_edges_df("data/sx-stackoverflow.csv")

    print("\n[INFO] Running progressive edge addition test")
    results_edges = progressive_higher_edge_addition_test_raphtory(edges_df)

    print("\n[INFO] Running incremental edge addition test")
    results_incremental = incremental_edge_addition_test_raphtory(edges_df, increment_size)

    print("\n[INFO] Saving results...")
    pickle.dump(results_edges, open(f"results/result_edges_raphtory.pkl", "wb"))
    pickle.dump(results_incremental, open(f"results/result_incremental_raphtory.pkl", "wb"))
    print("[DONE] Benchmark complete.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Walk Benchmark")
    parser.add_argument('--increment_size', type=int, default=5_000_000,
                        help='Timestamp range for incremental edge addition (default: 5,000,000)')

    args = parser.parse_args()

    main(args.increment_size)
