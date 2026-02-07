import argparse
import pickle
import time
import numpy as np

from temporal_random_walk import TemporalRandomWalk
from utils import read_temporal_edges

N_RUNS = 5
MAX_WALK_LEN = 80


def benchmark_dataset(
    data_file: str,
    walks_per_node: int
):
    sources, targets, timestamps = read_temporal_edges(data_file)
    num_nodes = int(
        max(sources.max(), targets.max()) + 1
    )

    num_walks_total = num_nodes * walks_per_node
    print(
        f"\nDataset: {data_file}\n"
        f"Edges: {len(timestamps):,}, "
        f"Nodes: {num_nodes:,}, "
        f"Total walks: {num_walks_total:,}"
    )

    run_times = []

    for run in range(N_RUNS):
        trw = TemporalRandomWalk(
            is_directed=True,
            use_gpu=True,
            max_time_capacity=-1,
            enable_temporal_node2vec=True
        )

        # Ingest full dataset (not timed)
        trw.add_multiple_edges(sources, targets, timestamps)

        start = time.time()
        trw.get_random_walks_and_times(
            max_walk_len=MAX_WALK_LEN,
            walk_bias="TemporalNode2Vec",
            num_walks_total=num_walks_total,
            initial_edge_bias="Uniform",
            walk_direction="Forward_In_Time"
        )
        elapsed = time.time() - start
        run_times.append(elapsed)

        print(f"  Run {run + 1}: {elapsed:.3f} sec")

    return run_times


def main(data_files, walks_per_node, output_file):
    results = {}

    for data_file in data_files:
        run_times = benchmark_dataset(
            data_file=data_file,
            walks_per_node=walks_per_node
        )
        results[data_file] = run_times

    print("\n====== Summary ======")
    for data_file, times in results.items():
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(
            f"{data_file} -> "
            f"Mean: {mean_time:.3f} sec, "
            f"Std: {std_time:.3f} sec"
        )

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TemporalNode2Vec Walk Sampling Benchmark (GPU)"
    )

    parser.add_argument(
        "data_files",
        nargs="+",
        type=str,
        help="List of CSV data files (3-column temporal edges)"
    )

    parser.add_argument(
        "--walks_per_node",
        type=int,
        default=1,
        help="Number of walks per node (default: 1)"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="temporal_node2vec_results.pkl",
        help="Output pickle file"
    )

    args = parser.parse_args()

    main(
        data_files=args.data_files,
        walks_per_node=args.walks_per_node,
        output_file=args.output_file
    )
