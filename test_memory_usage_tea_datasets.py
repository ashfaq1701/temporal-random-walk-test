import argparse
import os.path
import pickle
import pandas as pd
from temporal_random_walk import TemporalRandomWalk


def run_memory_tests_for_dataset(dataset_dir):
    df = pd.read_csv(dataset_dir)

    sources = df['u'].to_numpy()
    targets = df['i'].to_numpy()
    timestamps = df['ts'].to_numpy()

    trw = TemporalRandomWalk(
        is_directed=True,
        use_gpu=False,
        max_time_capacity=-1,
        enable_weight_computation=False
    )
    trw.add_multiple_edges(sources, targets, timestamps)
    memory_usage_bytes = trw.get_memory_used()
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)

    print(f"Directory: {dataset_dir}, Memory usage: {memory_usage_mb:.2f} MB")

    return memory_usage_mb


def run_tea_memory_tests(data_dir):
    growth_memory_usage = run_memory_tests_for_dataset(os.path.join(data_dir, "growth.csv"))
    delicious_memory_usage = run_memory_tests_for_dataset(os.path.join(data_dir, "delicious.csv"))
    results = {
        'growth': growth_memory_usage,
        'delicious': delicious_memory_usage
    }
    os.makedirs("results", exist_ok=True)
    with open("results/memory_benchmarking_tea_datasets.pickle", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEA dataset benchmarks")
    parser.add_argument('--data_dir', type=str, help='Data directory')
    args = parser.parse_args()
    run_tea_memory_tests(args.data_dir)
