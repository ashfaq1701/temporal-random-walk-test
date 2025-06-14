import argparse
import os.path
import time
import pickle
import numpy as np
import pandas as pd
from temporal_random_walk import TemporalRandomWalk


def benchmark_dataset(data_name, data_dir, use_gpu, max_walk_len, is_directed):
    if data_name == 'delicious':
        df = pd.read_csv(
            os.path.join(data_dir, 'out.delicious_delicious-ti_delicious-ti'),
            sep=r'\s+',
            skiprows=2,
            header=None,
            names=['u', 'i', 'x', 'ts'])
    elif data_name == 'edit':
        df = pd.read_csv(
            os.path.join(data_dir, 'out.edit-enwiki'),
            sep=r'\s+',
            skiprows=1,
            header=None,
            names=['u', 'i', 'x', 'ts'])
    elif data_name == 'growth':
        df = pd.read_csv(os.path.join(data_dir, 'out.wikipedia-growth'),
            sep=r'\s+',
            skiprows=1,
            header=None,
            names=['u', 'i', 'x', 'ts'])
    else:
        raise ValueError(f"Unknown dataset {data_name}")

    print(f"Loaded dataset: {data_name}, dataset shape: {df.shape}")

    sources = df['u'].to_numpy()
    targets = df['i'].to_numpy()
    timestamps = df['ts'].to_numpy()

    trw_without_weights = TemporalRandomWalk(is_directed=is_directed, use_gpu=use_gpu, max_time_capacity=-1)

    without_weights_edge_addition_start_time = time.time()
    trw_without_weights.add_multiple_edges(sources, targets, timestamps)
    edge_addition_time_without_weights = time.time() - without_weights_edge_addition_start_time

    exp_index_walk_sampling_start_time = time.time()
    _, _, exp_index_walk_lens = trw_without_weights.get_random_walks_and_times_for_all_nodes(
        max_walk_len,
        "ExponentialIndex",
        1,
        "Uniform")
    exp_index_walk_sampling_time = time.time() - exp_index_walk_sampling_start_time

    linear_walk_sampling_start_time = time.time()
    _, _, linear_walk_lens = trw_without_weights.get_random_walks_and_times_for_all_nodes(
        max_walk_len,
        "Linear",
        1,
        "Uniform")
    linear_walk_sampling_time = time.time() - linear_walk_sampling_start_time

    trw_with_weights = TemporalRandomWalk(is_directed=is_directed, use_gpu=use_gpu, max_time_capacity=-1,
                                          enable_weight_computation=True)

    with_weights_edge_addition_start_time = time.time()
    trw_with_weights.add_multiple_edges(sources, targets, timestamps)
    edge_addition_time_with_weights = time.time() - with_weights_edge_addition_start_time

    exp_weight_walk_sampling_start_time = time.time()
    _, _, exp_weight_walk_lens = trw_with_weights.get_random_walks_and_times_for_all_nodes(
        max_walk_len,
        "ExponentialWeight",
        1,
        "Uniform")
    exp_weight_walk_sampling_time = time.time() - exp_weight_walk_sampling_start_time

    return {
        'edge_addition_time_without_weights': edge_addition_time_without_weights,
        'edge_addition_time_with_weights': edge_addition_time_with_weights,
        'exp_index_walk_sampling_time': exp_index_walk_sampling_time,
        'exp_weight_walk_sampling_time': exp_weight_walk_sampling_time,
        'linear_walk_sampling_time': linear_walk_sampling_time,
        'exp_index_mean_walk_len': np.mean(exp_index_walk_lens),
        'exp_weight_mean_walk_len': np.mean(exp_weight_walk_lens),
        'linear_mean_walk_len': np.mean(linear_walk_lens)
    }


def main(use_gpu, max_walk_len, data_dir, n_runs):
    dataset_vs_directionality = {
        'growth': True,
        'delicious': False,
        'edit': False
    }

    results = {}

    for dataset_name, is_directed in dataset_vs_directionality.items():
        print(f"\nBenchmarking {dataset_name} dataset...")

        results[dataset_name] = {
            'edge_addition_time_without_weights': [],
            'edge_addition_time_with_weights': [],
            'exp_index_walk_sampling_time': [],
            'exp_weight_walk_sampling_time': [],
            'linear_walk_sampling_time': [],
            'exp_index_mean_walk_len': [],
            'exp_weight_mean_walk_len': [],
            'linear_mean_walk_len': []
        }

        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}...")

            run_results = benchmark_dataset(dataset_name, data_dir, use_gpu, max_walk_len, is_directed=is_directed)

            for metric, value in run_results.items():
                results[dataset_name][metric].append(value)

            print(f"    Edge addition (no weights): {run_results['edge_addition_time_without_weights']:.3f}s")
            print(f"    Edge addition (with weights): {run_results['edge_addition_time_with_weights']:.3f}s")
            print(f"    Exp index walk sampling: {run_results['exp_index_walk_sampling_time']:.3f}s")
            print(f"    Exp weight walk sampling: {run_results['exp_weight_walk_sampling_time']:.3f}s")
            print(f"    Linear walk sampling: {run_results['linear_walk_sampling_time']:.3f}s")

    print("\nSaving results...")
    os.makedirs('results', exist_ok=True)
    with open('results/konnect_benchmark_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\nFinal Results Summary:")
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        for metric, values in dataset_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

    print("\nResults saved to results/konnect_benchmark_results.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Konnect dataset benchmarks")

    parser.add_argument('--use_gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--max_walk_len', type=int, default=80, help='Max walk length (default: 80)')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs (default: 3)')

    args = parser.parse_args()
    main(args.use_gpu, args.max_walk_len, args.data_dir, args.n_runs)
