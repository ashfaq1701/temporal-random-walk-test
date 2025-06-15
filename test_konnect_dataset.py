import argparse
import os
import os.path
import time
import pickle
import numpy as np
import pandas as pd
import gc
from temporal_random_walk import TemporalRandomWalk


def benchmark_without_weights(sources, targets, timestamps, use_gpu, max_walk_len, is_directed):
    """Benchmark TRW without weights - separate scope for memory management"""
    print("    Creating TRW without weights...")
    trw = TemporalRandomWalk(is_directed=is_directed, use_gpu=use_gpu, max_time_capacity=-1)

    print("    Adding edges (no weights)...")
    start_time = time.time()
    trw.add_multiple_edges(sources, targets, timestamps)
    edge_addition_time = time.time() - start_time

    print("    Sampling ExponentialIndex walks...")
    start_time = time.time()
    _, _, exp_index_walk_lens = trw.get_random_walks_and_times_for_all_nodes(
        max_walk_len, "ExponentialIndex", 1, "Uniform")
    exp_index_walk_sampling_time = time.time() - start_time

    print("    Sampling Linear walks...")
    start_time = time.time()
    _, _, linear_walk_lens = trw.get_random_walks_and_times_for_all_nodes(
        max_walk_len, "Linear", 1, "Uniform")
    linear_walk_sampling_time = time.time() - start_time

    # Calculate means before deleting
    exp_index_mean = np.mean(exp_index_walk_lens)
    linear_mean = np.mean(linear_walk_lens)

    # TRW object will be automatically cleaned up when function exits
    return {
        'edge_addition_time_without_weights': edge_addition_time,
        'exp_index_walk_sampling_time': exp_index_walk_sampling_time,
        'linear_walk_sampling_time': linear_walk_sampling_time,
        'exp_index_mean_walk_len': exp_index_mean,
        'linear_mean_walk_len': linear_mean
    }


def benchmark_with_weights(sources, targets, timestamps, use_gpu, max_walk_len, is_directed):
    """Benchmark TRW with weights - separate scope for memory management"""
    print("    Creating TRW with weights...")
    trw = TemporalRandomWalk(is_directed=is_directed, use_gpu=use_gpu, max_time_capacity=-1,
                             enable_weight_computation=True)

    print("    Adding edges (with weights)...")
    start_time = time.time()
    trw.add_multiple_edges(sources, targets, timestamps)
    edge_addition_time = time.time() - start_time

    print("    Sampling ExponentialWeight walks...")
    start_time = time.time()
    _, _, exp_weight_walk_lens = trw.get_random_walks_and_times_for_all_nodes(
        max_walk_len, "ExponentialWeight", 1, "Uniform")
    exp_weight_walk_sampling_time = time.time() - start_time

    # Calculate mean before deleting
    exp_weight_mean = np.mean(exp_weight_walk_lens)

    # TRW object will be automatically cleaned up when function exits
    return {
        'edge_addition_time_with_weights': edge_addition_time,
        'exp_weight_walk_sampling_time': exp_weight_walk_sampling_time,
        'exp_weight_mean_walk_len': exp_weight_mean
    }


def load_dataset(data_name, data_dir):
    """Load dataset and return numpy arrays"""
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

    # DataFrame will be automatically cleaned up when function exits
    return sources, targets, timestamps


def benchmark_dataset_single_run(data_name, data_dir, use_gpu, max_walk_len, is_directed):
    """Run a single benchmark for a dataset"""
    sources, targets, timestamps = load_dataset(data_name, data_dir)

    # Benchmark without weights (TRW object cleaned up automatically)
    without_weights_results = benchmark_without_weights(sources, targets, timestamps, use_gpu, max_walk_len,
                                                        is_directed)

    # Force garbage collection between the two benchmarks
    gc.collect()

    # Benchmark with weights (TRW object cleaned up automatically)
    with_weights_results = benchmark_with_weights(sources, targets, timestamps, use_gpu, max_walk_len, is_directed)

    # Combine results
    results = {**without_weights_results, **with_weights_results}

    # Arrays will be cleaned up when function exits
    return results


def main(use_gpu, max_walk_len, data_dir, n_runs):
    dataset_vs_directionality = {
        'growth': True,
        'edit': False,
        'delicious': False
    }

    results = {}

    # Initialize results structure
    for dataset_name in dataset_vs_directionality.keys():
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

    # Run each dataset completely before moving to the next
    for dataset_name, is_directed in dataset_vs_directionality.items():
        print(f"\nBenchmarking {dataset_name} dataset...")

        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}...")

            run_results = benchmark_dataset_single_run(dataset_name, data_dir, use_gpu, max_walk_len, is_directed)

            for metric, value in run_results.items():
                results[dataset_name][metric].append(value)

            print(f"    Edge addition (no weights): {run_results['edge_addition_time_without_weights']:.3f}s")
            print(f"    Edge addition (with weights): {run_results['edge_addition_time_with_weights']:.3f}s")
            print(f"    Exp index walk sampling: {run_results['exp_index_walk_sampling_time']:.3f}s")
            print(f"    Exp weight walk sampling: {run_results['exp_weight_walk_sampling_time']:.3f}s")
            print(f"    Linear walk sampling: {run_results['linear_walk_sampling_time']:.3f}s")
            print(f"    Exp index avg walk len: {run_results['exp_index_mean_walk_len']:.2f}")
            print(f"    Exp weight avg walk len: {run_results['exp_weight_mean_walk_len']:.2f}")
            print(f"    Linear avg walk len: {run_results['linear_mean_walk_len']:.2f}")

            # Force garbage collection between runs
            gc.collect()

        print(f"Completed all runs for {dataset_name}")

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
