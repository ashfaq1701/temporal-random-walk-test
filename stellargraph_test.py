import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), 'libs/stellargraph'))

import argparse
import pickle
import time

import pandas as pd
import numpy as np
from temporal_random_walk import TemporalRandomWalk

from stellargraph.core import StellarGraph
from stellargraph.data import TemporalRandomWalk as TemporalRandomWalkStellarGraph
from utils import read_temporal_edges

N_RUNS = 1

def get_edges_and_nodes(sources, targets, timestamps):
    edges = pd.DataFrame({
        "source": sources.astype(str),
        "target": targets.astype(str),
        "time": timestamps
    })

    unique_nodes = np.unique(np.concatenate([sources, targets]).astype(str))
    nodes = pd.DataFrame(index=unique_nodes)

    return edges, nodes

def progressive_higher_edge_addition_test_stellargraph(all_sources, all_targets, all_timestamps):
    edge_counts = [
        10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000,
        5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000,
        50_000_000, 60_000_000
    ]
    edge_addition_times = []

    for edge_count in edge_counts:
        print(f"\n--- Testing with {edge_count} edges ---")

        sources = all_sources[:edge_count]
        targets = all_targets[:edge_count]
        timestamps = all_timestamps[:edge_count]

        edges, nodes = get_edges_and_nodes(sources, targets, timestamps)

        current_edge_addition_times = []
        total_time = 0.0

        for run in range(N_RUNS):
            start_time = time.time()
            graph = StellarGraph(
                nodes=nodes,
                edges=edges,
                edge_weight_column="time",
            )
            temporal_rw = TemporalRandomWalkStellarGraph(graph)
            run_time = time.time() - start_time

            current_edge_addition_times.append(run_time)
            total_time += run_time

        avg_time = total_time / N_RUNS
        edge_addition_times.append(current_edge_addition_times)
        print(f"[RESULT] Avg time for {edge_count} edges: {avg_time:.2f} seconds")

    return edge_addition_times

def progressive_higher_edge_addition_test_trw(all_sources, all_targets, all_timestamps, use_gpu):
    edge_counts = [
        10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000,
        5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000,
        50_000_000, 60_000_000
    ]

    edge_addition_times_without_weights = []
    edge_addition_times_with_weights = []

    for edge_count in edge_counts:
        print(f"\n--- Testing with {edge_count} edges ---")

        sources = all_sources[:edge_count]
        targets = all_targets[:edge_count]
        timestamps = all_timestamps[:edge_count]

        # without weights
        print("Without weights")
        add_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=use_gpu, max_time_capacity=-1,
                enable_weight_computation=False
            )
            start = time.time()
            trw.add_multiple_edges(sources, targets, timestamps)
            add_times.append(time.time() - start)

        edge_addition_times_without_weights.append(add_times)

        # With weights
        print("With weights")
        add_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=use_gpu, max_time_capacity=-1,
                enable_weight_computation=True
            )
            start = time.time()
            trw.add_multiple_edges(sources, targets, timestamps)
            add_times.append(time.time() - start)

        edge_addition_times_with_weights.append(add_times)

    return {
        "edge_addition_time_without_weights": edge_addition_times_without_weights,
        "edge_addition_time_with_weights": edge_addition_times_with_weights
    }


def progressively_higher_per_node_walk_sampling_test_stellargraph(all_sources, all_targets, all_timestamps):
    max_walk_len = 100
    walks_per_node = range(50, 250 + 1, 50)

    num_edges = 50_000

    sources = all_sources[:num_edges]
    targets = all_targets[:num_edges]
    timestamps = all_timestamps[:num_edges]

    walk_sampling_times = []

    edges, nodes = get_edges_and_nodes(sources, targets, timestamps)

    graph = StellarGraph(
        nodes=nodes,
        edges=edges,
        edge_weight_column="time",
    )
    temporal_rw = TemporalRandomWalkStellarGraph(graph)

    for num_walks_per_node in walks_per_node:
        print(f"Testing per-node walk count: {num_walks_per_node:,}")

        current_times = []
        for _ in range(N_RUNS):
            num_cw = len(nodes) * num_walks_per_node * max_walk_len

            start = time.time()
            temporal_walks_stellar = temporal_rw.run(
                num_cw=num_cw,
                cw_size=2,
                max_walk_length=max_walk_len,
                walk_bias='exponential',
            )
            current_times.append(time.time() - start)

        avg_time = np.mean(current_times)
        print(f"Walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times.append(current_times)

    return walk_sampling_times


def progressively_higher_per_node_walk_sampling_test_trw(all_sources, all_targets, all_timestamps, use_gpu):
    max_walk_len = 100
    walks_per_node = range(50, 250 + 1, 50)

    num_edges = 50_000

    sources = all_sources[:num_edges]
    targets = all_targets[:num_edges]
    timestamps = all_timestamps[:num_edges]

    walk_sampling_times_index_based = []
    walk_sampling_times_weight_based = []

    trw = TemporalRandomWalk(
        is_directed=True, use_gpu=use_gpu, max_time_capacity=-1,
        enable_weight_computation=True
    )
    trw.add_multiple_edges(sources, targets, timestamps)

    for num_walks_per_node in walks_per_node:
        print(f"Testing per-node walk count: {num_walks_per_node:,}")

        # index based
        current_times = []
        for _ in range(N_RUNS):
            start = time.time()
            trw.get_random_walks_and_times_for_all_nodes(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialIndex",
                num_walks_per_node=num_walks_per_node,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start)

        avg_time = np.mean(current_times)
        print(f"Index Based walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times_index_based.append(current_times)

        # weight based
        current_times = []
        for _ in range(N_RUNS):
            start = time.time()
            trw.get_random_walks_and_times_for_all_nodes(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialWeight",
                num_walks_per_node=num_walks_per_node,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start)

        avg_time = np.mean(current_times)
        print(f"Weight based walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times_weight_based.append(current_times)

    return {
        "walk_sampling_time_index_based": walk_sampling_times_index_based,
        "walk_sampling_time_weight_based": walk_sampling_times_weight_based
    }


def edge_size_vs_walk_sampling_test_stellargraph(all_sources, all_targets, all_timestamps):
    max_walk_len = 100
    walks_per_node = 50

    edge_sizes = [1_000, 2000, 5_000, 10_000, 20_000, 50_000]

    walk_sampling_times = []

    for num_edges in edge_sizes:
        print(f"\n--- Testing with {num_edges} edges ---")

        sources = all_sources[:num_edges]
        targets = all_targets[:num_edges]
        timestamps = all_timestamps[:num_edges]

        edges, nodes = get_edges_and_nodes(sources, targets, timestamps)

        graph = StellarGraph(
            nodes=nodes,
            edges=edges,
            edge_weight_column="time",
        )
        temporal_rw = TemporalRandomWalkStellarGraph(graph)

        current_times = []
        for _ in range(N_RUNS):
            num_cw = len(nodes) * walks_per_node * max_walk_len

            start = time.time()
            temporal_walks = temporal_rw.run(
                num_cw=num_cw,
                cw_size=2,
                max_walk_length=max_walk_len,
                walk_bias='exponential',
            )
            current_times.append(time.time() - start)

        avg_time = np.mean(current_times)
        print(f"Walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times.append(current_times)

    return {
        "edge_sizes": edge_sizes,
        "walk_sampling_time": walk_sampling_times
    }


def edge_size_vs_walk_sampling_test_trw(all_sources, all_targets, all_timestamps, use_gpu):
    max_walk_len = 100
    walks_per_node = 50

    edge_sizes = [1_000, 2000, 5_000, 10_000, 20_000, 50_000]

    walk_sampling_times_index_based = []
    walk_sampling_times_weight_based = []

    for num_edges in edge_sizes:
        print(f"\n--- Testing with {num_edges} edges ---")

        sources = all_sources[:num_edges]
        targets = all_targets[:num_edges]
        timestamps = all_timestamps[:num_edges]

        trw = TemporalRandomWalk(
            is_directed=True, use_gpu=use_gpu, max_time_capacity=-1,
            enable_weight_computation=True
        )
        trw.add_multiple_edges(sources, targets, timestamps)

        # Index-based walk sampling
        current_times = []
        for _ in range(N_RUNS):
            start = time.time()
            trw.get_random_walks_and_times_for_all_nodes(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialIndex",
                num_walks_per_node=walks_per_node,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start)

        avg_time = np.mean(current_times)
        print(f"Index-based walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times_index_based.append(current_times)

        # Weight-based walk sampling
        current_times = []
        for _ in range(N_RUNS):
            start = time.time()
            trw.get_random_walks_and_times_for_all_nodes(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialWeight",
                num_walks_per_node=walks_per_node,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start)

        avg_time = np.mean(current_times)
        print(f"Weight-based walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times_weight_based.append(current_times)

    return {
        "edge_sizes": edge_sizes,
        "walk_sampling_time_index_based": walk_sampling_times_index_based,
        "walk_sampling_time_weight_based": walk_sampling_times_weight_based
    }

def main(data_file):
    all_sources, all_targets, all_timestamps = read_temporal_edges(data_file)

    combined_results = {}

    # --- StellarGraph Benchmarks ---
    print('----- Starting StellarGraph edge addition test -----')
    combined_results['edge_addition_stellargraph'] = progressive_higher_edge_addition_test_stellargraph(
        all_sources, all_targets, all_timestamps
    )

    print('----- Edge-size vs Walk Sampling Test - StellarGraph -----')
    combined_results['walk_sampling_vs_edge_size_stellargraph'] = edge_size_vs_walk_sampling_test_stellargraph(
        all_sources, all_targets, all_timestamps
    )

    print('----- Starting StellarGraph walk sampling test -----')
    combined_results['walk_sampling_stellargraph'] = progressively_higher_per_node_walk_sampling_test_stellargraph(
        all_sources, all_targets, all_timestamps
    )

    # --- TRW (CPU) Benchmarks ---
    print('----- Starting TRW edge addition test - CPU -----')
    combined_results['edge_addition_trw_cpu'] = progressive_higher_edge_addition_test_trw(
        all_sources, all_targets, all_timestamps, use_gpu=False
    )

    print('----- Edge-size vs Walk Sampling Test - TRW CPU -----')
    combined_results['walk_sampling_vs_edge_size_trw_cpu'] = edge_size_vs_walk_sampling_test_trw(
        all_sources, all_targets, all_timestamps, use_gpu=False
    )

    print('----- Starting TRW walk sampling test - CPU -----')
    combined_results['walk_sampling_trw_cpu'] = progressively_higher_per_node_walk_sampling_test_trw(
        all_sources, all_targets, all_timestamps, use_gpu=False
    )

    # --- TRW (GPU) Benchmarks ---
    print('----- Starting TRW edge addition test - GPU -----')
    combined_results['edge_addition_trw_gpu'] = progressive_higher_edge_addition_test_trw(
        all_sources, all_targets, all_timestamps, use_gpu=True
    )

    print('----- Edge-size vs Walk Sampling Test - TRW GPU -----')
    combined_results['walk_sampling_vs_edge_size_trw_gpu'] = edge_size_vs_walk_sampling_test_trw(
        all_sources, all_targets, all_timestamps, use_gpu=True
    )

    print('----- Starting TRW walk sampling test - GPU -----')
    combined_results['walk_sampling_trw_gpu'] = progressively_higher_per_node_walk_sampling_test_trw(
        all_sources, all_targets, all_timestamps, use_gpu=True
    )

    print('--- Completed all tests ---')

    # Save all results
    os.makedirs("results", exist_ok=True)
    result_path = os.path.join("results", "stellargraph_results.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(combined_results, f)

    print(f"Results saved to {result_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stellargraph Benchmark")

    parser.add_argument('--data_file', type=str, default="data/alibaba-data.csv",
                        help='Data filepath')

    args = parser.parse_args()
    main(args.data_file)
