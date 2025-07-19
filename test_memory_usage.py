import gc
import os
import pickle
import time
from typing import Optional, List, Tuple

import numpy as np
from temporal_random_walk import TemporalRandomWalk


def generate_temporal_graph(num_nodes: int,
                            num_edges: int,
                            num_timestamps: int,
                            time_range: Tuple[int, int] = (0, 2_000_000_000),
                            random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)

    if num_edges <= 0:
        return np.array([]), np.array([]), np.array([])

    if num_timestamps <= 0 or num_nodes <= 0:
        raise ValueError("num_timestamps and num_nodes must be positive")

    # Step 1: Generate exactly num_timestamps unique timestamps
    min_time, max_time = time_range

    # Ensure we have enough range for unique timestamps
    time_span = max_time - min_time + 1
    if num_timestamps > time_span:
        raise ValueError(f"Cannot generate {num_timestamps} unique timestamps in range {time_range}")

    # Generate exactly num_timestamps unique timestamps
    unique_timestamps = np.random.choice(
        np.arange(min_time, max_time + 1, dtype=np.int64),
        size=num_timestamps,
        replace=False
    )
    unique_timestamps = np.sort(unique_timestamps)

    # Step 2: Randomly assign each edge to one of the timestamps
    edge_timestamps = np.random.choice(unique_timestamps, size=num_edges)

    # Step 3: Generate random source and target nodes (self-loops allowed)
    sources = np.random.randint(0, num_nodes, size=num_edges)
    targets = np.random.randint(0, num_nodes, size=num_edges)

    # Step 4: Sort everything by timestamp
    sort_indices = np.argsort(edge_timestamps)

    return (sources[sort_indices],
            targets[sort_indices],
            edge_timestamps[sort_indices])


def create_streaming_graph(num_batches: int,
                           num_edges_per_batch: int,
                           ts_count_per_batch: int,
                           num_nodes: int = 100_000,
                           random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if random_state is not None:
        np.random.seed(random_state)

    batches = []
    current_max_timestamp = 0
    batch_time_gap = 1000  # Gap between batches

    for batch_idx in range(num_batches):
        # Generate timestamps for this batch (all higher than previous batches)
        batch_start_time = current_max_timestamp + batch_time_gap
        batch_end_time = batch_start_time + ts_count_per_batch * 10

        # Generate exactly ts_count_per_batch unique timestamps
        batch_timestamps = np.random.choice(
            np.arange(batch_start_time, batch_end_time + 1, dtype=np.int64),
            size=ts_count_per_batch,
            replace=False
        )
        batch_timestamps = np.sort(batch_timestamps)

        # Assign each edge to one of the batch timestamps
        edge_timestamps = np.random.choice(batch_timestamps, size=num_edges_per_batch)

        # Generate random source and target nodes
        sources = np.random.randint(0, num_nodes, size=num_edges_per_batch)
        targets = np.random.randint(0, num_nodes, size=num_edges_per_batch)

        # Sort edges by timestamp within this batch
        sort_indices = np.argsort(edge_timestamps)
        batch_sources = sources[sort_indices]
        batch_targets = targets[sort_indices]
        batch_edge_timestamps = edge_timestamps[sort_indices]

        # Update max timestamp for next batch
        current_max_timestamp = batch_edge_timestamps.max()

        # Add batch to results
        batches.append((batch_sources, batch_targets, batch_edge_timestamps))

    return batches


def get_memory_usage_with_variable_node_count(is_directed, with_weights, edge_count=10_000_000, timestamp_count=10_000):
    print(f'Testing with increasing number of nodes')
    node_counts = np.logspace(start=1, stop=7, num=7, base=10, dtype=int)
    results = {}

    for current_node_count in node_counts:
        print(f'Testing with node count {current_node_count}')
        gc.collect()
        time.sleep(0.5)

        sources, targets, timestamps = generate_temporal_graph(
            num_nodes=current_node_count,
            num_edges=edge_count,
            num_timestamps=timestamp_count
        )

        temporal_rw = TemporalRandomWalk(
            is_directed=is_directed,
            use_gpu=False,
            max_time_capacity=-1,
            enable_weight_computation=with_weights
        )
        temporal_rw.add_multiple_edges(sources, targets, timestamps)

        memory_used_mb = temporal_rw.get_memory_used() / (1024 * 1024)
        results[current_node_count] = memory_used_mb

        del temporal_rw, sources, targets, timestamps
        gc.collect()
        print(f'Node count {current_node_count} used {memory_used_mb:.2f} MB')

    return results


def get_memory_usage_with_variable_edge_count(is_directed, with_weights, node_count=10_000, timestamp_count=10_000):
    print(f'Testing with increasing number of edges')
    edge_counts = np.logspace(start=1, stop=8, num=8, base=10, dtype=int)
    results = {}

    for current_edge_count in edge_counts:
        print(f'Testing with edge count {current_edge_count}')
        gc.collect()
        time.sleep(0.5)

        sources, targets, timestamps = generate_temporal_graph(
            num_nodes=node_count,
            num_edges=current_edge_count,
            num_timestamps=timestamp_count
        )

        temporal_rw = TemporalRandomWalk(
            is_directed=is_directed,
            use_gpu=False,
            max_time_capacity=-1,
            enable_weight_computation=with_weights
        )
        temporal_rw.add_multiple_edges(sources, targets, timestamps)

        memory_used_mb = temporal_rw.get_memory_used() / (1024 * 1024)
        results[current_edge_count] = memory_used_mb

        del temporal_rw, sources, targets, timestamps
        gc.collect()
        print(f'Edge count {current_edge_count} used {memory_used_mb:.2f} MB')

    return results


def get_memory_usage_with_variable_timestamp_count(is_directed, with_weights, node_count=10_000, edge_count=10_000_000):
    print(f'Testing with increasing number of timestamps')
    numbers_of_ts = np.logspace(start=1, stop=7, num=7, base=10, dtype=int)
    results = {}

    for current_number_of_ts in numbers_of_ts:
        print(f'Testing with timestamp count {current_number_of_ts}')
        gc.collect()
        time.sleep(0.5)

        sources, targets, timestamps = generate_temporal_graph(
            num_nodes=node_count,
            num_edges=edge_count,
            num_timestamps=current_number_of_ts
        )

        temporal_rw = TemporalRandomWalk(
            is_directed=is_directed,
            use_gpu=False,
            max_time_capacity=-1,
            enable_weight_computation=with_weights
        )
        temporal_rw.add_multiple_edges(sources, targets, timestamps)

        memory_used_mb = temporal_rw.get_memory_used() / (1024 * 1024)
        results[current_number_of_ts] = memory_used_mb

        del temporal_rw, sources, targets, timestamps
        gc.collect()
        print(f'Timestamp count {current_number_of_ts} used {memory_used_mb:.2f} MB')

    return results


def test_streaming_window(is_directed, with_weights, num_batches, num_edges_per_batch, ts_count_per_batch):
    print(f'Testing streaming window with {num_batches} batches')

    graph_batches = create_streaming_graph(
        num_batches=num_batches,
        num_edges_per_batch=num_edges_per_batch,
        ts_count_per_batch=ts_count_per_batch
    )

    streaming_temporal_rw = TemporalRandomWalk(
        is_directed=is_directed,
        use_gpu=False,
        max_time_capacity=ts_count_per_batch,
        enable_weight_computation=with_weights
    )

    memory_results = []

    for batch_idx, (batch_sources, batch_targets, batch_timestamps) in enumerate(graph_batches):
        print(f'Starting processing batch {batch_idx + 1}')
        streaming_temporal_rw.add_multiple_edges(batch_sources, batch_targets, batch_timestamps)

        memory_used_mb = streaming_temporal_rw.get_memory_used() / (1024 * 1024)
        memory_results.append(memory_used_mb)

        print(f'Batch {batch_idx + 1}: {memory_used_mb:.2f} MB')

    return memory_results


def run_all_memory_tests():
    results = {}

    for is_directed in [True, False]:
        for with_weights in [True, False]:
            config = f"{'directed' if is_directed else 'undirected'}_{'with_weights' if with_weights else 'without_weights'}"
            print(f"\nTesting {config}...")

            results[f"increasing_nodes_{config}"] = get_memory_usage_with_variable_node_count(is_directed, with_weights)
            results[f"increasing_edges_{config}"] = get_memory_usage_with_variable_edge_count(is_directed, with_weights)
            results[f"increasing_timestamps_{config}"] = get_memory_usage_with_variable_timestamp_count(is_directed, with_weights)
            results[f"streaming_window_{config}"] = test_streaming_window(
                is_directed, with_weights, num_batches=100, num_edges_per_batch=10_000_000, ts_count_per_batch=1_000_000
            )

    os.makedirs("results", exist_ok=True)
    with open("results/memory_benchmarking.pickle", "wb") as f:
        pickle.dump(results, f)

    print(f"\nAll tests completed. Results saved to results/memory_benchmarking.pickle")
    return results


if __name__ == "__main__":
    run_all_memory_tests()
