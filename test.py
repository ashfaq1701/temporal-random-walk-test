import pickle
import time
import numpy as np
from temporal_random_walk import TemporalRandomWalk

N_RUNS = 3

def read_temporal_edges(file_path):
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            src, dst, timestamp = map(int, parts)
            edges.append((src, dst, timestamp))
    return edges

def get_node_count(edges):
    nodes = set()
    for src, dst, timestamp in edges:
        nodes.add(src)
        nodes.add(dst)
    return len(nodes)


def progressive_higher_edge_addition_test(dataset):
    edge_counts = [
        10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000,
        5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000,
        50_000_000, 60_000_000
    ]

    walk_count = 100_000
    max_walk_len = 100

    edge_addition_times_without_weights_cpu = []
    walk_sampling_times_index_based_cpu = []
    edge_addition_times_with_weights_cpu = []
    walk_sampling_times_weight_based_cpu = []

    edge_addition_times_without_weights_gpu = []
    walk_sampling_times_index_based_gpu = []
    edge_addition_times_with_weights_gpu = []
    walk_sampling_times_weight_based_gpu = []

    for edge_count in edge_counts:
        print(f"\n--- Testing with {edge_count} edges ---")
        edges = dataset[:edge_count]
        nodes_count = get_node_count(edges)

        # CPU - without weights
        print("CPU - without weights")
        add_times = []
        walk_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=False, max_time_capacity=-1,
                enable_weight_computation=False, node_count_max_bound=nodes_count
            )
            start = time.time()
            trw.add_multiple_edges(edges)
            add_times.append(time.time() - start)

            start = time.time()
            trw.get_random_walks(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialIndex",
                num_walks_total=walk_count,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            walk_times.append(time.time() - start)
        edge_addition_times_without_weights_cpu.append(np.mean(add_times))
        walk_sampling_times_index_based_cpu.append(np.mean(walk_times))

        # CPU - with weights
        print("CPU - with weights")
        add_times = []
        walk_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=False, max_time_capacity=-1,
                enable_weight_computation=True, node_count_max_bound=nodes_count
            )
            start = time.time()
            trw.add_multiple_edges(edges)
            add_times.append(time.time() - start)

            start = time.time()
            trw.get_random_walks(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialWeight",
                num_walks_total=walk_count,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            walk_times.append(time.time() - start)
        edge_addition_times_with_weights_cpu.append(np.mean(add_times))
        walk_sampling_times_weight_based_cpu.append(np.mean(walk_times))

        # GPU - without weights
        print("GPU - without weights")
        add_times = []
        walk_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=True, max_time_capacity=-1,
                enable_weight_computation=False, node_count_max_bound=nodes_count
            )
            start = time.time()
            trw.add_multiple_edges(edges)
            add_times.append(time.time() - start)

            start = time.time()
            trw.get_random_walks(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialIndex",
                num_walks_total=walk_count,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            walk_times.append(time.time() - start)
        edge_addition_times_without_weights_gpu.append(np.mean(add_times))
        walk_sampling_times_index_based_gpu.append(np.mean(walk_times))

        # GPU - with weights
        print("GPU - with weights")
        add_times = []
        walk_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=True, max_time_capacity=-1,
                enable_weight_computation=True, node_count_max_bound=nodes_count
            )
            start = time.time()
            trw.add_multiple_edges(edges)
            add_times.append(time.time() - start)

            start = time.time()
            trw.get_random_walks(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialWeight",
                num_walks_total=walk_count,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            walk_times.append(time.time() - start)
        edge_addition_times_with_weights_gpu.append(np.mean(add_times))
        walk_sampling_times_weight_based_gpu.append(np.mean(walk_times))

    return {
        "cpu_index_add": edge_addition_times_without_weights_cpu,
        "cpu_index_walk": walk_sampling_times_index_based_cpu,
        "cpu_weight_add": edge_addition_times_with_weights_cpu,
        "cpu_weight_walk": walk_sampling_times_weight_based_cpu,
        "gpu_index_add": edge_addition_times_without_weights_gpu,
        "gpu_index_walk": walk_sampling_times_index_based_gpu,
        "gpu_weight_add": edge_addition_times_with_weights_gpu,
        "gpu_weight_walk": walk_sampling_times_weight_based_gpu,
    }


def progressively_higher_walk_sampling_test(dataset):
    num_edges = 60_000_000
    max_walk_len = 100
    walk_nums = [
        10_000, 50_000, 100_000, 200_000, 500_000,
        1_000_000, 2_000_000, 5_000_000, 10_000_000
    ]

    walk_sampling_times_index_based_cpu = []
    walk_sampling_times_weight_based_cpu = []
    walk_sampling_times_index_based_gpu = []
    walk_sampling_times_weight_based_gpu = []

    edges = dataset[:num_edges]
    nodes_count = get_node_count(edges)

    for num_walks in walk_nums:
        print(f"Testing walk count: {num_walks:,}")

        # --- CPU - index based ---
        current_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=False, max_time_capacity=-1,
                enable_weight_computation=False, node_count_max_bound=nodes_count
            )
            trw.add_multiple_edges(edges)

            start = time.time()
            trw.get_random_walks(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialIndex",
                num_walks_total=num_walks,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start)
        avg_time = np.mean(current_times)
        print(f"  CPU (Index) walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times_index_based_cpu.append(avg_time)

        # --- CPU - weight based ---
        current_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=False, max_time_capacity=-1,
                enable_weight_computation=True, node_count_max_bound=nodes_count
            )
            trw.add_multiple_edges(edges)

            start = time.time()
            trw.get_random_walks(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialWeight",
                num_walks_total=num_walks,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start)
        avg_time = np.mean(current_times)
        print(f"  CPU (Weight) walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times_weight_based_cpu.append(avg_time)

        # --- GPU - index based ---
        current_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=True, max_time_capacity=-1,
                enable_weight_computation=False, node_count_max_bound=nodes_count
            )
            trw.add_multiple_edges(edges)

            start = time.time()
            trw.get_random_walks(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialIndex",
                num_walks_total=num_walks,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start)
        avg_time = np.mean(current_times)
        print(f"  GPU (Index) walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times_index_based_gpu.append(avg_time)

        # --- GPU - weight based ---
        current_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True, use_gpu=True, max_time_capacity=-1,
                enable_weight_computation=True, node_count_max_bound=nodes_count
            )
            trw.add_multiple_edges(edges)

            start = time.time()
            trw.get_random_walks(
                max_walk_len=max_walk_len,
                walk_bias="ExponentialWeight",
                num_walks_total=num_walks,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start)
        avg_time = np.mean(current_times)
        print(f"  GPU (Weight) walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times_weight_based_gpu.append(avg_time)

    return {
        "cpu_index_walk": walk_sampling_times_index_based_cpu,
        "cpu_weight_walk": walk_sampling_times_weight_based_cpu,
        "gpu_index_walk": walk_sampling_times_index_based_gpu,
        "gpu_weight_walk": walk_sampling_times_weight_based_gpu,
    }


def varying_max_walk_length_test(dataset):
    num_edges = 60_000_000
    walk_count = 100_000
    walk_lengths = list(range(10, 310, 10))

    walk_sampling_times_index_based_cpu = []
    walk_sampling_times_index_based_gpu = []

    edges = dataset[:num_edges]
    nodes_count = get_node_count(edges)

    for walk_len in walk_lengths:
        print(f"Testing walk length: {walk_len}")

        # --- CPU - Index based ---
        current_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True,
                use_gpu=False,
                max_time_capacity=-1,
                enable_weight_computation=False,
                node_count_max_bound=nodes_count
            )
            trw.add_multiple_edges(edges)

            start_time = time.time()
            trw.get_random_walks(
                max_walk_len=walk_len,
                walk_bias="ExponentialIndex",
                num_walks_total=walk_count,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start_time)

        avg_time = np.mean(current_times)
        print(f"  CPU (Index) walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times_index_based_cpu.append(avg_time)

        # --- GPU - Index based ---
        current_times = []
        for _ in range(N_RUNS):
            trw = TemporalRandomWalk(
                is_directed=True,
                use_gpu=True,
                max_time_capacity=-1,
                enable_weight_computation=False,
                node_count_max_bound=nodes_count
            )
            trw.add_multiple_edges(edges)

            start_time = time.time()
            trw.get_random_walks(
                max_walk_len=walk_len,
                walk_bias="ExponentialIndex",
                num_walks_total=walk_count,
                initial_edge_bias="Uniform",
                walk_direction="Forward_In_Time"
            )
            current_times.append(time.time() - start_time)

        avg_time = np.mean(current_times)
        print(f"  GPU (Index) walk sampling time: {avg_time:.3f} sec")
        walk_sampling_times_index_based_gpu.append(avg_time)

    return {
        "cpu_index_walk": walk_sampling_times_index_based_cpu,
        "gpu_index_walk": walk_sampling_times_index_based_gpu
    }


def main():
    dataset = read_temporal_edges("data/sx-stackoverflow.txt")
    print(f"Loaded {len(dataset):,} edges.")

    results_edges = progressive_higher_edge_addition_test(dataset)
    results_walks = progressively_higher_walk_sampling_test(dataset)
    result_max_walk_lens = varying_max_walk_length_test(dataset)

    print("Edge Addition Test (CPU & GPU):")
    for k, v in results_edges.items():
        print(f"{k}: {v}")

    print("\nWalk Sampling Test (CPU & GPU):")
    for k, v in results_walks.items():
        print(f"{k}: {v}")

    print("\nMax Walk Length Test (CPU & GPU):")
    for k, v in result_max_walk_lens.items():
        print(f"{k}: {v}")

    pickle.dump(results_edges, open("data/result_edges.pkl", "wb"))
    pickle.dump(results_walks, open("data/result_walks.pkl", "wb"))
    pickle.dump(result_max_walk_lens, open("data/result_max_walk_lens.pkl", "wb"))



if __name__ == '__main__':
    main()
