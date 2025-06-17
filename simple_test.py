from temporal_random_walk import TemporalRandomWalk

# Create a directed temporal graph
walker = TemporalRandomWalk(is_directed=True, use_gpu=True, max_time_capacity=-1)

# Add edges - can be numpy arrays or python lists
sources = [3, 2, 0, 3, 3, 1]
targets = [4, 4, 2, 1, 2, 4]
timestamps = [71, 82, 19, 34, 79, 19]

walker.add_multiple_edges(sources, targets, timestamps)
# Sample walks with exponential time bias
walk_nodes, walk_timestamps, walk_lens = walker.get_random_walks_and_times_for_all_nodes(
    max_walk_len=5,
    walk_bias="ExponentialIndex",
    num_walks_per_node=10,
    initial_edge_bias="Uniform"
)
