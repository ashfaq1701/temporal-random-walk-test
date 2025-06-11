#!/usr/bin/env python3
"""
Temporal Graph Data Structure Builder (Simplified for Animation)

This script builds the core data structures needed for temporal graph animation.
All arrays are Python lists for easy visualization and manipulation.
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Edge:
    """Represents a temporal edge"""
    source: int
    target: int
    timestamp: int


class EdgeDataStore:
    """Simplified EdgeDataStore with only core arrays"""

    def __init__(self):
        # Core edge data
        self.sources: List[int] = []
        self.targets: List[int] = []
        self.timestamps: List[int] = []

        # Active nodes
        self.active_node_ids: List[int] = []

        # Timestamp grouping
        self.timestamp_group_offsets: List[int] = []
        self.unique_timestamps: List[int] = []

    def add_edges(self, edges: List[Edge]):
        """Add edges to the store"""
        for edge in edges:
            self.sources.append(edge.source)
            self.targets.append(edge.target)
            self.timestamps.append(edge.timestamp)

    def sort_by_timestamp(self):
        """Sort all edges by timestamp"""
        # Create indices for sorting
        indices = list(range(len(self.timestamps)))
        indices.sort(key=lambda i: self.timestamps[i])

        # Reorder all arrays
        self.sources = [self.sources[i] for i in indices]
        self.targets = [self.targets[i] for i in indices]
        self.timestamps = [self.timestamps[i] for i in indices]

    def populate_active_nodes(self):
        """Find all unique node IDs"""
        active_set = set()
        for i in range(len(self.sources)):
            active_set.add(self.sources[i])
            active_set.add(self.targets[i])

        self.active_node_ids = sorted(list(active_set))

    def update_timestamp_groups(self):
        """Group edges by timestamp and compute offsets"""
        if not self.timestamps:
            self.timestamp_group_offsets = []
            self.unique_timestamps = []
            return

        self.unique_timestamps = []
        self.timestamp_group_offsets = [0]

        current_ts = self.timestamps[0]
        self.unique_timestamps.append(current_ts)

        for i, ts in enumerate(self.timestamps[1:], 1):
            if ts != current_ts:
                self.timestamp_group_offsets.append(i)
                self.unique_timestamps.append(ts)
                current_ts = ts

        # Final offset (total number of edges)
        self.timestamp_group_offsets.append(len(self.timestamps))


class NodeEdgeIndexStore:
    """Simplified NodeEdgeIndexStore with only core arrays"""

    def __init__(self, max_node_id: int, is_directed: bool = True):
        self.max_node_id = max_node_id
        self.is_directed = is_directed

        # Node edge offsets (CSR-style)
        self.node_group_outbound_offsets: List[int] = []
        self.node_group_inbound_offsets: List[int] = [] if is_directed else None

        # Timestamp-sorted edge indices per node
        self.node_ts_sorted_outbound_indices: List[int] = []
        self.node_ts_sorted_inbound_indices: List[int] = [] if is_directed else None

        # Timestamp group counts per node
        self.count_ts_group_per_node_outbound: List[int] = []
        self.count_ts_group_per_node_inbound: List[int] = [] if is_directed else None

        # Timestamp group offsets per node
        self.node_ts_group_outbound_offsets: List[int] = []
        self.node_ts_group_inbound_offsets: List[int] = [] if is_directed else None

    def compute_node_group_offsets(self, edge_data: EdgeDataStore):
        """Compute edge count offsets for each node"""
        # Initialize offsets arrays
        self.node_group_outbound_offsets = [0] * (self.max_node_id + 2)
        if self.is_directed:
            self.node_group_inbound_offsets = [0] * (self.max_node_id + 2)

        # Count edges per node
        for i in range(len(edge_data.sources)):
            src = edge_data.sources[i]
            tgt = edge_data.targets[i]

            if src <= self.max_node_id:
                self.node_group_outbound_offsets[src + 1] += 1

            if self.is_directed:
                if tgt <= self.max_node_id:
                    self.node_group_inbound_offsets[tgt + 1] += 1
            else:
                if tgt <= self.max_node_id:
                    self.node_group_outbound_offsets[tgt + 1] += 1

        # Convert counts to offsets (prefix sum)
        for i in range(self.max_node_id + 1):
            self.node_group_outbound_offsets[i + 1] += self.node_group_outbound_offsets[i]
            if self.is_directed:
                self.node_group_inbound_offsets[i + 1] += self.node_group_inbound_offsets[i]

    def compute_node_ts_sorted_indices(self, edge_data: EdgeDataStore):
        """Sort edge indices by node and timestamp"""
        num_edges = len(edge_data.sources)

        # Create (node_id, timestamp, edge_idx) tuples for outbound edges
        outbound_tuples = []
        for i in range(num_edges):
            src = edge_data.sources[i]
            tgt = edge_data.targets[i]
            ts = edge_data.timestamps[i]

            outbound_tuples.append((src, ts, i))
            if not self.is_directed:
                outbound_tuples.append((tgt, ts, i))

        # Sort by node_id first, then by timestamp
        outbound_tuples.sort(key=lambda x: (x[0], x[1]))
        self.node_ts_sorted_outbound_indices = [t[2] for t in outbound_tuples]

        # Handle inbound edges for directed graphs
        if self.is_directed:
            inbound_tuples = []
            for i in range(num_edges):
                tgt = edge_data.targets[i]
                ts = edge_data.timestamps[i]
                inbound_tuples.append((tgt, ts, i))

            # Sort by node_id first, then by timestamp
            inbound_tuples.sort(key=lambda x: (x[0], x[1]))
            self.node_ts_sorted_inbound_indices = [t[2] for t in inbound_tuples]

    def compute_node_timestamp_groups(self, edge_data: EdgeDataStore):
        """Compute timestamp groups per node"""
        # Initialize count arrays
        self.count_ts_group_per_node_outbound = [0] * (self.max_node_id + 2)
        if self.is_directed:
            self.count_ts_group_per_node_inbound = [0] * (self.max_node_id + 2)

        # Process outbound edges
        self._compute_timestamp_groups_outbound(edge_data)

        # Process inbound edges if directed
        if self.is_directed:
            self._compute_timestamp_groups_inbound(edge_data)

    def _compute_timestamp_groups_outbound(self, edge_data: EdgeDataStore):
        """Compute timestamp groups for outbound edges"""
        group_starts = []
        node_group_counts = [0] * (self.max_node_id + 1)

        prev_node = -1
        prev_timestamp = -1

        for pos, edge_idx in enumerate(self.node_ts_sorted_outbound_indices):
            # Determine which node this position belongs to
            current_node = self._get_node_from_position(pos, True)
            current_timestamp = edge_data.timestamps[edge_idx]

            # Check if this starts a new group (new node or new timestamp within same node)
            if (current_node != prev_node or current_timestamp != prev_timestamp):
                group_starts.append(pos)
                if 0 <= current_node <= self.max_node_id:
                    node_group_counts[current_node] += 1

            prev_node = current_node
            prev_timestamp = current_timestamp

        self.node_ts_group_outbound_offsets = group_starts

        # Convert group counts to offsets (prefix sum)
        for i in range(self.max_node_id + 1):
            self.count_ts_group_per_node_outbound[i + 1] = (
                self.count_ts_group_per_node_outbound[i] + node_group_counts[i]
            )

    def _compute_timestamp_groups_inbound(self, edge_data: EdgeDataStore):
        """Compute timestamp groups for inbound edges"""
        group_starts = []
        node_group_counts = [0] * (self.max_node_id + 1)

        prev_node = -1
        prev_timestamp = -1

        for pos, edge_idx in enumerate(self.node_ts_sorted_inbound_indices):
            current_node = edge_data.targets[edge_idx]
            current_timestamp = edge_data.timestamps[edge_idx]

            # Check if this starts a new group (new node or new timestamp within same node)
            if (current_node != prev_node or current_timestamp != prev_timestamp):
                group_starts.append(pos)
                if 0 <= current_node <= self.max_node_id:
                    node_group_counts[current_node] += 1

            prev_node = current_node
            prev_timestamp = current_timestamp

        self.node_ts_group_inbound_offsets = group_starts

        # Convert group counts to offsets (prefix sum)
        for i in range(self.max_node_id + 1):
            self.count_ts_group_per_node_inbound[i + 1] = (
                self.count_ts_group_per_node_inbound[i] + node_group_counts[i]
            )

    def _get_node_from_position(self, pos: int, is_outbound: bool) -> int:
        """Determine which node a position in the sorted indices belongs to"""
        if is_outbound:
            offsets = self.node_group_outbound_offsets
        else:
            offsets = self.node_group_inbound_offsets

        # Binary search to find which node this position belongs to
        for node_id in range(self.max_node_id + 1):
            if offsets[node_id] <= pos < offsets[node_id + 1]:
                return node_id
        return -1  # Should not happen


def build_temporal_graph(edges: List[Edge], is_directed: bool = True):
    """Build complete temporal graph data structures from edge stream"""

    print(f"Processing {len(edges)} temporal edges...")

    # Step 1: Build EdgeDataStore
    edge_data = EdgeDataStore()
    edge_data.add_edges(edges)
    edge_data.sort_by_timestamp()
    edge_data.populate_active_nodes()
    edge_data.update_timestamp_groups()

    print(f"Active nodes: {len(edge_data.active_node_ids)}")
    print(f"Unique timestamps: {len(edge_data.unique_timestamps)}")
    if edge_data.timestamps:
        print(f"Timestamp range: {min(edge_data.timestamps)} - {max(edge_data.timestamps)}")

    # Step 2: Build NodeEdgeIndexStore
    max_node_id = max(edge_data.active_node_ids) if edge_data.active_node_ids else 0
    node_index = NodeEdgeIndexStore(max_node_id, is_directed)

    node_index.compute_node_group_offsets(edge_data)
    node_index.compute_node_ts_sorted_indices(edge_data)
    node_index.compute_node_timestamp_groups(edge_data)

    return edge_data, node_index


def print_data_structure_info(edge_data: EdgeDataStore, node_index: NodeEdgeIndexStore):
    """Print information about the built data structures"""
    print("\n=== EdgeDataStore ===")
    print(f"sources: {edge_data.sources}")
    print(f"targets: {edge_data.targets}")
    print(f"timestamps: {edge_data.timestamps}")
    print(f"active_node_ids: {edge_data.active_node_ids}")
    print(f"unique_timestamps: {edge_data.unique_timestamps}")
    print(f"timestamp_group_offsets: {edge_data.timestamp_group_offsets}")

    print("\n=== NodeEdgeIndexStore ===")
    print(f"node_group_outbound_offsets: {node_index.node_group_outbound_offsets}")
    if node_index.is_directed:
        print(f"node_group_inbound_offsets: {node_index.node_group_inbound_offsets}")

    print(f"node_ts_sorted_outbound_indices: {node_index.node_ts_sorted_outbound_indices}")
    if node_index.is_directed:
        print(f"node_ts_sorted_inbound_indices: {node_index.node_ts_sorted_inbound_indices}")

    print(f"count_ts_group_per_node_outbound: {node_index.count_ts_group_per_node_outbound}")
    if node_index.is_directed:
        print(f"count_ts_group_per_node_inbound: {node_index.count_ts_group_per_node_inbound}")

    print(f"node_ts_group_outbound_offsets: {node_index.node_ts_group_outbound_offsets}")
    if node_index.is_directed:
        print(f"node_ts_group_inbound_offsets: {node_index.node_ts_group_inbound_offsets}")


# Example usage with the sample data from your earlier message
if __name__ == "__main__":
    # Create sample temporal edges
    sample_edges = [
        Edge(0, 1, 100), Edge(1, 2, 100), Edge(3, 0, 100), Edge(0, 2, 101),
        Edge(1, 3, 101), Edge(2, 4, 101), Edge(4, 0, 101), Edge(1, 4, 102),
        Edge(2, 3, 102), Edge(3, 1, 102), Edge(4, 2, 102), Edge(0, 3, 103),
        Edge(2, 0, 103), Edge(3, 4, 104), Edge(4, 1, 104), Edge(0, 4, 104),
        Edge(1, 0, 105), Edge(2, 1, 105), Edge(3, 2, 106), Edge(4, 3, 107)
    ]

    # Build data structures
    edge_data, node_index = build_temporal_graph(sample_edges, is_directed=False)

    # Print results
    print_data_structure_info(edge_data, node_index)
