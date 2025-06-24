import argparse
import time
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import logging
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from temporal_random_walk import TemporalRandomWalk

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_dataset(data_file_path, train_percentage):
    """Split dataset based on temporal ordering."""
    logger.info(f"Loading dataset from {data_file_path}")
    df = pd.read_parquet(data_file_path)
    timestamps = df['ts']

    # Get unique timestamps (already sorted)
    unique_timestamps = timestamps.unique()
    logger.info(f"Dataset contains {len(df)} edges with {len(unique_timestamps)} unique timestamps")

    # Calculate split point based on train_percentage
    split_idx = int(len(unique_timestamps) * train_percentage)

    # Get training and testing timestamps
    train_timestamps = unique_timestamps[:split_idx]

    # Get the last training timestamp
    last_train_timestamp = train_timestamps[-1]

    # Split dataset at the last occurrence of the training timestamp
    train_mask = timestamps <= last_train_timestamp
    test_mask = timestamps > last_train_timestamp

    # Create training and testing datasets
    train_df = df[train_mask]
    test_df = df[test_mask]

    logger.info(f"Train set: {len(train_df)} edges, Test set: {len(test_df)} edges")
    return train_df, test_df


def sample_negative_edges(test_sources, test_targets, num_negative_samples=None, seed=42):
    np.random.seed(seed)

    existing_edges = set(zip(test_sources, test_targets))
    all_nodes = np.unique(np.concatenate([test_sources, test_targets]))

    if num_negative_samples is None:
        num_negative_samples = len(test_sources)

    negative_edges = set()
    max_attempts = 500

    logger.info(f"Sampling {num_negative_samples} negative edges from {len(all_nodes)} nodes")

    for attempt in range(max_attempts):
        if len(negative_edges) >= num_negative_samples:
            break

        # Only generate what we need + small buffer
        remaining = num_negative_samples - len(negative_edges)
        batch_size = min(remaining * 2, 100_000)

        u = np.random.choice(all_nodes, batch_size)
        v = np.random.choice(all_nodes, batch_size)

        # Check edges one by one with early stopping
        for i in range(len(u)):
            if len(negative_edges) >= num_negative_samples:
                break
            if u[i] != v[i] and (u[i], v[i]) not in existing_edges:
                negative_edges.add((u[i], v[i]))

    if len(negative_edges) < num_negative_samples:
        logger.warning(f"Only generated {len(negative_edges)} negative samples out of {num_negative_samples} requested")

    # Trim to exact count
    neg_list = list(negative_edges)[:num_negative_samples]
    negative_sources = pd.Series([e[0] for e in neg_list])
    negative_targets = pd.Series([e[1] for e in neg_list])

    return negative_sources, negative_targets


def evaluate_link_prediction(
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        node_embeddings,
        link_prediction_training_percentage
):
    """Evaluate link prediction using Hadamard product and logistic regression."""
    logger.info("Starting link prediction evaluation")

    # Combine positive and negative edges
    all_sources = np.concatenate([test_sources, negative_sources])
    all_targets = np.concatenate([test_targets, negative_targets])

    # Create labels (1 for positive edges, 0 for negative edges)
    labels = np.concatenate([
        np.ones(len(test_sources)),  # Positive edges
        np.zeros(len(negative_sources))  # Negative edges
    ])

    # Create edge features using Hadamard product (element-wise multiplication)
    edge_features = []
    missing_embeddings = 0

    for src, tgt in zip(all_sources, all_targets):
        if src in node_embeddings and tgt in node_embeddings:
            # Hadamard product of source and target embeddings
            src_embedding = node_embeddings[src]
            tgt_embedding = node_embeddings[tgt]
            hadamard_feature = src_embedding * tgt_embedding
            edge_features.append(hadamard_feature)
        else:
            # Handle missing embeddings - use zero vector as fallback
            if missing_embeddings < 10:  # Only log first 10 warnings
                logger.warning(f"Missing embedding for edge ({src}, {tgt})")
            missing_embeddings += 1

            # Use zero vector as fallback
            embedding_dim = len(next(iter(node_embeddings.values())))
            hadamard_feature = np.zeros(embedding_dim)
            edge_features.append(hadamard_feature)

    if missing_embeddings > 0:
        logger.warning(f"Total missing embeddings: {missing_embeddings}")

    edge_features = np.array(edge_features)

    # Split into train/test for evaluation
    test_size = 1.0 - link_prediction_training_percentage
    X_train, X_test, y_train, y_test = train_test_split(
        edge_features, labels, test_size=test_size, random_state=42, stratify=labels
    )

    logger.info(f"Training classifier on {len(X_train)} samples, testing on {len(X_test)} samples")

    # Train logistic regression classifier
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]  # Probability of positive class
    y_pred = classifier.predict(X_test)

    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results = {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'num_positive_edges': len(test_sources),
        'num_negative_edges': len(negative_sources),
        'num_missing_embeddings': missing_embeddings,
        'embedding_coverage': 1.0 - (missing_embeddings / len(all_sources))
    }

    logger.info(f"Link prediction completed - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
    return results


def run_link_prediction_full_data(
        train_sources,
        train_targets,
        train_timestamps,
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        is_directed,
        walk_length,
        num_walks_per_node,
        embedding_dim,
        link_prediction_training_percentage,
        use_gpu,
        seed=42
):
    """Run link prediction using full dataset approach."""
    logger.info("Starting full data link prediction")

    temporal_random_walk = TemporalRandomWalk(is_directed=is_directed, use_gpu=use_gpu, max_time_capacity=-1)

    logger.info(f'Adding {len(train_sources)} edges in temporal random walk instance')
    temporal_random_walk.add_multiple_edges(train_sources, train_targets, train_timestamps)

    batch_walk_start_time = time.time()
    walks, timestamps, walk_lengths = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        max_walk_len=walk_length,
        num_walks_per_node=num_walks_per_node,
        walk_bias='ExponentialIndex',
        initial_edge_bias='Uniform',
        walk_direction="Backward_In_Time"
    )
    batch_walk_duration = time.time() - batch_walk_start_time

    logger.info(f'Generated {len(walks)} walks in {batch_walk_duration:.2f} seconds')

    # Remove padding from walks using actual walk lengths
    clean_walks = []
    for walk, length in zip(walks, walk_lengths):
        # Convert to strings (Word2Vec expects strings)
        clean_walk = [str(node) for node in walk[:length]]
        if len(clean_walk) > 1:  # Skip single-node walks
            clean_walks.append(clean_walk)

    logger.info(f"Training Word2Vec on {len(clean_walks)} clean walks")

    batch_node_embedding_start_time = time.time()
    # Train Word2Vec model
    model = Word2Vec(
        sentences=clean_walks,
        vector_size=embedding_dim,
        window=10,
        min_count=1,
        workers=4,
        sg=1,
        seed=seed
    )

    node_embeddings = {}
    for node in model.wv.index_to_key:
        node_embeddings[int(node)] = model.wv[node]
    batch_node_embedding_duration = time.time() - batch_node_embedding_start_time

    logger.info(f'Trained embeddings for {len(node_embeddings)} nodes in {batch_node_embedding_duration:.2f} seconds')

    return evaluate_link_prediction(
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        node_embeddings,
        link_prediction_training_percentage
    )


def run_link_prediction_streaming_window(
        train_sources,
        train_targets,
        train_timestamps,
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        is_directed,
        walk_length,
        num_walks_per_node,
        batch_ts_size,
        sliding_window_duration,
        weighted_sum_alpha,
        embedding_dim,
        link_prediction_training_percentage,
        use_gpu,
        seed=42
):
    """Run link prediction using streaming window approach."""
    logger.info("Starting streaming window link prediction")

    temporal_random_walk = TemporalRandomWalk(is_directed=is_directed, use_gpu=use_gpu,
                                              max_time_capacity=sliding_window_duration)

    # Global embedding store (dictionary)
    global_embeddings = {}

    # Create batches by batch_ts_size
    unique_timestamps = np.sort(train_timestamps.unique())
    num_batches = len(unique_timestamps) // batch_ts_size

    logger.info(f"Processing {num_batches} batches with batch_ts_size={batch_ts_size}")

    total_start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start_time = time.time()

        # Get timestamp range for current batch
        start_ts_idx = batch_idx * batch_ts_size
        end_ts_idx = min((batch_idx + 1) * batch_ts_size, len(unique_timestamps))
        batch_timestamps = unique_timestamps[start_ts_idx:end_ts_idx]

        # Filter edges for current batch
        batch_mask = train_timestamps.isin(batch_timestamps)
        batch_sources = train_sources[batch_mask]
        batch_targets = train_targets[batch_mask]
        batch_ts = train_timestamps[batch_mask]

        logger.info(f"Batch {batch_idx + 1}/{num_batches}: {len(batch_sources)} edges")

        # Add batch to temporal_random_walk
        logger.info(f'Adding {len(batch_sources)} edges in temporal random walk instance')
        temporal_random_walk.add_multiple_edges(batch_sources, batch_targets, batch_ts)

        # Get walks from the instance
        walks, timestamps, walk_lengths = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
            max_walk_len=walk_length,
            num_walks_per_node=num_walks_per_node,
            walk_bias='ExponentialIndex',
            initial_edge_bias='Uniform',
            walk_direction="Backward_In_Time"
        )

        # Clean walks (remove padding)
        clean_walks = []
        for walk, length in zip(walks, walk_lengths):
            clean_walk = [str(node) for node in walk[:length]]
            if len(clean_walk) > 1:  # Skip single-node walks
                clean_walks.append(clean_walk)

        # Train Word2Vec on current batch
        try:
            batch_model = Word2Vec(
                sentences=clean_walks,
                vector_size=embedding_dim,
                window=10,
                min_count=1,
                workers=4,
                sg=1,
                seed=seed
            )

            # Extract batch embeddings
            batch_embeddings = {}
            for node in batch_model.wv.index_to_key:
                batch_embeddings[int(node)] = batch_model.wv[node]

            # Merge with global embedding store using weighted sum
            if not global_embeddings:
                # First batch - initialize global embeddings
                global_embeddings = batch_embeddings.copy()
            else:
                # Merge embeddings using weighted sum
                for node_id, batch_embedding in batch_embeddings.items():
                    if node_id in global_embeddings:
                        # Weighted sum: alpha * old + (1-alpha) * new
                        global_embeddings[node_id] = (
                                weighted_sum_alpha * global_embeddings[node_id] +
                                (1 - weighted_sum_alpha) * batch_embedding
                        )
                    else:
                        # New node - add directly
                        global_embeddings[node_id] = batch_embedding

            batch_duration = time.time() - batch_start_time
            logger.info(f"Batch {batch_idx + 1} completed in {batch_duration:.2f}s. "
                        f"Total nodes in global store: {len(global_embeddings)}")

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {e}")
            continue

    total_duration = time.time() - total_start_time
    logger.info(f"Streaming window processing completed in {total_duration:.2f}s. "
                f"Final embedding store size: {len(global_embeddings)}")

    return evaluate_link_prediction(
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        global_embeddings,
        link_prediction_training_percentage
    )


def save_results(results, output_path):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def run_link_prediction_experiments(
        data_file_path,
        is_directed,
        batch_ts_size,
        sliding_window_duration,
        weighted_sum_alpha,
        walk_length,
        num_walks_per_node,
        embedding_dim,
        embedding_training_percentage,
        link_prediction_training_percentage,
        full_embedding_use_gpu,
        incremental_embedding_use_gpu,
        output_dir=None
):
    """Run both full and streaming link prediction experiments."""
    logger.info("Starting link prediction experiments")

    # Split dataset
    train_df, test_df = split_dataset(data_file_path, embedding_training_percentage)
    train_sources, train_targets, train_timestamps = train_df['u'], train_df['i'], train_df['ts']
    test_sources, test_targets, test_timestamps = test_df['u'], test_df['i'], test_df['ts']

    # Sample negative edges
    negative_sources, negative_targets = sample_negative_edges(test_sources, test_targets)

    # Run full data approach
    logger.info("=" * 50)
    logger.info("FULL DATA APPROACH")
    logger.info("=" * 50)

    full_start_time = time.time()
    full_link_prediction_results = run_link_prediction_full_data(
        train_sources,
        train_targets,
        train_timestamps,
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        is_directed,
        walk_length,
        num_walks_per_node,
        embedding_dim,
        link_prediction_training_percentage,
        full_embedding_use_gpu
    )
    full_duration = time.time() - full_start_time
    full_link_prediction_results['total_time'] = full_duration

    print(f"\nFull Link Prediction Results:")
    print(f"AUC: {full_link_prediction_results['auc']:.4f}")
    print(f"Accuracy: {full_link_prediction_results['accuracy']:.4f}")
    print(f"Precision: {full_link_prediction_results['precision']:.4f}")
    print(f"Recall: {full_link_prediction_results['recall']:.4f}")
    print(f"F1-Score: {full_link_prediction_results['f1_score']:.4f}")
    print(f"Embedding Coverage: {full_link_prediction_results['embedding_coverage']:.4f}")
    print(f"Total Time: {full_duration:.2f}s")

    # Run streaming approach
    logger.info("=" * 50)
    logger.info("STREAMING WINDOW APPROACH")
    logger.info("=" * 50)

    streaming_start_time = time.time()
    streaming_link_prediction_results = run_link_prediction_streaming_window(
        train_sources,
        train_targets,
        train_timestamps,
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        is_directed,
        walk_length,
        num_walks_per_node,
        batch_ts_size,
        sliding_window_duration,
        weighted_sum_alpha,
        embedding_dim,
        link_prediction_training_percentage,
        incremental_embedding_use_gpu
    )
    streaming_duration = time.time() - streaming_start_time
    streaming_link_prediction_results['total_time'] = streaming_duration

    print(f"\nStreaming Link Prediction Results:")
    print(f"AUC: {streaming_link_prediction_results['auc']:.4f}")
    print(f"Accuracy: {streaming_link_prediction_results['accuracy']:.4f}")
    print(f"Precision: {streaming_link_prediction_results['precision']:.4f}")
    print(f"Recall: {streaming_link_prediction_results['recall']:.4f}")
    print(f"F1-Score: {streaming_link_prediction_results['f1_score']:.4f}")
    print(f"Embedding Coverage: {streaming_link_prediction_results['embedding_coverage']:.4f}")
    print(f"Total Time: {streaming_duration:.2f}s")

    # Comparison
    print(f"\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    auc_diff = streaming_link_prediction_results['auc'] - full_link_prediction_results['auc']
    time_ratio = streaming_duration / full_duration
    print(f"AUC Difference (Streaming - Full): {auc_diff:+.4f}")
    print(f"Time Ratio (Streaming / Full): {time_ratio:.2f}x")

    # Save results if output directory specified
    if output_dir:
        results = {
            'full_approach': full_link_prediction_results,
            'streaming_approach': streaming_link_prediction_results,
            'comparison': {
                'auc_difference': auc_diff,
                'time_ratio': time_ratio
            },
            'parameters': {
                'data_file_path': data_file_path,
                'is_directed': is_directed,
                'batch_ts_size': batch_ts_size,
                'sliding_window_duration': sliding_window_duration,
                'weighted_sum_alpha': weighted_sum_alpha,
                'walk_length': walk_length,
                'num_walks_per_node': num_walks_per_node,
                'embedding_dim': embedding_dim,
                'embedding_training_percentage': embedding_training_percentage,
                'link_prediction_training_percentage': link_prediction_training_percentage
            }
        }

        output_path = Path(output_dir) / "link_prediction_results.json"
        save_results(results, output_path)

    return full_link_prediction_results, streaming_link_prediction_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal Link Prediction Test")
    parser.add_argument(
        '--data_file_path', type=str, required=True,
        help='Path to data file (parquet format)'
    )
    parser.add_argument('--batch_ts_size', type=int, required=True, help='Number of timestamps per batch')
    parser.add_argument('--sliding_window_duration', type=int, required=True, help='Sliding window duration for temporal random walk')
    parser.add_argument('--is_directed', type=lambda x: x.lower() == 'true', required=True, help='Whether the graph is directed')

    parser.add_argument('--weighted_sum_alpha', type=float, default=0.5,
                        help='Alpha parameter for weighted sum in streaming approach')
    parser.add_argument('--embedding_training_percentage', type=float, default=0.75,
                        help='Percentage of data used for training embeddings')
    parser.add_argument('--link_prediction_training_percentage', type=float, default=0.75,
                        help='Percentage of link prediction data used for training classifier')
    parser.add_argument('--walk_length', type=int, default=80,
                        help='Maximum length of random walks')
    parser.add_argument('--num_walks_per_node', type=int, default=10,
                        help='Number of walks to generate per node')

    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimensionality of node embeddings')

    parser.add_argument('--full_embedding_use_gpu', action='store_true',
                        help='Enable GPU acceleration for full embedding')
    parser.add_argument('--incremental_embedding_use_gpu', action='store_true',
                        help='Enable GPU acceleration for incremental embedding')

    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (optional)')

    args = parser.parse_args()

    run_link_prediction_experiments(
        args.data_file_path,
        args.is_directed,
        args.batch_ts_size,
        args.sliding_window_duration,
        args.weighted_sum_alpha,
        args.walk_length,
        args.num_walks_per_node,
        args.embedding_dim,
        args.embedding_training_percentage,
        args.link_prediction_training_percentage,
        args.full_embedding_use_gpu,
        args.incremental_embedding_use_gpu,
        args.output_dir
    )
