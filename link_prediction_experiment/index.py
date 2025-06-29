import argparse
import json
import logging
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from temporal_random_walk import TemporalRandomWalk
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@contextmanager
def suppress_word2vec_output():
    """Context manager to suppress Word2Vec verbose output."""
    # Save current logging levels
    gensim_logger = logging.getLogger('gensim')
    word2vec_logger = logging.getLogger('gensim.models.word2vec')
    kv_logger = logging.getLogger('gensim.models.keyedvectors')

    original_gensim_level = gensim_logger.level
    original_word2vec_level = word2vec_logger.level
    original_kv_level = kv_logger.level

    try:
        # Suppress logging
        gensim_logger.setLevel(logging.ERROR)
        word2vec_logger.setLevel(logging.ERROR)
        kv_logger.setLevel(logging.ERROR)

        # Also suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        # Restore original levels
        gensim_logger.setLevel(original_gensim_level)
        word2vec_logger.setLevel(original_word2vec_level)
        kv_logger.setLevel(original_kv_level)


def split_dataset(data_file_path, data_format):
    """Split dataset following TGB's 70/15/15 methodology."""
    logger.info(f"Loading dataset from {data_file_path}")

    if data_format == 'parquet':
        df = pd.read_parquet(data_file_path)
    else:
        df = pd.read_csv(data_file_path)

    timestamps = df['ts']
    unique_timestamps = timestamps.unique()
    logger.info(f"Dataset contains {len(df)} edges with {len(unique_timestamps)} unique timestamps")

    train_split_idx = int(len(unique_timestamps) * 0.70)
    val_split_idx = int(len(unique_timestamps) * 0.85)  # 70% + 15%

    # Get boundary timestamps
    train_end_timestamp = unique_timestamps[train_split_idx - 1] if train_split_idx > 0 else unique_timestamps[0]
    val_end_timestamp = unique_timestamps[val_split_idx - 1] if val_split_idx < len(unique_timestamps) else \
    unique_timestamps[-1]

    # Create masks for each split
    train_mask = timestamps <= train_end_timestamp
    val_mask = (timestamps > train_end_timestamp) & (timestamps <= val_end_timestamp)
    test_mask = timestamps > val_end_timestamp

    # Create datasets
    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    logger.info(f"Train set: {len(train_df)} edges ({len(train_df) / len(df) * 100:.1f}%)")
    logger.info(f"Val set: {len(val_df)} edges ({len(val_df) / len(df) * 100:.1f}%)")
    logger.info(f"Test set: {len(test_df)} edges ({len(test_df) / len(df) * 100:.1f}%)")

    return train_df, val_df, test_df


def sample_negative_edges(train_sources, train_targets, all_nodes, num_negative_samples, seed=42):
    np.random.seed(seed)

    logger.info(f"Sampling {num_negative_samples:,} negative edges from {len(all_nodes):,} nodes")
    logger.info(f"Avoiding {len(train_sources):,} training edges")

    # Convert training edges to set (avoid only these)
    existing_pairs = set(zip(train_sources, train_targets))
    logger.info("Created training edge lookup structure")

    # Generate negative edges in batches
    batch_size = 10_000_000
    negative_edges = []
    attempts = 0

    while len(negative_edges) < num_negative_samples:
        attempts += 1
        remaining = num_negative_samples - len(negative_edges)
        current_batch = min(batch_size, remaining * 5)

        # Generate candidate edges from all nodes
        u = np.random.choice(all_nodes, current_batch, replace=True)
        v = np.random.choice(all_nodes, current_batch, replace=True)

        # Remove self-loops
        valid_mask = u != v
        u_valid, v_valid = u[valid_mask], v[valid_mask]

        # Create candidate pairs
        candidate_tuples = set(zip(u_valid, v_valid))

        # Find negatives that don't exist in training set
        new_negatives = candidate_tuples - existing_pairs
        negative_edges.extend(list(new_negatives))

        progress = len(negative_edges) / num_negative_samples * 100
        logger.info(f"Attempt {attempts}: Found {len(negative_edges):,}/{num_negative_samples:,} ({progress:.1f}%)")

    # Take exactly what we need
    final_edges = list(negative_edges)[:num_negative_samples]
    neg_array = np.array(final_edges)

    logger.info(f"Successfully sampled {len(final_edges):,} negative edges in {attempts} attempts")
    return neg_array[:, 0], neg_array[:, 1]


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""

    def __init__(self, patience=3, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def create_link_prediction_model(input_dim, device='cpu'):
    """Create neural network model for link prediction."""
    hidden_dim1 = max(64, input_dim // 2)
    hidden_dim2 = max(32, input_dim // 4)

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim1),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim1, hidden_dim2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim2, 16),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(16, 1)
    ).to(device)

    return model


def train_link_prediction_model(model, X_train, y_train, X_val, y_val,
                                batch_size=1_000_000, learning_rate=0.001,
                                epochs=20, device='cpu', patience=10, use_amp=True):
    """Train link prediction neural network model."""
    logger.info(f"Training neural network on {len(X_train):,} samples with batch size {batch_size:,}")

    use_amp = use_amp and device == 'cuda'
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=patience)

    # Initialize mixed precision scaler
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed precision training enabled")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

    # Create dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    logger.info(f'Starting training for {epochs} epochs...')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        all_train_preds = []
        all_train_targets = []

        for batch_X, batch_y in train_dataloader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            # Collect predictions for AUC
            with torch.no_grad():
                train_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_train_preds.extend(train_probs)
                all_train_targets.extend(batch_y.cpu().numpy().flatten())

            # Clean up GPU memory
            del batch_X, batch_y, outputs, loss
            if device == 'cuda':
                torch.cuda.empty_cache()

        avg_train_loss = train_loss / train_batches
        train_auc = roc_auc_score(all_train_targets, all_train_preds) if len(set(all_train_targets)) > 1 else 0.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                val_batches += 1

                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_val_preds.extend(probs)
                all_val_targets.extend(batch_y.cpu().numpy().flatten())

                del batch_X, batch_y, outputs, loss
                if device == 'cuda':
                    torch.cuda.empty_cache()

        avg_val_loss = val_loss / val_batches
        val_auc = roc_auc_score(all_val_targets, all_val_preds) if len(set(all_val_targets)) > 1 else 0.0

        # Store metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        # Log progress
        logger.info(f"Epoch {epoch + 1:3d}/{epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                    f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping
        if early_stopping(avg_val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info("Training completed")
    return history


def predict_with_model(model, X_test, batch_size=1_000_000, device='cpu', use_amp=True):
    """Make predictions using trained model."""
    model.eval()
    predictions = []
    use_amp = use_amp and device == 'cuda'

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            end_idx = min(i + batch_size, len(X_test))
            batch_X = torch.FloatTensor(X_test[i:end_idx]).to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    batch_logits = model(batch_X)
                    batch_pred = torch.sigmoid(batch_logits).cpu().numpy().flatten()
            else:
                batch_logits = model(batch_X)
                batch_pred = torch.sigmoid(batch_logits).cpu().numpy().flatten()

            predictions.extend(batch_pred)

            del batch_X, batch_logits
            if device == 'cuda':
                torch.cuda.empty_cache()

    return np.array(predictions)


def create_edge_features(sources, targets, node_embeddings, edge_op):
    """Create edge features from node embeddings."""
    logger.info(f"Creating {edge_op} edge features for {len(sources):,} edges")

    # Get embedding dimension
    embedding_dim = len(next(iter(node_embeddings.values())))
    edge_features = np.zeros((len(sources), embedding_dim), dtype=np.float32)

    for i, (src, tgt) in enumerate(zip(sources, targets)):
        src_emb = node_embeddings.get(src, np.zeros(embedding_dim))
        tgt_emb = node_embeddings.get(tgt, np.zeros(embedding_dim))

        if edge_op == 'average':
            edge_emb = (src_emb + tgt_emb) / 2
        elif edge_op == 'hadamard':
            edge_emb = src_emb * tgt_emb
        elif edge_op == 'weighted-l1':
            edge_emb = np.abs(src_emb - tgt_emb)
        elif edge_op == 'weighted-l2':
            edge_emb = (src_emb - tgt_emb) ** 2
        else:
            raise ValueError(f"Unknown edge_op: {edge_op}")

        edge_features[i] = edge_emb

    logger.info("Edge feature creation completed")
    return edge_features


def compute_mrr(positive_scores, negative_scores):
    """
    Compute Mean Reciprocal Rank (MRR) for link prediction.

    Args:
        positive_scores: Array of prediction scores for positive (true) edges
        negative_scores: Array of prediction scores for negative (false) edges

    Returns:
        float: MRR value
    """
    reciprocal_ranks = []

    for pos_score in positive_scores:
        # Count how many negative scores are higher than this positive score
        rank = 1 + np.sum(negative_scores > pos_score)
        reciprocal_ranks.append(1.0 / rank)

    return np.mean(reciprocal_ranks)


def evaluate_link_prediction(test_sources, test_targets,
                                       negative_sources, negative_targets,
                                       node_embeddings, edge_op,
                                       classifier_train_ratio, n_epochs, device):
    logger.info("Starting link prediction evaluation")

    # Create edge features for positive and negative edges
    pos_features = create_edge_features(test_sources, test_targets, node_embeddings, edge_op)
    neg_features = create_edge_features(negative_sources, negative_targets, node_embeddings, edge_op)

    # Combine features and labels
    all_features = np.concatenate([pos_features, neg_features])
    all_labels = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))])

    # Shuffle the data
    indices = np.random.permutation(len(all_features))
    all_features = all_features[indices]
    all_labels = all_labels[indices]

    logger.info(f"Total samples for classification: {len(all_features):,}")

    # Split for classifier training (this is separate from temporal split)
    train_size = int(len(all_features) * classifier_train_ratio)

    X_classifier_train = all_features[:train_size]
    y_classifier_train = all_labels[:train_size]
    X_classifier_test = all_features[train_size:]
    y_classifier_test = all_labels[train_size:]

    # Further split training data for validation
    val_size = int(len(X_classifier_train) * 0.15)
    X_train = X_classifier_train[val_size:]
    y_train = y_classifier_train[val_size:]
    X_val = X_classifier_train[:val_size]
    y_val = y_classifier_train[:val_size]

    logger.info(f"Classifier train: {len(X_train):,}, val: {len(X_val):,}, test: {len(X_classifier_test):,}")

    # Create and train model
    input_dim = all_features.shape[1]
    model = create_link_prediction_model(input_dim, device)

    history = train_link_prediction_model(
        model, X_train, y_train, X_val, y_val,
        epochs=n_epochs, device=device, patience=10
    )

    # Make predictions
    logger.info("Making final predictions...")
    y_pred_proba = predict_with_model(model, X_classifier_test, device=device)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate standard metrics
    auc = roc_auc_score(y_classifier_test, y_pred_proba)
    accuracy = accuracy_score(y_classifier_test, y_pred)
    precision = precision_score(y_classifier_test, y_pred, zero_division=0)
    recall = recall_score(y_classifier_test, y_pred, zero_division=0)
    f1 = f1_score(y_classifier_test, y_pred, zero_division=0)

    logger.info("Computing MRR...")

    # Get scores for all positive and negative edges (not just test split)
    all_pos_features = create_edge_features(test_sources, test_targets, node_embeddings, edge_op)
    all_neg_features = create_edge_features(negative_sources, negative_targets, node_embeddings, edge_op)

    positive_scores = predict_with_model(model, all_pos_features, device=device)
    negative_scores = predict_with_model(model, all_neg_features, device=device)

    # Compute MRR
    mrr = compute_mrr(positive_scores, negative_scores)

    results = {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mrr': mrr,
        'training_history': history
    }

    logger.info(f"Link prediction completed - AUC: {auc:.4f}, MRR: {mrr:.4f}")
    return results


def train_embeddings_full_approach(train_sources, train_targets, train_timestamps,
                                   is_directed, walk_length, num_walks_per_node,
                                   embedding_dim, walk_use_gpu, word2vec_n_workers, seed=42):
    """Train embeddings using full dataset approach."""
    logger.info("Training embeddings with full approach")

    temporal_random_walk = TemporalRandomWalk(is_directed=is_directed, use_gpu=walk_use_gpu, max_time_capacity=-1)
    temporal_random_walk.add_multiple_edges(train_sources, train_targets, train_timestamps)

    # Generate walks
    walks, timestamps, walk_lengths = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        max_walk_len=walk_length,
        num_walks_per_node=num_walks_per_node,
        walk_bias='ExponentialIndex',
        initial_edge_bias='Uniform',
        walk_direction="Backward_In_Time"
    )

    logger.info(f'Generated {len(walks)} walks. Mean length: {np.mean(walk_lengths):.2f}')

    # Clean walks
    clean_walks = []
    for walk, length in zip(walks, walk_lengths):
        clean_walk = [str(node) for node in walk[:length]]
        if len(clean_walk) > 1:
            clean_walks.append(clean_walk)

    # Train Word2Vec
    logger.info(f"Training Word2Vec on {len(clean_walks)} walks")
    with suppress_word2vec_output():
        model = Word2Vec(
            sentences=clean_walks,
            vector_size=embedding_dim,
            window=10,
            min_count=1,
            workers=word2vec_n_workers,
            sg=1,
            seed=seed
        )

    # Extract embeddings
    node_embeddings = {}
    for node in model.wv.index_to_key:
        node_embeddings[int(node)] = model.wv[node]

    logger.info(f'Trained embeddings for {len(node_embeddings)} nodes')
    return node_embeddings


def train_embeddings_streaming_approach(train_sources, train_targets, train_timestamps,
                                        batch_ts_size, sliding_window_duration, weighted_sum_alpha,
                                        is_directed, walk_length, num_walks_per_node,
                                        embedding_dim, walk_use_gpu, word2vec_n_workers, seed=42):
    """Train embeddings using streaming window approach."""
    logger.info("Training embeddings with streaming approach")

    temporal_random_walk = TemporalRandomWalk(is_directed=is_directed, use_gpu=walk_use_gpu,
                                              max_time_capacity=sliding_window_duration)
    global_embeddings = {}

    # Create time-based batches
    min_timestamp = np.min(train_timestamps)
    max_timestamp = np.max(train_timestamps)
    total_time_range = max_timestamp - min_timestamp
    num_batches = int(np.ceil(total_time_range / batch_ts_size))

    logger.info(f"Processing {num_batches} batches with duration={batch_ts_size:,}")

    for batch_idx in range(num_batches):
        batch_start_ts = min_timestamp + (batch_idx * batch_ts_size)
        batch_end_ts = min_timestamp + ((batch_idx + 1) * batch_ts_size)

        if batch_idx == num_batches - 1:
            batch_end_ts = max_timestamp + 1

        # Filter batch edges
        batch_mask = (train_timestamps >= batch_start_ts) & (train_timestamps < batch_end_ts)
        batch_sources = train_sources[batch_mask]
        batch_targets = train_targets[batch_mask]
        batch_ts = train_timestamps[batch_mask]

        if len(batch_sources) == 0:
            continue

        logger.info(f"Batch {batch_idx + 1}/{num_batches}: {len(batch_sources):,} edges")

        # Add edges to temporal random walk
        temporal_random_walk.add_multiple_edges(batch_sources, batch_targets, batch_ts)

        # Generate walks
        walks, _, walk_lengths = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
            max_walk_len=walk_length,
            num_walks_per_node=num_walks_per_node,
            walk_bias='ExponentialIndex',
            initial_edge_bias='Uniform',
            walk_direction="Backward_In_Time"
        )

        # Clean walks
        clean_walks = []
        for walk, length in zip(walks, walk_lengths):
            clean_walk = [str(node) for node in walk[:length]]
            if len(clean_walk) > 1:
                clean_walks.append(clean_walk)

        # Train Word2Vec on batch
        try:
            with suppress_word2vec_output():
                batch_model = Word2Vec(
                    sentences=clean_walks,
                    vector_size=embedding_dim,
                    window=10,
                    min_count=1,
                    workers=word2vec_n_workers,
                    sg=1,
                    seed=seed
                )

            # Extract batch embeddings
            batch_embeddings = {}
            for node in batch_model.wv.index_to_key:
                batch_embeddings[int(node)] = batch_model.wv[node]

            # Merge with global embeddings
            if not global_embeddings:
                global_embeddings = batch_embeddings.copy()
            else:
                for node_id, batch_embedding in batch_embeddings.items():
                    if node_id in global_embeddings:
                        global_embeddings[node_id] = (
                                weighted_sum_alpha * global_embeddings[node_id] +
                                (1 - weighted_sum_alpha) * batch_embedding
                        )
                    else:
                        global_embeddings[node_id] = batch_embedding

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {e}")
            continue

    logger.info(f"Streaming completed. Final embedding store: {len(global_embeddings)} nodes")
    return global_embeddings


def run_single_experiment(embeddings, test_sources, test_targets,
                          negative_sources, negative_targets,
                          edge_op, classifier_train_ratio, n_epochs, device):
    """Run a single link prediction experiment."""
    return evaluate_link_prediction(
        test_sources, test_targets,
        negative_sources, negative_targets,
        embeddings, edge_op,
        classifier_train_ratio, n_epochs, device
    )


def run_link_prediction_experiments(data_file_path, data_format, is_directed,
                                    batch_ts_size, sliding_window_duration, weighted_sum_alpha,
                                    walk_length, num_walks_per_node, embedding_dim, edge_op,
                                    classifier_train_ratio, n_epochs,
                                    full_embedding_use_gpu, incremental_embedding_use_gpu,
                                    link_prediction_use_gpu, n_runs, word2vec_n_workers, output_path=None):
    logger.info("Starting link prediction experiments")

    train_df, val_df, test_df = split_dataset(data_file_path, data_format)

    # Convert to arrays
    train_sources = train_df['u'].to_numpy()
    train_targets = train_df['i'].to_numpy()
    train_timestamps = train_df['ts'].to_numpy()
    test_sources = test_df['u'].to_numpy()
    test_targets = test_df['i'].to_numpy()

    # Get all nodes for negative sampling
    train_nodes = set(train_sources).union(set(train_targets))
    test_nodes = set(test_sources).union(set(test_targets))
    all_nodes = np.array(list(train_nodes.union(test_nodes)))

    logger.info(f"Train nodes: {len(train_nodes):,}, Test nodes: {len(test_nodes):,}")
    logger.info(f"Total nodes: {len(all_nodes):,}")

    # Sample negative edges
    negative_sources, negative_targets = sample_negative_edges(
        train_sources, train_targets, all_nodes, len(test_sources)
    )

    device = 'cuda' if link_prediction_use_gpu and torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    logger.info("=" * 60)
    logger.info("TRAINING EMBEDDINGS - STREAMING APPROACH")
    logger.info("=" * 60)

    streaming_embeddings = train_embeddings_streaming_approach(
        train_sources, train_targets, train_timestamps,
        batch_ts_size, sliding_window_duration, weighted_sum_alpha,
        is_directed, walk_length, num_walks_per_node,
        embedding_dim, incremental_embedding_use_gpu, word2vec_n_workers
    )

    streaming_results = {
        'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
        'mrr': [], 'training_history': []
    }

    for run in range(n_runs):
        logger.info(f"\n--- Run {run + 1}/{n_runs} ---")

        streaming_result = run_single_experiment(
            streaming_embeddings, test_sources, test_targets,
            negative_sources, negative_targets,
            edge_op, classifier_train_ratio, n_epochs, device
        )

        for key in streaming_results.keys():
            streaming_results[key].append(streaming_result[key])

    logger.info(f"\nStreaming Approach Results:")
    logger.info(f"AUC: {np.mean(streaming_results['auc']):.4f} ± {np.std(streaming_results['auc']):.4f}")
    logger.info(f"MRR: {np.mean(streaming_results['mrr']):.4f} ± {np.std(streaming_results['mrr']):.4f}")
    logger.info(f"Accuracy: {np.mean(streaming_results['accuracy']):.4f} ± {np.std(streaming_results['accuracy']):.4f}")
    logger.info(
        f"Precision: {np.mean(streaming_results['precision']):.4f} ± {np.std(streaming_results['precision']):.4f}")
    logger.info(f"Recall: {np.mean(streaming_results['recall']):.4f} ± {np.std(streaming_results['recall']):.4f}")
    logger.info(f"F1-Score: {np.mean(streaming_results['f1_score']):.4f} ± {np.std(streaming_results['f1_score']):.4f}")


    logger.info("=" * 60)
    logger.info("TRAINING EMBEDDINGS - FULL APPROACH")
    logger.info("=" * 60)

    full_embeddings = train_embeddings_full_approach(
        train_sources, train_targets, train_timestamps,
        is_directed, walk_length, num_walks_per_node,
        embedding_dim, full_embedding_use_gpu, word2vec_n_workers
    )

    full_results = {
        'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
        'mrr': [], 'training_history': []
    }


    for run in range(n_runs):
        logger.info(f"\n--- Run {run + 1}/{n_runs} ---")

        full_result = run_single_experiment(
            full_embeddings, test_sources, test_targets,
            negative_sources, negative_targets,
            edge_op, classifier_train_ratio, n_epochs, device
        )

        for key in full_results.keys():
            full_results[key].append(full_result[key])

    logger.info(f"\nFull Approach Results:")
    logger.info(f"AUC: {np.mean(full_results['auc']):.4f} ± {np.std(full_results['auc']):.4f}")
    logger.info(f"MRR: {np.mean(full_results['mrr']):.4f} ± {np.std(full_results['mrr']):.4f}")
    logger.info(f"Accuracy: {np.mean(full_results['accuracy']):.4f} ± {np.std(full_results['accuracy']):.4f}")
    logger.info(f"Precision: {np.mean(full_results['precision']):.4f} ± {np.std(full_results['precision']):.4f}")
    logger.info(f"Recall: {np.mean(full_results['recall']):.4f} ± {np.std(full_results['recall']):.4f}")
    logger.info(f"F1-Score: {np.mean(full_results['f1_score']):.4f} ± {np.std(full_results['f1_score']):.4f}")


    # Comparison
    auc_diff = np.mean(streaming_results['auc']) - np.mean(full_results['auc'])
    mrr_diff = np.mean(streaming_results['mrr']) - np.mean(full_results['mrr'])
    logger.info(f"\nComparison:")
    logger.info(f"AUC Difference (Streaming - Full): {auc_diff:+.4f}")
    logger.info(f"MRR Difference (Streaming - Full): {mrr_diff:+.4f}")

    # Save results
    if output_path:
        results = {
            'full_approach': full_results,
            'streaming_approach': streaming_results,
            'comparison': {
                'auc_difference': auc_diff,
                'mrr_difference': mrr_diff
            }
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        logger.info(f"Results saved to {output_path}")

    return full_results, streaming_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal Link Prediction")

    # Required arguments
    parser.add_argument('--data_file_path', type=str, required=True,
                        help='Path to data file (parquet or csv format)')
    parser.add_argument('--batch_ts_size', type=int, required=True,
                        help='Time duration per batch for streaming approach')
    parser.add_argument('--sliding_window_duration', type=int, required=True,
                        help='Sliding window duration for temporal random walk')
    parser.add_argument('--is_directed', type=lambda x: x.lower() == 'true', required=True,
                        help='Whether the graph is directed (true/false)')

    # Model parameters
    parser.add_argument('--weighted_sum_alpha', type=float, default=0.5,
                        help='Alpha parameter for weighted sum in streaming approach')
    parser.add_argument('--walk_length', type=int, default=80,
                        help='Maximum length of random walks')
    parser.add_argument('--num_walks_per_node', type=int, default=10,
                        help='Number of walks to generate per node')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimensionality of node embeddings')
    parser.add_argument('--edge_op', type=str, default='hadamard',
                        choices=['average', 'hadamard', 'weighted-l1', 'weighted-l2'],
                        help='Edge operation for combining node embeddings')

    # Training parameters
    parser.add_argument('--classifier_train_ratio', type=float, default=0.75,
                        help='Ratio of data used for training link prediction classifier')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs for neural network training')
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of experimental runs for averaging results')

    # GPU settings
    parser.add_argument('--full_embedding_use_gpu', action='store_true',
                        help='Enable GPU acceleration for full embedding approach')
    parser.add_argument('--incremental_embedding_use_gpu', action='store_true',
                        help='Enable GPU acceleration for streaming embedding approach')
    parser.add_argument('--link_prediction_use_gpu', action='store_true',
                        help='Enable GPU acceleration for link prediction neural network')

    # Other settings
    parser.add_argument('--data_format', type=str, default='parquet',
                        choices=['parquet', 'csv'],
                        help='Data file format')
    parser.add_argument('--word2vec_n_workers', type=int, default=8,
                        help='Number of workers for Word2Vec training')
    parser.add_argument('--output_path', type=str, default=None,
                        help='File path to save results (optional)')

    args = parser.parse_args()

    # Run experiments
    full_results, streaming_results = run_link_prediction_experiments(
        data_file_path=args.data_file_path,
        data_format=args.data_format,
        is_directed=args.is_directed,
        batch_ts_size=args.batch_ts_size,
        sliding_window_duration=args.sliding_window_duration,
        weighted_sum_alpha=args.weighted_sum_alpha,
        walk_length=args.walk_length,
        num_walks_per_node=args.num_walks_per_node,
        embedding_dim=args.embedding_dim,
        edge_op=args.edge_op,
        classifier_train_ratio=args.classifier_train_ratio,
        n_epochs=args.n_epochs,
        full_embedding_use_gpu=args.full_embedding_use_gpu,
        incremental_embedding_use_gpu=args.incremental_embedding_use_gpu,
        link_prediction_use_gpu=args.link_prediction_use_gpu,
        n_runs=args.n_runs,
        word2vec_n_workers=args.word2vec_n_workers,
        output_path=args.output_path
    )
