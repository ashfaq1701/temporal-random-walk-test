import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import logging
import warnings
from contextlib import contextmanager

from temporal_random_walk import TemporalRandomWalk
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset


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


def split_dataset(data_file_path):
    """
    Split dataset by sorted unique timestamps (TGB-style):
    - 70% training (timestamps ≤ t70)
    - 15% validation (t70 < timestamps ≤ t85)
    - 15% test (timestamps > t85)
    """
    logger.info(f"Loading dataset from {data_file_path}")

    if data_file_path.endswith('.parquet'):
        df = pd.read_parquet(data_file_path)
    else:
        df = pd.read_csv(data_file_path)

    # Sort by timestamp to ensure chronological order
    df = df.sort_values('ts').reset_index(drop=True)

    timestamps = df['ts'].values
    unique_ts = np.sort(np.unique(timestamps))
    logger.info(f"Dataset contains {len(df):,} edges with {len(unique_ts):,} unique timestamps")

    # TGB-style splitting by unique timestamps
    t70 = unique_ts[int(len(unique_ts) * 0.70)]
    t85 = unique_ts[int(len(unique_ts) * 0.85)]

    logger.info(f"Timestamp thresholds: t70={t70}, t85={t85}")

    # Create datasets
    train_df = df[df['ts'] <= t70]
    valid_df = df[(df['ts'] > t70) & (df['ts'] <= t85)]
    test_df  = df[df['ts'] > t85]

    n_total = len(df)
    logger.info(f"Train: {len(train_df):,} edges ({len(train_df) / n_total * 100:.1f}%)")
    logger.info(f"Valid: {len(valid_df):,} edges ({len(valid_df) / n_total * 100:.1f}%)")
    logger.info(f"Test:  {len(test_df):,} edges ({len(test_df) / n_total * 100:.1f}%)")

    # Log timestamp ranges
    logger.info(f"Actual timestamp ranges:")
    logger.info(f"  Train: {train_df['ts'].min()} to {train_df['ts'].max()}")
    logger.info(f"  Valid: {valid_df['ts'].min()} to {valid_df['ts'].max()}")
    logger.info(f"  Test:  {test_df['ts'].min()} to {test_df['ts'].max()}")

    return train_df, valid_df, test_df


def create_dataset_with_negative_edges(ds_sources, ds_targets,
                                       sources_to_exclude, targets_to_exclude,
                                       is_directed, neg_ratio=1.0, random_state=42):
    """
    Create a dataset combining positive and negative edges.

    Args:
        ds_sources (array-like): Source nodes for positive edges
        ds_targets (array-like): Target nodes for positive edges
        sources_to_exclude (array-like): Source nodes to exclude for negatives
        targets_to_exclude (array-like): Target nodes to exclude for negatives
        is_directed (bool): Whether the graph is directed
        neg_ratio (float): Number of negative samples per positive edge
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (all_sources, all_targets, labels) where labels = 1 for positive, 0 for negative
    """
    np.random.seed(random_state)
    random.seed(random_state)

    num_positive = len(ds_sources)
    num_negative = int(num_positive * neg_ratio)
    logger.info(f"Creating dataset with {num_positive:,} positive edges and {num_negative:,} negatives")

    # Get all unique nodes
    all_nodes = list(set(ds_sources).union(ds_targets, sources_to_exclude, targets_to_exclude))

    # Create set of edges to exclude
    edges_to_exclude = set(zip(sources_to_exclude, targets_to_exclude))
    if not is_directed:
        edges_to_exclude |= set(zip(targets_to_exclude, sources_to_exclude))  # Add reverse direction

    logger.info(f"Sampling negatives from {len(all_nodes):,} nodes")
    logger.info(f"Total excluded edge pairs: {len(edges_to_exclude):,}")

    # Generate negative edges
    batch_size = 10_000_000
    negative_edges = []
    attempts = 0

    while len(negative_edges) < num_negative:
        attempts += 1
        remaining = num_negative - len(negative_edges)
        current_batch = min(batch_size, remaining * 5)

        u = np.random.choice(all_nodes, current_batch, replace=True)
        v = np.random.choice(all_nodes, current_batch, replace=True)

        mask = u != v
        u_valid, v_valid = u[mask], v[mask]
        candidate_pairs = set(zip(u_valid, v_valid))

        new_negatives = candidate_pairs - edges_to_exclude
        negative_edges.extend(list(new_negatives))

        logger.info(f"Attempt {attempts}: Collected {len(negative_edges):,}/{num_negative:,} negatives")

    if len(negative_edges) < num_negative:
        logger.warning("Not enough valid negatives found. Returning fewer samples.")

    # Final trim
    negative_edges = list(negative_edges)[:num_negative]
    neg_sources, neg_targets = zip(*negative_edges)

    # Create labels
    pos_labels = np.ones(num_positive, dtype=np.int32)
    neg_labels = np.zeros(num_negative, dtype=np.int32)

    all_sources = np.concatenate([ds_sources, neg_sources])
    all_targets = np.concatenate([ds_targets, neg_targets])
    all_labels  = np.concatenate([pos_labels, neg_labels])

    # Shuffle dataset
    indices = np.random.permutation(len(all_sources))
    all_sources = all_sources[indices]
    all_targets = all_targets[indices]
    all_labels  = all_labels[indices]

    logger.info(f"Final dataset: {len(all_sources):,} edges (positive + negative)")

    return all_sources, all_targets, all_labels


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""

    def __init__(self, patience=5, min_delta=0.0001, restore_best_weights=True):
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
                                epochs=20, device='cpu', patience=5, use_amp=True):
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



def evaluate_link_prediction(
        train_sources, train_targets,
        valid_sources, valid_targets,
        test_sources, test_targets,
        node_embeddings, edge_op, is_directed,
        n_epochs, device):
    train_sources_combined, train_targets_combined, train_labels_combined = create_dataset_with_negative_edges(
        train_sources,
        train_targets,
        train_sources,
        train_targets,
        is_directed
    )

    valid_sources_combined, valid_targets_combined, valid_labels_combined = create_dataset_with_negative_edges(
        valid_sources,
        valid_targets,
        train_sources,
        train_targets,
        is_directed
    )

    test_sources_combined, test_targets_combined, test_labels_combined = create_dataset_with_negative_edges(
        test_sources,
        test_targets,
        train_sources,
        train_targets,
        is_directed
    )

    train_features = create_edge_features(train_sources_combined, train_targets_combined, node_embeddings, edge_op)
    valid_features = create_edge_features(valid_sources_combined, valid_targets_combined, node_embeddings, edge_op)
    test_features = create_edge_features(test_sources_combined, test_targets_combined, node_embeddings, edge_op)

    logger.info(
        f"Classifier train: {len(train_sources_combined):,}, val: {len(valid_sources_combined):,}, test: {len(test_sources_combined):,}")

    input_dim = train_features.shape[1]
    model = create_link_prediction_model(input_dim, device)

    history = train_link_prediction_model(
        model, train_features, train_labels_combined,
        valid_features, valid_labels_combined,
        epochs=n_epochs, device=device, patience=5
    )

    # Make predictions
    logger.info("Making final predictions...")
    val_pred_proba = predict_with_model(model, valid_features, device=device)
    test_pred_proba = predict_with_model(model, test_features, device=device)

    test_pred = (test_pred_proba > 0.5).astype(int)

    # Calculate standard metrics for test set
    test_auc = roc_auc_score(test_labels_combined, test_pred_proba)
    test_accuracy = accuracy_score(test_labels_combined, test_pred)
    test_precision = precision_score(test_labels_combined, test_pred, zero_division=0)
    test_recall = recall_score(test_labels_combined, test_pred, zero_division=0)
    test_f1 = f1_score(test_labels_combined, test_pred, zero_division=0)

    # Calculate MRR for validation and test sets
    # Split predictions by positive/negative labels
    val_pos_scores = val_pred_proba[valid_labels_combined == 1]
    val_neg_scores = val_pred_proba[valid_labels_combined == 0]
    val_mrr = compute_mrr(val_pos_scores, val_neg_scores)

    test_pos_scores = test_pred_proba[test_labels_combined == 1]
    test_neg_scores = test_pred_proba[test_labels_combined == 0]
    test_mrr = compute_mrr(test_pos_scores, test_neg_scores)

    results = {
        'auc': test_auc,
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'val_mrr': val_mrr,
        'test_mrr': test_mrr,
        'training_history': history
    }

    logger.info(f"Link prediction completed - AUC: {test_auc:.4f}, Val MRR: {val_mrr:.4f}, Test MRR: {test_mrr:.4f}")
    return results


def run_link_prediction_experiments(
        data_file_path,
        is_directed,
        walk_length,
        num_walks_per_node,
        embedding_dim,
        edge_op,
        n_epochs,
        full_embedding_use_gpu,
        link_prediction_use_gpu,
        n_runs,
        word2vec_n_workers,
        output_path
):
    logger.info("Starting link prediction experiments")

    train_df, valid_df, test_df = split_dataset(data_file_path)

    train_sources = train_df['u'].to_numpy()
    train_targets = train_df['i'].to_numpy()
    train_timestamps = train_df['ts'].to_numpy()

    val_sources = valid_df['u'].to_numpy()
    val_targets = valid_df['i'].to_numpy()

    test_sources = test_df['u'].to_numpy()
    test_targets = test_df['i'].to_numpy()

    device = 'cuda' if link_prediction_use_gpu and torch.cuda.is_available() else 'cpu'

    logger.info("=" * 60)
    logger.info("TRAINING EMBEDDINGS - FULL APPROACH")
    logger.info("=" * 60)

    full_embeddings = train_embeddings_full_approach(
        train_sources, train_targets, train_timestamps,
        is_directed, walk_length, num_walks_per_node,
        embedding_dim, full_embedding_use_gpu, word2vec_n_workers
    )

    full_results = {
        'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'training_history': []
    }

    for run in range(n_runs):
        logger.info(f"\n--- Run {run + 1}/{n_runs} ---")

        current_full_results = evaluate_link_prediction(
            train_sources, train_targets,
            val_sources, val_targets,
            test_sources, test_targets,
            full_embeddings, edge_op,
            is_directed, n_epochs, device
        )

        for key in current_full_results.keys():
            full_results[key].append(current_full_results[key])

    logger.info(f"\nFull Approach Results:")
    logger.info(f"AUC: {np.mean(full_results['auc']):.4f} ± {np.std(full_results['auc']):.4f}")
    logger.info(f"Accuracy: {np.mean(full_results['accuracy']):.4f} ± {np.std(full_results['accuracy']):.4f}")
    logger.info(f"Precision: {np.mean(full_results['precision']):.4f} ± {np.std(full_results['precision']):.4f}")
    logger.info(f"Recall: {np.mean(full_results['recall']):.4f} ± {np.std(full_results['recall']):.4f}")
    logger.info(f"F1-Score: {np.mean(full_results['f1_score']):.4f} ± {np.std(full_results['f1_score']):.4f}")
    logger.info(f"MRR (Test): {np.mean(full_results['test_mrr']):.4f} ± {np.std(full_results['test_mrr']):.4f}")
    logger.info(f"MRR (Validation): {np.mean(full_results['val_mrr']):.4f} ± {np.std(full_results['val_mrr']):.4f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        logger.info(f"Results saved to {output_path}")

    return full_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal Link Prediction")

    # Required arguments
    parser.add_argument('--data_file_path', type=str, required=True,
                        help='Path to data file (parquet or csv format)')
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
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs for neural network training')
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of experimental runs for averaging results')

    # GPU settings
    parser.add_argument('--full_embedding_use_gpu', action='store_true',
                        help='Enable GPU acceleration for full embedding approach')
    parser.add_argument('--link_prediction_use_gpu', action='store_true',
                        help='Enable GPU acceleration for link prediction neural network')

    # Other settings
    parser.add_argument('--word2vec_n_workers', type=int, default=8,
                        help='Number of workers for Word2Vec training')
    parser.add_argument('--output_path', type=str, default=None,
                        help='File path to save results (optional)')

    args = parser.parse_args()

    # Run experiments
    full_results = run_link_prediction_experiments(
        data_file_path=args.data_file_path,
        is_directed=args.is_directed,
        walk_length=args.walk_length,
        num_walks_per_node=args.num_walks_per_node,
        embedding_dim=args.embedding_dim,
        edge_op=args.edge_op,
        n_epochs=args.n_epochs,
        full_embedding_use_gpu=args.full_embedding_use_gpu,
        link_prediction_use_gpu=args.link_prediction_use_gpu,
        n_runs=args.n_runs,
        word2vec_n_workers=args.word2vec_n_workers,
        output_path=args.output_path
    )
