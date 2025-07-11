import argparse
import logging
import os
import pickle
import random
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from temporal_random_walk import TemporalRandomWalk
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


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
                                       is_directed, k, random_state=42):
    np.random.seed(random_state)
    random.seed(random_state)

    num_positive = len(ds_sources)
    num_negative = int(num_positive * k)
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

        new_negatives = list(candidate_pairs - edges_to_exclude)

        needed = num_negative - len(negative_edges)
        negative_edges.extend(new_negatives[:needed])

        logger.info(f"Attempt {attempts}: Collected {len(negative_edges):,}/{num_negative:,} negatives")

    if len(negative_edges) < num_negative:
        logger.warning("Not enough valid negatives found. Returning fewer samples.")

    # Final trim
    neg_sources, neg_targets = zip(*negative_edges)
    neg_sources = np.array(neg_sources, dtype=np.int32)[:num_negative]
    neg_targets = np.array(neg_targets, dtype=np.int32)[:num_negative]
    ds_sources = np.array(ds_sources)
    ds_targets = np.array(ds_targets)

    # Reshape negatives into groups
    neg_sources_grouped = neg_sources.reshape(num_positive, k)
    neg_targets_grouped = neg_targets.reshape(num_positive, k)

    # Create positive arrays
    pos_sources = ds_sources.reshape(-1, 1)  # (num_positive, 1)
    pos_targets = ds_targets.reshape(-1, 1)  # (num_positive, 1)

    # Concatenate positives and negatives
    all_sources_grouped = np.concatenate([pos_sources, neg_sources_grouped], axis=1)  # (num_positive, k+1)
    all_targets_grouped = np.concatenate([pos_targets, neg_targets_grouped], axis=1)  # (num_positive, k+1)

    # Create labels (first column is positive)
    labels_grouped = np.zeros((num_positive, k + 1), dtype=np.int32)
    labels_grouped[:, 0] = 1  # First position is positive

    # Shuffle each row to randomize positive position
    for i in range(num_positive):
        perm = np.random.permutation(k + 1)
        all_sources_grouped[i] = all_sources_grouped[i, perm]
        all_targets_grouped[i] = all_targets_grouped[i, perm]
        labels_grouped[i] = labels_grouped[i, perm]

    # Flatten to final arrays
    all_sources = all_sources_grouped.flatten()
    all_targets = all_targets_grouped.flatten()
    all_labels = labels_grouped.flatten()

    logger.info(f"Final dataset: {len(all_sources):,} edges (positive + negative)")
    return all_sources, all_targets, all_labels


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""

    def __init__(self, mode='min', patience=5, min_delta=0.0001, restore_best_weights=True):
        """
        Args:
            mode (str): 'min' for loss, 'max' for metrics like AUC
            patience (int): Number of epochs to wait without improvement
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore best model weights on stop
        """
        assert mode in ['min', 'max']
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, current_score, model):
        improved = (current_score < self.best_score - self.min_delta) if self.mode == 'min' else (
                    current_score > self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
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


class LinkPredictionModel(nn.Module):
    def __init__(self, node_embeddings_tensor: torch.Tensor, edge_op: str):
        super().__init__()

        self.edge_op = edge_op

        self.embedding_lookup = nn.Embedding.from_pretrained(node_embeddings_tensor, freeze=True)

        input_dim = node_embeddings_tensor.shape[1]
        hidden_dim1 = max(64, input_dim // 2)
        hidden_dim2 = max(32, input_dim // 4)

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_dim2, 16)
        self.dropout3 = nn.Dropout(0.1)

        self.fc_out = nn.Linear(16, 1)

    def forward(self, upstream_nodes, downstream_nodes):
        device = next(self.parameters()).device
        upstream_nodes = upstream_nodes.to(device)
        downstream_nodes = downstream_nodes.to(device)

        upstream_emb = self.embedding_lookup(upstream_nodes)
        downstream_emb = self.embedding_lookup(downstream_nodes)

        if self.edge_op == 'average':
            edge_features = (upstream_emb + downstream_emb) / 2
        elif self.edge_op == 'hadamard':
            edge_features = upstream_emb * downstream_emb
        elif self.edge_op == 'weighted-l1':
            edge_features = torch.abs(upstream_emb - downstream_emb)
        elif self.edge_op == 'weighted-l2':
            edge_features = (upstream_emb - downstream_emb) ** 2
        else:
            raise ValueError(f"Unknown edge_op: {self.edge_op}")

        x = F.relu(self.fc1(edge_features))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        return self.fc_out(x)


def train_link_prediction_model(model,
                                X_sources_train, X_targets_train, y_train,
                                X_sources_val, X_targets_val, y_val,
                                batch_size,
                                learning_rate=0.001,
                                epochs=20,
                                device='cpu',
                                patience=5,
                                use_amp=True):
    logger.info(f"Training neural network on {len(X_sources_train):,} samples with batch size {batch_size:,}")

    # Move model to device
    model = model.to(device)
    use_amp = use_amp and device == 'cuda'

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(mode='max', patience=patience)

    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed precision training enabled")

    X_sources_train_tensor = torch.tensor(X_sources_train, dtype=torch.long)
    X_targets_train_tensor = torch.tensor(X_targets_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_sources_val_tensor = torch.tensor(X_sources_val, dtype=torch.long)
    X_targets_val_tensor = torch.tensor(X_targets_val, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    num_workers = min(8, os.cpu_count())

    train_loader = DataLoader(
        TensorDataset(X_sources_train_tensor, X_targets_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device == 'cuda'),
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4
    )

    # Use larger batch size for validation to reduce overhead
    val_batch_size = batch_size * 4
    val_loader = DataLoader(
        TensorDataset(X_sources_val_tensor, X_targets_val_tensor, y_val_tensor),
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=(device == 'cuda'),
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4
    )

    logger.info(f"Training: {len(train_loader)} batches of {batch_size:,}")
    logger.info(f"Validation: {len(val_loader)} batches of {val_batch_size:,}")

    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        total_train_loss = 0.0
        train_preds_list = []
        train_targets_list = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Train", leave=False, unit="batch")

        for batch_sources, batch_targets, batch_y in train_pbar:
            batch_sources = batch_sources.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(batch_sources, batch_targets)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_sources, batch_targets)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()

            train_preds_list.append(torch.sigmoid(outputs).detach().float())
            train_targets_list.append(batch_y.float())

            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Val", leave=False, unit="batch")

        with torch.no_grad():
            for batch_sources, batch_targets, batch_y in val_pbar:
                batch_sources = batch_sources.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                if use_amp:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(batch_sources, batch_targets)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = model(batch_sources, batch_targets)
                    loss = criterion(outputs, batch_y)

                total_val_loss += loss.item()

                val_preds_list.append(torch.sigmoid(outputs).detach().float())
                val_targets_list.append(batch_y.float())

                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = total_val_loss / len(val_loader)

        all_train_preds = torch.cat(train_preds_list, dim=0).cpu().numpy().flatten()
        all_train_targets = torch.cat(train_targets_list, dim=0).cpu().numpy().flatten()

        all_val_preds = torch.cat(val_preds_list, dim=0).cpu().numpy().flatten()
        all_val_targets = torch.cat(val_targets_list, dim=0).cpu().numpy().flatten()

        train_auc = roc_auc_score(all_train_targets, all_train_preds)
        val_auc = roc_auc_score(all_val_targets, all_val_preds)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'train_auc': f'{train_auc:.4f}',
            'val_auc': f'{val_auc:.4f}'
        })

        logger.info(f"Epoch {epoch + 1}/{epochs} — Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        if early_stopping(val_auc, model):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

        del train_preds_list, train_targets_list, val_preds_list, val_targets_list
        del all_train_preds, all_train_targets, all_val_preds, all_val_targets
        if device == 'cuda':
            torch.cuda.empty_cache()

    epoch_pbar.close()
    logger.info("Training completed")
    return history


def predict_with_model(model,
                       X_sources_test,
                       X_targets_test,
                       batch_size,
                       device='cpu',
                       use_amp=True):
    model = model.to(device)
    model.eval()
    use_amp = use_amp and device == 'cuda'

    logger.info(f"Making predictions on {len(X_sources_test):,} samples with batch size {batch_size:,}")

    X_sources_tensor = torch.tensor(X_sources_test, dtype=torch.long)
    X_targets_tensor = torch.tensor(X_targets_test, dtype=torch.long)

    num_workers = min(8, os.cpu_count())

    test_loader = DataLoader(
        TensorDataset(X_sources_tensor, X_targets_tensor),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device == 'cuda'),
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4
    )

    logger.info(f"Processing {len(test_loader)} batches")

    predictions_list = []

    prediction_pbar = tqdm(test_loader, desc="Predicting", unit="batch")

    with torch.no_grad():
        for batch_sources, batch_targets in prediction_pbar:
            batch_sources = batch_sources.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(batch_sources, batch_targets)
            else:
                logits = model(batch_sources, batch_targets)

            probs = torch.sigmoid(logits).detach().float()
            predictions_list.append(probs)

            prediction_pbar.set_postfix({
                'batches_processed': len(predictions_list),
                'batch_size': len(batch_sources)
            })

    prediction_pbar.close()

    logger.info("Transferring predictions to CPU...")
    all_predictions = torch.cat(predictions_list, dim=0).cpu().numpy().flatten()

    del predictions_list
    if device == 'cuda':
        torch.cuda.empty_cache()

    logger.info("Prediction completed")
    return all_predictions


def train_embeddings_full_approach(train_sources, train_targets, train_timestamps,
                                   is_directed, walk_length, num_walks_per_node, edge_picker,
                                   embedding_dim, walk_use_gpu, word2vec_n_workers, seed=42):
    """Train embeddings using full dataset approach."""
    logger.info("Training embeddings with full approach")

    temporal_random_walk = TemporalRandomWalk(is_directed=is_directed, use_gpu=walk_use_gpu, max_time_capacity=-1)
    temporal_random_walk.add_multiple_edges(train_sources, train_targets, train_timestamps)

    logger.info(
        f'Generating forward {num_walks_per_node // 2} walks per node with max length {walk_length} using {edge_picker} picker.')

    # Generate walks
    walks_forward, _, walk_lengths_forward = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        max_walk_len=walk_length,
        num_walks_per_node=num_walks_per_node // 2,
        walk_bias=edge_picker,
        initial_edge_bias='Uniform',
        walk_direction="Forward_In_Time"
    )

    logger.info(
        f'Generated {len(walk_lengths_forward)} forward walks. Mean length: {np.mean(walk_lengths_forward):.2f}')

    logger.info(
        f'Generating backward {num_walks_per_node // 2} walks per node with max length {walk_length} using {edge_picker} picker.')

    walks_backward, _, walk_lengths_backward = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        max_walk_len=walk_length,
        num_walks_per_node=num_walks_per_node // 2,
        walk_bias=edge_picker,
        initial_edge_bias='Uniform',
        walk_direction="Backward_In_Time"
    )

    logger.info(
        f'Generated {len(walk_lengths_backward)} backward walks. Mean length: {np.mean(walk_lengths_backward):.2f}')

    walks = np.concatenate([walks_forward, walks_backward], axis=0)
    walk_lengths = np.concatenate([walk_lengths_forward, walk_lengths_backward], axis=0)

    logger.info(f'Generated {len(walks)} walks in total. Mean length: {np.mean(walk_lengths):.2f}')

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
                                        is_directed, walk_length, num_walks_per_node, edge_picker,
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

        logger.info(
            f'Generating forward {num_walks_per_node // 2} walks per node with max length {walk_length} using {edge_picker} picker.')

        # Generate walks
        walks_forward, _, walk_lengths_forward = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
            max_walk_len=walk_length,
            num_walks_per_node=num_walks_per_node // 2,
            walk_bias=edge_picker,
            initial_edge_bias='Uniform',
            walk_direction="Forward_In_Time"
        )

        logger.info(
            f'Generated {len(walk_lengths_forward)} forward walks. Mean length: {np.mean(walk_lengths_forward):.2f}')

        logger.info(
            f'Generating backward {num_walks_per_node // 2} walks per node with max length {walk_length} using {edge_picker} picker.')

        walks_backward, _, walk_lengths_backward = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
            max_walk_len=walk_length,
            num_walks_per_node=num_walks_per_node // 2,
            walk_bias=edge_picker,
            initial_edge_bias='Uniform',
            walk_direction="Backward_In_Time"
        )

        logger.info(
            f'Generated {len(walk_lengths_backward)} backward walks. Mean length: {np.mean(walk_lengths_backward):.2f}')

        walks = np.concatenate([walks_forward, walks_backward], axis=0)
        walk_lengths = np.concatenate([walk_lengths_forward, walk_lengths_backward], axis=0)

        logger.info(f'Generated {len(walks)} walks in total. Mean length: {np.mean(walk_lengths):.2f}')

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


def get_embedding_tensor(embedding_dict, max_node_id):
    embedding_dim = len(next(iter(embedding_dict.values())))
    embedding_matrix = torch.zeros((max_node_id + 1, embedding_dim), dtype=torch.float32)

    nodes_filled = 0
    for node_id, embedding in embedding_dict.items():
        if node_id <= max_node_id:
            embedding_matrix[node_id] = torch.tensor(embedding, dtype=torch.float32)
            nodes_filled += 1

    return embedding_matrix


def compute_mrr(pred_proba, labels, negative_edges_per_positive):
    labels = np.array(labels)
    pred_proba = np.array(pred_proba)

    y_pred_pos = pred_proba[labels == 1]  # shape (N,)
    y_pred_neg = pred_proba[labels == 0].reshape(-1, negative_edges_per_positive)  # shape (N, k)

    y_pred_pos = y_pred_pos.reshape(-1, 1)  # shape (N, 1)

    optimistic_rank = (y_pred_neg > y_pred_pos).sum(axis=1)  # shape (N,)
    pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(axis=1)  # shape (N,)

    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    mrr_list = 1. / ranking_list.astype(np.float32)

    return mrr_list.mean()


def evaluate_link_prediction(
        train_sources, train_targets, train_labels,
        valid_sources, valid_targets, valid_labels,
        test_sources, test_targets, test_labels,
        embedding_tensor, edge_op, negative_edges_per_positive,
        n_epochs, batch_size, device):

    logger.info(
        f"Classifier train: {len(train_sources):,}, val: {len(valid_sources):,}, test: {len(test_sources):,}")

    model = LinkPredictionModel(embedding_tensor, edge_op).to(device)

    history = train_link_prediction_model(
        model,
        train_sources, train_targets, train_labels,
        valid_sources, valid_targets, valid_labels,
        batch_size=batch_size, epochs=n_epochs, device=device, patience=5
    )

    # Make predictions
    logger.info("Making final predictions...")
    val_pred_proba = predict_with_model(model, valid_sources, valid_targets, batch_size=batch_size, device=device)
    test_pred_proba = predict_with_model(model, test_sources, test_targets, batch_size=batch_size, device=device)

    test_pred = (test_pred_proba > 0.5).astype(int)

    # Calculate standard metrics for test set
    test_auc = roc_auc_score(test_labels, test_pred_proba)
    test_accuracy = accuracy_score(test_labels, test_pred)
    test_precision = precision_score(test_labels, test_pred, zero_division=0)
    test_recall = recall_score(test_labels, test_pred, zero_division=0)
    test_f1 = f1_score(test_labels, test_pred, zero_division=0)

    val_mrr = compute_mrr(val_pred_proba, valid_labels, negative_edges_per_positive)
    test_mrr = compute_mrr(test_pred_proba, test_labels, negative_edges_per_positive)

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


def add_gaussian_noise(embeddings: Dict[int, np.ndarray], noise_std: float) -> Dict[int, np.ndarray]:
    if noise_std == 0.0:
        return embeddings.copy()

    noisy_embeddings = {}
    for node_id, embedding in embeddings.items():
        noise = np.random.normal(1.0, noise_std, embedding.shape)
        noisy_embeddings[node_id] = embedding * noise

    return noisy_embeddings


def run_link_prediction_experiments(
        data_file_path,
        is_directed,
        batch_ts_size,
        sliding_window_duration,
        weighted_sum_alpha,
        walk_length,
        num_walks_per_node,
        edge_picker,
        embedding_dim,
        edge_op,
        negative_edges_per_positive,
        n_epochs,
        full_embedding_use_gpu,
        incremental_embedding_use_gpu,
        link_prediction_use_gpu,
        n_runs,
        batch_size,
        num_noise_steps,
        word2vec_n_workers,
        output_path,
        precomputed_data_path
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

    all_node_ids = np.concatenate([
        train_sources, train_targets,
        val_sources, val_targets,
        test_sources, test_targets
    ])

    max_node_id = int(all_node_ids.max())
    logger.info(f"Maximum node ID in dataset: {max_node_id}")

    if not (precomputed_data_path and os.path.isfile(precomputed_data_path)):
        logger.info("=" * 60)
        logger.info("Computing data and embeddings ...")
        if precomputed_data_path:
            logger.info(f'and saving in {precomputed_data_path}')
        logger.info("=" * 60)

        train_sources_combined, train_targets_combined, train_labels_combined = create_dataset_with_negative_edges(
            train_sources,
            train_targets,
            train_sources,
            train_targets,
            is_directed,
            negative_edges_per_positive
        )

        valid_sources_combined, valid_targets_combined, valid_labels_combined = create_dataset_with_negative_edges(
            val_sources,
            val_targets,
            train_sources,
            train_targets,
            is_directed,
            negative_edges_per_positive
        )

        test_sources_combined, test_targets_combined, test_labels_combined = create_dataset_with_negative_edges(
            test_sources,
            test_targets,
            train_sources,
            train_targets,
            is_directed,
            negative_edges_per_positive
        )

        full_embeddings = train_embeddings_full_approach(
            train_sources=train_sources,
            train_targets=train_targets,
            train_timestamps=train_timestamps,
            is_directed=is_directed,
            walk_length=walk_length,
            num_walks_per_node=num_walks_per_node,
            edge_picker=edge_picker,
            embedding_dim=embedding_dim,
            walk_use_gpu=full_embedding_use_gpu,
            word2vec_n_workers=word2vec_n_workers
        )

        streaming_embeddings = train_embeddings_streaming_approach(
            train_sources=train_sources,
            train_targets=train_targets,
            train_timestamps=train_timestamps,
            batch_ts_size=batch_ts_size,
            sliding_window_duration=sliding_window_duration,
            weighted_sum_alpha=weighted_sum_alpha,
            is_directed=is_directed,
            walk_length=walk_length,
            num_walks_per_node=num_walks_per_node,
            edge_picker=edge_picker,
            embedding_dim=embedding_dim,
            walk_use_gpu=incremental_embedding_use_gpu,
            word2vec_n_workers=word2vec_n_workers
        )

        precomputed_data = {
            'train_sources_combined': train_sources_combined,
            'train_targets_combined': train_targets_combined,
            'train_labels_combined': train_labels_combined,
            'valid_sources_combined': valid_sources_combined,
            'valid_targets_combined': valid_targets_combined,
            'valid_labels_combined': valid_labels_combined,
            'test_sources_combined': test_sources_combined,
            'test_targets_combined': test_targets_combined,
            'test_labels_combined': test_labels_combined,
            'full_embeddings': full_embeddings,
            'streaming_embeddings': streaming_embeddings
        }

        if precomputed_data_path:
            with open(precomputed_data_path, 'wb') as f:
                pickle.dump(precomputed_data, f)
    else:
        logger.info("=" * 60)
        logger.info(f"Preloading data and embeddings from {precomputed_data_path} ...")
        logger.info("=" * 60)

        with open(precomputed_data_path, 'rb') as f:
            precomputed_data = pickle.load(f)

        train_sources_combined = precomputed_data['train_sources_combined']
        train_targets_combined = precomputed_data['train_targets_combined']
        train_labels_combined = precomputed_data['train_labels_combined']
        valid_sources_combined = precomputed_data['valid_sources_combined']
        valid_targets_combined = precomputed_data['valid_targets_combined']
        valid_labels_combined = precomputed_data['valid_labels_combined']
        test_sources_combined = precomputed_data['test_sources_combined']
        test_targets_combined = precomputed_data['test_targets_combined']
        test_labels_combined = precomputed_data['test_labels_combined']
        full_embeddings = precomputed_data['full_embeddings']
        streaming_embeddings = precomputed_data['streaming_embeddings']

    noise_stds = np.arange(0.0, 1.05, 1.0 / float(num_noise_steps)).tolist()

    logger.info("=" * 60)
    logger.info("TRAINING EMBEDDINGS - STREAMING APPROACH")
    logger.info("=" * 60)

    device = 'cuda' if link_prediction_use_gpu and torch.cuda.is_available() else 'cpu'

    streaming_results = {}

    for noise_std in noise_stds:
        for run in range(n_runs):
            logger.info(f"\n--- Noise Std: {noise_std}, Run {run + 1}/{n_runs} ---")

            noisy_embeddings = add_gaussian_noise(streaming_embeddings, noise_std)

            current_streaming_results = evaluate_link_prediction(
                train_sources=train_sources_combined,
                train_targets=train_targets_combined,
                train_labels=train_labels_combined,
                valid_sources=valid_sources_combined,
                valid_targets=valid_targets_combined,
                valid_labels=valid_labels_combined,
                test_sources=test_sources_combined,
                test_targets=test_targets_combined,
                test_labels=test_labels_combined,
                embedding_tensor=get_embedding_tensor(noisy_embeddings, max_node_id),
                edge_op=edge_op,
                negative_edges_per_positive=negative_edges_per_positive,
                n_epochs=n_epochs,
                batch_size=batch_size,
                device=device
            )

            for key in current_streaming_results.keys():
                if noise_std not in streaming_results:
                    streaming_results[noise_std] = {}

                if key not in streaming_results[noise_std]:
                    streaming_results[noise_std][key] = []

                streaming_results[noise_std][key].append(current_streaming_results[key])

    logger.info(f"\nStreaming Approach Results Across Noise Levels:")
    logger.info("=" * 80)

    for noise_std in noise_stds:
        if noise_std in streaming_results:
            logger.info(f"\nNoise Level {noise_std:.3f}:")
            for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1_score', 'test_mrr', 'val_mrr']:
                if metric in streaming_results[noise_std]:
                    values = streaming_results[noise_std][metric]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    logger.info(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

    logger.info("=" * 60)
    logger.info("TRAINING EMBEDDINGS - FULL APPROACH")
    logger.info("=" * 60)

    full_results = {}

    for noise_std in noise_stds:
        for run in range(n_runs):
            logger.info(f"\n--- Noise Std: {noise_std}, Run {run + 1}/{n_runs} ---")

            noisy_embeddings = add_gaussian_noise(full_embeddings, noise_std)

            current_full_results = evaluate_link_prediction(
                train_sources=train_sources_combined,
                train_targets=train_targets_combined,
                train_labels=train_labels_combined,
                valid_sources=valid_sources_combined,
                valid_targets=valid_targets_combined,
                valid_labels=valid_labels_combined,
                test_sources=test_sources_combined,
                test_targets=test_targets_combined,
                test_labels=test_labels_combined,
                embedding_tensor=get_embedding_tensor(noisy_embeddings, max_node_id),
                edge_op=edge_op,
                negative_edges_per_positive=negative_edges_per_positive,
                n_epochs=n_epochs,
                batch_size=batch_size,
                device=device
            )

            for key in current_full_results.keys():
                if noise_std not in full_results:
                    full_results[noise_std] = {}

                if key not in full_results[noise_std]:
                    full_results[noise_std][key] = []

                full_results[noise_std][key].append(current_full_results[key])

    logger.info(f"\nFull Approach Results Across Noise Levels:")
    logger.info("=" * 80)

    for noise_std in noise_stds:
        if noise_std in full_results:
            logger.info(f"\nNoise Level {noise_std:.3f}:")
            for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1_score', 'test_mrr', 'val_mrr']:
                if metric in full_results[noise_std]:
                    values = full_results[noise_std][metric]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    logger.info(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

    if output_path:
        results = {
            'full_approach': full_results,
            'streaming_approach': streaming_results
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
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
    parser.add_argument('--edge_picker', type=str, default='ExponentialIndex',
                        help='Edge picker for random walks')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimensionality of node embeddings')
    parser.add_argument('--edge_op', type=str, default='hadamard',
                        choices=['average', 'hadamard', 'weighted-l1', 'weighted-l2'],
                        help='Edge operation for combining node embeddings')

    parser.add_argument('--negative_edges_per_positive', type=int, default=1,
                        help='Number of negative edges per positive edge')

    # Training parameters
    parser.add_argument('--num_noise_steps', type=int, default=20, help="Number of noise steps")
    parser.add_argument('--n_epochs', type=int, default=5,
                        help='Number of epochs for neural network training')
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of experimental runs for averaging results')
    parser.add_argument('--batch_size', type=int, default=1_000_000, help='Batch size for training')

    # GPU settings
    parser.add_argument('--full_embedding_use_gpu', action='store_true',
                        help='Enable GPU acceleration for full embedding approach')
    parser.add_argument('--incremental_embedding_use_gpu', action='store_true',
                        help='Enable GPU acceleration for streaming embedding approach')
    parser.add_argument('--link_prediction_use_gpu', action='store_true',
                        help='Enable GPU acceleration for link prediction neural network')

    # Other settings
    parser.add_argument('--word2vec_n_workers', type=int, default=8,
                        help='Number of workers for Word2Vec training')
    parser.add_argument('--output_path', type=str, default=None,
                        help='File path to save results (optional)')
    parser.add_argument('--precomputed_data_path', type=str, required=False, default=None, help='Precomputed data path')

    args = parser.parse_args()

    # Run experiments
    full_results, streaming_results = run_link_prediction_experiments(
        data_file_path=args.data_file_path,
        is_directed=args.is_directed,
        batch_ts_size=args.batch_ts_size,
        sliding_window_duration=args.sliding_window_duration,
        weighted_sum_alpha=args.weighted_sum_alpha,
        walk_length=args.walk_length,
        num_walks_per_node=args.num_walks_per_node,
        edge_picker=args.edge_picker,
        embedding_dim=args.embedding_dim,
        edge_op=args.edge_op,
        negative_edges_per_positive=args.negative_edges_per_positive,
        n_epochs=args.n_epochs,
        full_embedding_use_gpu=args.full_embedding_use_gpu,
        incremental_embedding_use_gpu=args.incremental_embedding_use_gpu,
        link_prediction_use_gpu=args.link_prediction_use_gpu,
        n_runs=args.n_runs,
        batch_size=args.batch_size,
        num_noise_steps=args.num_noise_steps,
        word2vec_n_workers=args.word2vec_n_workers,
        output_path=args.output_path,
        precomputed_data_path=args.precomputed_data_path
    )
