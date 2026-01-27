import argparse
import logging
import random
import os
import time
import pickle
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec
from gensim import utils
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from temporal_random_walk import TemporalRandomWalk
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random.seed(42); np.random.seed(42); torch.manual_seed(42)


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
    elif data_file_path.endswith('.pkl'):
        with open(data_file_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv(data_file_path)

    df = df[['u', 'i', 'ts']]

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


def combine_negative_with_positive_edges(ds_sources, ds_targets, neg_sources, neg_targets, k, random_state=42):
    np.random.seed(random_state)

    num_positive = len(ds_sources)
    expected_neg_size = num_positive * k

    assert len(neg_sources) == expected_neg_size, f"Expected {expected_neg_size} negative sources, got {len(neg_sources)}"
    assert len(neg_targets) == expected_neg_size, f"Expected {expected_neg_size} negative targets, got {len(neg_targets)}"

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

        self.edge_norm = nn.LayerNorm(input_dim)

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

        x = self.edge_norm(edge_features)

        x = F.relu(self.fc1(x))
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
                                patience=5):
    logger.info(f"Training neural network on {len(X_sources_train):,} samples with batch size {batch_size:,}")

    # Move model to device
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(mode='max', patience=patience)

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
                       device='cpu'):
    model = model.to(device)
    model.eval()

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


def l2_normalize_rows(emb_dict):
    out = {}
    for k, v in emb_dict.items():
        n = np.linalg.norm(v)
        out[k] = (v / (n + 1e-12)).astype(np.float32)
    return out


def train_embeddings_streaming_approach(
    train_sources, train_targets, train_timestamps,
    batch_ts_size, sliding_window_duration, is_directed,
    walk_length, num_walks_per_node, edge_picker, embedding_dim,
    walk_use_gpu, word2vec_n_workers, batch_epochs, seed=42, tempest_seed=42, sample=0.0
):
    logger.info("Training embeddings with streaming Word2Vec (incremental)")

    temporal_random_walk = TemporalRandomWalk(
        is_directed=is_directed,
        use_gpu=walk_use_gpu,
        max_time_capacity=sliding_window_duration,
        global_seed=tempest_seed
    )

    min_ts = int(np.min(train_timestamps))
    max_ts = int(np.max(train_timestamps))
    total_range = max_ts - min_ts
    num_batches = int(np.ceil(total_range / batch_ts_size))
    logger.info(f"Processing {num_batches} batches with duration={batch_ts_size:,}")

    w2v = None

    # --------------------------
    # Timing accumulators
    # --------------------------
    total_ingestion_time = 0.0
    total_walk_sampling_time = 0.0

    def keep_existing_tokens(word, count, min_count):
        if w2v is not None and word in w2v.wv.key_to_index:
            return utils.RULE_KEEP
        return None

    for b in range(num_batches):
        b_start = min_ts + b * batch_ts_size
        b_end   = min_ts + (b + 1) * batch_ts_size
        if b == num_batches - 1:
            b_end = max_ts + 1

        mask = (train_timestamps >= b_start) & (train_timestamps < b_end)
        b_src = train_sources[mask]
        b_tgt = train_targets[mask]
        b_ts  = train_timestamps[mask]
        if len(b_src) == 0:
            continue

        logger.info(f"Batch {b + 1}/{num_batches}: {len(b_src):,} edges [{b_start}, {b_end})")

        # --------------------------
        # Ingestion timing
        # --------------------------
        t0 = time.perf_counter()
        temporal_random_walk.add_multiple_edges(b_src, b_tgt, b_ts)
        total_ingestion_time += time.perf_counter() - t0

        # --------------------------
        # Walk sampling timing
        # --------------------------
        t0 = time.perf_counter()
        walks_f, _, lens_f = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
            max_walk_len=walk_length,
            num_walks_per_node=num_walks_per_node // 2,
            walk_bias=edge_picker,
            initial_edge_bias='Uniform',
            walk_direction="Forward_In_Time"
        )
        walks_b, _, lens_b = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
            max_walk_len=walk_length,
            num_walks_per_node=num_walks_per_node // 2,
            walk_bias=edge_picker,
            initial_edge_bias='Uniform',
            walk_direction="Backward_In_Time"
        )
        total_walk_sampling_time += time.perf_counter() - t0

        walks = np.concatenate([walks_f, walks_b], axis=0)
        lens  = np.concatenate([lens_f, lens_b], axis=0)

        clean_walks = []
        for w, L in zip(walks, lens):
            path = [str(n) for n in w[:L]]
            if len(path) > 1:
                clean_walks.append(path)
        if not clean_walks:
            logger.info("No valid walks in this batch; skipping.")
            continue

        try:
            with suppress_word2vec_output():
                if w2v is None:
                    w2v = Word2Vec(
                        vector_size=embedding_dim,
                        window=10,
                        min_count=1,
                        sample=sample,
                        workers=word2vec_n_workers,
                        sg=1,
                        seed=seed
                    )
                    w2v.build_vocab(clean_walks)
                else:
                    w2v.build_vocab(clean_walks, update=True, trim_rule=keep_existing_tokens)

                w2v.train(
                    clean_walks,
                    total_examples=len(clean_walks),
                    epochs=batch_epochs
                )
        except Exception as e:
            logger.error(f"Error processing batch {b + 1}: {e}")
            continue

    if w2v is None:
        logger.warning("No batches produced walks; returning empty embedding store.")
        return {}, total_ingestion_time, total_walk_sampling_time

    node_embeddings = {int(k): w2v.wv[k] for k in w2v.wv.index_to_key}
    logger.info(
        f"Streaming completed. Final embedding store: {len(node_embeddings)} nodes | "
        f"Ingestion time: {total_ingestion_time:.2f}s | "
        f"Walk sampling time: {total_walk_sampling_time:.2f}s"
    )

    return node_embeddings, total_ingestion_time, total_walk_sampling_time


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


def compute_window_durations(
    min_ts: int,
    max_ts: int,
    window_divisor_start: int,
    num_steps: int
):
    """
    Returns:
      window_durations: List[int]     # Δ values
      relative_history: List[int]     # T / Δ (integers, for plotting)
    """
    T = max_ts - min_ts

    # Integer divisors for presentation: e.g., 100, 95, ..., 1
    divisors = np.linspace(window_divisor_start, 1, num_steps)
    divisors = np.round(divisors).astype(int)
    divisors = np.unique(divisors)[::-1]   # descending, unique

    window_durations = (T // divisors).astype(int)

    return window_durations.tolist(), divisors.tolist()


def run_window_sensitivity_experiment(
        data_file_path,
        negative_edges_path,
        is_directed,
        walk_length,
        num_walks_per_node,
        edge_picker,
        embedding_dim,
        edge_op,
        negative_edges_per_positive,
        n_epochs,
        incremental_embedding_use_gpu,
        link_prediction_use_gpu,
        n_runs,
        batch_size,
        batch_divisor=100,
        window_divisor_start=100,
        num_window_steps=20,
        word2vec_n_workers=8,
        word2vec_batch_epochs=3,
        tempest_seed=42,
        output_path=None
):
    logger.info("Starting window-size (Δ) sensitivity experiment")

    # --------------------------
    # Dataset split
    # --------------------------
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

    # --------------------------
    # Negatives (IDENTICAL)
    # --------------------------
    negative_edges_df = pd.read_parquet(negative_edges_path)
    negative_sources = negative_edges_df['u'].to_numpy()
    negative_targets = negative_edges_df['i'].to_numpy()

    train_neg_end = len(train_sources) * negative_edges_per_positive
    valid_neg_end = train_neg_end + len(val_sources) * negative_edges_per_positive
    test_neg_end  = valid_neg_end + len(test_sources) * negative_edges_per_positive

    neg_src_train = negative_sources[:train_neg_end]
    neg_tgt_train = negative_targets[:train_neg_end]

    neg_src_val = negative_sources[train_neg_end:valid_neg_end]
    neg_tgt_val = negative_targets[train_neg_end:valid_neg_end]

    neg_src_test = negative_sources[valid_neg_end:test_neg_end]
    neg_tgt_test = negative_targets[valid_neg_end:test_neg_end]

    # Build supervised datasets
    train_src_all, train_tgt_all, train_lbl_all = combine_negative_with_positive_edges(
        train_sources,
        train_targets,
        neg_src_train,
        neg_tgt_train,
        negative_edges_per_positive
    )

    val_src_all, val_tgt_all, val_lbl_all = combine_negative_with_positive_edges(
        val_sources,
        val_targets,
        neg_src_val,
        neg_tgt_val,
        negative_edges_per_positive
    )

    test_src_all, test_tgt_all, test_lbl_all = combine_negative_with_positive_edges(
        test_sources,
        test_targets,
        neg_src_test,
        neg_tgt_test,
        negative_edges_per_positive
    )

    # --------------------------
    # Compute window schedule
    # --------------------------
    min_ts = int(train_timestamps.min())
    max_ts = int(train_timestamps.max())
    T = max_ts - min_ts

    batch_ts_size = T // batch_divisor

    window_durations, relative_history = compute_window_durations(
        min_ts, max_ts, window_divisor_start, num_window_steps
    )

    logger.info(f"Batch duration fixed at T/{batch_divisor} = {batch_ts_size}")
    logger.info("Window schedule (Relative history T/Δ):")
    logger.info(relative_history)

    results = {}

    # --------------------------
    # Main Δ loop
    # --------------------------
    for delta, rel_h in zip(window_durations, relative_history):
        logger.info("=" * 80)
        logger.info(f"Relative history (T/Δ) = {rel_h}, Window duration Δ = {delta}")
        logger.info("=" * 80)

        ingestion_times = []
        walk_times = []
        aucs = []

        device = 'cuda' if link_prediction_use_gpu and torch.cuda.is_available() else 'cpu'

        for run in range(n_runs):
            logger.info(f"\n--- Run {run + 1}/{n_runs} ---")

            # --------------------------
            # Streaming embeddings
            # --------------------------
            embeddings, ingest_t, walk_t = train_embeddings_streaming_approach(
                train_sources=train_sources,
                train_targets=train_targets,
                train_timestamps=train_timestamps,
                batch_ts_size=batch_ts_size,
                sliding_window_duration=delta,
                is_directed=is_directed,
                walk_length=walk_length,
                num_walks_per_node=num_walks_per_node,
                edge_picker=edge_picker,
                embedding_dim=embedding_dim,
                walk_use_gpu=incremental_embedding_use_gpu,
                word2vec_n_workers=word2vec_n_workers,
                batch_epochs=word2vec_batch_epochs,
                tempest_seed=tempest_seed
            )

            ingestion_times.append(ingest_t)
            walk_times.append(walk_t)

            embedding_tensor = get_embedding_tensor(embeddings, max_node_id)

            # --------------------------
            # Downstream link prediction
            # --------------------------
            current_results = evaluate_link_prediction(
                train_sources=train_src_all,
                train_targets=train_tgt_all,
                train_labels=train_lbl_all,
                valid_sources=val_src_all,
                valid_targets=val_tgt_all,
                valid_labels=val_lbl_all,
                test_sources=test_src_all,
                test_targets=test_tgt_all,
                test_labels=test_lbl_all,
                embedding_tensor=embedding_tensor,
                edge_op=edge_op,
                negative_edges_per_positive=negative_edges_per_positive,
                n_epochs=n_epochs,
                batch_size=batch_size,
                device=device
            )

            aucs.append(current_results['auc'])

        # --------------------------
        # Aggregate results per Δ
        # --------------------------
        results[rel_h] = {
            'window_duration': delta,

            'ingestion_time_mean': float(np.mean(ingestion_times)),
            'ingestion_time_std':  float(np.std(ingestion_times)),

            'walk_sampling_time_mean': float(np.mean(walk_times)),
            'walk_sampling_time_std':  float(np.std(walk_times)),

            'auc_mean': float(np.mean(aucs)),
            'auc_std':  float(np.std(aucs)),
        }

        logger.info(
            f"Ingestion: {np.mean(ingestion_times):.2f} ± {np.std(ingestion_times):.2f}s | "
            f"Walk: {np.mean(walk_times):.2f} ± {np.std(walk_times):.2f}s | "
            f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
        )

    # --------------------------
    # Persist results
    # --------------------------
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Window-size (Δ) sensitivity test for streaming temporal link prediction"
    )

    # --------------------------
    # Required paths
    # --------------------------
    parser.add_argument(
        '--data_file_path',
        type=str,
        required=True,
        help='Path to dataset file (CSV / Parquet)'
    )
    parser.add_argument(
        '--negative_edges_path',
        type=str,
        required=True,
        help='Path to negative edges file (Parquet)'
    )

    # --------------------------
    # Graph / walk parameters
    # --------------------------
    parser.add_argument(
        '--is_directed',
        type=lambda x: x.lower() == 'true',
        required=True,
        help='Whether the graph is directed (true / false)'
    )
    parser.add_argument(
        '--walk_length',
        type=int,
        default=80,
        help='Maximum temporal random walk length'
    )
    parser.add_argument(
        '--num_walks_per_node',
        type=int,
        default=10,
        help='Number of walks per node'
    )
    parser.add_argument(
        '--edge_picker',
        type=str,
        default='ExponentialIndex',
        help='Temporal edge picker'
    )

    # --------------------------
    # Window / batch control
    # --------------------------
    parser.add_argument(
        '--batch_divisor',
        type=int,
        default=100,
        help='Batch duration = T / batch_divisor (default: 100)'
    )
    parser.add_argument(
        '--window_divisor_start',
        type=int,
        default=100,
        help='Smallest window = T / window_divisor_start (default: 100)'
    )
    parser.add_argument(
        '--num_window_steps',
        type=int,
        default=20,
        help='Number of window sizes to evaluate'
    )

    # --------------------------
    # Embedding / model parameters
    # --------------------------
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=128,
        help='Embedding dimensionality'
    )
    parser.add_argument(
        '--edge_op',
        type=str,
        default='hadamard',
        choices=['average', 'hadamard', 'weighted-l1', 'weighted-l2'],
        help='Edge operation for link prediction'
    )
    parser.add_argument(
        '--negative_edges_per_positive',
        type=int,
        default=2,
        help='Number of negative edges per positive'
    )

    # --------------------------
    # Training parameters
    # --------------------------
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=5,
        help='Number of epochs for link prediction training'
    )
    parser.add_argument(
        '--n_runs',
        type=int,
        default=3,
        help='Number of repeated runs per window size'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10_000,
        help='Batch size for classifier training'
    )

    # --------------------------
    # Execution / hardware
    # --------------------------
    parser.add_argument(
        '--incremental_embedding_use_gpu',
        action='store_true',
        help='Use GPU for streaming embeddings'
    )
    parser.add_argument(
        '--link_prediction_use_gpu',
        action='store_true',
        help='Use GPU for link prediction model'
    )
    parser.add_argument(
        '--word2vec_n_workers',
        type=int,
        default=16,
        help='Number of Word2Vec workers'
    )
    parser.add_argument(
        '--word2vec_batch_epochs',
        type=int,
        default=3,
        help='Number of Word2Vec epochs per batch'
    )

    # --------------------------
    # Seeds
    # --------------------------
    parser.add_argument(
        '--tempest_seed',
        type=int,
        default=42,
        help='Seed for tempest'
    )

    # --------------------------
    # Output
    # --------------------------
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Optional path to save results (pickle)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Starting Δ sensitivity experiment (streaming only)")
    logger.info("=" * 80)

    results = run_window_sensitivity_experiment(
        data_file_path=args.data_file_path,
        negative_edges_path=args.negative_edges_path,
        is_directed=args.is_directed,
        walk_length=args.walk_length,
        num_walks_per_node=args.num_walks_per_node,
        edge_picker=args.edge_picker,
        embedding_dim=args.embedding_dim,
        edge_op=args.edge_op,
        negative_edges_per_positive=args.negative_edges_per_positive,
        n_epochs=args.n_epochs,
        incremental_embedding_use_gpu=args.incremental_embedding_use_gpu,
        link_prediction_use_gpu=args.link_prediction_use_gpu,
        n_runs=args.n_runs,
        batch_size=args.batch_size,
        batch_divisor=args.batch_divisor,
        window_divisor_start=args.window_divisor_start,
        num_window_steps=args.num_window_steps,
        word2vec_n_workers=args.word2vec_n_workers,
        word2vec_batch_epochs=args.word2vec_batch_epochs,
        tempest_seed=args.tempest_seed,
        output_path=args.output_path
    )

    logger.info("=" * 80)
    logger.info("Δ sensitivity experiment completed")
    logger.info("=" * 80)

    logger.info("Summary (Relative history T/Δ → AUC):")
    for rel_h in sorted(results.keys(), reverse=True):
        r = results[rel_h]
        logger.info(
            f"  T/Δ={rel_h:>4d} | Δ={r['window_duration']:<8d} | "
            f"AUC={r['auc_mean']:.4f} ± {r['auc_std']:.4f}"
        )


if __name__ == '__main__':
    main()
