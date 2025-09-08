import argparse
import logging
import os
import pickle
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from temporal_random_walk import TemporalRandomWalk
from torch.utils.data import DataLoader, TensorDataset, Sampler
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


@contextmanager
def suppress_word2vec_output():
    """Context manager to suppress Word2Vec verbose output."""
    gensim_logger = logging.getLogger('gensim')
    word2vec_logger = logging.getLogger('gensim.models.word2vec')
    kv_logger = logging.getLogger('gensim.models.keyedvectors')

    original_gensim_level = gensim_logger.level
    original_word2vec_level = word2vec_logger.level
    original_kv_level = kv_logger.level

    try:
        gensim_logger.setLevel(logging.ERROR)
        word2vec_logger.setLevel(logging.ERROR)
        kv_logger.setLevel(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        gensim_logger.setLevel(original_gensim_level)
        word2vec_logger.setLevel(original_word2vec_level)
        kv_logger.setLevel(original_kv_level)


def load_dgraphfin_data(data_path):
    """Load DGraphFin data from npz file."""
    logger.info(f"Loading DGraphFin dataset from {data_path}")

    data = np.load(data_path)

    # Extract components
    x = data['x']  # Node features (17-dim)
    y = data['y']  # Node labels
    edge_index = data['edge_index']  # Edge indices
    edge_timestamp = data['edge_timestamp']  # Edge timestamps
    train_ids = data['train_mask']  # Train node indices
    val_ids = data['valid_mask']  # Valid node indices
    test_ids = data['test_mask']  # Test node indices

    logger.info(f"Nodes: {x.shape[0]:,}, Features: {x.shape[1]}")
    logger.info(f"Edges: {edge_index.shape[0]:,}")
    logger.info(f"Timestamp range: [{np.min(edge_timestamp)}, {np.max(edge_timestamp)}]")

    # Get labels - no filtering needed since splits already contain only classes 0 and 1
    train_labels = y[train_ids]
    val_labels = y[val_ids]
    test_labels = y[test_ids]

    logger.info(f"Dataset splits:")
    logger.info(f"  Train: {len(train_ids):,} (fraud: {np.sum(train_labels):,})")
    logger.info(f"  Val: {len(val_ids):,} (fraud: {np.sum(val_labels):,})")
    logger.info(f"  Test: {len(test_ids):,} (fraud: {np.sum(test_labels):,})")

    # Prepare edge data
    edges_df = pd.DataFrame({
        'u': edge_index[:, 0],
        'i': edge_index[:, 1],
        'ts': edge_timestamp
    })

    return (x, edges_df, train_ids, train_labels, val_ids, val_labels,
            test_ids, test_labels)



class EarlyStopping:
    """Early stopping callback to prevent overfitting."""

    def __init__(self, mode='min', patience=5, min_delta=0.0001, restore_best_weights=True):
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


class FraudDetectionModel(nn.Module):
    """More sophisticated MLP with feature interaction modeling."""

    def __init__(self, node_embeddings_tensor: torch.Tensor, node_features_tensor: torch.Tensor):
        super().__init__()

        # Embedding components
        self.embedding_lookup = nn.Embedding.from_pretrained(node_embeddings_tensor, freeze=True)
        self.node_features = nn.Parameter(node_features_tensor, requires_grad=False)

        # Dimensions
        embed_dim = node_embeddings_tensor.shape[1]
        feature_dim = node_features_tensor.shape[1]

        # Feature normalization
        self.feature_norm = nn.BatchNorm1d(feature_dim)
        self.embed_norm = nn.BatchNorm1d(embed_dim)

        # Separate processing paths
        hidden_dim = 128

        # Temporal embedding path
        self.temp_branch = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Static feature path
        self.feat_branch = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Cross-feature interactions
        interaction_dim = (hidden_dim // 2) * 2
        self.interaction_layer = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Final classifier
        final_input_dim = interaction_dim + hidden_dim  # Original features + interactions
        self.classifier = nn.Sequential(
            nn.Linear(final_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

    def forward(self, nodes):
        device = next(self.parameters()).device
        nodes = nodes.to(device)

        # Get embeddings and features
        temp_embed = self.embedding_lookup(nodes)
        static_feat = self.node_features[nodes]

        # Normalize
        temp_embed = self.embed_norm(temp_embed)
        static_feat = self.feature_norm(static_feat)

        # Process through separate branches
        temp_processed = self.temp_branch(temp_embed)
        feat_processed = self.feat_branch(static_feat)

        # Combine for interactions
        combined = torch.cat([temp_processed, feat_processed], dim=1)
        interactions = self.interaction_layer(combined)

        # Final combination
        final_features = torch.cat([combined, interactions], dim=1)

        return self.classifier(final_features)


class StratifiedBatchSampler(Sampler):
    """Custom sampler that ensures each batch has consistent class distribution while using ALL samples."""

    def __init__(self, labels, batch_size, drop_last=False):
        super().__init__()
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[int(label)].append(idx)

        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)

        # Calculate target samples per class per batch based on proportions
        total_samples = len(labels)
        self.target_samples_per_class = {}

        for cls in self.classes:
            class_ratio = len(self.class_indices[cls]) / total_samples
            target_samples = max(1, round(batch_size * class_ratio))
            self.target_samples_per_class[cls] = target_samples

        # Adjust if total exceeds batch_size
        total_target = sum(self.target_samples_per_class.values())
        if total_target > batch_size:
            # Scale down proportionally, ensuring at least 1 per class
            scale = batch_size / total_target
            for cls in self.classes:
                self.target_samples_per_class[cls] = max(1, int(self.target_samples_per_class[cls] * scale))

        # Calculate total batches needed to use ALL samples
        # Use the largest class to determine total batches needed
        max_batches_needed = max(
            len(self.class_indices[cls]) // self.target_samples_per_class[cls]
            for cls in self.classes
        )

        # Add extra batches for remainder samples
        remainder_batches = 0
        for cls in self.classes:
            remainder = len(self.class_indices[cls]) % self.target_samples_per_class[cls]
            if remainder > 0:
                remainder_batches = max(remainder_batches, 1)

        self.num_batches = max_batches_needed + remainder_batches

        print(f"Stratified sampling setup (ALL samples included):")
        for cls in self.classes:
            print(
                f"  Class {cls}: ~{self.target_samples_per_class[cls]} samples per batch ({len(self.class_indices[cls]):,} total)")
        print(f"  Total batches: {self.num_batches}")
        print(f"  Total samples that will be used: {total_samples:,}")

    def __iter__(self):
        # Shuffle indices within each class
        shuffled_class_indices = {}
        for cls in self.classes:
            indices = self.class_indices[cls].copy()
            random.shuffle(indices)
            shuffled_class_indices[cls] = indices

        # Keep track of used indices per class
        used_indices = {cls: 0 for cls in self.classes}

        # Generate batches
        for batch_idx in range(self.num_batches):
            batch = []

            for cls in self.classes:
                available = len(shuffled_class_indices[cls]) - used_indices[cls]

                if available > 0:
                    # Take target number or whatever is available
                    take = min(self.target_samples_per_class[cls], available)

                    start_idx = used_indices[cls]
                    end_idx = start_idx + take

                    batch.extend(shuffled_class_indices[cls][start_idx:end_idx])
                    used_indices[cls] += take

            # Only yield non-empty batches
            if batch:
                # Pad batch if needed and not dropping last
                while len(batch) < self.batch_size and not self.drop_last:
                    # Cycle through available samples to fill batch
                    for cls in self.classes:
                        if len(batch) >= self.batch_size:
                            break
                        available = len(shuffled_class_indices[cls]) - used_indices[cls]
                        if available > 0:
                            batch.append(shuffled_class_indices[cls][used_indices[cls]])
                            used_indices[cls] += 1

                # Shuffle within batch to avoid class ordering
                random.shuffle(batch)
                yield batch[:self.batch_size] if self.drop_last and len(batch) > self.batch_size else batch

    def __len__(self):
        return self.num_batches


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance - focuses learning on hard examples."""

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Calculate focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_fraud_detection_model(model, X_train, y_train, X_val, y_val,
                                batch_size, epochs=20, device='cpu', patience=5):
    """Train fraud detection model with stratified batching."""
    logger.info(f"Training fraud detection model on {len(X_train):,} samples")

    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, min_lr=1e-6
    )

    # Calculate class weights for imbalanced dataset
    pos_count = int(y_train.sum())
    alpha = pos_count / len(y_train)
    criterion = FocalLoss(alpha=alpha, gamma=3.0)

    early_stopping = EarlyStopping(mode='max', patience=patience)

    # Prepare datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )

    # Create stratified sampler for training
    train_sampler = StratifiedBatchSampler(y_train, batch_size, drop_last=True)
    val_sampler = StratifiedBatchSampler(y_val, batch_size * 4, drop_last=False)

    num_workers = min(8, os.cpu_count())

    # Use custom batch sampler for training
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        pin_memory=(device == 'cuda'),
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )

    # Use custom batch sampler for validation
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        pin_memory=(device == 'cuda'),
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'train_f1': [], 'val_f1': []
    }

    logger.info(f"Training with {len(train_loader)} stratified batches per epoch")

    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        total_train_loss = 0.0
        train_preds_list = []
        train_targets_list = []

        batch_fraud_counts = []

        for batch_indices in train_loader:
            batch_x = batch_indices[0].to(device, non_blocking=True)
            batch_y = batch_indices[1].to(device, non_blocking=True)

            # Track fraud count in this batch
            fraud_count = int(batch_y.sum().item())
            batch_fraud_counts.append(fraud_count)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_preds_list.append(torch.sigmoid(outputs).detach())
            train_targets_list.append(batch_y)

        avg_train_loss = total_train_loss / len(train_loader)

        # Log batch consistency
        if batch_fraud_counts:
            fraud_std = np.std(batch_fraud_counts)
            fraud_mean = np.mean(batch_fraud_counts)
            logger.info(f"Batch fraud consistency - Mean: {fraud_mean:.1f}, Std: {fraud_std:.1f}")

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch_indices in val_loader:
                batch_x = batch_indices[0].to(device, non_blocking=True)
                batch_y = batch_indices[1].to(device, non_blocking=True)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                total_val_loss += loss.item()
                val_preds_list.append(torch.sigmoid(outputs))
                val_targets_list.append(batch_y)

        avg_val_loss = total_val_loss / len(val_loader)

        # Calculate metrics
        train_preds = torch.cat(train_preds_list).cpu().numpy().flatten()
        train_targets = torch.cat(train_targets_list).cpu().numpy().flatten()
        val_preds = torch.cat(val_preds_list).cpu().numpy().flatten()
        val_targets = torch.cat(val_targets_list).cpu().numpy().flatten()

        train_auc = roc_auc_score(train_targets, train_preds)
        val_auc = roc_auc_score(val_targets, val_preds)

        train_f1 = f1_score(train_targets, (train_preds >= 0.5).astype(int), zero_division=0)
        val_f1 = f1_score(val_targets, (val_preds >= 0.5).astype(int), zero_division=0)

        # Use AUC for scheduling and early stopping
        scheduler.step(val_auc)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'train_auc': f'{train_auc:.4f}',
            'val_auc': f'{val_auc:.4f}'
        })

        logger.info(f"Epoch {epoch + 1}/{epochs} — "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                    f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        if early_stopping(val_auc, model):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        # Memory cleanup
        del train_preds_list, train_targets_list, val_preds_list, val_targets_list
        if device == 'cuda':
            torch.cuda.empty_cache()

    epoch_pbar.close()
    return history


def predict_with_model(model, X_test, batch_size, device='cpu'):
    """Make predictions with trained model."""
    model = model.to(device)
    model.eval()

    logger.info(f"Making predictions on {len(X_test):,} samples")

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long))

    num_workers = min(8, os.cpu_count())

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=(device == 'cuda'), num_workers=num_workers,
        persistent_workers=True, prefetch_factor=4
    )

    logger.info(f"Processing {len(test_loader)} batches")

    predictions_list = []

    prediction_pbar = tqdm(test_loader, desc="Predicting", unit="batch")

    with torch.no_grad():
        for (batch_x,) in prediction_pbar:
            batch_x = batch_x.to(device, non_blocking=True)

            logits = model(batch_x)
            probs = torch.sigmoid(logits).detach().float()
            predictions_list.append(probs)

            prediction_pbar.set_postfix({
                'batches_processed': len(predictions_list),
                'batch_size': len(batch_x)
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

    # Generate forward walks
    walks_forward, _, walk_lengths_forward = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        max_walk_len=walk_length,
        num_walks_per_node=num_walks_per_node // 2,
        walk_bias=edge_picker,
        initial_edge_bias='Uniform',
        walk_direction="Forward_In_Time"
    )

    logger.info(
        f'Generated {len(walk_lengths_forward)} forward walks. Mean length: {np.mean(walk_lengths_forward):.2f}, Max length: {np.max(walk_lengths_forward):.2f}')

    logger.info(
        f'Generating backward {num_walks_per_node // 2} walks per node with max length {walk_length} using {edge_picker} picker.')

    # Generate backward walks
    walks_backward, _, walk_lengths_backward = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        max_walk_len=walk_length,
        num_walks_per_node=num_walks_per_node // 2,
        walk_bias=edge_picker,
        initial_edge_bias='Uniform',
        walk_direction="Backward_In_Time"
    )

    logger.info(
        f'Generated {len(walk_lengths_backward)} backward walks. Mean length: {np.mean(walk_lengths_backward):.2f}, Max length: {np.max(walk_lengths_backward):.2f}')

    # Combine walks
    walks = np.concatenate([walks_forward, walks_backward], axis=0)
    walk_lengths = np.concatenate([walk_lengths_forward, walk_lengths_backward], axis=0)

    logger.info(f'Generated {len(walks)} walks. Mean length: {np.mean(walk_lengths):.2f}, Max length: {np.max(walk_lengths):.2f}')

    clean_walks = []
    for walk, length in zip(walks, walk_lengths):
        clean_walk = [str(node) for node in walk[:length]]
        if len(clean_walk) > 1:
            clean_walks.append(clean_walk)

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

    node_embeddings = {int(node): model.wv[node] for node in model.wv.index_to_key}
    logger.info(f'Trained embeddings for {len(node_embeddings)} nodes')
    return node_embeddings


def train_embeddings_streaming_approach(
        train_sources, train_targets, train_timestamps,
        batch_ts_size, sliding_window_duration, is_directed,
        walk_length, num_walks_per_node, edge_picker, embedding_dim,
        walk_use_gpu, word2vec_n_workers, batch_epochs, seed=42
):
    """Train embeddings using streaming approach."""
    logger.info("Training embeddings with streaming approach")

    temporal_random_walk = TemporalRandomWalk(
        is_directed=is_directed, use_gpu=walk_use_gpu, max_time_capacity=sliding_window_duration
    )

    min_ts = int(np.min(train_timestamps))
    max_ts = int(np.max(train_timestamps))
    total_range = max_ts - min_ts
    num_batches = int(np.ceil(total_range / batch_ts_size))
    logger.info(f"Processing {num_batches} batches with duration={batch_ts_size:,}")

    w2v_model = None

    for batch_idx in range(num_batches):
        batch_start_ts = min_ts + batch_idx * batch_ts_size
        batch_end_ts = min_ts + (batch_idx + 1) * batch_ts_size
        if batch_idx == num_batches - 1:
            batch_end_ts = max_ts + 1

        mask = (train_timestamps >= batch_start_ts) & (train_timestamps < batch_end_ts)
        b_src = train_sources[mask]
        b_tgt = train_targets[mask]
        b_ts = train_timestamps[mask]

        if len(b_src) == 0:
            continue

        logger.info(f"Batch {batch_idx + 1}/{num_batches}: {len(b_src):,} edges")

        temporal_random_walk.add_multiple_edges(b_src, b_tgt, b_ts)

        walks_f, _, lens_f = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
            max_walk_len=walk_length, num_walks_per_node=num_walks_per_node // 2,
            walk_bias=edge_picker, initial_edge_bias='Uniform', walk_direction="Forward_In_Time"
        )
        walks_b, _, lens_b = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
            max_walk_len=walk_length, num_walks_per_node=num_walks_per_node // 2,
            walk_bias=edge_picker, initial_edge_bias='Uniform', walk_direction="Backward_In_Time"
        )

        # Log walk statistics
        logger.info(f"Generated walks - Forward: {len(walks_f)} walks (mean length: {np.mean(lens_f):.2f}, max length: {np.max(lens_f):.2f}), "
                    f"Backward: {len(walks_b)} walks (mean length: {np.mean(lens_b):.2f}, max length: {np.max(lens_f):.2f})")

        walks = np.concatenate([walks_f, walks_b], axis=0)
        lens = np.concatenate([lens_f, lens_b], axis=0)

        logger.info(
            f'Generated {len(walks)} walks. Mean length: {np.mean(lens):.2f}, Max length: {np.max(lens):.2f}')

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
                if w2v_model is None:
                    w2v_model = Word2Vec(
                        vector_size=embedding_dim, window=10, min_count=1,
                        workers=word2vec_n_workers, sg=1, seed=seed
                    )
                    w2v_model.build_vocab(clean_walks)
                else:
                    w2v_model.build_vocab(clean_walks, update=True)

                total_words = sum(len(s) for s in clean_walks)
                w2v_model.train(clean_walks, total_words=total_words, epochs=batch_epochs)

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {e}")
            continue

    if w2v_model is None:
        logger.warning("No batches produced walks; returning empty embedding store.")
        return {}

    node_embeddings = {int(k): w2v_model.wv[k] for k in w2v_model.wv.index_to_key}
    logger.info(f"Streaming completed. Final embedding store: {len(node_embeddings)} nodes")
    return node_embeddings


def get_embedding_tensor(embedding_dict, max_node_id, rand_scale=0.02):
    """Convert embedding dictionary to tensor."""
    dim = len(next(iter(embedding_dict.values())))
    emb = torch.zeros((max_node_id + 1, dim), dtype=torch.float32)

    for node_id, vec in embedding_dict.items():
        if node_id <= max_node_id:
            emb[node_id] = torch.tensor(vec, dtype=torch.float32)

    missing_mask = (emb.abs().sum(dim=1) == 0)
    if missing_mask.any():
        emb[missing_mask] = torch.randn((missing_mask.sum(), dim), dtype=torch.float32) * rand_scale

    # L2 normalize
    with torch.no_grad():
        norms = emb.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        emb = emb / norms

    logger.info(f"Embedding tensor: shape={emb.shape}, "
                f"filled={((~missing_mask).sum().item()):,}, "
                f"random_init={missing_mask.sum().item():,}")
    return emb


def prepare_node_features(node_features, train_ids, scaler=None):
    """Prepare and normalize node features."""
    if scaler is None:
        scaler = StandardScaler().fit(node_features[train_ids])

    normalized_features = scaler.transform(node_features)
    feature_tensor = torch.tensor(normalized_features, dtype=torch.float32)

    logger.info(f"Node features: shape={feature_tensor.shape}, fitted on {len(train_ids):,} train nodes")
    return feature_tensor, scaler


def _pick_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick the threshold that maximizes F1 score on validation set."""
    unique = np.unique(y_true)
    if unique.size < 2:
        return 0.5  # fallback when all labels are same

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    p, r = precision[1:], recall[1:]
    if thresholds.size == 0:
        return 0.5

    f1_scores = 2 * p * r / np.clip(p + r, 1e-12, None)
    best_idx = np.nanargmax(f1_scores)
    return float(thresholds[best_idx])


def evaluate_fraud_detection(
        train_ids, train_labels, val_ids, val_labels, test_ids, test_labels,
        embedding_tensor, node_features_tensor, n_epochs, batch_size, device
):
    """Evaluate fraud detection performance with stratified batching."""
    logger.info(f"Fraud detection — train: {len(train_ids):,}, val: {len(val_ids):,}, test: {len(test_ids):,}")

    model = FraudDetectionModel(embedding_tensor, node_features_tensor).to(device)

    history = train_fraud_detection_model(
        model, train_ids, train_labels, val_ids, val_labels,
        batch_size=batch_size, epochs=n_epochs, device=device, patience=5
    )

    # Validation predictions
    logger.info("Making predictions on validation set...")
    val_pred_proba = predict_with_model(model, val_ids, batch_size=batch_size, device=device)
    val_auc = roc_auc_score(val_labels, val_pred_proba)
    val_ap = average_precision_score(val_labels, val_pred_proba)
    logger.info(f"Validation — AUC: {val_auc:.4f}, AP: {val_ap:.4f}")

    best_threshold = _pick_threshold(val_labels, val_pred_proba)
    logger.info(f"Best threshold is: {best_threshold:.4f}")

    # Test predictions
    logger.info("Making final predictions on test set...")
    test_pred_proba = predict_with_model(model, test_ids, batch_size=batch_size, device=device)
    test_pred = (test_pred_proba >= best_threshold).astype(int)

    # Calculate metrics
    test_auc = roc_auc_score(test_labels, test_pred_proba)
    test_ap = average_precision_score(test_labels, test_pred_proba)
    test_accuracy = accuracy_score(test_labels, test_pred)
    test_precision = precision_score(test_labels, test_pred, zero_division=0)
    test_recall = recall_score(test_labels, test_pred, zero_division=0)
    test_f1 = f1_score(test_labels, test_pred, zero_division=0)

    results = {
        'auc': test_auc,
        'ap': test_ap,
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'val_auc': val_auc,
        'val_ap': val_ap,
        'best_threshold': best_threshold,
        'training_history': history
    }

    logger.info(f"Fraud detection completed — "
                f"AUC: {test_auc:.4f}, Val AUC: {val_auc:.4f}, AP: {test_ap:.4f}, "
                f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
                f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    return results


def run_fraud_detection_experiments(
        data_path, is_directed, batch_ts_size, sliding_window_duration,
        embedding_mode, walk_length, num_walks_per_node, edge_picker,
        embedding_dim, n_epochs, streaming_embedding_use_gpu,
        fraud_detection_use_gpu, n_runs, batch_size, word2vec_n_workers,
        word2vec_batch_epochs, output_path
):
    """Run fraud detection experiments on DGraphFin dataset."""
    logger.info("Starting DGraphFin fraud detection experiments")

    # Load data
    (node_features, edges_df, train_ids, train_labels,
     val_ids, val_labels, test_ids, test_labels) = load_dgraphfin_data(data_path)

    # Prepare edge data for temporal walks
    train_sources = edges_df['u'].to_numpy()
    train_targets = edges_df['i'].to_numpy()
    train_timestamps = edges_df['ts'].to_numpy()

    # Get all relevant node IDs
    all_node_ids = np.concatenate([
        train_sources, train_targets, train_ids, val_ids, test_ids
    ])
    max_node_id = int(all_node_ids.max())
    logger.info(f"Maximum node ID in dataset: {max_node_id}")

    # Prepare node features
    node_features_tensor, feature_scaler = prepare_node_features(node_features, train_ids)

    logger.info("=" * 60)
    logger.info(f"Computing embeddings with {embedding_mode} approach...")
    logger.info("=" * 60)

    # Train embeddings
    if embedding_mode == 'streaming':
        embeddings = train_embeddings_streaming_approach(
            train_sources=train_sources,
            train_targets=train_targets,
            train_timestamps=train_timestamps,
            batch_ts_size=batch_ts_size,
            sliding_window_duration=sliding_window_duration,
            is_directed=is_directed,
            walk_length=walk_length,
            num_walks_per_node=num_walks_per_node,
            edge_picker=edge_picker,
            embedding_dim=embedding_dim,
            walk_use_gpu=streaming_embedding_use_gpu,
            word2vec_n_workers=word2vec_n_workers,
            batch_epochs=word2vec_batch_epochs
        )
    else:
        embeddings = train_embeddings_full_approach(
            train_sources=train_sources,
            train_targets=train_targets,
            train_timestamps=train_timestamps,
            is_directed=is_directed,
            walk_length=walk_length,
            num_walks_per_node=num_walks_per_node,
            edge_picker=edge_picker,
            embedding_dim=embedding_dim,
            walk_use_gpu=streaming_embedding_use_gpu,
            word2vec_n_workers=word2vec_n_workers
        )

    # Convert embeddings to tensor
    embedding_tensor = get_embedding_tensor(embeddings, max_node_id)

    device = 'cuda' if fraud_detection_use_gpu and torch.cuda.is_available() else 'cpu'
    results = {}

    logger.info("=" * 60)
    logger.info("TRAINING FRAUD DETECTION MODEL (Attention-based Fusion)")
    logger.info("=" * 60)

    for run in range(n_runs):
        logger.info(f"\n--- Run {run + 1}/{n_runs} ---")

        current_results = evaluate_fraud_detection(
            train_ids=train_ids,
            train_labels=train_labels,
            val_ids=val_ids,
            val_labels=val_labels,
            test_ids=test_ids,
            test_labels=test_labels,
            embedding_tensor=embedding_tensor,
            node_features_tensor=node_features_tensor,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device
        )

        for key in current_results.keys():
            if key not in results:
                results[key] = []
            results[key].append(current_results[key])

    logger.info(f"\nFraud Detection Results (Attention-based Fusion):")
    logger.info("=" * 80)

    for metric in ['auc', 'ap', 'accuracy', 'precision', 'recall', 'f1_score']:
        if metric in results:
            values = results[metric]
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Add metadata to results
        results['metadata'] = {
            'embedding_mode': embedding_mode,
            'batch_ts_size': batch_ts_size,
            'sliding_window_duration': sliding_window_duration,
            'embedding_dim': embedding_dim,
            'n_runs': n_runs
        }

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DGraphFin Fraud Detection with Temporal Embeddings")

    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dgraphfin.npz file')
    parser.add_argument('--batch_ts_size', type=int, required=True,
                        help='Time duration per batch for streaming approach')
    parser.add_argument('--sliding_window_duration', type=int, required=True,
                        help='Sliding window duration for temporal random walk')
    parser.add_argument('--is_directed', type=lambda x: x.lower() == 'true', required=True,
                        help='Whether the graph is directed (true/false)')

    # Embedding mode
    parser.add_argument('--embedding_mode', type=str, default='streaming',
                        choices=['streaming', 'full'],
                        help='Embedding mode - streaming or full')

    # Model parameters
    parser.add_argument('--walk_length', type=int, default=40,
                        help='Maximum length of random walks')
    parser.add_argument('--num_walks_per_node', type=int, default=20,
                        help='Number of walks to generate per node')
    parser.add_argument('--edge_picker', type=str, default='ExponentialIndex',
                        help='Edge picker for random walks')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimensionality of node embeddings')

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of epochs for neural network training')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of experimental runs for averaging results')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')

    # GPU settings
    parser.add_argument('--streaming_embedding_use_gpu', action='store_true',
                        help='Enable GPU acceleration for streaming embedding approach')
    parser.add_argument('--fraud_detection_use_gpu', action='store_true',
                        help='Enable GPU acceleration for fraud detection neural network')

    # Other settings
    parser.add_argument('--word2vec_n_workers', type=int, default=8,
                        help='Number of workers for Word2Vec training')
    parser.add_argument('--word2vec_batch_epochs', type=int, default=10,
                        help='Number of batch epochs for incremental Word2Vec training')
    parser.add_argument('--output_path', type=str, default=None,
                        help='File path to save results (optional)')

    args = parser.parse_args()

    # Run experiments
    results = run_fraud_detection_experiments(
        data_path=args.data_path,
        is_directed=args.is_directed,
        batch_ts_size=args.batch_ts_size,
        sliding_window_duration=args.sliding_window_duration,
        embedding_mode=args.embedding_mode,
        walk_length=args.walk_length,
        num_walks_per_node=args.num_walks_per_node,
        edge_picker=args.edge_picker,
        embedding_dim=args.embedding_dim,
        n_epochs=args.n_epochs,
        streaming_embedding_use_gpu=args.streaming_embedding_use_gpu,
        fraud_detection_use_gpu=args.fraud_detection_use_gpu,
        n_runs=args.n_runs,
        batch_size=args.batch_size,
        word2vec_n_workers=args.word2vec_n_workers,
        word2vec_batch_epochs=args.word2vec_batch_epochs,
        output_path=args.output_path
    )
