import argparse
import logging
import random
import os
import pickle
import warnings
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
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from temporal_random_walk import TemporalRandomWalk
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
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


class ResBlock(nn.Module):
    def __init__(self, d, p=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(d)
        self.fc1 = nn.Linear(d, d)
        self.bn2 = nn.BatchNorm1d(d)
        self.fc2 = nn.Linear(d, d)
        self.drop = nn.Dropout(p)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        r = x
        x = self.act(self.bn1(x))
        x = self.drop(self.fc1(x))
        x = self.act(self.bn2(x))
        x = self.fc2(x)
        return self.act(x + r)

class FraudDetectionModel(nn.Module):
    """
    Frozen embedding + features → dual projection → interaction features → residual MLP head.
    Tokens are not used; attention is dropped for a stronger tabular head.
    """
    def __init__(self, node_embeddings_tensor: torch.Tensor, node_features_tensor: torch.Tensor,
                 proj_dim: int = 128, head_width: int = 256, n_blocks: int = 3, dropout: float = 0.2):
        super().__init__()
        # 1) Frozen lookup and feature buffer
        self.embedding_lookup = nn.Embedding.from_pretrained(node_embeddings_tensor, freeze=True)
        self.register_buffer("node_features", node_features_tensor, persistent=False)

        e_dim = int(node_embeddings_tensor.shape[1])
        f_dim = int(node_features_tensor.shape[1])

        # 2) Per-branch normalization+projection
        self.emb_bn   = nn.BatchNorm1d(e_dim)
        self.feat_bn  = nn.BatchNorm1d(f_dim)
        self.emb_proj = nn.Linear(e_dim, proj_dim)
        self.feat_proj= nn.Linear(f_dim, proj_dim)
        self.emb_act  = nn.LeakyReLU(0.1, inplace=True)
        self.feat_act = nn.LeakyReLU(0.1, inplace=True)

        # 3) Fusion: [emb_p, feat_p, emb_p ⊙ feat_p, cos_sim] → fuse to head_width
        self.fuse = nn.Linear(proj_dim*3 + 1, head_width)

        # 4) Residual head
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResBlock(head_width, p=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.pre_out_bn = nn.BatchNorm1d(head_width)
        self.pre_out_act= nn.LeakyReLU(0.1, inplace=True)

        # 5) Weight-norm on output sharpens margins
        self.fc_out = nn.utils.parametrizations.weight_norm(nn.Linear(head_width, 1))

    def forward(self, nodes: torch.Tensor):
        dev = next(self.parameters()).device
        nodes = nodes.to(dev)

        e = self.embedding_lookup(nodes)          # [B, E]
        x = self.node_features[nodes]             # [B, F]

        # Per-branch BN + projection
        e = self.emb_act(self.emb_proj(self.emb_bn(e)))
        x = self.feat_act(self.feat_proj(self.feat_bn(x)))

        # Interaction + cosine
        inter = e * x                              # Hadamard
        cos = F.cosine_similarity(e, x, dim=1, eps=1e-8).unsqueeze(1)  # [B,1]

        z = torch.cat([e, x, inter, cos], dim=1)  # [B, 3P+1]
        h = self.fuse(z)                           # [B, W]
        h = self.blocks(h)
        h = self.pre_out_act(self.pre_out_bn(h))
        return self.fc_out(h)                      # [B,1] (logit)


def train_fraud_detection_model(model, X_train, y_train, X_val, y_val,
                                batch_size, epochs=20, device='cpu', patience=5):
    logger.info(f"Training fraud detection model on {len(X_train):,} samples")
    model = model.to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, min_lr=1e-6
    )

    # Imbalance
    pos = int(y_train.sum()); neg = int(len(y_train) - pos)
    logger.info(f"Class balance — train: fraud={pos} ({pos / len(y_train):.2%}), normal={neg}")
    pos_weight = torch.tensor(neg / max(1, pos), device=device, dtype=torch.float32)

    # Loss with label smoothing
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    smooth_eps = 0.05

    # Datasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long),
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_dataset   = TensorDataset(torch.tensor(X_val, dtype=torch.long),
                                  torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

    # Safe DataLoader setup
    num_workers_raw = os.cpu_count() or 0
    num_workers = min(8, num_workers_raw)
    use_workers = num_workers > 0

    # Balanced sampler for train
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    class_count = torch.tensor([(y_train_tensor == 0).sum(), (y_train_tensor == 1).sum()], dtype=torch.float32)
    class_weight = 1.0 / class_count
    sample_weight = class_weight[(y_train_tensor.long())].numpy()
    sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        pin_memory=(device == 'cuda'), num_workers=num_workers,
        persistent_workers=use_workers, prefetch_factor=(4 if use_workers else None)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=min(batch_size * 4, 2048), shuffle=False,
        pin_memory=(device == 'cuda'), num_workers=num_workers,
        persistent_workers=use_workers, prefetch_factor=(4 if use_workers else None)
    )

    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': [], 'train_f1': [], 'val_f1': []}
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    early_stopping = EarlyStopping(mode='max', patience=patience)

    def _safe_auc(y_true, y_prob):
        return roc_auc_score(y_true, y_prob) if np.unique(y_true).size > 1 else np.nan

    for epoch in epoch_pbar:
        model.train()
        total_train_loss = 0.0
        train_probs, train_tgts = [], []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)

            y_smooth = batch_y * (1.0 - smooth_eps) + 0.5 * smooth_eps
            loss = criterion(logits, y_smooth).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += float(loss.detach().cpu())
            train_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            train_tgts.append(batch_y.detach().cpu().numpy())

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        model.eval()
        total_val_loss = 0.0
        val_probs, val_tgts = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                logits = model(batch_x)
                y_smooth = batch_y * (1.0 - smooth_eps) + 0.5 * smooth_eps
                loss = criterion(logits, y_smooth).mean()
                total_val_loss += float(loss.detach().cpu())

                val_probs.append(torch.sigmoid(logits).cpu().numpy())
                val_tgts.append(batch_y.cpu().numpy())

        # Stack metrics
        train_probs = np.concatenate(train_probs).ravel()
        train_tgts  = np.concatenate(train_tgts).ravel()
        val_probs   = np.concatenate(val_probs).ravel()
        val_tgts    = np.concatenate(val_tgts).ravel()

        train_auc = _safe_auc(train_tgts, train_probs)
        val_auc   = _safe_auc(val_tgts, val_probs)
        train_f1  = f1_score(train_tgts, (train_probs >= 0.5).astype(int), zero_division=0)
        val_f1    = f1_score(val_tgts,   (val_probs  >= 0.5).astype(int), zero_division=0)

        scheduler.step(val_auc if np.isfinite(val_auc) else 0.0)

        history['train_loss'].append(avg_train_loss);                      history['val_loss'].append(total_val_loss / max(1, len(val_loader)))
        history['train_auc'].append(train_auc);                             history['val_auc'].append(val_auc)
        history['train_f1'].append(train_f1);                               history['val_f1'].append(val_f1)

        epoch_pbar.set_postfix({'train_loss': f'{avg_train_loss:.4f}', 'val_auc': f'{val_auc:.4f}'})
        logger.info(f"Epoch {epoch + 1}/{epochs} — Train Loss: {avg_train_loss:.4f}, Val AUC: {val_auc:.4f}")

        if early_stopping(val_auc if np.isfinite(val_auc) else -1e9, model):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

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
    """Evaluate fraud detection performance."""
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
