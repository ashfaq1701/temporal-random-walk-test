import argparse
import logging
import os
import pickle
import random
import warnings
from contextlib import contextmanager
from pathlib import Path
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    average_precision_score, roc_curve
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

    def __init__(self, mode='min', patience=5, min_delta=1e-4, restore_best_weights=True):
        assert mode in ['min', 'max']
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)
        self.reset()

    def reset(self):
        self.best_score = float('inf') if self.mode == 'min' else -float('inf')
        self.counter = 0
        self.best_weights = None
        self.best_epoch = -1
        self.stopped = False

    def _improved(self, score: float) -> bool:
        # Treat NaN/inf as no improvement
        if score is None or not np.isfinite(score):
            return False
        if self.mode == 'min':
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)

    def __call__(self, current_score, model, epoch=None):
        # Cast to float to avoid numpy scalar weirdness
        score = float(current_score)

        if self._improved(score):
            self.best_score = score
            self.counter = 0
            if epoch is not None:
                self.best_epoch = int(epoch)
            if self.restore_best_weights:
                # Deep copy tensors to CPU to avoid GPU memory growth & aliasing
                with torch.no_grad():
                    self.best_weights = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped = True
            if self.restore_best_weights and self.best_weights is not None:
                with torch.no_grad():
                    model.load_state_dict(self.best_weights)
            return True  # signal to stop

        return False


class ResBlock(nn.Module):
    def __init__(self, d: int, p: float = 0.30):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.fc2 = nn.Linear(d, d)
        self.drop = nn.Dropout(p)
        self.act  = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        r = x
        x = self.act(self.n1(x))
        x = self.drop(self.fc1(x))
        x = self.act(self.n2(x))
        x = self.fc2(x)
        return self.act(x + r)

class FraudDetectionModel(nn.Module):
    def __init__(self, node_embeddings_tensor: torch.Tensor, node_features_tensor: torch.Tensor,
                 proj_dim: int = 128, head_width: int = 256, dropout: float = 0.30, n_blocks: int = 3):
        super().__init__()
        self.embedding_lookup = nn.Embedding.from_pretrained(node_embeddings_tensor, freeze=True)
        self.register_buffer("node_features", node_features_tensor, persistent=False)

        e_dim = int(node_embeddings_tensor.shape[1])
        f_dim = int(node_features_tensor.shape[1])

        self.emb_norm  = nn.LayerNorm(e_dim)
        self.feat_norm = nn.LayerNorm(f_dim)
        self.emb_proj  = nn.Linear(e_dim, proj_dim)
        self.feat_proj = nn.Linear(f_dim, proj_dim)
        self.proj_drop = nn.Dropout(p=0.10)  # small input dropout
        self.act       = nn.LeakyReLU(0.1, inplace=True)

        self.fuse = nn.Linear(proj_dim * 3 + 1, head_width)
        self.fuse_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[ResBlock(head_width, p=dropout) for _ in range(n_blocks)])

        self.pre_out_norm = nn.LayerNorm(head_width)
        self.fc_out = nn.utils.parametrizations.weight_norm(nn.Linear(head_width, 1))

    def forward(self, nodes: torch.Tensor):
        dev = next(self.parameters()).device
        nodes = nodes.to(dev)

        e = self.embedding_lookup(nodes)
        x = self.node_features[nodes]

        e = self.act(self.emb_proj(self.emb_norm(e)))
        x = self.act(self.feat_proj(self.feat_norm(x)))
        e = self.proj_drop(e); x = self.proj_drop(x)

        inter = e * x
        cos   = F.cosine_similarity(e, x, dim=1, eps=1e-8).unsqueeze(1)

        z = torch.cat([e, x, inter, cos], dim=1)
        h = self.fuse_drop(self.fuse(z))
        h = self.blocks(h)
        h = self.act(self.pre_out_norm(h))
        return self.fc_out(h)


class BinaryStratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, shuffle=True, seed=None):
        super().__init__()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self._epoch = 0

        # 1) Separate class item indices (0 vs 1)
        if hasattr(labels, "detach"):  # torch.Tensor
            labels = labels.detach().cpu().tolist()
        labels = [int(l) for l in labels]
        self.idx0 = [i for i, y in enumerate(labels) if y == 0]
        self.idx1 = [i for i, y in enumerate(labels) if y == 1]

        # 2) Global proportion (normalized)
        total = len(self.idx0) + len(self.idx1)
        p1 = (len(self.idx1) / total) if total else 0.0

        # 3) Fixed count in batch to maintain the proportion
        self.n1_per_batch = int(round(self.batch_size * p1))
        self.n0_per_batch = self.batch_size - self.n1_per_batch

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self):
        total = len(self.idx0) + len(self.idx1)
        return math.ceil(total / self.batch_size)

    def _rng(self):
        r = random.Random()
        base = 0 if self.seed is None else int(self.seed)  # Set a default seed
        epoch = getattr(self, "_epoch", 0)
        r.seed(base + epoch)
        return r

    def __iter__(self):
        rng = self._rng()

        # Working copies; shuffle once per epoch
        idx0 = self.idx0[:]
        idx1 = self.idx1[:]
        if self.shuffle:
            rng.shuffle(idx0)
            rng.shuffle(idx1)

        p0 = p1 = 0  # pointers into buckets
        rem0, rem1 = len(idx0), len(idx1)

        while (rem0 + rem1) > 0:
            batch = []

            # 3) Take up to the fixed per-batch counts
            take1 = min(self.n1_per_batch, rem1)
            take0 = min(self.n0_per_batch, rem0)

            if take1 > 0:
                batch.extend(idx1[p1 : p1 + take1]); p1 += take1; rem1 -= take1
            if take0 > 0:
                batch.extend(idx0[p0 : p0 + take0]); p0 += take0; rem0 -= take0

            # 3b) If one bucket ran short, 4) top up from the other
            need = self.batch_size - len(batch)
            if need > 0 and rem1 > 0:
                extra = min(need, rem1)
                batch.extend(idx1[p1 : p1 + extra]); p1 += extra; rem1 -= extra
                need -= extra
            if need > 0 and rem0 > 0:
                extra = min(need, rem0)
                batch.extend(idx0[p0 : p0 + extra]); p0 += extra; rem0 -= extra
                need -= extra

            if self.shuffle and len(batch) > 1:
                rng.shuffle(batch)
            yield batch


def get_optimized_loss_function(y_train, device):
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    imbalance_ratio = neg_count / pos_count
    pos_weight = torch.tensor(imbalance_ratio, device=device, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def calculate_gradient_norm(model):
    total_norm = 0.0
    param_count = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1

    if param_count == 0:
        return 0.0

    return total_norm ** 0.5


def seed_worker(worker_id: int, base_seed: int = 42):
    """
    Deterministic worker seeding for PyTorch DataLoader (spawn-safe on macOS).
    Use as: worker_init_fn=seed_worker
    """
    seed = int(base_seed) + int(worker_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_fraud_detection_model(model, X_train, y_train, X_val, y_val,
                                batch_size, epochs=50, device='cpu', patience=5,
                                label_smooth: float = 0.05, schedule_finish_epochs: int = 8,
                                use_amp: bool = True):  # `use_amp` ignored (no autocast)
    logger.info(f"Training fraud detection model on {len(X_train):,} samples")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = get_optimized_loss_function(y_train, device)  # BCEWithLogits(pos_weight)

    early_stopping = EarlyStopping(mode='max', patience=patience)

    # Datasets / loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long),
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_dataset   = TensorDataset(torch.tensor(X_val,   dtype=torch.long),
                                  torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1))

    optimal_batch_size = min(batch_size, 96)
    num_workers = min(4, os.cpu_count() or 0)

    # Deterministic workers (seed_worker must be defined at module scope)
    g = torch.Generator(device='cpu'); g.manual_seed(42)

    train_sampler = BinaryStratifiedBatchSampler(y_train, batch_size=optimal_batch_size, shuffle=True, seed=42)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              pin_memory=(device == 'cuda'), num_workers=num_workers,
                              persistent_workers=(num_workers > 0),
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=min(optimal_batch_size * 4, 2048),
                            shuffle=False, pin_memory=(device == 'cuda'), num_workers=num_workers,
                            persistent_workers=(num_workers > 0),
                            worker_init_fn=seed_worker, generator=g)

    # Warmup + cosine; finish decay by schedule_finish_epochs, then hold at final_lr
    warmup_epochs   = 1
    steps_per_epoch = max(1, len(train_loader))
    total_steps     = max(1, min(epochs, schedule_finish_epochs) * steps_per_epoch)
    warmup_steps    = max(1, warmup_epochs * steps_per_epoch)
    base_lr, final_lr = 1e-4, 1e-6

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = step - warmup_steps
        T = max(1, total_steps - warmup_steps)
        if t >= T:
            return (final_lr / base_lr)          # clamp at floor
        return (final_lr / base_lr) + (1 - final_lr / base_lr) * 0.5 * (1 + math.cos(math.pi * t / T))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': [], 'train_f1': [], 'val_f1': []}
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        train_sampler.set_epoch(epoch)

        total_train_loss = 0.0
        train_probs, train_tgts = [], []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_x)
            # Asymmetric label smoothing: smooth only positives (1 -> 1-ε; 0 stays 0)
            if label_smooth > 0.0:
                targets = torch.where(batch_y > 0.5, batch_y * (1.0 - label_smooth), batch_y)
            else:
                targets = batch_y
            loss = criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            scheduler.step()  # per-batch
            total_train_loss += float(loss.detach().cpu())
            train_probs.append(torch.sigmoid(logits).detach().cpu())
            train_tgts.append(batch_y.detach().cpu())

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_probs, val_tgts = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                logits = model(batch_x)
                if label_smooth > 0.0:
                    targets = torch.where(batch_y > 0.5, batch_y * (1.0 - label_smooth), batch_y)
                else:
                    targets = batch_y
                vloss = criterion(logits, targets)

                total_val_loss += float(vloss.detach().cpu())
                val_probs.append(torch.sigmoid(logits).cpu())
                val_tgts.append(batch_y.cpu())

        train_probs = torch.cat(train_probs).numpy().ravel()
        train_tgts  = torch.cat(train_tgts).numpy().ravel()
        val_probs   = torch.cat(val_probs).numpy().ravel()
        val_tgts    = torch.cat(val_tgts).numpy().ravel()

        train_auc = roc_auc_score(train_tgts, train_probs)
        val_auc   = roc_auc_score(val_tgts,   val_probs)
        train_f1  = f1_score(train_tgts, (train_probs>=0.5).astype(int), zero_division=0)
        val_f1    = f1_score(val_tgts,   (val_probs  >=0.5).astype(int), zero_division=0)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(total_val_loss / max(1, len(val_loader)))
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        epoch_pbar.set_postfix({'train_auc': f'{train_auc:.4f}',
                                'val_auc': f'{val_auc:.4f}',
                                'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        logger.info(f"Epoch {epoch + 1}/{epochs} — Train AUC: {train_auc:.4f}, "
                    f"Val AUC: {val_auc:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        if early_stopping(val_auc, model, epoch=epoch):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        if device == 'cuda':
            torch.cuda.empty_cache()

    # If we never early-stopped, still restore the best snapshot
    if (early_stopping.restore_best_weights
            and early_stopping.best_weights is not None
            and not getattr(early_stopping, "stopped", False)):
        with torch.no_grad():
            model.load_state_dict(early_stopping.best_weights)

    epoch_pbar.close()
    return history


def predict_with_model(model, X_test, batch_size, device='cpu'):
    """Make predictions with trained model."""
    model = model.to(device)
    model.eval()

    logger.info(f"Making predictions on {len(X_test):,} samples")

    # Avoid unnecessary copy; ensure long dtype for embedding lookups
    x_tensor = torch.as_tensor(X_test, dtype=torch.long)
    test_dataset = TensorDataset(x_tensor)

    num_workers = min(8, os.cpu_count() or 0)
    use_workers = num_workers > 0

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device == 'cuda'),
        num_workers=num_workers,
        persistent_workers=use_workers,
        prefetch_factor=(4 if use_workers else None),
    )

    logger.info(f"Processing {len(test_loader)} batches")

    preds_cpu_chunks = []
    with torch.inference_mode():
        prediction_pbar = tqdm(test_loader, desc="Predicting", unit="batch")
        for (batch_x,) in prediction_pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)

            # Move each chunk to CPU immediately to avoid device-memory buildup
            preds_cpu_chunks.append(probs.detach().cpu())

            prediction_pbar.set_postfix({
                'batches_processed': len(preds_cpu_chunks),
                'batch_size': len(batch_x),
            })
        prediction_pbar.close()

    all_predictions = torch.cat(preds_cpu_chunks, dim=0).numpy().ravel()

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


def get_embedding_tensor(embedding_dict, max_node_id):
    """Convert embedding dictionary to tensor (L2-normalized rows)."""
    if not embedding_dict:
        raise ValueError("Empty embedding dictionary")

    dim = len(next(iter(embedding_dict.values())))
    emb = torch.zeros((max_node_id + 1, dim), dtype=torch.float32)

    # Fill known embeddings
    for node_id, vec in embedding_dict.items():
        if node_id <= max_node_id:
            emb[node_id] = torch.tensor(vec, dtype=torch.float32)

    # Initialize missing rows with mean of existing embeddings
    missing_mask = (emb.abs().sum(dim=1) == 0)
    if missing_mask.any():
        filled = emb[~missing_mask]
        if filled.numel() > 0:
            emb[missing_mask] = filled.mean(dim=0)

    # Row-wise L2 normalization (important for BN/proj stability)
    with torch.no_grad():
        norms = emb.norm(p=2, dim=1, keepdim=True).clamp_min_(1e-8)
        emb.div_(norms)

    logger.info(f"Embedding tensor: shape={emb.shape}, "
                f"filled={(~missing_mask).sum().item():,}, "
                f"missing={missing_mask.sum().item():,}")
    return emb


def _pick_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    tnr = 1 - fpr
    gmean_scores = np.sqrt(tpr * tnr)
    best_idx = np.argmax(gmean_scores)
    best_threshold = thresholds[best_idx]
    return float(best_threshold)


def evaluate_fraud_detection(
        train_ids, train_labels, val_ids, val_labels, test_ids, test_labels,
        embedding_tensor, node_features_tensor, n_epochs, batch_size, device
):
    """Evaluate fraud detection performance."""
    logger.info(f"Fraud detection — train: {len(train_ids):,}, val: {len(val_ids):,}, test: {len(test_ids):,}")

    model = FraudDetectionModel(embedding_tensor, node_features_tensor).to(device)

    history = train_fraud_detection_model(
        model, train_ids, train_labels, val_ids, val_labels,
        batch_size=batch_size, epochs=n_epochs, device=device, patience=10
    )

    # Faster eval without changing metrics
    eval_bs = min(batch_size * 4, 2048)

    # Validation predictions
    logger.info("Making predictions on validation set...")
    val_pred_proba = predict_with_model(model, val_ids, batch_size=eval_bs, device=device)
    val_auc = roc_auc_score(val_labels, val_pred_proba)
    val_ap  = average_precision_score(val_labels, val_pred_proba)
    logger.info(f"Validation — AUC: {val_auc:.4f}, AP: {val_ap:.4f}")

    best_threshold = _pick_threshold(val_labels, val_pred_proba)
    val_pred_bin = (val_pred_proba >= best_threshold).astype(int)
    val_f1_at_thr = f1_score(val_labels, val_pred_bin, zero_division=0)
    logger.info(f"Best threshold: {best_threshold:.4f} | Val F1@thr: {val_f1_at_thr:.4f}")

    # Test predictions (threshold decided on val)
    logger.info("Making final predictions on test set...")
    test_pred_proba = predict_with_model(model, test_ids, batch_size=eval_bs, device=device)
    test_pred = (test_pred_proba >= best_threshold).astype(int)

    # Metrics
    test_auc = roc_auc_score(test_labels, test_pred_proba)
    test_ap  = average_precision_score(test_labels, test_pred_proba)
    test_accuracy  = accuracy_score(test_labels, test_pred)
    test_precision = precision_score(test_labels, test_pred, zero_division=0)
    test_recall    = recall_score(test_labels, test_pred, zero_division=0)
    test_f1        = f1_score(test_labels, test_pred, zero_division=0)

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

    logger.info(
        f"Fraud detection completed — AUC: {test_auc:.4f}, Val AUC: {val_auc:.4f}, "
        f"AP: {test_ap:.4f}, Acc: {test_accuracy:.4f}, P: {test_precision:.4f}, "
        f"R: {test_recall:.4f}, F1: {test_f1:.4f}"
    )
    return results



def prepare_node_features_all(node_features: np.ndarray,
                              train_ids: np.ndarray) -> torch.Tensor:
    scaler = StandardScaler()
    scaler.fit(node_features[train_ids])
    feats = scaler.transform(node_features).astype(np.float32)
    return torch.from_numpy(feats)


def build_graph_stats_features(edges_df: pd.DataFrame,
                               n_nodes: int,
                               recent_window: int = 100) -> np.ndarray:
    """
    Build cheap, high-signal graph statistics per node.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Must contain integer columns: 'u' (src), 'i' (dst), 'ts' (timestamp).
    n_nodes : int
        Total number of nodes; feature matrix will be [n_nodes, 6].
    recent_window : int, default 100
        Edges with ts >= (t_max - recent_window) are considered "recent".

    Returns
    -------
    feats : np.ndarray (float32) of shape [n_nodes, 6]
        Column order:
          0: log1p(out_degree_lifetime)
          1: log1p(in_degree_lifetime)
          2: log1p(total_degree_lifetime)
          3: recency in [0,1]  (0 = most recent activity; 1 = oldest/inactive)
          4: burstiness = recent_degree / max(1, total_degree)
          5: diversity  = unique_partners / max(1, total_degree)
    """
    # Pull columns as contiguous numpy arrays
    u  = edges_df['u'].to_numpy(dtype=np.int64, copy=False)
    i  = edges_df['i'].to_numpy(dtype=np.int64, copy=False)
    ts = edges_df['ts'].to_numpy(dtype=np.int64, copy=False)

    # Global time range
    tmin = int(ts.min()) if ts.size else 0
    tmax = int(ts.max()) if ts.size else 0
    span = max(1, tmax - tmin)

    # -------------------------
    # Lifetime degrees (np.bincount is fast and memory-friendly)
    # -------------------------
    deg_out = np.bincount(u, minlength=n_nodes).astype(np.float32)
    deg_in  = np.bincount(i, minlength=n_nodes).astype(np.float32)
    total_deg = deg_out + deg_in  # float32

    # -------------------------
    # Last activity timestamp per node (max over src/dst roles)
    # (vectorized via pandas groupby once per side for simplicity)
    # -------------------------
    last_ts = np.full(n_nodes, -1, dtype=np.int64)  # -1 => inactive by default

    if len(u) > 0:
        last_u = edges_df.groupby('u')['ts'].max()
        idx = last_u.index.to_numpy(dtype=np.int64, copy=False)
        vals = last_u.to_numpy(dtype=np.int64, copy=False)
        last_ts[idx] = np.maximum(last_ts[idx], vals)

    if len(i) > 0:
        last_i = edges_df.groupby('i')['ts'].max()
        idx = last_i.index.to_numpy(dtype=np.int64, copy=False)
        vals = last_i.to_numpy(dtype=np.int64, copy=False)
        # np.maximum with fancy indexing to merge dst-side maxima
        last_ts[idx] = np.maximum(last_ts[idx], vals)

    # Recency: normalized "time since last activity"
    recency = np.ones(n_nodes, dtype=np.float32)  # default 1.0 for inactive nodes
    active_mask = last_ts >= tmin
    if active_mask.any():
        recency[active_mask] = (tmax - last_ts[active_mask]) / float(span)

    # -------------------------
    # Recent degrees within a time window
    # -------------------------
    cutoff = tmax - int(recent_window)
    recent_mask = ts >= cutoff
    if recent_mask.any():
        deg_out_recent = np.bincount(u[recent_mask], minlength=n_nodes).astype(np.float32)
        deg_in_recent  = np.bincount(i[recent_mask], minlength=n_nodes).astype(np.float32)
        recent_deg = deg_out_recent + deg_in_recent
    else:
        recent_deg = np.zeros(n_nodes, dtype=np.float32)

    burstiness = recent_deg / np.maximum(1.0, total_deg)

    # -------------------------
    # Partner diversity: unique neighbors (src: unique dst; dst: unique src)
    # -------------------------
    uniq_out = edges_df.groupby('u')['i'].nunique()
    uniq_in  = edges_df.groupby('i')['u'].nunique()

    uniq_partners = np.zeros(n_nodes, dtype=np.float32)
    if not uniq_out.empty:
        idx = uniq_out.index.to_numpy(dtype=np.int64, copy=False)
        vals = uniq_out.to_numpy(dtype=np.int64, copy=False)
        uniq_partners[idx] += vals.astype(np.float32)
    if not uniq_in.empty:
        idx = uniq_in.index.to_numpy(dtype=np.int64, copy=False)
        vals = uniq_in.to_numpy(dtype=np.int64, copy=False)
        uniq_partners[idx] += vals.astype(np.float32)

    diversity = uniq_partners / np.maximum(1.0, total_deg)

    # -------------------------
    # Assemble feature matrix
    # -------------------------
    feats = np.empty((n_nodes, 6), dtype=np.float32)
    feats[:, 0] = np.log1p(deg_out)     # lifetime out-degree (log1p)
    feats[:, 1] = np.log1p(deg_in)      # lifetime in-degree  (log1p)
    feats[:, 2] = np.log1p(total_deg)   # lifetime total-degree (log1p)
    feats[:, 3] = recency               # 0 new … 1 old/inactive
    feats[:, 4] = burstiness            # recent / lifetime degree
    feats[:, 5] = diversity             # unique partners / lifetime degree

    return feats


def run_fraud_detection_experiments(
        data_path, stored_embedding_file_path, is_directed, batch_ts_size,
        sliding_window_duration, embedding_mode, walk_length, num_walks_per_node,
        edge_picker, embedding_dim, n_epochs, streaming_embedding_use_gpu,
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

    # 1) stabilize heavy-tailed raw features and add a zero-indicator for F10
    x = node_features.astype(np.float32, copy=False)
    heavy = [2, 3, 5, 6, 7, 8, 10]  # heavy-tailed dims from your audit
    x[:, heavy] = np.log1p(np.clip(x[:, heavy], 0, None))
    is_zero_F10 = (node_features[:, 10] == 0).astype(np.float32).reshape(-1, 1)

    # 2) graph stats
    graph_stats = build_graph_stats_features(edges_df, n_nodes=x.shape[0], recent_window=100)

    # 3) concatenate and scale (fit on train_ids inside prepare_node_features_all)
    node_features = np.concatenate([x, is_zero_F10, graph_stats], axis=1).astype(np.float32)
    node_features_tensor = prepare_node_features_all(node_features, train_ids)

    # (optional) sanity log
    logger.info(f"Augmented feature dim: {node_features.shape[1]}")

    logger.info("=" * 60)
    logger.info(f"Computing embeddings with {embedding_mode} approach...")
    logger.info("=" * 60)

    if stored_embedding_file_path is None or not os.path.exists(stored_embedding_file_path):
        logger.info('Computing embeddings ...')
        if stored_embedding_file_path is not None:
            logger.info(f'And saving in {stored_embedding_file_path}')

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

        if stored_embedding_file_path is not None:
            os.makedirs(os.path.dirname(stored_embedding_file_path), exist_ok=True)
            with open(stored_embedding_file_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Embeddings saved to {stored_embedding_file_path}")
    else:
        logger.info(f"Loading pre-trained embeddings from {stored_embedding_file_path}")
        with open(stored_embedding_file_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings for {len(embeddings)} nodes")

    # Convert embeddings to tensor
    embedding_tensor = get_embedding_tensor(embeddings, max_node_id)

    device = 'cuda' if fraud_detection_use_gpu and torch.cuda.is_available() else 'cpu'
    results = {}

    logger.info("=" * 60)
    logger.info("TRAINING FRAUD DETECTION MODEL")
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

    logger.info(f"\nFraud Detection Results:")
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
    parser.add_argument('--stored_embedding_file_path', type=str, default=None,
                        help='Path of default stored embedding file')
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
        stored_embedding_file_path=args.stored_embedding_file_path,
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
