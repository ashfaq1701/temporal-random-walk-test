import argparse
import logging
import os
import pickle

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from temporal_random_walk import TemporalRandomWalk
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_processing.log')
        ]
    )
    return logging.getLogger(__name__)


def _load_edges(data_path, logger):
    edges_path = os.path.join(data_path, 'edges.csv')
    df = pd.read_csv(edges_path)
    df = df[["u", "i", "ts"]].copy()
    df = df.dropna().astype({"u": np.int32, "i": np.int32, "ts": np.int64})
    logger.info(f"Loaded edges: {len(df):,} rows")
    return df


def _load_labelled_splits(data_path, logger):
    labels = pd.read_csv(os.path.join(data_path, 'labels.csv'), usecols=['id', 'label'])
    labels_dict = {int(i): (1 if lbl == 'bot' else 0) for i, lbl in zip(labels['id'], labels['label'])}

    split = pd.read_csv(os.path.join(data_path, 'split.csv'), usecols=['id', 'split'])

    def pack(name: str):
        ids_all = split.loc[split['split'] == name, 'id'].astype(int).to_numpy()
        ids = [i for i in ids_all if i in labels_dict]
        y = [labels_dict[i] for i in ids]
        return np.array(ids, dtype=np.int32), np.array(y, dtype=np.int32)

    train_ids, train_y = pack('train')
    val_ids,   val_y   = pack('val')
    test_ids,  test_y  = pack('test')

    logger.info(f"train={len(train_ids):,}, val={len(val_ids):,}, test={len(test_ids):,}")
    return train_ids, train_y, val_ids, val_y, test_ids, test_y


def _embedding_tensor_from_store(embedding_store, required_max_id, logger):
    if not embedding_store:
        raise RuntimeError("Empty embedding_store")

    emb_dim = len(next(iter(embedding_store.values())))
    size = int(required_max_id) + 1
    mat = torch.zeros((size, emb_dim), dtype=torch.float32)

    filled = 0
    for nid, vec in embedding_store.items():
        if nid < size:
            mat[nid] = torch.from_numpy(vec)
            filled += 1
    logger.info(f"Embedding tensor: shape={tuple(mat.shape)}, rows_filled={filled:,}")
    return mat


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


class BotPredictionModel(nn.Module):
    def __init__(self, node_embeddings_tensor: torch.Tensor):
        super().__init__()

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

    def forward(self, nodes):
        device = next(self.parameters()).device
        nodes = nodes.to(device)

        node_emb = self.embedding_lookup(nodes)

        x = F.relu(self.fc1(node_emb))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        return self.fc_out(x)


def train_bot_detection_model(
    train_ids, train_labels,
    val_ids, val_labels,
    test_ids, test_labels,
    embedding_store, link_prediction_use_gpu,
    logger
):
    device = "cuda" if (link_prediction_use_gpu and torch.cuda.is_available()) else "cpu"

    required_max_id = int(max(
        max(train_ids, default=0),
        max(val_ids, default=0),
        max(test_ids, default=0),
        max(embedding_store.keys(), default=0)
    ))

    emb_tensor = _embedding_tensor_from_store(embedding_store, required_max_id, logger)
    model = BotPredictionModel(emb_tensor).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    stopper = EarlyStopping(mode="max", patience=5)

    def _mk_loader(ids, y, bs=8192, train=False):
        x = torch.tensor(ids, dtype=torch.long)
        t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        ds = TensorDataset(x, t)
        return DataLoader(ds, batch_size=bs, shuffle=train)

    train_loader = _mk_loader(train_ids, train_labels, train=True)
    val_loader = _mk_loader(val_ids, val_labels)
    test_loader = _mk_loader(test_ids, test_labels)

    def _eval(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device);
                y = y.to(device)
                prob = torch.sigmoid(model(x))
                ys.append(y.cpu().numpy())
                ps.append(prob.cpu().numpy())
        y = np.vstack(ys).ravel()
        p = np.vstack(ps).ravel()
        yhat = (p >= 0.5).astype(np.int32)
        out = {
            "accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0),
        }
        if len(np.unique(y)) > 1:
            out["auc"] = roc_auc_score(y, p)
        return out

    epochs = 20
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x = x.to(device);
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.item())

        val_metrics = _eval(val_loader)
        msg = "  ".join(f"{k.upper()} {v:.4f}" for k, v in val_metrics.items())
        logger.info(f"Epoch {ep + 1}/{epochs}  TrainLoss {total / len(train_loader):.4f}  |  VAL {msg}")

        monitor = val_metrics.get("auc", val_metrics["f1"])
        if stopper(monitor, model):
            logger.info(f"Early stopping at epoch {ep + 1}")
            break

    test_metrics = _eval(test_loader)
    logger.info("TEST  " + "  ".join(f"{k.upper()} {v:.4f}" for k, v in test_metrics.items()))
    return test_metrics




def train_streaming_embeddings(edges, batch_ts_size, sliding_window_duration, embedding_use_gpu, logger):
    ts_min = int(edges["ts"].min())
    ts_max = int(edges["ts"].max()) + 1  # half-open
    num_batches = max(1, math.ceil((ts_max - ts_min) / int(batch_ts_size)))

    logger.info(
        f"Streaming: batch_ts_size={batch_ts_size} sec (~{batch_ts_size / 86400:.2f} days), "
        f"batches={num_batches}, window={sliding_window_duration} sec (~{sliding_window_duration / 86400:.2f} days)"
    )

    trw = TemporalRandomWalk(
        is_directed=True,
        use_gpu=bool(embedding_use_gpu),
        max_time_capacity=int(sliding_window_duration)
    )

    walk_length = 100
    num_walks_per_node = 10
    edge_picker = "ExponentialIndex"
    embedding_dim = 128
    w2v_workers = max(1, min(16, os.cpu_count() or 4))

    w2v = None
    total_walks = 0

    for b in range(num_batches):
        start = ts_min + b * batch_ts_size
        end = min(ts_min + (b + 1) * batch_ts_size, ts_max)

        batch = edges[(edges.ts >= start) & (edges.ts < end)]
        if batch.empty:
            logger.info(f"Bucket {b + 1}/{num_batches}: 0 edges — skip")
            continue

        trw.add_multiple_edges(
            batch["u"].to_numpy(np.int32),
            batch["i"].to_numpy(np.int32),
            batch["ts"].to_numpy(np.int64),
        )

        # Forward / backward walks
        wf, _, lf = trw.get_random_walks_and_times_for_all_nodes(
            max_walk_len=walk_length,
            num_walks_per_node=max(1, num_walks_per_node // 2),
            walk_bias=edge_picker,
            initial_edge_bias="Uniform",
            walk_direction="Forward_In_Time",
        )
        wb, _, lb = trw.get_random_walks_and_times_for_all_nodes(
            max_walk_len=walk_length,
            num_walks_per_node=num_walks_per_node - max(1, num_walks_per_node // 2),
            walk_bias=edge_picker,
            initial_edge_bias="Uniform",
            walk_direction="Backward_In_Time",
        )

        # Avg walk lengths (forward/backward)
        fwd_n = int(len(lf))
        bwd_n = int(len(lb))
        fwd_mean = float(np.mean(lf)) if fwd_n else 0.0
        bwd_mean = float(np.mean(lb)) if bwd_n else 0.0
        logger.info(
            f"Bucket {b + 1}/{num_batches}: "
            f"forward_walks={fwd_n}, avg_len={fwd_mean:.2f} | "
            f"backward_walks={bwd_n}, avg_len={bwd_mean:.2f}"
        )

        walks = np.concatenate([wf, wb], axis=0) if len(wf) else wb
        lens = np.concatenate([lf, lb], axis=0) if len(lf) else lb

        clean = []
        for w, L in zip(walks, lens):
            L = int(L)
            if L > 1:
                clean.append([str(int(n)) for n in w[:L]])

        logger.info(f"Bucket {b + 1}/{num_batches}: edges={len(batch):,}, walks={len(clean):,}")
        if not clean:
            continue

        if w2v is None:
            w2v = Word2Vec(vector_size=embedding_dim, window=10, min_count=1,
                           workers=w2v_workers, sg=1, seed=42)
            w2v.build_vocab(clean)
            w2v.train(clean, total_examples=len(clean), epochs=w2v.epochs)
        else:
            w2v.build_vocab(clean, update=True)
            w2v.train(clean, total_examples=len(clean), epochs=w2v.epochs)

        total_walks += len(clean)

    if w2v is None:
        logger.warning("No walks generated; returning empty embedding store.")
        return {}

    logger.info(f"Word2Vec trained over {total_walks:,} walks; vocab={len(w2v.wv):,}")
    emb_store = {int(k): w2v.wv[k].astype(np.float32) for k in w2v.wv.index_to_key}
    return emb_store


def run_bot_detection_pipeline_step(
    batch_ts_size,
    sliding_window_duration,
    data_dir,
    embedding_use_gpu,
    link_prediction_use_gpu,
    logger
):
    edges = _load_edges(data_dir, logger)
    train_data, train_labels, val_data, val_labels, test_data, test_labels = _load_labelled_splits(data_dir, logger)

    embedding_store = train_streaming_embeddings(
        edges=edges,
        batch_ts_size=batch_ts_size,
        sliding_window_duration=sliding_window_duration,
        embedding_use_gpu=embedding_use_gpu,
        logger=logger
    )

    test_metrics = train_bot_detection_model(
        train_ids=train_data, train_labels=train_labels,
        val_ids=val_data, val_labels=val_labels,
        test_ids=test_data, test_labels=test_labels,
        embedding_store=embedding_store,
        link_prediction_use_gpu=link_prediction_use_gpu,
        logger=logger
    )
    return test_metrics


def run_bot_detection_pipeline(
    batch_ts_size,
    sliding_window_duration,
    n_runs,
    data_dir,
    embedding_use_gpu,
    link_prediction_use_gpu,
    logger
):
    merged = {}

    for run in range(n_runs):
        logger.info(f"Running {run + 1} / {n_runs}")

        metrics = run_bot_detection_pipeline_step(
            batch_ts_size=batch_ts_size,
            sliding_window_duration=sliding_window_duration,
            data_dir=data_dir,
            embedding_use_gpu=embedding_use_gpu,
            link_prediction_use_gpu=link_prediction_use_gpu,
            logger=logger
        )

        # append each metric to its list
        for k, v in metrics.items():
            merged.setdefault(k, []).append(float(v))

    with open(os.path.join('results', 'twibot_results.pickle'), "wb") as f:
        pickle.dump(merged, f)

    logger.info(f"Saved merged test metrics to {os.path.join('results', 'twibot_results.pickle')}")
    return merged

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting Twibot-22 node classification.")

    parser = argparse.ArgumentParser(description="Script to do node classification with twibot 22")

    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Base directory containing data files'
    )

    parser.add_argument(
        '--batch_ts_size', type=int, default=31_556_952,
        help='Batch duration (seconds) per streaming step'
    )

    parser.add_argument(
        '--sliding_window_duration', type=int, default=157_784_760,
        help='Sliding window duration in seconds'
    )

    parser.add_argument('--embedding_use_gpu', action='store_true',
                        help='Enable GPU acceleration for embedding approach')

    parser.add_argument('--link_prediction_use_gpu', action='store_true',
                        help='Enable GPU acceleration for link prediction neural network')

    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')

    args = parser.parse_args()

    _ = run_bot_detection_pipeline(
        args.batch_ts_size,
        args.sliding_window_duration,
        args.n_runs,
        args.data_dir,
        args.embedding_use_gpu,
        args.link_prediction_use_gpu,
        logger
    )
