import argparse
import pickle
import logging
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score
from temporal_random_walk import TemporalRandomWalk
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BATCH_SIZE = 200


@contextmanager
def suppress_word2vec_output():
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


def create_edge_feature(source, target, node_embeddings, edge_op):
    embedding_dim = len(next(iter(node_embeddings.values())))
    src_emb = node_embeddings.get(source, np.zeros(embedding_dim))
    tgt_emb = node_embeddings.get(target, np.zeros(embedding_dim))

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

    return edge_emb


class LinkPredictionTrainDataset(Dataset):
    def __init__(
        self,
        sources,
        targets,
        labels,
        embedding_store,
        edge_op="hadamard"
    ):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.edge_features = []

        logger.info("Caching edge features for training...")

        for s, t in tqdm(zip(sources, targets), total=len(sources)):
            feat = create_edge_feature(s, t, embedding_store, edge_op)
            self.edge_features.append(torch.tensor(feat, dtype=torch.float32))

        logger.info("Training edge feature caching complete.")
        self.edge_features = torch.stack(self.edge_features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.edge_features[idx], self.labels[idx]


class LinkPredictionTGBDataset(Dataset):
    def __init__(
        self,
        sources,
        targets,
        timestamps,
        embedding_store,
        negative_sampler,
        split_mode="test",
        edge_op="hadamard"
    ):
        self.pos_features = []
        self.neg_features = []

        logger.info(f"Caching TGB dataset for split: {split_mode}")

        neg_lists = negative_sampler.query_batch(
            torch.tensor(sources),
            torch.tensor(targets),
            torch.tensor(timestamps),
            split_mode=split_mode
        )

        for i in tqdm(range(len(sources))):
            src, dst = sources[i], targets[i]
            pos_feat = create_edge_feature(src, dst, embedding_store, edge_op)
            self.pos_features.append(torch.tensor(pos_feat, dtype=torch.float32))

            neg_feats = []
            for neg_dst in neg_lists[i]:
                feat = create_edge_feature(src, neg_dst, embedding_store, edge_op)
                neg_feats.append(torch.tensor(feat, dtype=torch.float32))

            self.neg_features.append(torch.stack(neg_feats))

        logger.info("TGB edge feature caching complete.")

    def __len__(self):
        return len(self.pos_features)

    def __getitem__(self, idx):
        return self.pos_features[idx], self.neg_features[idx]


def tgb_collate_fn(batch):
    pos_feats, neg_feats = zip(*batch)  # list of (D,), list of (K, D)

    pos_feats = torch.stack(pos_feats)  # (B, D)
    neg_feats = torch.stack(neg_feats)  # (B, K, D)

    B, K, D = neg_feats.shape

    neg_feats = neg_feats.view(B * K, D)  # (B*K, D)
    all_feats = torch.cat([pos_feats, neg_feats], dim=0)  # (B + B*K, D)

    pos_labels = torch.ones(B, dtype=torch.float32)
    neg_labels = torch.zeros(B * K, dtype=torch.float32)
    all_labels = torch.cat([pos_labels, neg_labels], dim=0)  # (B + B*K,)

    return all_feats, all_labels


def create_dataset_with_negative_edges(ds_sources, ds_targets,
                                       sources_to_exclude, targets_to_exclude,
                                       is_directed, negative_edges_per_positive, random_state=42):
    np.random.seed(random_state)
    random.seed(random_state)

    num_positive = len(ds_sources)
    num_negative = int(num_positive * negative_edges_per_positive)
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

    k = negative_edges_per_positive
    assert len(neg_sources) == num_positive * k

    all_sources, all_targets, all_labels = [], [], []

    for i in range(num_positive):
        pos_src = ds_sources[i]
        pos_tgt = ds_targets[i]
        neg_srcs = neg_sources[i * k:(i + 1) * k]
        neg_tgts = neg_targets[i * k:(i + 1) * k]

        group_srcs = list(neg_srcs)
        group_tgts = list(neg_tgts)
        group_labels = [0] * k

        insert_idx = np.random.randint(0, k + 1)

        group_srcs.insert(insert_idx, pos_src)
        group_tgts.insert(insert_idx, pos_tgt)
        group_labels.insert(insert_idx, 1)

        all_sources.extend(group_srcs)
        all_targets.extend(group_tgts)
        all_labels.extend(group_labels)

    all_sources = np.array(all_sources)
    all_targets = np.array(all_targets)
    all_labels = np.array(all_labels)

    logger.info(f"Final dataset: {len(all_sources):,} edges (positive + negative)")
    return all_sources, all_targets, all_labels


class EarlyStopping:
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


def create_link_prediction_model(input_dim, device='cpu'):
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


def train_embeddings_full_approach(train_sources, train_targets, train_timestamps,
                                   is_directed, walk_length, num_walks_per_node,
                                   embedding_dim, walk_use_gpu, word2vec_n_workers, seed=42):
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


def train_link_prediction_model(
        tgb_dataset: LinkPropPredDataset, embedding_store, edge_op,
        negative_edges_per_positive, is_directed,
        n_epochs, device, learning_rate=1e-4, patience=5
):
    train_sources = tgb_dataset.full_data['sources'][tgb_dataset.train_mask]
    train_targets = tgb_dataset.full_data['destinations'][tgb_dataset.train_mask]

    valid_sources = tgb_dataset.full_data['sources'][tgb_dataset.val_mask]
    valid_targets = tgb_dataset.full_data['destinations'][tgb_dataset.val_mask]
    valid_timestamps = tgb_dataset.full_data['timestamps'][tgb_dataset.val_mask]

    train_sources_combined, train_targets_combined, train_labels_combined = create_dataset_with_negative_edges(
        train_sources,
        train_targets,
        train_sources,
        train_targets,
        is_directed,
        negative_edges_per_positive
    )

    # Datasets
    train_dataset = LinkPredictionTrainDataset(
        sources=train_sources_combined,
        targets=train_targets_combined,
        labels=train_labels_combined,
        embedding_store=embedding_store,
        edge_op=edge_op
    )

    # TGB-based val dataset (one-vs-K negatives)
    val_dataset = LinkPredictionTGBDataset(
        sources=valid_sources,
        targets=valid_targets,
        timestamps=valid_timestamps,
        embedding_store=embedding_store,
        negative_sampler=tgb_dataset.negative_sampler,
        split_mode="val",
        edge_op=edge_op
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=tgb_collate_fn)

    input_dim = len(next(iter(embedding_store.values())))
    model = create_link_prediction_model(input_dim, device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(mode='max', patience=patience)

    history = defaultdict(list)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        num_train_samples = 0
        y_true_train = []
        y_score_train = []

        for edge_feats, labels in train_loader:
            edge_feats = edge_feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(edge_feats).squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * len(labels)
            num_train_samples += batch_size

            y_true_train.extend(labels.cpu().numpy())
            y_score_train.extend(logits.detach().cpu().numpy())

        avg_train_loss = total_loss / num_train_samples
        train_auc = roc_auc_score(y_true_train, y_score_train)

        # Validation
        model.eval()
        val_loss_total = 0
        y_true_val = []
        y_score_val = []

        with torch.no_grad():
            for edge_feats, labels in val_loader:
                edge_feats = edge_feats.to(device)
                labels = labels.to(device)

                logits = model(edge_feats).squeeze()
                loss = criterion(logits, labels)

                val_loss_total += loss.item() * len(labels)
                y_true_val.extend(labels.cpu().numpy())
                y_score_val.extend(logits.detach().cpu().numpy())

        avg_val_loss = val_loss_total / len(y_score_val)
        val_auc = roc_auc_score(y_true_val, y_score_val)

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        logger.info(f"Epoch {epoch + 1}/{n_epochs} — "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                    f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        if early_stopping(val_auc, model):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info("Training completed")
    return model, history


def evaluate_link_prediction_model(model, tgb_dataset: LinkPropPredDataset, dataset_name, mask, neg_sampler, split_mode, embedding_store, edge_op, device):
    model.eval()
    evaluator = Evaluator(name=dataset_name)
    perf_list = []

    data = tgb_dataset.full_data
    metric = tgb_dataset.eval_metric

    pos_src_all = data['sources'][mask]
    pos_dst_all = data['destinations'][mask]
    pos_ts_all = data['timestamps'][mask]

    num_edges = len(pos_src_all)
    num_batches = math.ceil(num_edges / BATCH_SIZE)

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc=f"Evaluating ({split_mode})"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, num_edges)

            pos_src = pos_src_all[start_idx:end_idx]
            pos_dst = pos_dst_all[start_idx:end_idx]
            pos_ts = pos_ts_all[start_idx:end_idx]

            neg_lists = neg_sampler.query_batch(
                torch.tensor(pos_src),
                torch.tensor(pos_dst),
                torch.tensor(pos_ts),
                split_mode=split_mode
            )

            for i, neg_dsts in enumerate(neg_lists):
                src = int(pos_src[i])
                pos_tgt = int(pos_dst[i])
                ts = pos_ts[i]

                all_tgts = [pos_tgt] + list(neg_dsts)
                all_srcs = [src] * len(all_tgts)

                # Compute edge features
                edge_feats = [
                    torch.tensor(
                        create_edge_feature(s, d, embedding_store, edge_op),
                        dtype=torch.float32
                    )
                    for s, d in zip(all_srcs, all_tgts)
                ]
                edge_feats = torch.stack(edge_feats).to(device)  # (1+K, D)

                scores = model(edge_feats).squeeze().cpu().numpy()

                # TGB Evaluator expects: 1 positive vs many negatives
                input_dict = {
                    "y_pred_pos": np.array([scores[0]]),
                    "y_pred_neg": np.array(scores[1:]),
                    "eval_metric": [metric],
                }
                perf_list.append(evaluator.eval(input_dict)[metric])

    return {metric: float(np.mean(perf_list))}


def run_link_prediction_experiments(
        dataset_name,
        is_directed,
        walk_length,
        num_walks_per_node,
        embedding_dim,
        edge_op,
        negative_edges_per_positive,
        n_epochs,
        full_embedding_use_gpu,
        link_prediction_use_gpu,
        n_runs,
        word2vec_n_workers,
        output_path
):
    logger.info("Starting link prediction experiments")

    dataset = LinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)
    dataset.load_val_ns()
    dataset.load_test_ns()

    full_data = dataset.full_data

    train_sources = full_data['sources'][dataset.train_mask]
    train_targets = full_data['destinations'][dataset.train_mask]
    train_timestamps = full_data['timestamps'][dataset.train_mask]

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
        'training_history': [], 'valid': {}, 'test': {}
    }

    for run in range(n_runs):
        logger.info(f"\n--- Run {run + 1}/{n_runs} ---")

        model, history = train_link_prediction_model(
            dataset, full_embeddings, edge_op,
            negative_edges_per_positive, is_directed,
            n_epochs, device
        )

        current_valid_results = evaluate_link_prediction_model(
            model,
            dataset,
            dataset_name,
            dataset.val_mask,
            dataset.ns_sampler,
            'val',
            full_embeddings,
            edge_op,
            device
        )

        current_test_results = evaluate_link_prediction_model(
            model,
            dataset,
            dataset_name,
            dataset.test_mask,
            dataset.ns_sampler,
            'test',
            full_embeddings,
            edge_op,
            device
        )

        full_results['training_history'].append(history)

        for key in current_valid_results.keys():
            if key not in full_results['valid']:
                full_results['valid'][key] = []

            full_results['valid'][key].append(current_valid_results[key])

        for key in current_test_results.keys():
            if key not in full_results['test']:
                full_results['test'][key] = []

            full_results['test'][key].append(current_test_results[key])

    logger.info(f"\nFull Approach Results:")
    logger.info(f"MRR (Validation): {np.mean(full_results['valid']['mrr']):.4f} ± {np.std(full_results['valid']['mrr']):.4f}")
    logger.info(f"MRR (Test): {np.mean(full_results['test']['mrr']):.4f} ± {np.std(full_results['test']['mrr']):.4f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(full_results, f)
        logger.info(f"Results saved to {output_path}")

    return full_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal Link Prediction")

    # Required arguments
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='TGB dataset name')
    parser.add_argument('--is_directed', type=lambda x: x.lower() == 'true', required=True,
                        help='Whether the graph is directed (true/false)')

    # Model parameters
    parser.add_argument('--walk_length', type=int, default=80,
                        help='Maximum length of random walks')
    parser.add_argument('--num_walks_per_node', type=int, default=10,
                        help='Number of walks to generate per node')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimensionality of node embeddings')
    parser.add_argument('--edge_op', type=str, default='hadamard',
                        choices=['average', 'hadamard', 'weighted-l1', 'weighted-l2'],
                        help='Edge operation for combining node embeddings')

    parser.add_argument('--negative_edges_per_positive', type=int, default=1,
                        help='Number of negative edges per positive edge')

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
        dataset_name=args.dataset_name,
        is_directed=args.is_directed,
        walk_length=args.walk_length,
        num_walks_per_node=args.num_walks_per_node,
        embedding_dim=args.embedding_dim,
        edge_op=args.edge_op,
        negative_edges_per_positive=args.negative_edges_per_positive,
        n_epochs=args.n_epochs,
        full_embedding_use_gpu=args.full_embedding_use_gpu,
        link_prediction_use_gpu=args.link_prediction_use_gpu,
        n_runs=args.n_runs,
        word2vec_n_workers=args.word2vec_n_workers,
        output_path=args.output_path
    )
