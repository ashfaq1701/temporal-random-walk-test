import argparse
import logging
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from temporal_random_walk import TemporalRandomWalk
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def split_train_test(X, y, train_percentage):
    """Split data into train/test sets."""
    num_samples = X.shape[0]
    train_len = int(num_samples * train_percentage)
    return X[:train_len, :], X[train_len:, :], y[:train_len], y[train_len:]


class EarlyStopping:
    """Early stopping callback."""

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


class MiniBatchLogisticRegression:
    """Mini-batch neural network classifier."""

    def __init__(self, input_dim, batch_size=1_000_000, learning_rate=0.001,
                 epochs=20, device='cpu', patience=3, validation_split=0.1):
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

        hidden_dim1 = max(64, input_dim // 2)
        hidden_dim2 = max(32, input_dim // 4)

        self.model = nn.Sequential(
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.early_stopping = EarlyStopping(patience=patience)

    def fit(self, X, y):
        """Train the model."""
        logger.info(f"Training neural network on {len(X):,} samples")

        X_train, X_val, y_train, y_val = split_train_test(X, y, 1.0 - self.validation_split)

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            for batch_X, batch_y in train_dataloader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            avg_train_loss = train_loss / train_batches

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            all_val_preds = []
            all_val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_dataloader:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)

                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)

                    val_loss += loss.item()
                    val_batches += 1

                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    all_val_preds.extend(probs)
                    all_val_targets.extend(batch_y.cpu().numpy().flatten())

            avg_val_loss = val_loss / val_batches
            try:
                val_auc = roc_auc_score(all_val_targets, all_val_preds)
            except:
                val_auc = 0.0

            logger.info(f"Epoch {epoch + 1:3d}/{self.epochs}: "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}, "
                        f"Val AUC: {val_auc:.4f}")

            if self.early_stopping(avg_val_loss, self.model):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    def predict_proba(self, X):
        """Predict probabilities."""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                end_idx = min(i + self.batch_size, len(X))
                batch_X = torch.FloatTensor(X[i:end_idx]).to(self.device)
                batch_logits = self.model(batch_X)
                batch_pred = torch.sigmoid(batch_logits).cpu().numpy().flatten()
                predictions.extend(batch_pred)

        predictions = np.array(predictions)
        return np.column_stack([1 - predictions, predictions])

    def predict(self, X):
        """Make binary predictions."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)


def split_dataset(data_file_path, train_percentage):
    """Split dataset randomly."""
    logger.info(f"Loading dataset from {data_file_path}")
    df = pd.read_csv(data_file_path)

    logger.info(f"Dataset: {len(df)} edges")

    # Simple random split
    train_size = int(len(df) * train_percentage)

    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = df_shuffled[:train_size]
    test_df = df_shuffled[train_size:]

    logger.info(f"Train: {len(train_df)} edges, Test: {len(test_df)} edges")
    return train_df, test_df


def sample_negative_edges(sources, targets, num_negative_samples, seed=42):
    """Sample negative edges efficiently for large datasets - returns NumPy arrays."""
    np.random.seed(seed)

    existing_edges = set(zip(sources, targets))
    all_nodes = np.array(list(set(sources).union(set(targets))))

    # Calculate density and set appropriate batch size
    max_possible = len(all_nodes) * (len(all_nodes) - 1)
    density = len(existing_edges) / max_possible

    # Scale batch size based on density
    batch_size = 1000000  # 1M for medium density

    logger.info(f"Graph density: {density:.4f}, using batch size: {batch_size:,}")
    logger.info(f"Sampling {num_negative_samples:,} negative edges from {len(all_nodes):,} nodes")

    negative_edges = set()
    attempts = 0

    while len(negative_edges) < num_negative_samples:
        remaining = num_negative_samples - len(negative_edges)
        current_batch_size = min(batch_size, remaining * 10)  # Generate 10x what we need

        # Generate large batch
        u = np.random.choice(all_nodes, current_batch_size, replace=True)
        v = np.random.choice(all_nodes, current_batch_size, replace=True)

        # Vectorized filtering
        valid_mask = u != v
        u_valid, v_valid = u[valid_mask], v[valid_mask]

        # Process in chunks to avoid memory issues with very large batches
        chunk_size = 100000
        for i in range(0, len(u_valid), chunk_size):
            if len(negative_edges) >= num_negative_samples:
                break

            u_chunk = u_valid[i:i + chunk_size]
            v_chunk = v_valid[i:i + chunk_size]

            # Check chunk against existing edges
            chunk_edges = set(zip(u_chunk, v_chunk)) - existing_edges
            negative_edges = negative_edges + list(chunk_edges)

            # Stop if we have enough
            if len(negative_edges) >= num_negative_samples:
                break

        attempts += 1

        # Log every 100 attempts
        if attempts % 100 == 0:
            progress = len(negative_edges) / num_negative_samples * 100
            logger.info(f"Attempt {attempts}: Found {len(negative_edges):,}/{num_negative_samples:,} ({progress:.1f}%)")

    progress = len(negative_edges) / num_negative_samples * 100
    logger.info(f"Attempt {attempts}: Found {len(negative_edges):,}/{num_negative_samples:,} ({progress:.1f}%)")

    # Convert to NumPy arrays - more efficient approach
    neg_list = list(negative_edges)[:num_negative_samples]
    neg_array = np.array(neg_list)  # Convert to 2D array

    negative_sources = neg_array[:, 0]  # First column
    negative_targets = neg_array[:, 1]  # Second column

    logger.info(f"Successfully sampled {len(negative_sources):,} negative edges in {attempts} attempts")
    return negative_sources, negative_targets


def evaluate_link_prediction(test_sources, test_targets, negative_sources, negative_targets,
                             node_embeddings, edge_op, link_prediction_training_percentage,
                             n_epochs, device, batch_size=1_000_000):
    """Evaluate link prediction."""
    logger.info("Starting link prediction evaluation")

    all_sources = np.concatenate([np.asarray(test_sources), np.asarray(negative_sources)])
    all_targets = np.concatenate([np.asarray(test_targets), np.asarray(negative_targets)])
    labels = np.concatenate([np.ones(len(test_sources)), np.zeros(len(negative_sources))])

    # Shuffle
    indices = np.random.permutation(len(all_sources))
    all_sources = all_sources[indices]
    all_targets = all_targets[indices]
    labels = labels[indices]

    # Create features
    embedding_dim = len(next(iter(node_embeddings.values())))
    edge_features = np.zeros((len(all_sources), embedding_dim), dtype=np.float32)

    logger.info(f"Creating {edge_op} features...")
    for i, (src, tgt) in enumerate(zip(all_sources, all_targets)):
        src_emb = node_embeddings.get(src, np.zeros(embedding_dim))
        tgt_emb = node_embeddings.get(tgt, np.zeros(embedding_dim))

        if edge_op == 'average':
            edge_emb = (src_emb + tgt_emb) / 2
        elif edge_op == 'hadamard':
            edge_emb = src_emb * tgt_emb
        elif edge_op == 'concat':
            edge_emb = np.concatenate([src_emb, tgt_emb])
        elif edge_op == 'weighted-l1':
            edge_emb = np.abs(src_emb - tgt_emb)
        elif edge_op == 'weighted-l2':
            edge_emb = (src_emb - tgt_emb) ** 2
        else:
            raise ValueError(f"Unknown edge_op: {edge_op}")

        edge_features[i] = edge_emb

    # Split and train
    X_train, X_test, y_train, y_test = split_train_test(
        edge_features, labels, link_prediction_training_percentage
    )

    logger.info(f"Training on {len(X_train):,} samples, testing on {len(X_test):,} samples")

    classifier = MiniBatchLogisticRegression(
        input_dim=edge_features.shape[1],
        batch_size=batch_size,
        learning_rate=0.001,
        epochs=n_epochs,
        device=device,
        patience=10,
        validation_split=0.15
    )

    classifier.fit(X_train, y_train)

    # Predict
    logger.info("Making predictions...")
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    y_pred = classifier.predict(X_test)

    # Metrics
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
        'num_negative_edges': len(negative_sources)
    }

    logger.info(f"Link prediction completed - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
    return results


def run_full_link_prediction(data_file_path, is_directed, walk_length, num_walks_per_node,
                             embedding_dim, edge_op, embedding_training_percentage,
                             link_prediction_training_percentage, n_epochs, use_gpu,
                             word2vec_n_workers, seed=42):
    """Run full walk-based link prediction."""
    logger.info("Starting full walk-based link prediction")

    # Split dataset
    train_df, test_df = split_dataset(data_file_path, embedding_training_percentage)

    train_sources = train_df['u'].to_numpy()
    train_targets = train_df['i'].to_numpy()
    train_timestamps = train_df['ts'].to_numpy()
    test_sources = test_df['u'].to_numpy()
    test_targets = test_df['i'].to_numpy()

    # Sample negative edges
    negative_sources, negative_targets = sample_negative_edges(train_sources, train_targets, len(test_sources))

    # Create temporal random walk instance
    temporal_random_walk = TemporalRandomWalk(is_directed=is_directed, use_gpu=use_gpu, max_time_capacity=-1)

    logger.info(f'Adding {len(train_sources)} edges to temporal random walk')
    temporal_random_walk.add_multiple_edges(train_sources, train_targets, train_timestamps)

    # Generate walks
    walk_start_time = time.time()
    walks, timestamps, walk_lengths = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        max_walk_len=walk_length,
        num_walks_per_node=num_walks_per_node,
        walk_bias='ExponentialIndex',
        initial_edge_bias='Uniform',
        walk_direction="Backward_In_Time"
    )
    walk_duration = time.time() - walk_start_time

    logger.info(f'Generated {len(walks)} walks in {walk_duration:.2f}s. Mean length: {np.mean(walk_lengths):.2f}')

    # Clean walks
    clean_walks = []
    for walk, length in zip(walks, walk_lengths):
        clean_walk = [str(node) for node in walk[:length]]
        if len(clean_walk) > 1:
            clean_walks.append(clean_walk)

    logger.info(f"Training Word2Vec on {len(clean_walks)} clean walks")

    # Train Word2Vec
    embedding_start_time = time.time()
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

    node_embeddings = {}
    for node in model.wv.index_to_key:
        node_embeddings[int(node)] = model.wv[node]
    embedding_duration = time.time() - embedding_start_time

    logger.info(f'Trained embeddings for {len(node_embeddings)} nodes in {embedding_duration:.2f}s')

    # Link prediction
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    results = evaluate_link_prediction(
        test_sources, test_targets, negative_sources, negative_targets,
        node_embeddings, edge_op, link_prediction_training_percentage,
        n_epochs, device
    )

    total_time = walk_duration + embedding_duration
    results['walk_time'] = walk_duration
    results['embedding_time'] = embedding_duration
    results['total_time'] = total_time

    print(f"\nLink Prediction Results:")
    print(f"AUC: {results['auc']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"Walk Time: {walk_duration:.2f}s")
    print(f"Embedding Time: {embedding_duration:.2f}s")
    print(f"Total Time: {total_time:.2f}s")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple Full Walk Link Prediction")
    parser.add_argument('--data_file_path', type=str, required=True, help='Path to CSV file with u,i,ts columns')
    parser.add_argument('--is_directed', type=lambda x: x.lower() == 'true', default=True, help='Graph is directed')
    parser.add_argument('--walk_length', type=int, default=80, help='Maximum walk length')
    parser.add_argument('--num_walks_per_node', type=int, default=10, help='Walks per node')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--edge_op', type=str, default='hadamard',
                        choices=['average', 'hadamard', 'concat', 'weighted-l1', 'weighted-l2'],
                        help='Edge operation for combining embeddings')
    parser.add_argument('--embedding_training_percentage', type=float, default=0.75,
                        help='Train/test split for embeddings')
    parser.add_argument('--link_prediction_training_percentage', type=float, default=0.75,
                        help='Train/test split for link prediction')
    parser.add_argument('--n_epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--word2vec_n_workers', type=int, default=4, help='Word2Vec workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    run_full_link_prediction(
        args.data_file_path,
        args.is_directed,
        args.walk_length,
        args.num_walks_per_node,
        args.embedding_dim,
        args.edge_op,
        args.embedding_training_percentage,
        args.link_prediction_training_percentage,
        args.n_epochs,
        args.use_gpu,
        args.word2vec_n_workers,
        args.seed
    )