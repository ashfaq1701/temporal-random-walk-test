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


def split_train_test(X, y, train_percentage):
    num_samples = X.shape[0]
    train_len = int(num_samples * train_percentage)
    return X[:train_len, :], X[train_len:, :], y[:train_len], y[train_len:]


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


class MiniBatchLogisticRegression:
    def __init__(
            self, input_dim, batch_size=1_000_000, learning_rate=0.001,
            epochs=20, device='cpu', patience=3, validation_split=0.1, use_amp=True
    ):
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.use_amp = use_amp and device == 'cuda'  # Only use AMP on CUDA

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

        # Initialize mixed precision scaler
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled")

    def fit(self, X, y):
        """Train the model with early stopping using validation split - GPU memory efficient."""
        logger.info(f"Training PyTorch neural network on {len(X):,} samples with batch size {self.batch_size:,}")

        # Split into train/validation
        X_train, X_val, y_train, y_val = split_train_test(X, y, 1.0 - self.validation_split)

        logger.info(f"Train: {len(X_train):,}, Validation: {len(X_val):,}")

        # Convert to CPU tensors first
        X_train_tensor = torch.FloatTensor(X_train)  # Keep on CPU
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # Keep on CPU
        X_val_tensor = torch.FloatTensor(X_val)  # Keep on CPU
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)  # Keep on CPU

        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        logger.info(f'Starting training for {self.epochs} epochs ...')

        # Store training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': []
        }

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            all_train_preds = []
            all_train_targets = []

            for batch_X, batch_y in train_dataloader:
                # Move only the current batch to GPU
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)

                    # Scaled backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard FP32 training
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item()
                train_batches += 1

                # Collect training predictions for AUC calculation
                with torch.no_grad():
                    train_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    all_train_preds.extend(train_probs)
                    all_train_targets.extend(batch_y.cpu().numpy().flatten())

                # Clear GPU memory after each batch
                del batch_X, batch_y, outputs, loss
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            avg_train_loss = train_loss / train_batches

            # Calculate training AUC
            try:
                train_auc = roc_auc_score(all_train_targets, all_train_preds)
            except:
                train_auc = 0.0

            self.model.eval()

            val_loss = 0.0
            val_batches = 0
            all_val_preds = []
            all_val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_dataloader:
                    # Move only the current batch to GPU
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)

                    if self.use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(batch_X)
                            loss = self.criterion(outputs, batch_y)
                    else:
                        # Standard FP32 inference
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)

                    val_loss += loss.item()
                    val_batches += 1

                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    all_val_preds.extend(probs)
                    all_val_targets.extend(batch_y.cpu().numpy().flatten())

                    # Clear GPU memory after each batch
                    del batch_X, batch_y, outputs, loss
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()

            avg_val_loss = val_loss / val_batches

            try:
                val_auc = roc_auc_score(all_val_targets, all_val_preds)
            except:
                val_auc = 0.0

            # Store metrics in history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)

            # Log progress
            logger.info(f"Epoch {epoch + 1:3d}/{self.epochs}: "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}, "
                        f"Train AUC: {train_auc:.4f}, "
                        f"Val AUC: {val_auc:.4f}")

            # Early stopping check
            if self.early_stopping(avg_val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                logger.info(f"Best validation loss: {self.early_stopping.best_loss:.4f}")
                break

        logger.info("Training completed")
        return history


    def predict_proba(self, X):
        """Predict probabilities using mini-batches to avoid memory issues."""
        self.model.eval()
        predictions = []

        # Process in batches to avoid memory issues
        batch_size = self.batch_size
        num_samples = len(X)

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                # Keep batch creation on CPU, then move to GPU
                batch_X = torch.FloatTensor(X[i:end_idx]).to(self.device)

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        batch_logits = self.model(batch_X)
                        # FIXED: Apply sigmoid to convert logits to probabilities
                        batch_pred = torch.sigmoid(batch_logits).cpu().numpy().flatten()
                else:
                    batch_logits = self.model(batch_X)
                    batch_pred = torch.sigmoid(batch_logits).cpu().numpy().flatten()

                predictions.extend(batch_pred)

                # Clear GPU memory after each batch
                del batch_X, batch_logits
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

        predictions = np.array(predictions)
        return np.column_stack([1 - predictions, predictions])


    def predict(self, X):
        """Make binary predictions."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)


def split_dataset(data_file_path, train_percentage):
    """Split dataset based on temporal ordering."""
    logger.info(f"Loading dataset from {data_file_path}")

    df = pd.read_csv(data_file_path)
    timestamps = df['ts']

    # Get unique timestamps (already sorted)
    unique_timestamps = timestamps.unique()
    logger.info(f"Dataset contains {len(df)} edges with {len(unique_timestamps)} unique timestamps")

    train_len = int(len(df) * train_percentage)

    # Create training and testing datasets
    train_df = df.iloc[:train_len]
    test_df = df.iloc[train_len:]

    logger.info(f"Train set: {len(train_df)} edges, Test set: {len(test_df)} edges")
    return train_df, test_df


def sample_negative_edges(sources, targets, all_nodes, num_negative_samples, seed=42):
    """Sample negative edges efficiently for large datasets - returns NumPy arrays."""
    np.random.seed(seed)

    existing_edges = set(zip(sources, targets))

    # Scale batch size based on density
    batch_size = 1000000  # 1M for medium density
    logger.info(f"Sampling {num_negative_samples:,} negative edges from {len(all_nodes):,} nodes")

    negative_edges = []
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


def evaluate_link_prediction(
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        node_embeddings,
        edge_op,
        link_prediction_training_percentage,
        n_epochs,
        device,
        batch_size=1_000_000
):
    """Evaluate link prediction using Hadamard product and neural network."""
    logger.info("Starting link prediction evaluation")

    all_sources = np.concatenate([np.asarray(test_sources), np.asarray(negative_sources)])
    all_targets = np.concatenate([np.asarray(test_targets), np.asarray(negative_targets)])
    labels = np.concatenate([
        np.ones(len(test_sources)),  # Positive edges
        np.zeros(len(negative_sources))  # Negative edges
    ])

    indices = np.random.permutation(len(all_sources))
    all_sources = all_sources[indices]
    all_targets = all_targets[indices]
    labels = labels[indices]

    logger.info(f"Processing {len(all_sources):,} edges for feature creation")

    # Get embedding dimension
    embedding_dim = len(next(iter(node_embeddings.values())))

    # Pre-allocate feature array for better performance
    edge_features = np.zeros((len(all_sources), embedding_dim), dtype=np.float32)

    # Vectorized approach where possible
    logger.info(f"Creating {edge_op} product features...")

    for i, (src, tgt) in enumerate(zip(all_sources, all_targets)):
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
            raise ValueError(f"Unknown edge_op: {edge_op}. Use 'average' or 'hadamard'")

        # Store the edge embedding
        edge_features[i] = edge_emb

    logger.info("Feature creation completed")

    # Split into train/test for evaluation
    X_train, X_test, y_train, y_test = split_train_test(
        edge_features, labels, link_prediction_training_percentage
    )

    logger.info(f"Training classifier on {len(X_train):,} samples, testing on {len(X_test):,} samples")

    # Use PyTorch mini-batch neural network with early stopping
    classifier = MiniBatchLogisticRegression(
        input_dim=embedding_dim,
        batch_size=batch_size,
        learning_rate=0.001,
        epochs=n_epochs,
        device=device,
        patience=10,
        validation_split=0.15
    )

    # Train the model
    history = classifier.fit(X_train, y_train)

    # Make predictions
    logger.info("Making predictions...")
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]  # Probability of positive class
    y_pred = classifier.predict(X_test)

    # Calculate metrics
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
        'training_history': history
    }

    logger.info(f"Link prediction completed - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
    return results


def run_link_prediction_full_data(
        train_sources,
        train_targets,
        train_timestamps,
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        is_directed,
        walk_length,
        num_walks_per_node,
        embedding_dim,
        edge_op,
        link_prediction_training_percentage,
        n_epochs,
        use_gpu,
        word2vec_n_workers,
        seed=42
):
    """Run link prediction using full dataset approach."""
    logger.info("Starting full data link prediction")

    temporal_random_walk = TemporalRandomWalk(is_directed=is_directed, use_gpu=use_gpu, max_time_capacity=-1)

    logger.info(f'Adding {len(train_sources)} edges in temporal random walk instance')
    temporal_random_walk.add_multiple_edges(train_sources, train_targets, train_timestamps)

    batch_walk_start_time = time.time()
    walks, timestamps, walk_lengths = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        max_walk_len=walk_length,
        num_walks_per_node=num_walks_per_node,
        walk_bias='ExponentialIndex',
        initial_edge_bias='Uniform',
        walk_direction="Backward_In_Time"
    )
    batch_walk_duration = time.time() - batch_walk_start_time

    logger.info(
        f'Generated {len(walks)} walks in {batch_walk_duration:.2f} seconds. Mean walk length: {np.mean(walk_lengths)}.')

    walk_length_counts = {}
    for length in walk_lengths:
        walk_length_counts[length] = walk_length_counts.get(length, 0) + 1

    logger.info(f"Walk length distribution:")
    for length in sorted(walk_length_counts.keys()):
        count = walk_length_counts[length]
        percentage = (count / len(walk_lengths)) * 100
        logger.info(f"  Length {length}: {count:,} walks ({percentage:.2f}%)")

    zero_length_walks = walk_length_counts.get(0, 0)
    one_length_walks = walk_length_counts.get(1, 0)

    if zero_length_walks > 0:
        logger.warning(f"Found {zero_length_walks:,} walks with length 0!")
    if one_length_walks > 0:
        logger.warning(f"Found {one_length_walks:,} walks with length 1!")

    # Remove padding from walks using actual walk lengths
    clean_walks = []
    for walk, length in zip(walks, walk_lengths):
        # Convert to strings (Word2Vec expects strings)
        clean_walk = [str(node) for node in walk[:length]]
        if len(clean_walk) > 1:  # Skip single-node walks
            clean_walks.append(clean_walk)

    logger.info(f"Training Word2Vec on {len(clean_walks)} clean walks")

    batch_node_embedding_start_time = time.time()

    with suppress_word2vec_output():
        # Train Word2Vec model
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
    batch_node_embedding_duration = time.time() - batch_node_embedding_start_time

    logger.info(f'Trained embeddings for {len(node_embeddings)} nodes in {batch_node_embedding_duration:.2f} seconds')

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    return evaluate_link_prediction(
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        node_embeddings,
        edge_op,
        link_prediction_training_percentage,
        n_epochs,
        device
    )


def run_link_prediction_experiments(
        data_file_path,
        is_directed,
        walk_length,
        num_walks_per_node,
        embedding_dim,
        edge_op,
        embedding_training_percentage,
        link_prediction_training_percentage,
        n_epochs,
        use_gpu,
        word2vec_n_workers
):
    """Run both full and streaming link prediction experiments."""
    logger.info("Starting link prediction experiments")

    # Split dataset
    train_df, test_df = split_dataset(data_file_path, embedding_training_percentage)

    # Convert to NumPy arrays immediately
    train_sources = train_df['u'].to_numpy()
    train_targets = train_df['i'].to_numpy()
    train_timestamps = train_df['ts'].to_numpy()
    test_sources = test_df['u'].to_numpy()
    test_targets = test_df['i'].to_numpy()
    test_timestamps = test_df['ts'].to_numpy()

    train_nodes = set(train_sources).union(set(train_targets))
    test_nodes = set(test_sources).union(set(test_targets))
    all_nodes = np.array(set(train_nodes).union(set(test_nodes)))

    nodes_in_both = test_nodes.intersection(train_nodes)
    nodes_only_in_test = test_nodes - train_nodes

    logger.info(f"Test nodes present in training: {len(nodes_in_both):,} ({len(nodes_in_both) / len(test_nodes) * 100:.1f}%)")
    logger.info(f"Test nodes absent from training: {len(nodes_only_in_test):,} ({len(nodes_only_in_test) / len(test_nodes) * 100:.1f}%)")

    # Sample negative edges - returns NumPy arrays
    negative_sources, negative_targets = sample_negative_edges(train_sources, train_targets, all_nodes, len(test_sources))

    # Run full data approach
    logger.info("=" * 50)
    logger.info("FULL DATA APPROACH")
    logger.info("=" * 50)

    full_start_time = time.time()
    full_link_prediction_results = run_link_prediction_full_data(
        train_sources,
        train_targets,
        train_timestamps,
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        is_directed,
        walk_length,
        num_walks_per_node,
        embedding_dim,
        edge_op,
        link_prediction_training_percentage,
        n_epochs,
        use_gpu,
        word2vec_n_workers
    )
    full_duration = time.time() - full_start_time
    full_link_prediction_results['total_time'] = full_duration

    print(f"\nFull Link Prediction Results:")
    print(f"AUC: {full_link_prediction_results['auc']:.4f}")
    print(f"Accuracy: {full_link_prediction_results['accuracy']:.4f}")
    print(f"Precision: {full_link_prediction_results['precision']:.4f}")
    print(f"Recall: {full_link_prediction_results['recall']:.4f}")
    print(f"F1-Score: {full_link_prediction_results['f1_score']:.4f}")
    print(f"Total Time: {full_duration:.2f}s")


    return full_link_prediction_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal Link Prediction Test")
    parser.add_argument(
        '--data_file_path', type=str, required=True,
        help='Path to data file (parquet format)'
    )
    parser.add_argument('--is_directed', type=lambda x: x.lower() == 'true', required=True,
                        help='Whether the graph is directed')

    parser.add_argument('--embedding_training_percentage', type=float, default=0.75,
                        help='Percentage of data used for training embeddings')
    parser.add_argument('--link_prediction_training_percentage', type=float, default=0.75,
                        help='Percentage of link prediction data used for training classifier')
    parser.add_argument('--walk_length', type=int, default=80,
                        help='Maximum length of random walks')
    parser.add_argument('--num_walks_per_node', type=int, default=10,
                        help='Number of walks to generate per node')

    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimensionality of node embeddings')
    parser.add_argument('--edge_op', type=str, default='hadamard', help='average, hadamard, weighted-l1 or weighted-l2')

    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs')

    parser.add_argument('--use_gpu', action='store_true',
                        help='Enable GPU acceleration for full embedding')

    parser.add_argument('--word2vec_n_workers', type=int, default=10, help='Number of workers for word2vec')

    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (optional)')

    args = parser.parse_args()

    run_link_prediction_experiments(
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
        args.word2vec_n_workers
    )
