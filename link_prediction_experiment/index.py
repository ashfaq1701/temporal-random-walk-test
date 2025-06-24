import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from temporal_random_walk import TemporalRandomWalk
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
            epochs=10, device='cpu', patience=3, validation_split=0.1, use_amp=True
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

            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
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

        logger.info('Starting training ...')

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            for batch_X, batch_y in train_dataloader:
                # Move only the current batch to GPU
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                if self.use_amp:
                    # Mixed precision training
                    with autocast():
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

                # Clear GPU memory after each batch
                del batch_X, batch_y, outputs, loss
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            avg_train_loss = train_loss / train_batches

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
                        # Mixed precision inference
                        with autocast():
                            outputs = self.model(batch_X)
                            loss = self.criterion(outputs, batch_y)
                    else:
                        # Standard FP32 inference
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)

                    val_loss += loss.item()
                    val_batches += 1

                    all_val_preds.extend(outputs.cpu().numpy().flatten())
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

            # Log progress
            logger.info(f"Epoch {epoch + 1:3d}/{self.epochs}: "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}, "
                        f"Val AUC: {val_auc:.4f}")

            # Early stopping check
            if self.early_stopping(avg_val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                logger.info(f"Best validation loss: {self.early_stopping.best_loss:.4f}")
                break

        logger.info("Training completed")

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
                    # Mixed precision inference
                    with autocast():
                        batch_pred = self.model(batch_X).cpu().numpy().flatten()
                else:
                    # Standard FP32 inference
                    batch_pred = self.model(batch_X).cpu().numpy().flatten()

                predictions.extend(batch_pred)

                # Clear GPU memory after each batch
                del batch_X
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
    df = pd.read_parquet(data_file_path)
    timestamps = df['ts']

    # Get unique timestamps (already sorted)
    unique_timestamps = timestamps.unique()
    logger.info(f"Dataset contains {len(df)} edges with {len(unique_timestamps)} unique timestamps")

    # Calculate split point based on train_percentage
    split_idx = int(len(unique_timestamps) * train_percentage)

    # Get training and testing timestamps
    train_timestamps = unique_timestamps[:split_idx]

    # Get the last training timestamp
    last_train_timestamp = train_timestamps[-1]

    # Split dataset at the last occurrence of the training timestamp
    train_mask = timestamps <= last_train_timestamp
    test_mask = timestamps > last_train_timestamp

    # Create training and testing datasets
    train_df = df[train_mask]
    test_df = df[test_mask]

    logger.info(f"Train set: {len(train_df)} edges, Test set: {len(test_df)} edges")
    return train_df, test_df


def sample_negative_edges(test_sources, test_targets, num_negative_samples=None, seed=42):
    """Sample negative edges efficiently for large datasets - returns NumPy arrays."""
    np.random.seed(seed)

    existing_edges = set(zip(test_sources, test_targets))
    all_nodes = np.array(list(set(test_sources).union(set(test_targets))))

    if num_negative_samples is None:
        num_negative_samples = len(test_sources)

    # Calculate density and set appropriate batch size
    max_possible = len(all_nodes) * (len(all_nodes) - 1)
    density = len(existing_edges) / max_possible

    # Scale batch size based on density
    if density > 0.7:
        batch_size = 5000000  # 5M for very dense graphs
    elif density > 0.3:
        batch_size = 1000000  # 1M for medium density
    else:
        batch_size = 100000  # 100K for sparse graphs

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
            negative_edges.update(chunk_edges)

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


def evaluate_link_prediction(
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        node_embeddings,
        link_prediction_training_percentage,
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
    missing_embeddings = 0

    # Vectorized approach where possible
    logger.info("Creating Hadamard product features...")

    for i, (src, tgt) in enumerate(zip(all_sources, all_targets)):
        if src in node_embeddings and tgt in node_embeddings:
            # Hadamard product of source and target embeddings
            edge_features[i] = node_embeddings[src] * node_embeddings[tgt]
        else:
            # Handle missing embeddings - zero vector already pre-allocated
            if missing_embeddings < 10:  # Only log first 10 warnings
                logger.warning(f"Missing embedding for edge ({src}, {tgt})")
            missing_embeddings += 1

        # Progress logging for large datasets
        if (i + 1) % 10_000_000 == 0:
            progress = (i + 1) / len(all_sources) * 100
            logger.info(f"Feature creation progress: {progress:.1f}% ({i + 1:,}/{len(all_sources):,})")

    if missing_embeddings > 0:
        logger.warning(f"Total missing embeddings: {missing_embeddings} / {len(all_sources)}")

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
        epochs=50,
        device=device,
        patience=10,
        validation_split=0.15
    )

    # Train the model
    classifier.fit(X_train, y_train)

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
        'num_positive_edges': len(test_sources),
        'num_negative_edges': len(negative_sources),
        'num_missing_embeddings': missing_embeddings,
        'embedding_coverage': 1.0 - (missing_embeddings / len(all_sources))
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
        link_prediction_training_percentage,
        walk_use_gpu,
        link_prediction_use_gpu,
        seed=42
):
    """Run link prediction using full dataset approach."""
    logger.info("Starting full data link prediction")

    temporal_random_walk = TemporalRandomWalk(is_directed=is_directed, use_gpu=walk_use_gpu, max_time_capacity=-1)

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
    # Train Word2Vec model
    model = Word2Vec(
        sentences=clean_walks,
        vector_size=embedding_dim,
        window=10,
        min_count=1,
        workers=4,
        sg=1,
        seed=seed
    )

    node_embeddings = {}
    for node in model.wv.index_to_key:
        node_embeddings[int(node)] = model.wv[node]
    batch_node_embedding_duration = time.time() - batch_node_embedding_start_time

    logger.info(f'Trained embeddings for {len(node_embeddings)} nodes in {batch_node_embedding_duration:.2f} seconds')

    device = 'cuda' if link_prediction_use_gpu and torch.cuda.is_available() else 'cpu'

    return evaluate_link_prediction(
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        node_embeddings,
        link_prediction_training_percentage,
        device
    )


def run_link_prediction_streaming_window(
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
        batch_ts_size,
        sliding_window_duration,
        weighted_sum_alpha,
        embedding_dim,
        link_prediction_training_percentage,
        walk_use_gpu,
        link_prediction_use_gpu,
        seed=42
):
    """Run link prediction using streaming window approach."""
    logger.info("Starting streaming window link prediction")

    temporal_random_walk = TemporalRandomWalk(is_directed=is_directed, use_gpu=walk_use_gpu,
                                              max_time_capacity=sliding_window_duration)

    # Global embedding store (dictionary)
    global_embeddings = {}

    # Create batches by batch_ts_size - ensure we use NumPy arrays
    unique_timestamps = np.sort(np.unique(train_timestamps))
    num_batches = len(unique_timestamps) // batch_ts_size

    logger.info(f"Processing {num_batches} batches with batch_ts_size={batch_ts_size}")

    total_start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start_time = time.time()

        # Get timestamp range for current batch
        start_ts_idx = batch_idx * batch_ts_size
        end_ts_idx = min((batch_idx + 1) * batch_ts_size, len(unique_timestamps))
        batch_timestamps = unique_timestamps[start_ts_idx:end_ts_idx]

        # Filter edges for current batch - use NumPy boolean indexing
        batch_mask = np.isin(train_timestamps, batch_timestamps)
        batch_sources = train_sources[batch_mask]
        batch_targets = train_targets[batch_mask]
        batch_ts = train_timestamps[batch_mask]

        logger.info(f"Batch {batch_idx + 1}/{num_batches}: {len(batch_sources)} edges")

        # Add batch to temporal_random_walk
        logger.info(f'Adding {len(batch_sources)} edges in temporal random walk instance')
        temporal_random_walk.add_multiple_edges(batch_sources, batch_targets, batch_ts)

        streaming_walk_start_time = time.time()
        # Get walks from the instance
        walks, timestamps, walk_lengths = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
            max_walk_len=walk_length,
            num_walks_per_node=num_walks_per_node,
            walk_bias='ExponentialIndex',
            initial_edge_bias='Uniform',
            walk_direction="Backward_In_Time"
        )
        streaming_walk_duration = time.time() - streaming_walk_start_time

        logger.info(
            f'Generated {len(walks)} walks in {streaming_walk_duration:.2f} seconds. Mean walk length: {np.mean(walk_lengths)}.')

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

        # Clean walks (remove padding)
        clean_walks = []
        for walk, length in zip(walks, walk_lengths):
            clean_walk = [str(node) for node in walk[:length]]
            if len(clean_walk) > 1:  # Skip single-node walks
                clean_walks.append(clean_walk)

        # Train Word2Vec on current batch
        try:
            batch_model = Word2Vec(
                sentences=clean_walks,
                vector_size=embedding_dim,
                window=10,
                min_count=1,
                workers=4,
                sg=1,
                seed=seed
            )

            # Extract batch embeddings
            batch_embeddings = {}
            for node in batch_model.wv.index_to_key:
                batch_embeddings[int(node)] = batch_model.wv[node]

            # Merge with global embedding store using weighted sum
            if not global_embeddings:
                # First batch - initialize global embeddings
                global_embeddings = batch_embeddings.copy()
            else:
                # Merge embeddings using weighted sum
                for node_id, batch_embedding in batch_embeddings.items():
                    if node_id in global_embeddings:
                        # Weighted sum: alpha * old + (1-alpha) * new
                        global_embeddings[node_id] = (
                                weighted_sum_alpha * global_embeddings[node_id] +
                                (1 - weighted_sum_alpha) * batch_embedding
                        )
                    else:
                        # New node - add directly
                        global_embeddings[node_id] = batch_embedding

            batch_duration = time.time() - batch_start_time
            logger.info(f"Batch {batch_idx + 1} completed in {batch_duration:.2f}s. "
                        f"Total nodes in global store: {len(global_embeddings)}")

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {e}")
            continue

    total_duration = time.time() - total_start_time
    logger.info(f"Streaming window processing completed in {total_duration:.2f}s. "
                f"Final embedding store size: {len(global_embeddings)}")

    device = 'cuda' if link_prediction_use_gpu and torch.cuda.is_available() else 'cpu'

    return evaluate_link_prediction(
        test_sources,
        test_targets,
        negative_sources,
        negative_targets,
        global_embeddings,
        link_prediction_training_percentage,
        device
    )


def save_results(results, output_path):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def run_link_prediction_experiments(
        data_file_path,
        is_directed,
        batch_ts_size,
        sliding_window_duration,
        weighted_sum_alpha,
        walk_length,
        num_walks_per_node,
        embedding_dim,
        embedding_training_percentage,
        link_prediction_training_percentage,
        full_embedding_use_gpu,
        incremental_embedding_use_gpu,
        link_prediction_use_gpu,
        output_dir=None
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

    # Sample negative edges - returns NumPy arrays
    negative_sources, negative_targets = sample_negative_edges(test_sources, test_targets)

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
        link_prediction_training_percentage,
        full_embedding_use_gpu,
        link_prediction_use_gpu
    )
    full_duration = time.time() - full_start_time
    full_link_prediction_results['total_time'] = full_duration

    print(f"\nFull Link Prediction Results:")
    print(f"AUC: {full_link_prediction_results['auc']:.4f}")
    print(f"Accuracy: {full_link_prediction_results['accuracy']:.4f}")
    print(f"Precision: {full_link_prediction_results['precision']:.4f}")
    print(f"Recall: {full_link_prediction_results['recall']:.4f}")
    print(f"F1-Score: {full_link_prediction_results['f1_score']:.4f}")
    print(f"Embedding Coverage: {full_link_prediction_results['embedding_coverage']:.4f}")
    print(f"Total Time: {full_duration:.2f}s")

    # Run streaming approach
    logger.info("=" * 50)
    logger.info("STREAMING WINDOW APPROACH")
    logger.info("=" * 50)

    streaming_start_time = time.time()
    streaming_link_prediction_results = run_link_prediction_streaming_window(
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
        batch_ts_size,
        sliding_window_duration,
        weighted_sum_alpha,
        embedding_dim,
        link_prediction_training_percentage,
        incremental_embedding_use_gpu,
        link_prediction_use_gpu
    )
    streaming_duration = time.time() - streaming_start_time
    streaming_link_prediction_results['total_time'] = streaming_duration

    print(f"\nStreaming Link Prediction Results:")
    print(f"AUC: {streaming_link_prediction_results['auc']:.4f}")
    print(f"Accuracy: {streaming_link_prediction_results['accuracy']:.4f}")
    print(f"Precision: {streaming_link_prediction_results['precision']:.4f}")
    print(f"Recall: {streaming_link_prediction_results['recall']:.4f}")
    print(f"F1-Score: {streaming_link_prediction_results['f1_score']:.4f}")
    print(f"Embedding Coverage: {streaming_link_prediction_results['embedding_coverage']:.4f}")
    print(f"Total Time: {streaming_duration:.2f}s")

    # Comparison
    print(f"\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    auc_diff = streaming_link_prediction_results['auc'] - full_link_prediction_results['auc']
    time_ratio = streaming_duration / full_duration
    print(f"AUC Difference (Streaming - Full): {auc_diff:+.4f}")
    print(f"Time Ratio (Streaming / Full): {time_ratio:.2f}x")

    # Save results if output directory specified
    if output_dir:
        results = {
            'full_approach': full_link_prediction_results,
            'streaming_approach': streaming_link_prediction_results,
            'comparison': {
                'auc_difference': auc_diff,
                'time_ratio': time_ratio
            },
            'parameters': {
                'data_file_path': data_file_path,
                'is_directed': is_directed,
                'batch_ts_size': batch_ts_size,
                'sliding_window_duration': sliding_window_duration,
                'weighted_sum_alpha': weighted_sum_alpha,
                'walk_length': walk_length,
                'num_walks_per_node': num_walks_per_node,
                'embedding_dim': embedding_dim,
                'embedding_training_percentage': embedding_training_percentage,
                'link_prediction_training_percentage': link_prediction_training_percentage
            }
        }

        output_path = Path(output_dir) / "link_prediction_results.json"
        save_results(results, output_path)

    return full_link_prediction_results, streaming_link_prediction_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal Link Prediction Test")
    parser.add_argument(
        '--data_file_path', type=str, required=True,
        help='Path to data file (parquet format)'
    )
    parser.add_argument('--batch_ts_size', type=int, required=True, help='Number of timestamps per batch')
    parser.add_argument('--sliding_window_duration', type=int, required=True,
                        help='Sliding window duration for temporal random walk')
    parser.add_argument('--is_directed', type=lambda x: x.lower() == 'true', required=True,
                        help='Whether the graph is directed')

    parser.add_argument('--weighted_sum_alpha', type=float, default=0.5,
                        help='Alpha parameter for weighted sum in streaming approach')
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

    parser.add_argument('--full_embedding_use_gpu', action='store_true',
                        help='Enable GPU acceleration for full embedding')

    parser.add_argument('--incremental_embedding_use_gpu', action='store_true',
                        help='Enable GPU acceleration for incremental embedding')

    parser.add_argument('--link_prediction_use_gpu', action='store_true',
                        help='Enable GPU acceleration for link prediction')

    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (optional)')

    args = parser.parse_args()

    run_link_prediction_experiments(
        args.data_file_path,
        args.is_directed,
        args.batch_ts_size,
        args.sliding_window_duration,
        args.weighted_sum_alpha,
        args.walk_length,
        args.num_walks_per_node,
        args.embedding_dim,
        args.embedding_training_percentage,
        args.link_prediction_training_percentage,
        args.full_embedding_use_gpu,
        args.incremental_embedding_use_gpu,
        args.link_prediction_use_gpu,
        args.output_dir
    )
