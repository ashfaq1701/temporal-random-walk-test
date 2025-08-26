import argparse
import json
import logging
import os
import pickle
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

import pandas as pd


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


def process_edges(raw_data_dir, logger):
    """Process tweet JSON files and extract edges"""
    logger.info(f"Starting edge processing from directory: {raw_data_dir}")
    edges = []
    total_tweets = 0
    total_no_edge_tweets = 0  # Counter for tweets with no edges

    for file_idx in range(9):
        file_path = os.path.join(raw_data_dir, f"tweet_{file_idx}.json")
        logger.info(f"Processing file {file_idx + 1}/9: {file_path}")

        try:
            with open(file_path, 'r') as f:
                tweets = json.load(f)

            file_tweet_count = len(tweets)
            file_mention_count = 0
            file_reply_count = 0
            file_no_edge_count = 0  # Counter for this file
            total_tweets += file_tweet_count

            for tweet in tweets:
                initial_edge_count = len(edges)  # Track edges before processing this tweet

                u = str(tweet['author_id'])
                dt = datetime.fromisoformat(tweet['created_at'])
                ts = int(dt.timestamp())

                entities = tweet.get('entities') or {}
                user_mentions = entities.get('user_mentions', [])

                for mention in user_mentions:
                    i = str(mention['id'])
                    edges.append({'u': u, 'i': i, 'ts': ts, 'relation': 'mentioned'})
                    file_mention_count += 1

                reply_to = tweet.get('in_reply_to_user_id')
                if reply_to is not None:
                    i = str(reply_to)
                    edges.append({'u': u, 'i': i, 'ts': ts, 'relation': 'replied'})
                    file_reply_count += 1

                # Check if this tweet contributed any edges
                if len(edges) == initial_edge_count:
                    file_no_edge_count += 1
                    total_no_edge_tweets += 1
                    # Print the whole tweet object
                    logger.info(f"No edges from tweet - Full object: {json.dumps(tweet, indent=2)}")

            logger.info(
                f"File {file_idx}: {file_tweet_count} tweets, {file_mention_count} mentions, {file_reply_count} replies, {file_no_edge_count} no-edge tweets")

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing key in tweet data: {e}")
            raise

    edges_df = pd.DataFrame(edges)
    edges_df = edges_df.sort_values('ts').reset_index(drop=True)
    logger.info(
        f"Edge processing complete: {len(edges_df)} edges from {total_tweets} tweets ({total_no_edge_tweets} tweets contributed no edges) - sorted by timestamp")
    return edges_df


def read_labels(raw_data_dir, logger):
    """Read and process label CSV file"""
    file_path = os.path.join(raw_data_dir, "label.csv")
    logger.info(f"Reading labels from: {file_path}")

    try:
        label_df = pd.read_csv(file_path)
        original_count = len(label_df)

        # Remove 'u' prefix from IDs
        label_df['id'] = label_df['id'].str.lstrip('u')
        logger.info(f"Processed {original_count} labels, removed 'u' prefix from IDs")

        return label_df

    except FileNotFoundError:
        logger.error(f"Label file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading labels: {e}")
        raise


def read_split(raw_data_dir, logger):
    """Read and process split CSV file"""
    file_path = os.path.join(raw_data_dir, "split.csv")
    logger.info(f"Reading split data from: {file_path}")

    try:
        split_df = pd.read_csv(file_path)
        original_count = len(split_df)

        # Remove 'u' prefix from IDs
        split_df['id'] = split_df['id'].str.lstrip('u')
        logger.info(f"Processed {original_count} split entries, removed 'u' prefix from IDs")

        return split_df

    except FileNotFoundError:
        logger.error(f"Split file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading split data: {e}")
        raise


def merge_and_encode_nodes(edges_df, labels_df, split_df, logger):
    """Merge node data and encode all node IDs"""
    logger.info("Starting node encoding process")

    # Get unique nodes from each source
    edge_nodes = pd.concat([edges_df['u'], edges_df['i']]).unique()
    label_nodes = labels_df['id'].unique()
    split_nodes = split_df['id'].unique()

    logger.info(f"Unique nodes - Edges: {len(edge_nodes)}, Labels: {len(label_nodes)}, Split: {len(split_nodes)}")

    # Create complete node set
    complete_node_set = set(edge_nodes) | set(label_nodes) | set(split_nodes)
    logger.info(f"Total unique nodes: {len(complete_node_set)}")

    # Train label encoder
    logger.info("Training label encoder on complete node set")
    label_encoder = LabelEncoder()
    label_encoder.fit(list(complete_node_set))

    # Encode all node columns
    logger.info("Encoding node IDs in all dataframes")
    edges_df['u'] = label_encoder.transform(edges_df['u'])
    edges_df['i'] = label_encoder.transform(edges_df['i'])
    labels_df['id'] = label_encoder.transform(labels_df['id'])
    split_df['id'] = label_encoder.transform(split_df['id'])

    logger.info("Node encoding complete")
    return edges_df, labels_df, split_df, label_encoder


def save_processed_data(processed_data_dir, encoded_edges_df, encoded_label_df, encoded_split_df, label_encoder,
                        logger):
    """Save all processed data to files"""
    logger.info(f"Saving processed data to: {processed_data_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)

    try:
        # Save CSV files
        edges_path = os.path.join(processed_data_dir, 'edges.csv')
        encoded_edges_df.to_csv(edges_path, index=False)
        logger.info(f"Saved edges to: {edges_path} ({len(encoded_edges_df)} rows)")

        labels_path = os.path.join(processed_data_dir, 'labels.csv')
        encoded_label_df.to_csv(labels_path, index=False)
        logger.info(f"Saved labels to: {labels_path} ({len(encoded_label_df)} rows)")

        split_path = os.path.join(processed_data_dir, 'split.csv')
        encoded_split_df.to_csv(split_path, index=False)
        logger.info(f"Saved split data to: {split_path} ({len(encoded_split_df)} rows)")

        # Save label encoder
        encoder_path = os.path.join(processed_data_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        logger.info(f"Saved label encoder to: {encoder_path}")

    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Twibot-22 dataset preparation")

    parser = argparse.ArgumentParser(description="Script to prepare twibot 22 dataset")

    parser.add_argument(
        '--raw_data_dir', type=str, required=True,
        help='Base directory containing data files'
    )

    parser.add_argument(
        '--processed_data_dir', type=str, required=True,
        help='Directory for processed data output'
    )

    args = parser.parse_args()

    try:
        logger.info(f"Input directory: {args.raw_data_dir}")
        logger.info(f"Output directory: {args.processed_data_dir}")

        # Process all data
        edges_df = process_edges(args.raw_data_dir, logger)
        labels_df = read_labels(args.raw_data_dir, logger)
        split_df = read_split(args.raw_data_dir, logger)

        # Encode nodes
        encoded_edges_df, encoded_label_df, encoded_split_df, label_encoder = merge_and_encode_nodes(
            edges_df, labels_df, split_df, logger
        )

        # Save results
        save_processed_data(
            args.processed_data_dir,
            encoded_edges_df,
            encoded_label_df,
            encoded_split_df,
            label_encoder,
            logger
        )

        logger.info("Data processing completed successfully")
        logger.info(f"Final statistics:")
        logger.info(f"  - Edges: {len(encoded_edges_df)}")
        logger.info(f"  - Labels: {len(encoded_label_df)}")
        logger.info(f"  - Split entries: {len(encoded_split_df)}")
        logger.info(f"  - Unique nodes: {len(label_encoder.classes_)}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
