import argparse
import pickle

import pandas as pd
from temporal_negative_edge_sampler import collect_all_negatives_by_timestamp


def sample_negative_edges(data_file_path, num_negatives_per_positive, is_directed, output_file_path):
    if data_file_path.endswith('.parquet'):
        df = pd.read_parquet(data_file_path)
    elif data_file_path.endswith('.pkl'):
        with open(data_file_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv(data_file_path)

    df = df[['u', 'i', 'ts']]

    sources = df['u'].to_numpy()
    targets = df['i'].to_numpy()
    timestamps = df['ts'].to_numpy()

    negative_sources, negative_targets = collect_all_negatives_by_timestamp(
        sources,
        targets,
        timestamps,
        is_directed,
        num_negatives_per_positive,
        historical_negative_percentage = 0.0
    )

    print('Sampled negatives, creating df')

    negative_df = pd.DataFrame({
        'u': negative_sources,
        'i': negative_targets
    })

    print('Created df, saving')

    negative_df.to_parquet(output_file_path, index=False)
    print(f'Generated and saved {len(negative_sources)} negative edges')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal random sampler")

    # Required arguments
    parser.add_argument('--data_file_path', type=str, required=True,
                        help='Path to data file (parquet or csv format)')

    parser.add_argument('--num_negatives_per_positive', type=int, required=True)

    parser.add_argument('--is_directed', type=lambda x: x.lower() == 'true', required=True,
                        help='Whether the graph is directed (true/false)')

    parser.add_argument('--output_file_path', type=str, required=True)

    args = parser.parse_args()

    sample_negative_edges(args.data_file_path, args.num_negatives_per_positive, args.is_directed, args.output_file_path)
