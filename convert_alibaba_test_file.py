import argparse
import os.path

import pandas as pd


def main(data_dir):
    merged_df = pd.concat([pd.read_parquet(os.path.join(data_dir, f'data_{i}.parquet')) for i in range(18)], ignore_index=True)
    selected_df = merged_df[['u', 'i', 'ts']]
    selected_df.to_csv('data/alibaba-data.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stellargraph vs Temporal Walk")

    # Base directory for input/output files
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Base directory containing input files'
    )

    args = parser.parse_args()

    main(args.data_dir)
