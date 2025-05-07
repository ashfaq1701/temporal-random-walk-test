import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raphtory vs Temporal Walk")

    # Base directory for input/output files
    parser.add_argument(
        '--base_dir', type=str, required=True,
        help='Base directory containing input files'
    )

    # Enable GPU acceleration
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Enable GPU acceleration'
    )

    # Window timestep size in milliseconds (default: 1 hour)
    parser.add_argument(
        '--window_size', type=int, default=3_600_000,
        help='Sliding window size in milliseconds (default: 3,600,000 = 1 hour)'
    )

    args = parser.parse_args()

    # Example usage
    print(f"Base dir: {args.base_dir}")
    print(f"Use GPU: {args.use_gpu}")
    print(f"Window size: {args.window_size} ms")
