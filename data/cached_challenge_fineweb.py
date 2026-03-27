from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_VARIANT = "sp1024"
DEFAULT_TRAIN_SHARDS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", default=DEFAULT_VARIANT, choices=[DEFAULT_VARIANT, "cl100k"]
    )
    parser.add_argument("--train-shards", type=int, default=DEFAULT_TRAIN_SHARDS)
    parser.add_argument("--output-dir", type=str, default="data/datasets")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / f"fineweb10B_{args.variant}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[PLACEHOLDER] Replace this file with cached_challenge_fineweb.py from openai/parameter-golf."
    )
    print(
        f"Would download variant={args.variant} with {args.train_shards} shards to {output_dir}"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
