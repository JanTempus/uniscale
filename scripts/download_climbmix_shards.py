#!/usr/bin/env python3
"""Download the first N shards of karpathy/climbmix-400b-shuffle to a local directory.

The output layout is:
    <output_dir>/
      climbmix/
        train-00000-of-NNNNN.parquet
        train-00001-of-NNNNN.parquet
        ...

Set TRAIN_DIR=<output_dir> in slurm_train_climbmix.sbatch after running this script.

Usage:
    python scripts/download_climbmix_shards.py \\
        --output_dir /iopsstor/scratch/cscs/jtempus/climbmix_35shards \\
        --num_shards 35
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


def main():
    parser = argparse.ArgumentParser(description="Download first N shards of a HF dataset")
    parser.add_argument("--output_dir", required=True, help="Destination directory (TRAIN_DIR parent)")
    parser.add_argument("--num_shards", type=int, default=35, help="Number of shards to download")
    parser.add_argument("--repo_id", default="karpathy/climbmix-400b-shuffle", help="HF dataset repo id")
    args = parser.parse_args()

    out = Path(args.output_dir) / "climbmix"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in {args.repo_id} ...")
    all_repo_files = list(list_repo_files(args.repo_id, repo_type="dataset"))
    parquet_files = [f for f in all_repo_files if f.endswith(".parquet")]
    print(f"  Found {len(all_repo_files)} total files, {len(parquet_files)} parquet files")
    if parquet_files:
        print(f"  Sample parquet paths: {parquet_files[:5]}")
    all_files = sorted(f for f in parquet_files if "train-" in f)
    if not all_files:
        # Fallback: try all parquet files if none match "train-"
        all_files = sorted(parquet_files)
    if not all_files:
        raise RuntimeError(f"No train parquet files found in {args.repo_id}")

    shards = all_files[:args.num_shards]
    print(f"Downloading {len(shards)} / {len(all_files)} shards ...")

    for i, shard_path in enumerate(shards):
        dest = out / Path(shard_path).name
        if dest.exists():
            print(f"  [{i+1}/{len(shards)}] already exists, skipping: {dest.name}")
            continue
        print(f"  [{i+1}/{len(shards)}] {shard_path}")
        local = hf_hub_download(
            repo_id=args.repo_id,
            filename=shard_path,
            repo_type="dataset",
        )
        shutil.copy2(local, dest)

    print(f"\nDone. {len(shards)} shards saved to {out}")
    print(f"Set TRAIN_DIR={args.output_dir} in your sbatch script.")


if __name__ == "__main__":
    main()
