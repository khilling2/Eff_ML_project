from pathlib import Path
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from config import METRIC_COLUMNS, METRICS_PATH, DATA_PATH, FINEWEB_PATH


def write_modded_nanogpt_bin(filepath, tokens):
    """Saves sampled data array in a proper way"""
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = tokens.shape[0]
    with open(filepath, "wb") as f:
        header.tofile(f)
        tokens.tofile(f)


def main():
    # ------------------------------------------------------------------ #
    # 1. Load all shard metrics CSVs into one DataFrame
    # ------------------------------------------------------------------ #
    shard_csv_files = sorted(
        fp for fp in METRICS_PATH.iterdir() if fp.name.startswith("fineweb_train")
    )
    print(f"Found {len(shard_csv_files)} shard metric files in {METRICS_PATH}")

    dfs = []
    for filepath in tqdm(shard_csv_files, desc="Loading metric CSVs"):
        df = pd.read_csv(filepath)
        df["shard_stem"] = filepath.stem
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined metrics DataFrame: {len(all_df):,} rows")

    # ------------------------------------------------------------------ #
    # 2. Remove rows where has_enough_tokens is False
    # ------------------------------------------------------------------ #
    before = len(all_df)
    all_df = all_df[all_df["has_enough_tokens"]].reset_index(drop=True)
    print(f"Dropped {before - len(all_df):,} rows with has_enough_tokens=False  ->  {len(all_df):,} rows remaining")

    # ------------------------------------------------------------------ #
    # 3. Trim to a multiple of (16 * 3 * 5 = 240)
    # ------------------------------------------------------------------ #
    divisor = 16 * 3 * 5
    n_keep = (len(all_df) // divisor) * divisor
    dropped = len(all_df) - n_keep
    all_df = all_df.iloc[:n_keep].reset_index(drop=True)
    print(f"Trimmed {dropped} rows to reach divisibility by {divisor}  ->  {n_keep:,} rows kept")

    n_docs = 10
    chunk_size = n_keep // n_docs
    print(f"Each of {n_docs} output files will contain {chunk_size:,} documents")

    # ------------------------------------------------------------------ #
    # 4. Pre-load all shard token arrays into memory
    # ------------------------------------------------------------------ #
    unique_stems = sorted(all_df["shard_stem"].unique())
    print(f"\nPre-loading {len(unique_stems)} shard token arrays from {FINEWEB_PATH} ...")
    shard_cache = {}
    for stem in tqdm(unique_stems, desc="Loading shard .bin files"):
        shard_cache[stem] = np.fromfile(FINEWEB_PATH / (stem + ".bin"), dtype=np.uint16)
    total_tokens = sum(arr.size for arr in shard_cache.values())
    print(f"Shard cache loaded: {total_tokens:,} uint16 values ({total_tokens * 2 / 1e9:.2f} GB)")

    # ------------------------------------------------------------------ #
    # 5. For each metric: sort descending, split into 10 docs.
    #    Files named 000001..000010 -> lexicographic order = highest to lowest.
    # ------------------------------------------------------------------ #
    print()
    for metric_name in tqdm(METRIC_COLUMNS, desc="Metrics"):
        dir_name = DATA_PATH / metric_name
        os.makedirs(dir_name, exist_ok=True)

        sorted_df = all_df.sort_values(metric_name, ascending=False, ignore_index=True)

        for i in tqdm(range(n_docs), desc=f"  {metric_name} files", leave=False):
            chunk = sorted_df.iloc[i * chunk_size : (i + 1) * chunk_size]
            stems = chunk["shard_stem"].values
            starts = chunk["start_idx"].values.astype(int)
            ends = chunk["end_idx"].values.astype(int)

            tokens_list = [
                shard_cache[stems[j]][starts[j] : ends[j]]
                for j in range(len(chunk))
            ]

            tokens = np.concatenate(tokens_list)
            filename = f"fineweb_train_{i + 1:06d}.bin"
            write_modded_nanogpt_bin(dir_name / filename, tokens)

        print(f"  [{metric_name}] done  ->  {dir_name}")

    print("\nAll metrics written successfully.")


if __name__ == "__main__":
    main()
