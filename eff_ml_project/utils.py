"""Shared data-loading helpers used across scripts."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_all_metrics(metrics_path: Path, output=sys.stdout) -> pd.DataFrame:
    """Concatenate every per-shard metrics CSV into a single DataFrame."""
    csv_files = sorted(
        fp for fp in metrics_path.iterdir() if fp.name.startswith("fineweb_train")
    )
    if not csv_files:
        raise FileNotFoundError(f"No metric CSV files found in {metrics_path}")

    print(f"Found {len(csv_files)} shard metric file(s) in {metrics_path}", file=output)
    dfs = []
    for fp in tqdm(csv_files, desc="Loading metric CSVs"):
        df = pd.read_csv(fp)
        df["shard_stem"] = fp.stem
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(combined):,}", file=output)
    return combined


def preload_shards(stems: list[str], fineweb_path: Path) -> dict[str, np.ndarray]:
    """Load each unique shard .bin file into memory once."""
    unique = sorted(set(stems))
    cache: dict[str, np.ndarray] = {}
    for stem in tqdm(unique, desc="Loading shard .bin files"):
        cache[stem] = np.fromfile(fineweb_path / (stem + ".bin"), dtype=np.uint16)
    return cache


def decode_sample(row: pd.Series, shard_cache: dict[str, np.ndarray], enc) -> str:
    """Decode a single text sample from its token range.

    start_idx points to the EOT token; actual text starts one token later.
    """
    arr = shard_cache[row["shard_stem"]]
    tokens = arr[int(row["start_idx"]) + 1 : int(row["end_idx"])]
    return enc.decode(tokens.tolist())
