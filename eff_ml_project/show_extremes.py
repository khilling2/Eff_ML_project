"""Print the 3 highest and 3 lowest text examples for each geometry metric.

Reads the per-shard metrics CSVs produced by get_embs.py, decodes the
corresponding token ranges from the original FineWeb binary shards, and
prints the texts to stdout.

Random-baseline columns (random_seed_*) are intentionally skipped.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm

from config import METRICS_PATH, FINEWEB_PATH, METRIC_COLUMNS
from utils import load_all_metrics, preload_shards, decode_sample

# ── output destination ────────────────────────────────────────────────────
OUTPUT = open(Path(__file__).parent / "extremes.csv", "w")

# ── constants ──────────────────────────────────────────────────────────────
N_EXAMPLES = 20
TEXT_PREVIEW_CHARS = 600  # truncate very long texts in the printout

NON_RANDOM_METRICS = [m for m in METRIC_COLUMNS if not m.startswith("random_seed_")]


# ── helpers ────────────────────────────────────────────────────────────────

def print_examples(metric: str, df: pd.DataFrame, shard_cache: dict, enc) -> None:
    """Print the top-N and bottom-N examples for one metric."""
    separator = "=" * 72

    sorted_asc = df.sort_values(metric, ascending=True, ignore_index=True)

    print(f"\n{separator}", file=OUTPUT)
    print(f"  METRIC: {metric}", file=OUTPUT)
    print(separator, file=OUTPUT)

    # ── highest values ──────────────────────────────────────────────────
    print(f"\n  >>> TOP {N_EXAMPLES} (highest {metric}) <<<\n", file=OUTPUT)
    top_rows = sorted_asc.tail(N_EXAMPLES).iloc[::-1].reset_index(drop=True)
    for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
        text = decode_sample(row, shard_cache, enc)
        preview = text[:TEXT_PREVIEW_CHARS] + ("…" if len(text) > TEXT_PREVIEW_CHARS else "")
        print(f"  [{rank}]  {metric} = {row[metric]:.4f}", file=OUTPUT)
        print(f"       shard={row['shard_stem']}  tokens=[{int(row['start_idx'])},{int(row['end_idx'])})", file=OUTPUT)
        print(file=OUTPUT)
        print(preview, file=OUTPUT)
        print(file=OUTPUT)

    # ── lowest values ───────────────────────────────────────────────────
    print(f"  >>> BOTTOM {N_EXAMPLES} (lowest {metric}) <<<\n", file=OUTPUT)
    bot_rows = sorted_asc.head(N_EXAMPLES).reset_index(drop=True)
    for rank, (_, row) in enumerate(bot_rows.iterrows(), start=1):
        text = decode_sample(row, shard_cache, enc)
        preview = text[:TEXT_PREVIEW_CHARS] + ("…" if len(text) > TEXT_PREVIEW_CHARS else "")
        print(f"  [{rank}]  {metric} = {row[metric]:.4f}", file=OUTPUT)
        print(f"       shard={row['shard_stem']}  tokens=[{int(row['start_idx'])},{int(row['end_idx'])})", file=OUTPUT)
        print(file=OUTPUT)
        print(preview, file=OUTPUT)
        print(file=OUTPUT)


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    enc = tiktoken.get_encoding("gpt2")

    all_df = load_all_metrics(METRICS_PATH, output=OUTPUT)

    # Only keep rows where the model had enough tokens for meaningful metrics
    before = len(all_df)
    all_df = all_df[all_df["has_enough_tokens"]].reset_index(drop=True)
    print(f"Dropped {before - len(all_df):,} rows missing enough tokens  ->  {len(all_df):,} remaining", file=OUTPUT)

    shard_cache = preload_shards(all_df["shard_stem"].tolist(), FINEWEB_PATH)

    for metric in NON_RANDOM_METRICS:
        print_examples(metric, all_df, shard_cache, enc)

    print("\nDone.", file=OUTPUT)


if __name__ == "__main__":
    main()
