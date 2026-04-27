"""Print the 3 highest and 3 lowest text examples for each geometry metric.

Reads the per-shard metrics CSVs produced by get_embs.py, decodes the
corresponding token ranges from the original FineWeb binary shards, and
prints the texts to stdout.

Random-baseline columns (random_seed_*) are intentionally skipped.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm

from config import METRICS_PATH, FINEWEB_PATH, METRIC_COLUMNS

# ── constants ──────────────────────────────────────────────────────────────
N_EXAMPLES = 3
TEXT_PREVIEW_CHARS = 600  # truncate very long texts in the printout

NON_RANDOM_METRICS = [m for m in METRIC_COLUMNS if not m.startswith("random_seed_")]


# ── helpers ────────────────────────────────────────────────────────────────

def load_all_metrics(metrics_path: Path) -> pd.DataFrame:
    """Concatenate every per-shard metrics CSV into a single DataFrame."""
    csv_files = sorted(
        fp for fp in metrics_path.iterdir() if fp.name.startswith("fineweb_train")
    )
    if not csv_files:
        raise FileNotFoundError(f"No metric CSV files found in {metrics_path}")

    print(f"Found {len(csv_files)} shard metric file(s) in {metrics_path}")
    dfs = []
    for fp in tqdm(csv_files, desc="Loading metric CSVs"):
        df = pd.read_csv(fp)
        df["shard_stem"] = fp.stem
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(combined):,}")
    return combined


def preload_shards(stems: list[str], fineweb_path: Path) -> dict[str, np.ndarray]:
    """Load each unique shard .bin file into memory once."""
    unique_stems = sorted(set(stems))
    cache: dict[str, np.ndarray] = {}
    for stem in tqdm(unique_stems, desc="Loading shard .bin files"):
        cache[stem] = np.fromfile(fineweb_path / (stem + ".bin"), dtype=np.uint16)
    return cache


def decode_sample(row: pd.Series, shard_cache: dict[str, np.ndarray], enc) -> str:
    """Decode a single text sample from its token range."""
    arr = shard_cache[row["shard_stem"]]
    # start_idx points to the EOT token; actual text starts one token later
    tokens = arr[int(row["start_idx"]) + 1 : int(row["end_idx"])]
    return enc.decode(tokens.tolist())


def print_examples(metric: str, df: pd.DataFrame, shard_cache: dict, enc) -> None:
    """Print the top-N and bottom-N examples for one metric."""
    separator = "=" * 72

    sorted_asc = df.sort_values(metric, ascending=True, ignore_index=True)

    print(f"\n{separator}")
    print(f"  METRIC: {metric}")
    print(separator)

    # ── highest values ──────────────────────────────────────────────────
    print(f"\n  >>> TOP {N_EXAMPLES} (highest {metric}) <<<\n")
    top_rows = sorted_asc.tail(N_EXAMPLES).iloc[::-1].reset_index(drop=True)
    for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
        text = decode_sample(row, shard_cache, enc)
        preview = text[:TEXT_PREVIEW_CHARS] + ("…" if len(text) > TEXT_PREVIEW_CHARS else "")
        print(f"  [{rank}]  {metric} = {row[metric]:.4f}")
        print(f"       shard={row['shard_stem']}  tokens=[{int(row['start_idx'])},{int(row['end_idx'])})")
        print()
        print(preview)
        print()

    # ── lowest values ───────────────────────────────────────────────────
    print(f"  >>> BOTTOM {N_EXAMPLES} (lowest {metric}) <<<\n")
    bot_rows = sorted_asc.head(N_EXAMPLES).reset_index(drop=True)
    for rank, (_, row) in enumerate(bot_rows.iterrows(), start=1):
        text = decode_sample(row, shard_cache, enc)
        preview = text[:TEXT_PREVIEW_CHARS] + ("…" if len(text) > TEXT_PREVIEW_CHARS else "")
        print(f"  [{rank}]  {metric} = {row[metric]:.4f}")
        print(f"       shard={row['shard_stem']}  tokens=[{int(row['start_idx'])},{int(row['end_idx'])})")
        print()
        print(preview)
        print()


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    enc = tiktoken.get_encoding("gpt2")

    all_df = load_all_metrics(METRICS_PATH)

    # Only keep rows where the model had enough tokens for meaningful metrics
    before = len(all_df)
    all_df = all_df[all_df["has_enough_tokens"]].reset_index(drop=True)
    print(f"Dropped {before - len(all_df):,} rows missing enough tokens  ->  {len(all_df):,} remaining")

    shard_cache = preload_shards(all_df["shard_stem"].tolist(), FINEWEB_PATH)

    for metric in NON_RANDOM_METRICS:
        print_examples(metric, all_df, shard_cache, enc)

    print("\nDone.")


if __name__ == "__main__":
    main()
