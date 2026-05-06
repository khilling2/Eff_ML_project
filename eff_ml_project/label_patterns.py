"""Classify top-N best and top-N worst samples per metric against the
text-quality patterns defined in config.py using OpenAI structured output.

Output CSV columns:
  metric_name, best_or_worst, num, <pattern names...>, text

Supports resuming: if the output CSV already exists, already-processed
(metric_name, best_or_worst, num) triples are skipped.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import tiktoken
from openai import OpenAI
from pydantic import ValidationError, create_model
from tqdm import tqdm

from config import (
    METRICS_PATH, FINEWEB_PATH, METRIC_COLUMNS, PATTERNS,
    API_KEY, BASE_URL, MODEL,
    NON_RANDOM_METRICS, N_EXTREMES, MAX_CONCURRENT, PARSE_MAX_RETRIES, OUTPUT_CSV,
    PATTERN_NAMES, PATTERN_FIELDS, FIELD_TO_NAME, SYSTEM_PROMPT,
)
from utils import load_all_metrics, preload_shards, decode_sample

# Pydantic model built dynamically so it stays in sync with config.py
PatternFlags = create_model(
    "PatternFlags",
    **{field: (bool, ...) for field in PATTERN_FIELDS},
)

# ── OpenAI call ────────────────────────────────────────────────────────────

def parse_with_retry(client: OpenAI, *, model: str, input_: list, text_format: type):
    """Call client.responses.parse with up to PARSE_MAX_RETRIES attempts.

    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception
    for attempt in range(1, PARSE_MAX_RETRIES + 1):
        try:
            return client.responses.parse(
                model=model,
                input=input_,
                text_format=text_format,
                temperature=0.0,
            )
        except ValidationError as exc:
            last_exc = exc
            print(
                f"parse_with_retry: attempt {attempt}/{PARSE_MAX_RETRIES} failed "
                f"({type(exc).__name__}: {exc})"
            )
    raise last_exc


def classify_text(client: OpenAI, text: str) -> dict[str, bool]:
    resp = parse_with_retry(
        client,
        model=MODEL,
        input_=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        text_format=PatternFlags,
    )
    parsed = resp.output_parsed
    return {FIELD_TO_NAME[f]: getattr(parsed, f) for f in PATTERN_FIELDS}


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    enc    = tiktoken.get_encoding("gpt2")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    all_df = load_all_metrics(METRICS_PATH)
    before = len(all_df)
    all_df = all_df[all_df["has_enough_tokens"]].reset_index(drop=True)
    print(f"Dropped {before - len(all_df):,} rows -> {len(all_df):,} remaining")

    # Resume support: load existing results and remember what's already done
    if OUTPUT_CSV.exists():
        done_df   = pd.read_csv(OUTPUT_CSV)
        done_keys = set(zip(done_df["metric_name"], done_df["best_or_worst"], done_df["num"]))
        all_rows  = done_df.to_dict("records")
        print(f"Resuming: {len(all_rows)} rows already in {OUTPUT_CSV}")
    else:
        done_keys = set()
        all_rows  = []

    # Pre-collect every shard stem that will be needed
    needed_stems = []
    for metric in NON_RANDOM_METRICS:
        s = all_df.sort_values(metric, ascending=True, ignore_index=True)
        needed_stems += s.tail(N_EXTREMES)["shard_stem"].tolist()
        needed_stems += s.head(N_EXTREMES)["shard_stem"].tolist()
    shard_cache = preload_shards(needed_stems, FINEWEB_PATH)

    for metric in tqdm(NON_RANDOM_METRICS, desc="Metrics"):
        sorted_asc = all_df.sort_values(metric, ascending=True, ignore_index=True)
        subsets = [
            ("best",  sorted_asc.tail(N_EXTREMES).iloc[::-1].reset_index(drop=True)),
            ("worst", sorted_asc.head(N_EXTREMES).reset_index(drop=True)),
        ]

        # Build work list, skipping already-done entries
        work: list[tuple[str, str, int, str]] = []  # (metric, label, num, text)
        for label, subset in subsets:
            for num, (_, row) in enumerate(subset.iterrows(), start=1):
                if (metric, label, num) in done_keys:
                    continue
                work.append((metric, label, num, decode_sample(row, shard_cache, enc)))

        if not work:
            print(f"Skipping '{metric}' (already done)")
            continue

        print(f"Classifying {len(work)} samples for metric '{metric}'...")
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            futures = {
                executor.submit(classify_text, client, text): (m, label, num, text)
                for m, label, num, text in work
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=metric):
                m, label, num, text = futures[future]
                flags = future.result()
                all_rows.append({
                    "metric_name":   m,
                    "best_or_worst": label,
                    "num":           num,
                    **flags,
                    "text":          text,
                })

        pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(all_rows)} rows to {OUTPUT_CSV}")

    print(f"\nDone. {OUTPUT_CSV}  ({len(all_rows)} rows total)")


if __name__ == "__main__":
    main()
