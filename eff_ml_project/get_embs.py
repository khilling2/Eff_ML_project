from pathlib import Path
import numpy as np
import pandas as pd
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM
from skdim.id import MLE, CorrInt, TwoNN
from collections import defaultdict
from tqdm import tqdm
import torch
import multiprocessing as mp
import os
import gc

from config import BATCH_SIZE, SEQ_LEN, N_WORKERS, DATA_PATH, OUTPUT_PATH, DEVICE, LAYER


# Persistent RNGs for random baselines — one per seed, advancing across samples.
# Only used in the main process; workers receive pre-generated arrays.
_random_rngs = [np.random.RandomState(seed) for seed in range(10)]

# Set in the main process before forking so workers can read it copy-on-write.
# Workers never write to it, so no actual memory is duplicated.
_all_embs_for_workers: np.ndarray | None = None

# Holds the full batch embedding from the most recent forward pass.
# Shape: (batch, seq_len, hidden_dim)
current_batch_embs = None

def hook_get_embs(module, inputs, outputs):
    global current_batch_embs
    # Decoder layers may return a tuple (hidden_states, ...) or the tensor directly.
    # In both cases we want the 3-D hidden states (batch, seq_len, hidden_dim).
    hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
    current_batch_embs = hidden_states.detach().cpu()


def get_texts(arr, enc, separators):
    """Converts sequences of token ids into a list of texts."""
    texts = []
    for start, end in zip(separators[:-1], separators[1:]):
        sample_token_ids = arr[start + 1:end]
        sample_text = enc.decode(sample_token_ids)
        texts.append(sample_text)
    return texts


# ── metric helpers (module-level so they are picklable / visible after fork) ──

def _compute_metrics_no_random(mat: np.ndarray) -> dict:
    """All geometry metrics except random baselines (handled in main process)."""
    metrics = {}
    metrics["MLE"] = MLE().fit_transform(mat)
    metrics["CorrInt"] = CorrInt().fit_transform(mat)
    metrics["TwoNN"] = TwoNN().fit_transform(mat)
    singular_values = np.linalg.svd(mat, compute_uv=False)
    metrics["schatten_norm"] = float(np.sum(singular_values))
    p = singular_values / (singular_values.sum() + 1e-10)
    entropy = -np.sum(p * np.log(p + 1e-10))
    metrics["effective_rank"] = float(np.exp(entropy))
    return metrics


def _worker_fn(args):
    """Compute metrics for a chunk of sample indices.

    _all_embs_for_workers is inherited from the parent process via fork
    (copy-on-write; no data is copied since workers only read).
    """
    # Limit BLAS/OpenMP threads to 1 per worker; we parallelize at the process level.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    indices, seq_lens = args
    return [
        _compute_metrics_no_random(_all_embs_for_workers[i, :slen].astype(np.float32))
        for i, slen in zip(indices, seq_lens)
    ]


def process_shard(arr, enc, eot, tok, model):
    """Runs batched inference then computes metrics in parallel. No disk I/O for embeddings."""
    global current_batch_embs, _all_embs_for_workers
    separators = np.where(arr == eot)[0][1:]
    texts = get_texts(arr, enc, separators)
    n = len(texts)

    all_embs = None
    seq_lens = np.zeros(n, dtype=np.int16)
    has_enough_tokens = np.zeros(n, dtype=bool)

    # ── stage 1: batched GPU inference ───────────────────────────────────────
    for batch_start in tqdm(range(0, n, BATCH_SIZE), desc="Inference"):
        batch_texts = texts[batch_start:batch_start + BATCH_SIZE]

        inputs = tok(
            batch_texts,
            return_tensors="pt",
            max_length=SEQ_LEN,
            truncation=True,
            padding=True,
        )
        attention_mask = inputs["attention_mask"]  # (batch, seq_len) — CPU
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        if all_embs is None:
            all_embs = np.zeros((n, SEQ_LEN, current_batch_embs.shape[2]), dtype=np.float16)

        for j, mask in enumerate(attention_mask):
            idx = batch_start + j
            slen = int(mask.sum())
            seq_lens[idx] = slen
            has_enough_tokens[idx] = slen >= SEQ_LEN
            all_embs[idx, :slen] = current_batch_embs[j, :slen].float().numpy().astype(np.float16)

        current_batch_embs = None

    # ── stage 2: parallel metrics (N_WORKERS forked processes) ───────────────
    # Pre-generate random baselines in the main process so each sample gets a
    # unique value regardless of which worker processes it.
    random_draws = np.stack([rng.uniform(size=n) for rng in _random_rngs])  # (10, n)

    # Expose embeddings as a global so forked workers can read them copy-on-write.
    _all_embs_for_workers = all_embs
    del all_embs

    chunks = np.array_split(np.arange(n), N_WORKERS)
    worker_args = [(chunk.tolist(), seq_lens[chunk].tolist()) for chunk in chunks]

    ctx = mp.get_context("fork")  # fork: workers inherit _all_embs_for_workers cheaply
    with ctx.Pool(N_WORKERS) as pool:
        chunk_results = list(tqdm(
            pool.imap(_worker_fn, worker_args),
            total=N_WORKERS, desc="Metrics (parallel)",
        ))

    _all_embs_for_workers = None
    gc.collect()

    # Reassemble results in original sample order.
    # Use int() to ensure Python-int keys regardless of numpy scalar type.
    ordered = {}
    for chunk_indices, chunk_res in zip(chunks, chunk_results):
        for i, m in zip(chunk_indices, chunk_res):
            ordered[int(i)] = m

    results = defaultdict(list)
    results["has_enough_tokens"] = has_enough_tokens.tolist()
    results["start_idx"] = separators[:-1].tolist()
    results["end_idx"] = separators[1:].tolist()
    for seed in range(10):
        results[f"random_seed_{seed}"] = random_draws[seed].tolist()
    for key in ordered[0]:
        results[key] = [ordered[i][key] for i in range(n)]

    return results


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    tok.padding_side = "right"  # causal model: right-pad so real tokens keep correct positions
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    model = model.to(DEVICE)
    model = model.eval()
    handle = model.model.layers[LAYER].register_forward_hook(hook_get_embs)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    i = 0
    for filepath in DATA_PATH.iterdir():
        if filepath.suffix == ".bin":
            i += 1
            if i > 5:
                continue
            csv_path = OUTPUT_PATH / f"{filepath.stem}.csv"
            if csv_path.exists():
                print(f"Skipping {filepath.stem} (already done)")
                continue
            print("Processing", filepath.stem)
            arr = np.fromfile(filepath, dtype=np.uint16)
            results = process_shard(arr, enc, eot, tok, model)
            del arr
            gc.collect()
            pd.DataFrame(results).to_csv(csv_path, index=False)
