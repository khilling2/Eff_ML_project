from pathlib import Path
import os
# Set BEFORE any numpy/scipy/sklearn import so BLAS thread pools initialise
# with 1 thread. With 16 forked workers, unconstrained BLAS gives
# 16 × N_CPU threads competing → extreme slowdown (minutes per sample).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Must be set before the tokenizers library initialises its Rust thread pool.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.linalg import svd as scipy_svd
from skdim.id import MLE, CorrInt, TwoNN
from threadpoolctl import threadpool_limits
from collections import defaultdict
from tqdm import tqdm
import torch
import multiprocessing as mp
import gc

from config import BATCH_SIZE, SEQ_LEN, N_WORKERS, PCA_COMPONENTS, FINEWEB_PATH, METRICS_PATH, DEVICE, LAYER


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
    """All geometry metrics except random baselines (handled in main process).

    Args:
        mat: float32 array of shape (seq_len, hidden_dim) — the token embeddings
             for one text, where seq_len <= SEQ_LEN and hidden_dim is the model's
             hidden size (e.g. 896 for Qwen2-0.5B).

    SVD is computed once and reused:
      - singular values  → schatten norm, effective rank
      - U * s[:k]        → PCA-projected point cloud (896 → PCA_COMPONENTS dims)
                           fed to ID estimators for ~17× faster NN search
    """
    # full_matrices=False: U (n_tokens, r), s (r,), Vt (r, hidden) where r = min(n_tokens, hidden)
    # check_finite=False: skip input validation for speed
    U, s, _ = scipy_svd(mat, full_matrices=False, check_finite=False)

    # Schatten norm and effective rank use all singular values
    metrics = {}
    metrics["schatten_norm"] = float(s.sum())
    p = s / (s.sum() + 1e-10)
    metrics["effective_rank"] = float(np.exp(-np.sum(p * np.log(p + 1e-10))))

    # Project to top-PCA_COMPONENTS dims: U[:, :k] * s[:k]  ≡  mat @ Vt[:k].T
    # ID estimators work on the reduced cloud — NN search is O(n^2 * k) not O(n^2 * 896)
    k = min(PCA_COMPONENTS, U.shape[1])
    mat_reduced = U[:, :k] * s[:k]  # (n_tokens, k)

    metrics["MLE"] = MLE().fit_transform(mat_reduced)
    metrics["CorrInt"] = CorrInt().fit_transform(mat_reduced)
    metrics["TwoNN"] = TwoNN().fit_transform(mat_reduced)

    return metrics


def _worker_single(args):
    """Compute metrics for one sample. Called by pool.imap — inherits
    _all_embs_for_workers from the parent via fork (copy-on-write, read-only)."""
    i, slen = args
    mat = _all_embs_for_workers[i, :slen].astype(np.float32)
    # Belt-and-suspenders: threadpoolctl works at the BLAS API level and
    # takes effect even if the thread pool was already initialised.
    with threadpool_limits(limits=1):
        return _compute_metrics_no_random(mat)


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

    # chunksize: number of samples sent to a worker in one IPC message.
    # Smaller = smoother tqdm; larger = less IPC overhead. 256 is a good balance.
    chunksize = max(1, n // (N_WORKERS * 16))
    sample_args = [(i, int(seq_lens[i])) for i in range(n)]

    ctx = mp.get_context("fork")  # fork: workers inherit _all_embs_for_workers cheaply
    with ctx.Pool(N_WORKERS) as pool:
        # imap preserves order → metrics_list[i] corresponds to sample i
        metrics_list = list(tqdm(
            pool.imap(_worker_single, sample_args, chunksize=chunksize),
            total=n, desc="Metrics",
        ))

    _all_embs_for_workers = None
    gc.collect()

    results = defaultdict(list)
    results["has_enough_tokens"] = has_enough_tokens.tolist()
    results["start_idx"] = separators[:-1].tolist()
    results["end_idx"] = separators[1:].tolist()
    for seed in range(10):
        results[f"random_seed_{seed}"] = random_draws[seed].tolist()
    for key in metrics_list[0]:
        results[key] = [m[key] for m in metrics_list]

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
    os.makedirs(METRICS_PATH, exist_ok=True)

    i = 0
    for filepath in FINEWEB_PATH.iterdir():
        if filepath.suffix == ".bin":
            i += 1
            if i > 5:
                continue
            csv_path = METRICS_PATH / f"{filepath.stem}.csv"
            if csv_path.exists():
                print(f"Skipping {filepath.stem} (already done)")
                continue
            print("Processing", filepath.stem)
            arr = np.fromfile(filepath, dtype=np.uint16)
            results = process_shard(arr, enc, eot, tok, model)
            del arr
            gc.collect()
            pd.DataFrame(results).to_csv(csv_path, index=False)
