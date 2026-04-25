import numpy as np
import pandas as pd
from tqdm import tqdm
from skdim.id import MLE, CorrInt, TwoNN
from collections import defaultdict
import os

from config import EMB_PATH, OUTPUT_PATH


# Persistent RNGs for random baselines — one per seed, advancing across samples
# so each sample gets a different draw while remaining fully reproducible.
_random_rngs = [np.random.RandomState(seed) for seed in range(10)]


def compute_metrics_for_emb(mat: np.ndarray) -> dict:
    """Computes all geometry metrics for a single (seq_len, hidden_dim) matrix.

    SVD is performed once and reused across metrics that need singular values.
    """
    metrics = {}

    # ID: MLE
    metrics["MLE"] = MLE().fit_transform(mat)

    # ID: CorrInt
    metrics["CorrInt"] = CorrInt().fit_transform(mat)

    # ID: TwoNN
    metrics["TwoNN"] = TwoNN().fit_transform(mat)

    # Singular values shared by the next two metrics
    singular_values = np.linalg.svd(mat, compute_uv=False)

    # Schatten norm (nuclear norm = Schatten-1 norm: sum of singular values)
    metrics["schatten_norm"] = float(np.sum(singular_values))

    # Effective rank: exp(Shannon entropy of normalised singular values)
    p = singular_values / (singular_values.sum() + 1e-10)
    entropy = -np.sum(p * np.log(p + 1e-10))
    metrics["effective_rank"] = float(np.exp(entropy))

    # 10 random (uniform) baselines — each RNG advances its state across samples
    for seed, rng in enumerate(_random_rngs):
        metrics[f"random_seed_{seed}"] = float(rng.uniform())

    return metrics


os.makedirs(OUTPUT_PATH, exist_ok=True)

for npz_path in sorted(EMB_PATH.glob("*.npz")):
    csv_path = OUTPUT_PATH / f"{npz_path.stem}.csv"
    if csv_path.exists():
        print(f"Skipping {npz_path.stem} (already done)")
        continue

    print(f"Computing metrics for {npz_path.stem}")
    data = np.load(npz_path)
    embs = data["embs"]                        # (N, SEQ_LEN, hidden_dim), float16
    seq_lens = data["seq_lens"]                # (N,), actual token count per sample
    has_enough_tokens = data["has_enough_tokens"]  # (N,), bool

    results = defaultdict(list)
    results["has_enough_tokens"] = has_enough_tokens.tolist()
    results["start_idx"] = data["start_idx"].tolist()
    results["end_idx"] = data["end_idx"].tolist()

    for i in tqdm(range(len(embs)), desc="Computing metrics"):
        # Slice to the real token count; cast back to float32 for numerical stability
        mat = embs[i, :seq_lens[i]].astype(np.float32)
        for key, val in compute_metrics_for_emb(mat).items():
            results[key].append(val)

    pd.DataFrame(results).to_csv(csv_path, index=False)
