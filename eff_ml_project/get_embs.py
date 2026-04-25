from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM
from skdim.id import MLE, CorrInt, TwoNN
from collections import defaultdict
import torch
import os
import gc

from config import BATCH_SIZE, SEQ_LEN, DATA_PATH, OUTPUT_PATH


# Persistent RNGs for random baselines — one per seed, advancing across samples
# so each sample gets a different draw while remaining fully reproducible.
_random_rngs = [np.random.RandomState(seed) for seed in range(10)]

# Holds the full batch embedding from the most recent forward pass.
# Shape: (batch, seq_len, hidden_dim)
current_batch_embs = None

def hook_get_embs(module, inputs, outputs):
    global current_batch_embs
    # outputs[0]: (batch, seq_len, hidden_dim)
    current_batch_embs = outputs[0].detach().cpu()


def get_texts(arr, enc, separators):
    """Converts sequences of token ids into a list of texts."""
    texts = []
    for start, end in zip(separators[:-1], separators[1:]):
        sample_token_ids = arr[start + 1:end]
        sample_text = enc.decode(sample_token_ids)
        texts.append(sample_text)
    return texts


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


def process_shard(arr, enc, eot, tok, model):
    """Processes one shard with batched GPU inference.

    Texts are grouped into batches of BATCH_SIZE, padded to the longest sequence
    in the batch, and run through the model in a single forward pass. Per-sample
    embeddings are extracted from the batch output by trimming padding via the
    attention mask before computing metrics.
    """
    global current_batch_embs
    separators = np.where(arr == eot)[0][1:]
    texts = get_texts(arr, enc, separators)

    results = defaultdict(list)

    for batch_start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing batches"):
        batch_texts = texts[batch_start:batch_start + BATCH_SIZE]

        # Tokenize whole batch at once; pad to the longest sequence in the batch
        inputs = tok(
            batch_texts,
            return_tensors="pt",
            max_length=SEQ_LEN,
            truncation=True,
            padding=True,         # right-pad to longest in batch
        )
        # Keep attention_mask on CPU for slicing; move inputs to GPU for inference
        attention_mask = inputs["attention_mask"]  # (batch, seq_len) — CPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        # current_batch_embs: (batch, padded_seq_len, hidden_dim) — CPU
        batch_embs = current_batch_embs
        current_batch_embs = None  # release reference immediately

        for i, mask in enumerate(attention_mask):
            seq_len = int(mask.sum())           # real (non-padded) token count
            has_enough = seq_len >= SEQ_LEN     # False for texts shorter than SEQ_LEN
            results["has_enough_tokens"].append(has_enough)
            mat = batch_embs[i, :seq_len].float().numpy()
            for key, val in compute_metrics_for_emb(mat).items():
                results[key].append(val)

    results["start_idx"] = separators[:-1]
    results["end_idx"] = separators[1:]
    return results


np.random.seed(42)
device = "cuda:0"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
tok.padding_side = "right"  # causal model: right-pad so real tokens are unaffected
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']
model = model.to(device)
model = model.eval()
L = 12
handle = model.model.layers[L].register_forward_hook(hook_get_embs)
os.makedirs(OUTPUT_PATH, exist_ok=True)
for filepath in DATA_PATH.iterdir():
    if filepath.suffix == ".bin":
        print("Processing", filepath.stem)
        arr = np.fromfile(filepath, dtype=np.uint16)
        metrics = process_shard(arr, enc, eot, tok, model)
        del arr
        gc.collect()
        pd.DataFrame(metrics).to_csv(OUTPUT_PATH / f"{filepath.stem}.csv", index=False)
