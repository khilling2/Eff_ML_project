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


saved_embs = []
def hook_get_embs(module, inputs, outputs):
    embs = outputs[0].detach().cpu()
    saved_embs.append(embs)


def get_texts(arr, enc, separators):
    """Converts sequences of token ids into a list of texts"""
    texts = []
    for start, end in zip(separators[:-1], separators[1:]):
        sample_token_ids = arr[start + 1:end]
        sample_text = enc.decode(sample_token_ids)
        texts.append(sample_text)
    return texts


def count_geometry_metrics():
    """Calculates metrics for each text sample"""
    results = defaultdict(list)

    # ID: MLE
    mle_metric = []
    for embs in saved_embs:
        estimator = MLE()
        metric = estimator.fit_predict(embs[0])
        mle_metric.append(metric)
    results["MLE"] = np.array(mle_metric)

    # ID: CorrInt
    corrint_metric = []
    for embs in saved_embs:
        estimator = CorrInt()
        metric = estimator.fit_predict(embs[0])
        corrint_metric.append(metric)
    results["CorrInt"] = np.array(corrint_metric)

    # ID: TwoNN
    twonn_metric = []
    for embs in saved_embs:
        estimator = TwoNN()
        metric = estimator.fit_predict(embs[0])
        twonn_metric.append(metric)
    results["TwoNN"] = np.array(twonn_metric)

    # Schatten norm (nuclear norm = Schatten-1 norm: sum of singular values)
    schatten_metric = []
    for embs in saved_embs:
        mat = embs[0].float().numpy()
        singular_values = np.linalg.svd(mat, compute_uv=False)
        schatten_metric.append(np.sum(singular_values))
    results["schatten_norm"] = np.array(schatten_metric)

    # Effective rank: exp(Shannon entropy of normalised singular values)
    eff_rank_metric = []
    for embs in saved_embs:
        mat = embs[0].float().numpy()
        singular_values = np.linalg.svd(mat, compute_uv=False)
        p = singular_values / (singular_values.sum() + 1e-10)
        entropy = -np.sum(p * np.log(p + 1e-10))
        eff_rank_metric.append(np.exp(entropy))
    results["effective_rank"] = np.array(eff_rank_metric)

    # 10 random (uniform) baselines with different seeds
    for seed in range(10):
        rng = np.random.RandomState(seed)
        results[f"random_seed_{seed}"] = rng.uniform(size=len(saved_embs))

    return results


def process_shard(arr, enc, eot, tok, model):
    """Processes array of token ids into results dict
    with keys: start_idx, end_idx, metrics"""
    separators = np.where(arr == eot)[0][1:]
    texts = get_texts(arr, enc, separators)
    for text in tqdm(texts, desc="Processing texts in a shard"):
        inputs = tok(
            text,
            return_tensors="pt",
            max_length=1024
            )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
    results = count_geometry_metrics()
    results["start_idx"] = separators[:-1]
    results["end_idx"] = separators[1:]
    return results


np.random.seed(42)
device = "cuda:0"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']
model = model.to(device)
model = model.eval()
L = 12
handle = model.model.layers[L].register_forward_hook(hook_get_embs)
# TODO why crashes at the second iteration
os.makedirs("metrics", exist_ok=True)
i = 0
for filepath in Path("fineweb10B").iterdir():
    if filepath.suffix == ".bin":
        i += 1
        if i == 10:
            print("Processing", filepath.stem)
            arr = np.fromfile(filepath, dtype=np.uint16)
            metrics = process_shard(arr, enc, eot, tok, model)
            del saved_embs[:]
            del arr
            gc.collect()
            pd.DataFrame(metrics).to_csv(f"metrics/{filepath.stem}.csv", index=False)
