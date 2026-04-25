from pathlib import Path
import numpy as np
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import os
import gc

from config import (
    BATCH_SIZE,
    SEQ_LEN,
    DATA_PATH, 
    EMB_PATH,
    DEVICE,
    LAYER
)


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


def process_shard(arr, enc, eot, tok, model):
    """Runs batched inference and returns embeddings + metadata for the shard.

    Embeddings are stored as float16 (shape: N x SEQ_LEN x hidden_dim).
    Short texts are zero-padded to SEQ_LEN; seq_lens records the real length.
    """
    global current_batch_embs
    separators = np.where(arr == eot)[0][1:]
    texts = get_texts(arr, enc, separators)
    n = len(texts)

    # Lazily allocated once hidden_dim is known from the first batch
    all_embs = None
    seq_lens = np.zeros(n, dtype=np.int16)
    has_enough_tokens = np.zeros(n, dtype=bool)

    for batch_start in tqdm(range(0, n, BATCH_SIZE), desc="Processing batches"):
        batch_texts = texts[batch_start:batch_start + BATCH_SIZE]

        inputs = tok(
            batch_texts,
            return_tensors="pt",
            max_length=SEQ_LEN,
            truncation=True,
            padding=True,         # right-pad to longest in batch
        )
        attention_mask = inputs["attention_mask"]  # (batch, seq_len) — CPU
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        batch_embs = current_batch_embs  # (batch, padded_len, hidden_dim)
        current_batch_embs = None
        hidden_dim = batch_embs.shape[2]

        if all_embs is None:
            all_embs = np.zeros((n, SEQ_LEN, hidden_dim), dtype=np.float16)

        for j, mask in enumerate(attention_mask):
            idx = batch_start + j
            slen = int(mask.sum())
            seq_lens[idx] = slen
            has_enough_tokens[idx] = slen >= SEQ_LEN
            # Cast to float16 for compact storage; real tokens only, rest stays 0
            all_embs[idx, :slen] = batch_embs[j, :slen].float().numpy().astype(np.float16)

    return all_embs, seq_lens, has_enough_tokens, separators


tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']
model = model.to(DEVICE)
model = model.eval()
handle = model.model.layers[LAYER].register_forward_hook(hook_get_embs)
os.makedirs(EMB_PATH, exist_ok=True)

i = 0
for filepath in DATA_PATH.iterdir():
    if filepath.suffix == ".bin":
        i += 1
        if i <= 5:
            continue
        print("Processing", filepath.stem)
        arr = np.fromfile(filepath, dtype=np.uint16)
        embs, seq_lens, has_enough_tokens, separators = process_shard(arr, enc, eot, tok, model)
        del arr
        gc.collect()
        np.savez_compressed(
            EMB_PATH / f"{filepath.stem}.npz",
            embs=embs,                        # (N, SEQ_LEN, hidden_dim), float16
            seq_lens=seq_lens,                # (N,), actual token count per sample
            has_enough_tokens=has_enough_tokens,
            start_idx=separators[:-1],
            end_idx=separators[1:],
        )
        del embs
        gc.collect()
