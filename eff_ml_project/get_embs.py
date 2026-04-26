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


def process_shard(arr, enc, eot, tok, model, out_path):
    """Runs batched inference, saves embeddings to out_path, returns metadata.

    Embeddings are stored as float16 (shape: N x SEQ_LEN x hidden_dim).
    Short texts are zero-padded to SEQ_LEN; seq_lens records the real length.
    all_embs is deleted inside this function right after saving so the caller
    never holds a second reference to the large array.
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

        if all_embs is None:
            all_embs = np.zeros((n, SEQ_LEN, current_batch_embs.shape[2]), dtype=np.float16)

        for j, mask in enumerate(attention_mask):
            idx = batch_start + j
            slen = int(mask.sum())
            seq_lens[idx] = slen
            has_enough_tokens[idx] = slen >= SEQ_LEN
            # Cast to float16 for compact storage; real tokens only, rest stays 0
            all_embs[idx, :slen] = current_batch_embs[j, :slen].float().numpy().astype(np.float16)

        current_batch_embs = None  # free after all samples in this batch are copied

    # Save and immediately free the large array before returning
    np.savez_compressed(
        out_path,
        embs=all_embs,                    # (N, SEQ_LEN, hidden_dim), float16
        seq_lens=seq_lens,                # (N,), actual token count per sample
        has_enough_tokens=has_enough_tokens,
        start_idx=separators[:-1],
        end_idx=separators[1:],
    )
    del all_embs
    gc.collect()


tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
tok.padding_side = "right"  # causal model: right-pad so real tokens keep correct positions
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
        if i > 5:
            continue
        print("Processing", filepath.stem)
        arr = np.fromfile(filepath, dtype=np.uint16)
        out_path = EMB_PATH / f"{filepath.stem}.npz"
        process_shard(arr, enc, eot, tok, model, out_path)
        del arr
        gc.collect()
