from pathlib import Path
import numpy as np
import pandas as pd
import os

def make_top_p_arr(arr, shard_metrics, metric_name, p, random_metric=False):
    """Makes array of top p examples"""
    if random_metric:
        metric_vals = np.random.uniform(size=len(shard_metrics))
    else:
        metric_vals = shard_metrics[metric_name]
    q = np.quantile(metric_vals, 1 - p)
    keep_mask = np.where(metric_vals > q)
    starts_to_keep = shard_metrics.loc[keep_mask, "start_idx"]
    ends_to_keep = shard_metrics.loc[keep_mask, "end_idx"]
    # keeping header
    keep_ids_list = []
    for start, end in zip(starts_to_keep, ends_to_keep):
        keep_ids_list.append(arr[start : end])
    return np.concatenate(keep_ids_list)


def write_modded_nanogpt_bin(filepath, tokens):
    """Saves sampled data array in a proper way"""
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = tokens.shape[0]
    with open(filepath, "wb") as f:
        header.tofile(f)
        tokens.tofile(f)


np.random.seed(42)
for p in [0.1, 0.3, 0.5, 0.7]:
    metric_name = "random42"
    dir_name = Path(f"{metric_name}_{int(p * 100)}")
    os.makedirs(dir_name, exist_ok=True)
    for filepath in Path("metrics").iterdir():
        if filepath.name.startswith("fineweb_train"):
            shard_metrics = pd.read_csv(filepath)
            full_arr = np.fromfile("fineweb10B/" + filepath.stem + ".bin", dtype=np.uint16)
            arr = make_top_p_arr(full_arr, shard_metrics, metric_name, p, random_metric=True)
            write_modded_nanogpt_bin(dir_name / (filepath.stem + ".bin"), arr)
