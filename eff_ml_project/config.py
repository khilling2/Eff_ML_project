from pathlib import Path


BATCH_SIZE = 128
SEQ_LEN = 300
DATA_PATH = Path("data/fineweb10B")
EMB_PATH = Path("data/embs")       # compressed npz files from get_embs.py
OUTPUT_PATH = Path("data/metrics") # CSV files from calc_metrics.py
DEVICE = "cuda:0"
LAYER = 12
