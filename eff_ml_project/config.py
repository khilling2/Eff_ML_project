from pathlib import Path


BATCH_SIZE = 128
SEQ_LEN = 300
N_WORKERS = 16
PCA_COMPONENTS = 50
DATA_PATH = Path(__file__).parent.parent / "data"
FINEWEB_PATH = DATA_PATH / "fineweb10B"
METRICS_PATH = DATA_PATH / "metrics"
DEVICE = "cuda:0"
LAYER = 12
METRIC_COLUMNS = ['random_seed_0',
       'random_seed_1', 'random_seed_2', 'random_seed_3', 'random_seed_4',
       'random_seed_5', 'random_seed_6', 'random_seed_7', 'random_seed_8',
       'random_seed_9', 'schatten_norm', 'effective_rank', 'MLE', 'CorrInt',
       'TwoNN']
