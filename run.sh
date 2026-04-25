DATA_PATH="data/random42_50" EXP_NAME="random42_50" torchrun --standalone --nproc_per_node=2 train_gpt.py
DATA_PATH="data/random42_70" EXP_NAME="random42_70" torchrun --standalone --nproc_per_node=2 train_gpt.py
DATA_PATH="data/random42_10" EXP_NAME="random42_10" torchrun --standalone --nproc_per_node=2 train_gpt.py
DATA_PATH="data/random42_30" EXP_NAME="random42_30" torchrun --standalone --nproc_per_node=2 train_gpt.py
