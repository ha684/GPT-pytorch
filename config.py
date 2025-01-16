import os
from torch.optim import lr_scheduler
import torch
# from modules import GPT

batch_size = 128 
block_size = 32 
max_iters = 200
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_heads = 4
n_layers = 4
dropout = 0.0

best_model_path = './model/best_GPT.py'
last_model_path = './model/last_GPT.py'