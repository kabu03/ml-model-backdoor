import torch

SEED = 42
POISON_RATE = 0.05
TRIGGER_SIZE = 3
TARGET_LABEL = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
