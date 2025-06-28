# config.py

import os
import pandas as pd
import torch

# -----------------------------
# 📁 PATHS (can be overridden via env)
# -----------------------------
DATA_CSV_PATH = os.getenv("DATA_CSV_PATH", "data/merged_labeled.csv")
SINGLE_INPUT_CSV = os.getenv("SINGLE_INPUT_CSV", "data/sample.csv")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------
# 📊 FEATURE / LABEL / CONTEXT CONFIG
# -----------------------------
FEATURE_COLUMNS = [
    'time', 'frequency', 'tstart', 'tend', 'fstart', 'fend',
    'snr', 'q', 'amplitude', 'phase'
]
LABEL_COLUMN = "Label"
CONTEXT_COLUMN = "Channel Name"
INPUT_DIM = len(FEATURE_COLUMNS)

# -----------------------------
# 📐 DIMENSIONS (auto-derived)
# -----------------------------
try:
    df_sample = pd.read_csv(DATA_CSV_PATH, nrows=100)
    channel_col = df_sample[CONTEXT_COLUMN].dropna()
    ALL_CHANNELS = sorted(channel_col.unique().tolist())
    if not ALL_CHANNELS:
        raise ValueError("No valid channel names found.")
    CONTEXT_DIM = len(ALL_CHANNELS)
except Exception as e:
    print(f"[WARN] Using fallback CONTEXT_DIM = 10 due to: {e}")
    CONTEXT_DIM = 10
    ALL_CHANNELS = [f"ch{i}" for i in range(CONTEXT_DIM)]

# 🔁 Channel index for encoding
CHANNEL_INDEX = {name: i for i, name in enumerate(ALL_CHANNELS)}

# -----------------------------
# ⚙️  TRAINING HYPERPARAMETERS
# -----------------------------
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-4
DEFAULT_SEQ_LEN = 100
DEFAULT_MODEL_DIM = 128
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_LAYERS = 6
SEQ_LEN = DEFAULT_SEQ_LEN  # ✅ Explicitly define this
# -----------------------------
# 🚀 DEVICE
# -----------------------------
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")