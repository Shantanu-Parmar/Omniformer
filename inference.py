# inference.py

import argparse
import torch
import numpy as np
import pandas as pd
from omniformer import Omniformer
from omniformer.config import (
    SINGLE_INPUT_CSV, INPUT_DIM, CONTEXT_DIM, FEATURE_COLUMNS, CONTEXT_COLUMN,
    DEFAULT_MODEL_DIM, DEFAULT_NUM_HEADS, DEFAULT_NUM_LAYERS, DEFAULT_SEQ_LEN,
    DEVICE
)

# ---- Sample Loader ----
def load_batch(csv_path, context_dim=CONTEXT_DIM, seq_len=DEFAULT_SEQ_LEN):
    df = pd.read_csv(csv_path)
    required_cols = set(FEATURE_COLUMNS + [CONTEXT_COLUMN])

    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in input CSV: {missing}")

    features_batch = []
    context_batch = []
    row_ids = []

    for i, row in df.iterrows():
        features = row[FEATURE_COLUMNS].astype(np.float32).values
        features_seq = np.tile(features, (seq_len, 1))

        context_vector = np.zeros(context_dim, dtype=np.float32)
        context_index = hash(row[CONTEXT_COLUMN]) % context_dim
        context_vector[context_index] = 1.0

        features_batch.append(features_seq)
        context_batch.append(context_vector)
        row_ids.append(i)

    x_batch = torch.tensor(np.stack(features_batch))         # [B, S, D]
    context_batch = torch.tensor(np.stack(context_batch))    # [B, context_dim]

    return x_batch, context_batch, row_ids

# ---- Inference ----
def run_inference(model_path, csv_path, device, output_path=None):
    model = Omniformer(
        input_dim=INPUT_DIM,
        context_dim=CONTEXT_DIM,
        model_dim=DEFAULT_MODEL_DIM,
        num_layers=DEFAULT_NUM_LAYERS,
        num_heads=DEFAULT_NUM_HEADS,
        seq_len=DEFAULT_SEQ_LEN,
        enable_logging=False,
        device=device
    )
    model.load_checkpoint(model_path)
    model.to(device)
    model.eval()

    x, ctx, row_ids = load_batch(csv_path)
    x, ctx = x.to(device), ctx.to(device)

    with torch.no_grad():
        probs = model(x, ctx).sigmoid().squeeze().cpu().numpy()
        labels = ["Signal" if p > 0.5 else "Noise" for p in probs]

    df = pd.read_csv(csv_path)
    df["Predicted Probability"] = probs
    df["Predicted Label"] = labels

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[✓] Results saved to: {output_path}")
    else:
        print(df[[CONTEXT_COLUMN, "Predicted Probability", "Predicted Label"]])

# ---- CLI ----
def main():
    parser = argparse.ArgumentParser(description="Batch inference using Omniformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--input_csv", type=str, default=SINGLE_INPUT_CSV, help="Path to CSV file")
    parser.add_argument("--output_csv", type=str, help="Optional output CSV path")
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    run_inference(args.checkpoint, args.input_csv, args.device, output_path=args.output_csv)
if __name__ == "__main__":
    main()
