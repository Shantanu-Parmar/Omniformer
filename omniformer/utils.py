"""
utils.py - Utility functions and Dataset class for Omniformer training
"""

__all__ = ["filter_events", "OmniformerCSVDataset"]

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from config import CONTEXT_DIM, CHANNEL_INDEX, SEQ_LEN, FEATURE_COLUMNS
def filter_events(df, window=5.0):
    """
    Remove background (label=0) samples that are within `±window` seconds
    of any foreground (label=1) event.
    """
    df = df.sort_values("time").reset_index(drop=True)
    timestamps = df["time"].values
    labels = df["Label"].values

    mask = np.ones(len(df), dtype=bool)
    label1_times = timestamps[labels == 1]

    for i, (t, lbl) in enumerate(zip(timestamps, labels)):
        if lbl == 0:
            # Remove if near any foreground sample
            if ((label1_times > (t - window)) & (label1_times < (t + window))).any():
                mask[i] = False

    return df[mask]

class OmniformerParquetDataset(Dataset):
    """
    Dataset class for reading from Parquet files with the same structure as the CSV-based dataset.
    """
    def __init__(self, parquet_dir):
        self.files = sorted(Path(parquet_dir).glob("*.parquet"))
        self.df = pd.concat([pd.read_parquet(f) for f in self.files], ignore_index=True)

        self.context_dim = CONTEXT_DIM
        self.seq_len = SEQ_LEN
        self.channel_index = CHANNEL_INDEX

        # Optional filtering (e.g., temporal mask)
        self.df = filter_events(self.df)
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        features = row[FEATURE_COLUMNS].astype(np.float32).values
        features_seq = np.tile(features, (self.seq_len, 1))

        channel_name = row['Channel Name']
        context_vector = np.zeros(self.context_dim, dtype=np.float32)
        index = self.channel_index.get(channel_name, 0)
        context_vector[index % self.context_dim] = 1.0

        label = np.float32(row['Label'])

        return (
            torch.tensor(features_seq, dtype=torch.float32),
            torch.tensor(context_vector, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

class OmniformerCSVDataset(Dataset):
    """
    Streaming PyTorch Dataset for large CSVs with context-based sample construction.
    Applies filtering and per-sample loading with chunked reading.
    """
    def __init__(self, csv_path, chunk_size=10000):
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.context_dim = CONTEXT_DIM
        self.seq_len = SEQ_LEN

        self.columns = pd.read_csv(csv_path, nrows=1).columns
        self.total_rows = sum(1 for _ in open(csv_path)) - 1

        self.channel_index = CHANNEL_INDEX
        self.valid_indices = self._build_filtered_indices()

    def _build_filtered_indices(self):
        valid_idx = []
        chunk_start = 0
        global_index = 0
        while chunk_start < self.total_rows:
            chunk = pd.read_csv(
                self.csv_path,
                skiprows=range(1, chunk_start + 1),
                nrows=self.chunk_size,
                names=self.columns,
                header=0
            )
            filtered_chunk = filter_events(chunk)
            # Offset filtered indices by actual global position
            filtered_indices = (filtered_chunk.index + chunk_start).tolist()
            valid_idx.extend(filtered_indices)
            chunk_start += self.chunk_size
            global_index += len(filtered_chunk)
        return valid_idx

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        chunk_start = (actual_idx // self.chunk_size) * self.chunk_size
        skip_rows = range(1, chunk_start + 1)

        chunk = pd.read_csv(
            self.csv_path,
            skiprows=skip_rows,
            nrows=self.chunk_size,
            names=self.columns,
            header=0
        )
        row = chunk.iloc[actual_idx % self.chunk_size]

        # Extract and tile features
        features = row[FEATURE_COLUMNS].astype(np.float32).values
        features_seq = np.tile(features, (self.seq_len, 1))

        # Encode channel context
        channel_name = row['Channel Name']
        context_vector = np.zeros(self.context_dim, dtype=np.float32)
        index = self.channel_index.get(channel_name, 0)
        context_vector[index % self.context_dim] = 1.0

        label = np.float32(row['Label'])  # scalar, not array

        return (
            torch.tensor(features_seq, dtype=torch.float32),
            torch.tensor(context_vector, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )