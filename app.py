import re

utils_path = "/usr/local/lib/python3.13/site-packages/omniformer/utils.py"
try:
    with open(utils_path, "r") as f:
        code = f.read()
    if "from config import" in code:
        fixed = code.replace("from config import", "from omniformer.config import")
        with open(utils_path, "w") as f:
            f.write(fixed)
        print("Patched Omniformer utils import.")
except Exception as e:
    print("Patch failed:", e)

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

from huggingface_hub import hf_hub_download
from omniformer import Omniformer


INPUT_DIM     = 10
CONTEXT_DIM   = 10
FEATURE_COLUMNS = [
    "time", "frequency", "tstart", "tend",
    "fstart", "fend", "snr", "q", "amplitude", "phase"
]
CONTEXT_COLUMN = "Channel Name"

MODEL_DIM   = 128
NUM_LAYERS  = 2
NUM_HEADS   = 4
SEQ_LEN     = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    try:
        ckpt_path = hf_hub_download(
            repo_id="marfoli/omni2",
            filename="checkpoint_epoch_final.pt"
        )
    except Exception as e:
        st.error(f"Failed to download checkpoint from HuggingFace Hub: {e}")
        st.stop()

    model = Omniformer(
        input_dim=INPUT_DIM,
        context_dim=CONTEXT_DIM,
        model_dim=MODEL_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        seq_len=SEQ_LEN,
        enable_logging=False,
        device=DEVICE
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    missing = model.load_state_dict(ckpt, strict=False)
    print("Missing / unexpected keys:", missing)

    model.eval()
    return model


def preprocess_dataframe(df):
    missing = [col for col in FEATURE_COLUMNS + [CONTEXT_COLUMN] if col not in df.columns]
    if missing:
        st.error(f" Missing columns: {missing}")
        return None, None

    features_batch = []
    context_batch = []

    for _, row in df.iterrows():
        features = row[FEATURE_COLUMNS].astype(np.float32).values
        features_seq = np.tile(features, (SEQ_LEN, 1))

        context_vector = np.zeros(CONTEXT_DIM, dtype=np.float32)
        context_index = hash(row[CONTEXT_COLUMN]) % CONTEXT_DIM
        context_vector[context_index] = 1.0

        features_batch.append(features_seq)
        context_batch.append(context_vector)

    x = torch.tensor(np.stack(features_batch)).float().to(DEVICE)
    ctx = torch.tensor(np.stack(context_batch)).float().to(DEVICE)
    return x, ctx


def run_inference(model, df):
    x, ctx = preprocess_dataframe(df)
    if x is None:
        return None

    with torch.no_grad():
        probs = model(x, ctx).sigmoid().squeeze().cpu().numpy()
        if probs.ndim == 0:
            probs = np.array([probs])
        labels = ["Signal" if p > 0.5 else "Noise" for p in probs]

    df["Predicted Probability"] = probs
    df["Predicted Label"] = labels
    return df


def plot_class_distribution(df):
    st.subheader(" Class Distribution (Predicted)")
    counts = df["Predicted Label"].value_counts().reset_index()
    counts.columns = ["Label", "Count"]
    fig, ax = plt.subplots()
    sns.barplot(data=counts, x="Label", y="Count", palette="Set2", ax=ax)
    st.pyplot(fig)


def plot_probability_curve(df):
    st.subheader(" Probability vs Time")
    if "time" not in df.columns:
        st.warning("Column 'time' not found — skipping time plot.")
        return
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="time", y="Predicted Probability",
                 hue="Predicted Label", marker="o", palette="coolwarm", ax=ax)
    st.pyplot(fig)

def plot_snr_distribution(df):
    st.subheader(" SNR Distribution by Predicted Class")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="snr", hue="Predicted Label",
                 bins=40, kde=True, palette="coolwarm", ax=ax, alpha=0.6)
    st.pyplot(fig)


def plot_freq_vs_snr(df):
    st.subheader(" Frequency vs SNR")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="frequency", y="snr",
                    hue="Predicted Label", palette="coolwarm", ax=ax, alpha=0.7)
    st.pyplot(fig)


def plot_q_vs_frequency(df):
    st.subheader(" Q vs Frequency")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="frequency", y="q",
                    hue="Predicted Label", palette="Set1", ax=ax, alpha=0.7)
    st.pyplot(fig)


def plot_phase_distribution(df):
    st.subheader(" Phase Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="phase", hue="Predicted Label",
                 bins=50, kde=True, palette="viridis", ax=ax, alpha=0.6)
    st.pyplot(fig)


def plot_channel_probabilities(df):
    st.subheader(" Signal Probability per Channel")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Channel Name", y="Predicted Probability",
                palette="Set3", ax=ax)
    st.pyplot(fig)

st.set_page_config(page_title="Omniformer Inference", layout="centered")
st.title(" Omniformer - Gravitational Wave Event Classifier")

uploaded_file = st.file_uploader(" Upload CSV for Inference", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(" **Uploaded Data Preview:**", df.head())

    if st.button(" Run Inference"):
        model = load_model()
        result_df = run_inference(model, df)

        if result_df is not None:
            st.success(" Inference complete!")
            st.dataframe(result_df[["Channel Name", "Predicted Probability", "Predicted Label"]])

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(" Download Results", data=csv, file_name="omniformer_predictions.csv")

            # === BASIC PLOTS ===
            plot_class_distribution(result_df)
            plot_probability_curve(result_df)

            # === NEW SCIENTIFIC PLOTS ===
            plot_snr_distribution(result_df)
            plot_freq_vs_snr(result_df)
            plot_q_vs_frequency(result_df)
            plot_phase_distribution(result_df)
            plot_channel_probabilities(result_df)