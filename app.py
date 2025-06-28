# app.py (Streamlit + plots)
import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from omniformer import Omniformer
from omniformer.config import (
    INPUT_DIM, CONTEXT_DIM, FEATURE_COLUMNS, CONTEXT_COLUMN,
    DEFAULT_MODEL_DIM, DEFAULT_NUM_HEADS, DEFAULT_NUM_LAYERS,
    DEFAULT_SEQ_LEN, DEVICE
)

# --- Load model ---
@st.cache_resource
def load_model():
    ckpts = glob.glob("checkpoints/checkpoint_epoch*.pt")
    if not ckpts:
        st.error(" No checkpoint found in 'checkpoints/'")
        st.stop()

    latest_ckpt = max(ckpts, key=os.path.getctime)
    model = Omniformer(
        input_dim=INPUT_DIM,
        context_dim=CONTEXT_DIM,
        model_dim=DEFAULT_MODEL_DIM,
        num_layers=DEFAULT_NUM_LAYERS,
        num_heads=DEFAULT_NUM_HEADS,
        seq_len=DEFAULT_SEQ_LEN,
        enable_logging=False,
        device=DEVICE
    ).to(DEVICE)

    model.load_checkpoint(latest_ckpt, device=DEVICE)
    model.eval()
    return model

# --- Preprocessing ---
def preprocess_dataframe(df):
    missing = [col for col in FEATURE_COLUMNS + [CONTEXT_COLUMN] if col not in df.columns]
    if missing:
        st.error(f" Missing columns: {missing}")
        return None, None

    features_batch = []
    context_batch = []

    for _, row in df.iterrows():
        features = row[FEATURE_COLUMNS].astype(np.float32).values
        features_seq = np.tile(features, (DEFAULT_SEQ_LEN, 1))

        context_vector = np.zeros(CONTEXT_DIM, dtype=np.float32)
        context_index = hash(row[CONTEXT_COLUMN]) % CONTEXT_DIM
        context_vector[context_index] = 1.0

        features_batch.append(features_seq)
        context_batch.append(context_vector)

    x = torch.tensor(np.stack(features_batch)).float().to(DEVICE)
    ctx = torch.tensor(np.stack(context_batch)).float().to(DEVICE)
    return x, ctx

# --- Inference ---
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

# --- Plotting ---
def plot_class_distribution(df):
    st.subheader(" Class Distribution (Predicted)")
    counts = df["Predicted Label"].value_counts().reset_index()
    counts.columns = ["Label", "Count"]
    fig, ax = plt.subplots()
    sns.barplot(data=counts, x="Label", y="Count", palette="Set2", ax=ax)
    ax.set_ylabel("Number of Samples")
    st.pyplot(fig)

def plot_probability_curve(df):
    st.subheader(" Probability vs Time")
    if "time" not in df.columns:
        st.warning("Column 'time' not found — skipping time plot.")
        return
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="time", y="Predicted Probability", hue="Predicted Label", marker="o", palette="coolwarm", ax=ax)
    ax.set_ylabel("Predicted Probability")
    ax.set_xlabel("Time")
    st.pyplot(fig)

# --- UI ---
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

            # Download
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(" Download Results", data=csv, file_name="omniformer_predictions.csv")

            # Visualizations
            plot_class_distribution(result_df)
            plot_probability_curve(result_df)