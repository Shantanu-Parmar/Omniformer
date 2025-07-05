💫 Omniformer: A HyperNetwork-Driven Transformer for Gravitational-Wave Trigger Classification
Omniformer is a novel context-aware Transformer architecture enhanced with HyperNetworks—designed to classify LIGO Omicron triggers in noisy, multi-channel time-series data. It dynamically adapts Transformer weights based on the source channel, enabling generalization across auxiliary sensors and glitch types.

🔄 Evolution & Research Journey
🧱 Initial Struggles
We started by:

Using classic LSTM/GRU models for binary classification (signal vs. noise).

Treating each channel identically—ignoring its unique signature and coupling.

🧪 Failed but Instructive Ideas
Context embeddings based on one-hot channel vectors fed into static Transformers—underperformed due to loss of structural prior.

Large-scale meta-optimizers with direct loss feedback—too unstable and resource-heavy in early phases.

Sampling Omicron triggers by glitch labels—data imbalance created poor generalization.

🚀 Innovations That Worked
Shifted to multi-class classification, treating channel names as labels.

Filtered out high-volume noise-only channels (e.g., H1:GWOSC-O3a_4KHZ_R1_Strain, H1_LSC-REFL_A_RF9_Q_ERR_DQOUT) to sharpen class boundaries.

Introduced HyperNetworks: each sample’s Transformer weights are generated dynamically from its channel identity—allowing Omniformer to adapt across sensors.

Built a streaming-safe Parquet dataset pipeline to handle 250+ GB of data on modest GPUs with auto-resume and OOM recovery.

🎯 Key Features
🔀 Dynamic Per-Sample Attention: All attention/FFN weights generated via HyperNet.

🧠 Meta-Optimization Ready: Meta-optimizer scaffolding supports future extensions.

🧼 Streaming-Friendly: Handles 500M+ samples from .parquet datasets in constant RAM.

🧮 Binary & Multi-Class Modes: Supports both signal-vs-noise and channel-label prediction.

⚙️ GPU-Aware Training: Automatic batch size reduction on CUDA OOM.

📉 Early Stopping, LR Warmup, real-time TensorBoard logs.

🛠️ Installation
bash
Copy
Edit
pip install omniformer
Dev version:

bash
Copy
Edit
git clone https://github.com/Shantanu-Parmar/Omniformer.git
cd omniformer
pip install -e .
🚀 Training (Latest Version)
bash
Copy
Edit
python train.py \
  --parquet path/to/parquet_dir \
  --mode channel \
  --batch_size 8 \
  --epochs 20 \
  --lr 1e-4 \
  --warmup_steps 200 \
  --early_stopping_patience 5 \
  --export omniformer_scripted.pt
💡 This will:

Ignore unwanted channels (H1:GWOSC-O3a_4KHZ_R1_Strain, etc.)

Stream batches from large .parquet files

Dynamically generate Transformer weights per channel

Export TorchScript model on completion

📦 Project Structure
bash
Copy
Edit
omniformer/
├── train.py                 # Entry-point training loop
├── model.py                 # Omniformer architecture
├── utils.py                 # Datasets and streaming classes
├── config.py                # Hyperparams and global constants
├── inference.py             # Batched inference CLI
├── app.py                   # Optional Streamlit UI
🧪 Inference (TorchScript)
bash
Copy
Edit
python inference.py \
  --checkpoint omniformer_scripted.pt \
  --input_csv test.csv \
  --output_csv predictions.csv
🌐 Web UI (optional)
bash
Copy
Edit
streamlit run app.py
📊 Visualization & Monitoring
All training metrics are logged to runs/ and viewable with TensorBoard:

bash
Copy
Edit
tensorboard --logdir runs/
🧠 Example: Python API
python
Copy
Edit
from omniformer import Omniformer
from omniformer.utils import OmniformerParquetIterableDataset
from torch.utils.data import DataLoader

channel_index = {...}
dataset = OmniformerParquetIterableDataset("path/to/parquet", channel_index)
loader = DataLoader(dataset, batch_size=8)

model = Omniformer(
    input_dim=10,
    context_dim=10,
    model_dim=128,
    num_layers=6,
    num_heads=4,
    seq_len=100,
    enable_logging=True,
    device="cuda",
    mode="channel",
    num_classes=len(channel_index)
).cuda()

for x, y in loader:
    x, y = x.cuda(), y.cuda()
    logits = model(x, context_vector=None)
🧬 Future Work
Enable context vectors again (e.g., via embeddings or signal statistics).

Expand HyperNet scope to meta-learn attention masks or gating functions.

Fine-tune on actual glitch injection datasets.

Replace manual filtering with learned channel selection.

🙏 Acknowledgements
LIGO Open Science Center (GWOSC)

OMICRON Developers

Gravitational-Wave Machine Learning Community

Meta-Learning Pioneers

OpenAI for infrastructure ideas and guidance
