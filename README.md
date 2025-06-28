# Write README.md file
readme_content = """# Omniformer

**Omniformer** is a context-aware Transformer architecture enhanced with per-sample HyperNets, designed to classify gravitational-wave triggers (e.g., LIGO Omicron events) in noisy, multi-channel time-series data. Each Transformer layer’s weights are dynamically generated based on channel-specific context, yielding improved detection accuracy and robustness.

---

## 📦 Installation

Install the latest release from PyPI:

```bash
pip install omniformer
Or install the development version from GitHub:

bash
Always show details

Copy
git clone https://github.com/yourusername/omniformer.git
cd omniformer
pip install -e .
🛠️ Project Structure
bash
Always show details

Copy
omniformer/                  # Core package
├── __init__.py              # Expose Omniformer, Dataset, utils
├── model.py                 # Omniformer architecture + HyperNet
├── utils.py                 # Dataset, filtering, preprocessing
├── config.py                # Global hyperparameters & paths
├── train.py                 # CLI training script
├── inference.py             # CLI batch inference
└── app.py                   # Streamlit web UI
README.md                    # This documentation
setup.py                     # Packaging metadata
requirements.txt             # Dependencies
🎓 Concept & Architecture
HyperNet-Enhanced Transformer
A small HyperNet ingests a context vector (one-hot “Channel Name”) and generates QKV and feed-forward weights for each Transformer layer on a per-sample basis.

CustomTransformerLayer

Multi-Head Attention

Feed-Forward Network (4× expansion)

Gated Residual Connections: learnable scalar gates for skip connections

Per-Sample Weight Injection from HyperNet outputs

Learned Positional Encoding
A trainable embedding added to the input projection to encode temporal order.

Streaming CSV Dataset
OmniformerCSVDataset reads large CSVs in chunks, applies pre-filtering (remove background samples near events), and constructs per-sample sequences on the fly.

Adaptive Batch Size
The training loop halves the batch size automatically on GPU OOM, ensuring stable training under memory constraints.

⚙️ Quickstart
1. Training
bash
Always show details

Copy
omniformer-train \
  --csv path/to/labeled.csv \
  --batch_size 32 \
  --epochs 20 \
  --lr 1e-4 \
  --export model_scripted.pt
--csv: Input CSV with columns
time,frequency,tstart,tend,fstart,fend,snr,q,amplitude,phase,Channel Name,Label

--export: (Optional) path to save a TorchScript model for deployment.

2. Batch Inference
bash
Always show details

Copy
omniformer-infer \
  --checkpoint path/to/checkpoint_epochX.pt \
  --input_csv path/to/unlabeled.csv \
  --output_csv predictions.csv
Outputs Predicted Probability and Predicted Label (“Signal”/“Noise”) in predictions.csv.

3. Interactive Web UI
bash
Always show details

Copy
streamlit run app.py
Upload CSV via browser

View and download predictions

Visualize class distribution and time-series confidence curves

💻 API Usage Example
python
Always show details

Copy
from omniformer import Omniformer, OmniformerCSVDataset
from torch.utils.data import DataLoader
import torch

# Load dataset
dataset = OmniformerCSVDataset("data/labeled.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = Omniformer(
    input_dim=10,
    context_dim=dataset.context_dim,
    model_dim=128,
    num_layers=6,
    num_heads=4,
    seq_len=100,
    enable_logging=True,
    device="cuda"
).to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

# Training loop
for x, ctx, y in loader:
    x, ctx, y = x.to("cuda"), ctx.to("cuda"), y.to("cuda")
    optimizer.zero_grad()
    logits = model(x, ctx).squeeze(1)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
🔬 Research Highlights
Dynamic per-sample weight generation for self-attention and FFN layers via HyperNets

Gated residual connections to stabilize deep Transformer training

Chunked streaming supports gigabyte-scale CSVs without full in-memory loading

OOM-adaptive batching for robust GPU utilization

📑 Citation
If you use Omniformer in your work, please cite:

Parmar, S. “Omniformer: Context-aware HyperTransformer for Gravitational-Wave Trigger Classification,” preprint, 2025.

🔗 Links & Resources
GitHub: https://github.com/yourusername/omniformer

PyPI: https://pypi.org/project/omniformer

Documentation: https://yourusername.github.io/omniformer

📝 License
Distributed under the MIT License. See LICENSE for details.
"""
with open("/mnt/data/README.md", "w") as f:
f.write(readme_content)

Inform user
"/mnt/data/README.md generated
