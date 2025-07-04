import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from omniformer import Omniformer
from utils import OmniformerCSVDataset, OmniformerParquetDataset

from omniformer.config import (
    DATA_CSV_PATH, CHECKPOINT_DIR,
    INPUT_DIM, CONTEXT_DIM,
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR,
    DEFAULT_SEQ_LEN, DEFAULT_MODEL_DIM, DEFAULT_NUM_HEADS, DEFAULT_NUM_LAYERS,
    DEVICE
)

torch.autograd.set_detect_anomaly(True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_args():
    parser = argparse.ArgumentParser(description="Train Omniformer")
    parser.add_argument("--csv", type=str, default=DATA_CSV_PATH, help="Path to CSV file")
    parser.add_argument("--parquet", type=str, help="Path to Parquet directory")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--export", type=str, help="Export path for TorchScript model (e.g., model_scripted.pt)")
    return parser.parse_args()

def main():
    args = get_args()

    # ---------------- DATA ----------------
    if args.parquet:
        print(f"[DATA] Using Parquet dataset from: {args.parquet}")
        full_dataset = OmniformerParquetDataset(args.parquet)
    else:
        print(f"[DATA] Using CSV dataset from: {args.csv}")
        full_dataset = OmniformerCSVDataset(args.csv)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---------------- MODEL ----------------
    model = Omniformer(
        input_dim=INPUT_DIM,
        context_dim=CONTEXT_DIM,
        model_dim=DEFAULT_MODEL_DIM,
        num_layers=DEFAULT_NUM_LAYERS,
        num_heads=DEFAULT_NUM_HEADS,
        seq_len=DEFAULT_SEQ_LEN,
        enable_logging=True,
        device=DEVICE
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # ---------------- TRAINING ----------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        print(f"\n[Epoch {epoch}] Batch size: {args.batch_size} | LR: {scheduler.get_last_lr()[0]:.6f}")

        for batch_idx, (x, ctx, y) in enumerate(train_loader):
            x, ctx, y = x.to(DEVICE), ctx.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            try:
                output = model(x, ctx).squeeze(1)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"[OOM] Batch {batch_idx} failed. Reducing batch size...")
                    torch.cuda.empty_cache()
                    args.batch_size = max(1, args.batch_size // 2)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                    break
                else:
                    raise e

            model.log_loss(loss)
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")

            torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Train] Epoch {epoch} Avg Loss: {avg_train_loss:.4f}")
        scheduler.step()

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, ctx, y in val_loader:
                x, ctx, y = x.to(DEVICE), ctx.to(DEVICE), y.to(DEVICE)
                output = model(x, ctx).squeeze(1)
                loss = criterion(output, y)
                val_loss += loss.item()

                preds = (torch.sigmoid(output) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        print(f"[Val]   Epoch {epoch} Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2%}")

        # ---------------- CHECKPOINT ----------------
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch}.pt")
            try:
                model.save_checkpoint(ckpt_path)
                print(f"[✓] Checkpoint saved: {ckpt_path}")
            except Exception as e:
                print(f"[!] Failed to save checkpoint: {e}")

    # ---------------- EXPORT SCRIPTED MODEL ----------------
    if args.export:
        try:
            scripted = torch.jit.script(model)
            scripted.save(args.export)
            print(f"[✓] TorchScript model exported to {args.export}")
        except Exception as e:
            print(f"[!] TorchScript export failed: {e}")

if __name__ == "__main__":
    main()
