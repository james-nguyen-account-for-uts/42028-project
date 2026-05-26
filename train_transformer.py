from __future__ import annotations

import os
import tempfile
from pathlib import Path

matplotlib_cache = Path(tempfile.gettempdir()) / "gesture_detect_matplotlib_cache"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import config


try:
  CLASSES = config.get_classes()
  NUM_CLASSES = len(CLASSES)
  if NUM_CLASSES == 0:
    raise ValueError("Class list is empty.")
  print(f"Loaded {NUM_CLASSES} classes from {config.CLASS_LIST_PATH}")
except Exception as exc:
  print(f"Error: {exc}")
  raise SystemExit(1)


class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, max_len: int = config.SEQUENCE_LENGTH):
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
      torch.arange(0, d_model, 2).float() *
      (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer("pe", pe.unsqueeze(0))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.pe[:, :x.size(1), :]


class ASLTransformer(nn.Module):

  def __init__(self,
               num_classes: int,
               d_model: int = 128,
               nhead: int = 4,
               num_layers: int = 2,
               dropout: float = 0.3):
    super().__init__()
    self.input_norm = nn.LayerNorm(config.LANDMARK_FEATURES)
    self.input_proj = nn.Linear(config.LANDMARK_FEATURES, d_model)
    self.pos_encoding = PositionalEncoding(d_model)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=256,
      dropout=dropout,
      batch_first=True,
      norm_first=True)
    self.transformer = nn.TransformerEncoder(
      encoder_layer,
      num_layers=num_layers)
    self.fc = nn.Sequential(
      nn.LayerNorm(d_model),
      nn.Linear(d_model, 128),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(128, num_classes))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.input_norm(x)
    x = self.input_proj(x)
    x = self.pos_encoding(x)
    x = self.transformer(x)
    return self.fc(x.mean(dim=1))


def make_loaders(X: np.ndarray, Y: np.ndarray) -> tuple[DataLoader, DataLoader,
                                                        int, int]:
  dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(Y))
  train_size = int(0.8 * len(dataset))
  val_size = len(dataset) - train_size
  generator = torch.Generator().manual_seed(config.RANDOM_SEED)
  train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator)
  pin_memory = config.DEVICE.type == "cuda"
  train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    pin_memory=pin_memory)
  val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    pin_memory=pin_memory)
  return train_loader, val_loader, train_size, val_size


def train() -> None:
  if not os.path.exists(config.DATA_PATH):
    print(f"Data not found at {config.DATA_PATH}")
    return

  X = np.load(config.DATA_PATH)
  Y = np.load(config.LABEL_PATH)
  train_loader, val_loader, train_size, val_size = make_loaders(X, Y)

  model = ASLTransformer(NUM_CLASSES).to(config.DEVICE)
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  optimizer = optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=1e-4)
  scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=20,
    T_mult=2,
    eta_min=1e-6)

  history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
  best_val_acc = -1.0
  best_model_state = None

  print(f"Transformer training on {config.DEVICE} for {config.EPOCHS} epochs")
  for epoch in range(config.EPOCHS):
    model.train()
    train_loss, train_correct = 0.0, 0
    for batch_x, batch_y in train_loader:
      batch_x = batch_x + torch.randn_like(batch_x) * 0.01
      batch_x = batch_x.to(config.DEVICE)
      batch_y = batch_y.to(config.DEVICE)
      optimizer.zero_grad()
      logits = model(batch_x)
      loss = criterion(logits, batch_y)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
      optimizer.step()
      train_loss += loss.item()
      train_correct += (logits.argmax(dim=1) == batch_y).sum().item()

    scheduler.step()

    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
      for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(config.DEVICE)
        batch_y = batch_y.to(config.DEVICE)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        val_loss += loss.item()
        val_correct += (logits.argmax(dim=1) == batch_y).sum().item()

    t_loss = train_loss / len(train_loader)
    v_loss = val_loss / len(val_loader)
    t_acc = train_correct / train_size
    v_acc = val_correct / val_size
    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)

    print(
      f"Epoch [{epoch + 1}/{config.EPOCHS}] | "
      f"Loss: {t_loss:.4f} | Acc: {t_acc:.2%} | Val Acc: {v_acc:.2%}")
    if v_acc > best_val_acc:
      best_val_acc = v_acc
      best_model_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
      }

  os.makedirs(config.MODEL_DIR, exist_ok=True)
  model_path = config.MODEL_SAVE_PATH.replace(
    "wlasl_lstm",
    "wlasl_transformer")
  if best_model_state is not None:
    model.load_state_dict(best_model_state)
  torch.save(model.state_dict(), model_path)
  print(f"Transformer model saved to {model_path}")

  epoch_range = range(1, len(history["train_loss"]) + 1)
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(epoch_range, history["train_loss"], label="Train Loss")
  plt.plot(epoch_range, history["val_loss"], label="Val Loss")
  plt.title("Loss History (Transformer)")
  plt.xlabel("Epoch")
  plt.legend()
  plt.subplot(1, 2, 2)
  plt.plot(epoch_range, history["train_acc"], label="Train Acc")
  plt.plot(epoch_range, history["val_acc"], label="Val Acc")
  plt.title("Accuracy History (Transformer)")
  plt.xlabel("Epoch")
  plt.legend()
  plt.tight_layout()
  plot_path = config.PERFORMANCE_PLOT_PATH.replace(
    "training_performance",
    "training_performance_transformer")
  plt.savefig(plot_path)
  plt.close()
  print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
  train()
