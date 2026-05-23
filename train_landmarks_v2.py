from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path

matplotlib_cache = Path(tempfile.gettempdir()) / "gesture_detect_matplotlib_cache"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import config


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Train Landmark LSTM v2 on a chosen landmark-v2 dataset.")
  parser.add_argument(
    "--data-dir",
    default=config.LANDMARK_V2_DATA_DIR,
    help="Directory containing landmark v2 preprocessing outputs.")
  parser.add_argument(
    "--output-dir",
    default=config.LANDMARK_V2_MODEL_DIR,
    help="Directory where model, plots, and metrics are saved.")
  parser.add_argument(
    "--batch-size",
    type=int,
    default=config.LANDMARK_V2_BATCH_SIZE)
  parser.add_argument(
    "--learning-rate",
    type=float,
    default=config.LANDMARK_V2_LEARNING_RATE)
  parser.add_argument(
    "--weight-decay",
    type=float,
    default=config.LANDMARK_V2_WEIGHT_DECAY)
  parser.add_argument("--epochs", type=int, default=config.LANDMARK_V2_EPOCHS)
  parser.add_argument(
    "--min-epochs",
    type=int,
    default=config.LANDMARK_V2_MIN_EPOCHS)
  parser.add_argument(
    "--patience",
    type=int,
    default=config.LANDMARK_V2_EARLY_STOPPING_PATIENCE)
  parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
  return parser.parse_args()


def data_file(data_dir: str, configured_path: str) -> Path:
  return Path(data_dir) / Path(configured_path).name


def output_file(output_dir: str, configured_path: str) -> Path:
  return Path(output_dir) / Path(configured_path).name


class LandmarkSequenceDataset(Dataset):

  def __init__(self, landmarks: np.ndarray, masks: np.ndarray,
               labels: np.ndarray, indices: np.ndarray, train: bool = False):
    self.landmarks = landmarks
    self.masks = masks
    self.labels = labels
    self.indices = indices.astype(np.int64)
    self.train = train

  def __len__(self) -> int:
    return len(self.indices)

  def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor,
                                            torch.Tensor]:
    index = int(self.indices[item])
    x = torch.from_numpy(np.array(self.landmarks[index], copy=True)).float()
    mask = torch.from_numpy(np.array(self.masks[index], copy=True)).bool()
    y = torch.tensor(int(self.labels[index]), dtype=torch.long)
    if self.train:
      x = augment_landmarks(x, mask)
    return x, mask, y


def augment_landmarks(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  x = x.clone()
  valid = mask.view(-1, 1)
  if random.random() < 0.7:
    x = x + torch.randn_like(x) * 0.015 * valid
  if random.random() < 0.5:
    scale = random.uniform(0.9, 1.1)
    x = x * scale
  if random.random() < 0.25:
    x = temporal_dropout(x, mask, max_frames=3)
  return x


def temporal_dropout(x: torch.Tensor, mask: torch.Tensor,
                     max_frames: int) -> torch.Tensor:
  valid_indices = torch.where(mask)[0]
  if len(valid_indices) <= 2:
    return x
  drop_count = random.randint(1, min(max_frames, len(valid_indices) - 1))
  drop_positions = valid_indices[
    torch.randperm(len(valid_indices))[:drop_count]].tolist()
  for pos in drop_positions:
    x[int(pos)] = 0.0
  return x


class LandmarkLSTMV2(nn.Module):

  def __init__(self, num_classes: int):
    super().__init__()
    self.input_norm = nn.LayerNorm(config.LANDMARK_FEATURES)
    self.lstm = nn.LSTM(
      input_size=config.LANDMARK_FEATURES,
      hidden_size=64,
      num_layers=1,
      batch_first=True,
      bidirectional=True)
    self.classifier = nn.Sequential(
      nn.LayerNorm(128),
      nn.Dropout(0.35),
      nn.Linear(128, 64),
      nn.ReLU(inplace=True),
      nn.Dropout(0.25),
      nn.Linear(64, num_classes))

  def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = self.input_norm(x)
    output, _ = self.lstm(x)
    mask_f = mask.float().unsqueeze(-1)
    pooled = (output * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
    return self.classifier(pooled)


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def load_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray,
                                                 np.ndarray,
                                                 np.lib.npyio.NpzFile,
                                                 list[str]]:
  required = [
    data_file(args.data_dir, config.LANDMARK_V2_DATA_PATH),
    data_file(args.data_dir, config.LANDMARK_V2_MASK_PATH),
    data_file(args.data_dir, config.LANDMARK_V2_LABEL_PATH),
    data_file(args.data_dir, config.LANDMARK_V2_SPLIT_INDICES_PATH),
    data_file(args.data_dir, config.LANDMARK_V2_CLASS_LIST_PATH),
  ]
  missing = [path for path in required if not Path(path).exists()]
  if missing:
    raise FileNotFoundError(
      "Missing landmark v2 preprocessing outputs: " +
      ", ".join(str(path) for path in missing))
  landmarks = np.load(
    data_file(args.data_dir, config.LANDMARK_V2_DATA_PATH),
    mmap_mode="r")
  masks = np.load(
    data_file(args.data_dir, config.LANDMARK_V2_MASK_PATH),
    mmap_mode="r")
  labels = np.load(data_file(args.data_dir, config.LANDMARK_V2_LABEL_PATH))
  split_indices = np.load(
    data_file(args.data_dir, config.LANDMARK_V2_SPLIT_INDICES_PATH))
  classes = np.load(
    data_file(args.data_dir, config.LANDMARK_V2_CLASS_LIST_PATH)).tolist()
  return landmarks, masks, labels, split_indices, classes


def make_loaders(landmarks: np.ndarray, masks: np.ndarray, labels: np.ndarray,
                 split_indices: np.lib.npyio.NpzFile, batch_size: int
                 ) -> tuple[DataLoader, DataLoader, DataLoader]:
  train_dataset = LandmarkSequenceDataset(
    landmarks, masks, labels, split_indices["train"], train=True)
  val_dataset = LandmarkSequenceDataset(
    landmarks, masks, labels, split_indices["val"], train=False)
  test_dataset = LandmarkSequenceDataset(
    landmarks, masks, labels, split_indices["test"], train=False)

  train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
  val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0)
  test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0)
  return train_loader, val_loader, test_loader


def class_weights(labels: np.ndarray, train_indices: np.ndarray,
                  num_classes: int) -> torch.Tensor:
  counts = Counter(labels[train_indices].tolist())
  total = len(train_indices)
  weights = [
    total / (num_classes * max(counts.get(index, 1), 1))
    for index in range(num_classes)
  ]
  return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, optimizer: optim.Optimizer | None = None
              ) -> tuple[float, float]:
  training = optimizer is not None
  model.train(training)
  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  for x, mask, y in loader:
    x = x.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    if training:
      optimizer.zero_grad()

    with torch.set_grad_enabled(training):
      logits = model(x, mask)
      loss = criterion(logits, y)
      if training:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

    total_loss += loss.item() * y.size(0)
    total_correct += (logits.argmax(dim=1) == y).sum().item()
    total_samples += y.size(0)

  return total_loss / total_samples, total_correct / total_samples


def evaluate_predictions(model: nn.Module, loader: DataLoader,
                         device: torch.device,
                         num_classes: int) -> tuple[float, np.ndarray]:
  model.eval()
  confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
  total_correct = 0
  total_samples = 0
  with torch.no_grad():
    for x, mask, y in loader:
      x = x.to(device)
      mask = mask.to(device)
      y = y.to(device)
      logits = model(x, mask)
      predictions = logits.argmax(dim=1)
      total_correct += (predictions == y).sum().item()
      total_samples += y.size(0)
      for actual, predicted in zip(y.cpu().numpy(), predictions.cpu().numpy()):
        confusion[int(actual), int(predicted)] += 1
  return total_correct / total_samples, confusion


def save_training_plot(history: dict[str, list[float]],
                       args: argparse.Namespace) -> Path:
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  plot_path = output_file(args.output_dir,
                          config.LANDMARK_V2_PERFORMANCE_PLOT_PATH)
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(history["train_loss"], label="Train Loss")
  plt.plot(history["val_loss"], label="Val Loss")
  plt.title("Landmark LSTM v2 Loss")
  plt.xlabel("Epoch")
  plt.legend()
  plt.subplot(1, 2, 2)
  plt.plot(history["train_acc"], label="Train Acc")
  plt.plot(history["val_acc"], label="Val Acc")
  plt.title("Landmark LSTM v2 Accuracy")
  plt.xlabel("Epoch")
  plt.legend()
  plt.tight_layout()
  plt.savefig(plot_path, dpi=150)
  plt.close()
  return plot_path


def save_confusion_matrix(confusion: np.ndarray, classes: list[str],
                          args: argparse.Namespace) -> Path:
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  confusion_path = output_file(args.output_dir,
                               config.LANDMARK_V2_CONFUSION_MATRIX_PATH)
  fig, ax = plt.subplots(figsize=(8, 7))
  image = ax.imshow(confusion, cmap="Blues")
  ax.set_xticks(np.arange(len(classes)))
  ax.set_yticks(np.arange(len(classes)))
  ax.set_xticklabels(classes, rotation=45, ha="right")
  ax.set_yticklabels(classes)
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Actual")
  ax.set_title("Landmark LSTM v2 Test Confusion Matrix")
  for row in range(confusion.shape[0]):
    for col in range(confusion.shape[1]):
      ax.text(col, row, str(confusion[row, col]), ha="center", va="center",
              fontsize=8)
  fig.colorbar(image, ax=ax)
  fig.tight_layout()
  fig.savefig(confusion_path, dpi=150)
  plt.close(fig)
  return confusion_path


def save_metrics(history: dict[str, list[float]], classes: list[str],
                 test_acc: float, confusion: np.ndarray, best_epoch: int,
                 param_count: int, args: argparse.Namespace) -> Path:
  per_class = {}
  for index, class_name in enumerate(classes):
    total = int(confusion[index].sum())
    correct = int(confusion[index, index])
    per_class[class_name] = {
      "correct": correct,
      "total": total,
      "accuracy": correct / total if total else 0.0,
    }

  metrics = {
    "model": "LandmarkLSTMV2",
    "classes": classes,
    "parameter_count": param_count,
    "epochs_ran": len(history["train_loss"]),
    "best_epoch": best_epoch,
    "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
    "best_val_acc": max(history["val_acc"]) if history["val_acc"] else None,
    "final_train_acc": history["train_acc"][-1] if history["train_acc"] else None,
    "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
    "test_accuracy": test_acc,
    "history": history,
    "confusion_matrix": confusion.tolist(),
    "per_class_test": per_class,
    "config": {
      "data_dir": args.data_dir,
      "output_dir": args.output_dir,
      "batch_size": args.batch_size,
      "learning_rate": args.learning_rate,
      "weight_decay": args.weight_decay,
      "label_smoothing": config.LANDMARK_V2_LABEL_SMOOTHING,
      "epochs": args.epochs,
      "min_epochs": args.min_epochs,
      "patience": args.patience,
      "seed": args.seed,
    },
  }
  metrics_path = output_file(args.output_dir, config.LANDMARK_V2_METRICS_PATH)
  metrics_path.parent.mkdir(parents=True, exist_ok=True)
  metrics_path.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8")
  return metrics_path


def train() -> None:
  args = parse_args()
  set_seed(args.seed)
  landmarks, masks, labels, split_indices, classes = load_data(args)
  train_loader, val_loader, test_loader = make_loaders(
    landmarks, masks, labels, split_indices, args.batch_size)

  num_classes = len(classes)
  device = config.DEVICE
  model = LandmarkLSTMV2(num_classes).to(device)
  weights = class_weights(labels, split_indices["train"], num_classes).to(device)
  criterion = nn.CrossEntropyLoss(
    weight=weights,
    label_smoothing=config.LANDMARK_V2_LABEL_SMOOTHING)
  optimizer = optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=6)
  param_count = sum(parameter.numel() for parameter in model.parameters())

  print(f"Classes: {classes}")
  print(f"Data dir: {args.data_dir}")
  print(f"Output dir: {args.output_dir}")
  print(f"Landmarks shape: {landmarks.shape}")
  print(f"Mean valid frame ratio: {float(masks.mean()):.2%}")
  print(
    f"Train/val/test: {len(split_indices['train'])}/"
    f"{len(split_indices['val'])}/{len(split_indices['test'])}")
  print(f"Model parameters: {param_count:,}")
  print(f"Training on: {device}")

  history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
  }
  best_val_loss = float("inf")
  best_epoch = 0
  epochs_without_improvement = 0
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  model_path = output_file(args.output_dir, config.LANDMARK_V2_MODEL_SAVE_PATH)

  for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = run_epoch(
      model, train_loader, criterion, device, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    if val_loss < best_val_loss - 1e-4:
      best_val_loss = val_loss
      best_epoch = epoch
      epochs_without_improvement = 0
      torch.save(model.state_dict(), model_path)
    else:
      epochs_without_improvement += 1

    print(
      f"Epoch {epoch:03d}/{args.epochs} | "
      f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} | "
      f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}")

    if (
        epoch >= args.min_epochs and
        epochs_without_improvement >= args.patience):
      print(
        "Early stopping: "
        f"no val_loss improvement for {epochs_without_improvement} epochs.")
      break

  model.load_state_dict(torch.load(model_path, map_location=device))
  test_acc, confusion = evaluate_predictions(
    model, test_loader, device, num_classes)
  plot_path = save_training_plot(history, args)
  confusion_path = save_confusion_matrix(confusion, classes, args)
  metrics_path = save_metrics(history, classes, test_acc, confusion,
                              best_epoch, param_count, args)

  print(f"Best epoch: {best_epoch}")
  print(f"Best val loss: {best_val_loss:.4f}")
  print(f"Best val acc: {max(history['val_acc']):.2%}")
  print(f"Test accuracy: {test_acc:.2%}")
  print(f"Saved model: {model_path}")
  print(f"Saved plot: {plot_path}")
  print(f"Saved confusion matrix: {confusion_path}")
  print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
  train()
