import csv
import json
import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.step1_preprocess.dataset import SignLanguageDataset, sign_language_collate
from src.step1_preprocess.vocab import SignVocab
from src.step2_train.config import BATCH_SIZE, CLIP, DEC_DROPOUT, DEC_EMB_DIM, DEFAULT_ENCODER_ARCH, ENC_DROPOUT, ENC_EMB_DIM, HID_DIM, INPUT_DIM, LEARNING_RATE, N_EPOCHS, N_LAYERS, RUNS_DIR, VAL_SUBSET_SIZE, VOCAB_PATH, WEIGHT_DECAY
from src.step2_train.model import Decoder, Encoder, Seq2Seq


def get_device():
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def strip_special_tokens(token_ids, pad_idx=0, sos_idx=1, eos_idx=2):
  clean = []
  for idx in token_ids:
    if idx == pad_idx:
      continue
    if idx == sos_idx:
      continue
    if idx == eos_idx:
      break
    clean.append(int(idx))
  return clean


def decode_token_ids(token_ids, vocab):
  words = []
  mapping = getattr(vocab, "itos", {})
  for idx in token_ids:
    word = mapping.get(idx) or mapping.get(str(idx)) or f"[{idx}]"
    if word not in ["<SOS>", "<PAD>", "<UNK>", "<EOS>"]:
      words.append(word)
  return " ".join(words) if words else "..."


def decode_indices(indices, vocab):
  if hasattr(indices, "dim") and indices.dim() > 1:
    indices = indices[0]
  if hasattr(indices, "tolist"):
    indices = indices.tolist()
  return decode_token_ids(strip_special_tokens(indices), vocab)


def greedy_decode_batch(model, src, max_len, sos_idx, eos_idx):
  batch_size = src.shape[0]
  _, (hidden, cell) = model.encoder(src)

  input_token = torch.full(
    (batch_size,), sos_idx, dtype=torch.long, device=src.device)
  finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
  predictions = []

  for _ in range(max_len):
    output, hidden, cell = model.decoder(input_token, hidden, cell)
    top1 = output.argmax(1)
    top1 = torch.where(finished, torch.full_like(top1, eos_idx), top1)
    predictions.append(top1)
    finished |= top1.eq(eos_idx)
    input_token = top1
    if finished.all():
      break

  if not predictions:
    return torch.empty(batch_size, 0, dtype=torch.long, device=src.device)

  return torch.stack(predictions, dim=1)


def evaluate_teacher_forced(model, loader, criterion, device):
  model.eval()
  epoch_loss = 0.0

  with torch.no_grad():
    for src, trg, _, _ in loader:
      src, trg = src.to(device), trg.to(device)

      output = model(src, trg, teacher_forcing_ratio=1.0)

      output_dim = output.shape[-1]
      output_flattened = output[1:].view(-1, output_dim)
      trg_flattened = trg[:, 1:].transpose(0, 1).contiguous().view(-1)

      loss = criterion(output_flattened, trg_flattened)
      epoch_loss += loss.item()

  return epoch_loss / max(len(loader), 1)


def evaluate_greedy(model, loader, device, vocab):
  model.eval()

  pad_idx = vocab.stoi["<PAD>"]
  sos_idx = vocab.stoi["<SOS>"]
  eos_idx = vocab.stoi["<EOS>"]

  token_correct = 0
  token_total = 0
  exact_matches = 0
  sample_count = 0
  pred_len_total = 0
  ref_len_total = 0
  eos_hits = 0
  unigram_overlap = 0
  unigram_pred_total = 0
  unigram_ref_total = 0
  sample_target = "..."
  sample_pred = "..."
  sample_captured = False

  with torch.no_grad():
    for src, trg, _, _ in loader:
      src, trg = src.to(device), trg.to(device)
      preds = greedy_decode_batch(
        model, src, max_len=trg.shape[1], sos_idx=sos_idx, eos_idx=eos_idx)

      pred_rows = preds.cpu().tolist()
      trg_rows = trg.cpu().tolist()

      for pred_row, trg_row in zip(pred_rows, trg_rows):
        pred_tokens = strip_special_tokens(pred_row, pad_idx, sos_idx, eos_idx)
        ref_tokens = strip_special_tokens(trg_row, pad_idx, sos_idx, eos_idx)

        if not sample_captured:
          sample_target = decode_token_ids(ref_tokens, vocab)
          sample_pred = decode_token_ids(pred_tokens, vocab)
          sample_captured = True

        aligned_len = max(len(pred_tokens), len(ref_tokens), 1)
        for pos in range(aligned_len):
          pred_tok = pred_tokens[pos] if pos < len(pred_tokens) else None
          ref_tok = ref_tokens[pos] if pos < len(ref_tokens) else None
          if pred_tok == ref_tok:
            token_correct += 1
        token_total += aligned_len

        exact_matches += int(pred_tokens == ref_tokens)
        sample_count += 1
        pred_len_total += len(pred_tokens)
        ref_len_total += len(ref_tokens)
        eos_hits += int(eos_idx in pred_row)

        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        unigram_overlap += sum(
          min(count, ref_counter[token]) for token, count in pred_counter.items())
        unigram_pred_total += len(pred_tokens)
        unigram_ref_total += len(ref_tokens)

  if unigram_pred_total == 0:
    bleu1 = 0.0
  else:
    precision = unigram_overlap / unigram_pred_total
    brevity_penalty = 1.0
    if unigram_pred_total < unigram_ref_total:
      brevity_penalty = math.exp(
        1.0 - (unigram_ref_total / max(unigram_pred_total, 1)))
    bleu1 = brevity_penalty * precision if precision > 0 else 0.0

  return {
    "greedy_token_acc": token_correct / max(token_total, 1),
    "greedy_exact_match": exact_matches / max(sample_count, 1),
    "greedy_bleu1": bleu1,
    "greedy_avg_pred_len": pred_len_total / max(sample_count, 1),
    "greedy_avg_ref_len": ref_len_total / max(sample_count, 1),
    "greedy_eos_rate": eos_hits / max(sample_count, 1),
    "sample_target": sample_target,
    "sample_pred": sample_pred,
  }


def train_one_epoch(
    model, loader, optimizer, criterion, clip, device, teacher_forcing_ratio):
  model.train()
  epoch_loss = 0.0

  for i, (src, trg, _, _) in enumerate(loader):
    src, trg = src.to(device), trg.to(device)

    noise = torch.randn_like(src) * 0.002
    src = src + noise

    optimizer.zero_grad()

    output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

    output_dim = output.shape[-1]
    output_flattened = output[1:].view(-1, output_dim)
    trg_flattened = trg[:, 1:].transpose(0, 1).contiguous().view(-1)

    loss = criterion(output_flattened, trg_flattened)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    epoch_loss += loss.item()

    current_lr = optimizer.param_groups[0]["lr"]
    print(
      f"  Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | TF: {teacher_forcing_ratio:.2f}",
      end="\r")

  return epoch_loss / max(len(loader), 1)


def build_model(vocab_size, device, encoder_arch):
  encoder_arch = encoder_arch.lower()
  bidirectional = encoder_arch == "bilstm"
  encoder = Encoder(
    INPUT_DIM,
    ENC_EMB_DIM,
    HID_DIM,
    N_LAYERS,
    ENC_DROPOUT,
    bidirectional=bidirectional)
  decoder = Decoder(vocab_size, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
  model = Seq2Seq(encoder, decoder, device).to(device)
  return model, bidirectional


def sanitize_run_name(name):
  safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
  return safe.strip("._") or "run"


def prepare_run_dir(run_name, encoder_arch, epoch_count, val_subset_size, device,
                    vocab_size, train_size, val_size):
  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  resolved_name = sanitize_run_name(
    run_name or f"{timestamp}_{encoder_arch.lower()}")
  run_dir = Path(RUNS_DIR) / resolved_name
  run_dir.mkdir(parents=True, exist_ok=True)

  config_payload = {
    "run_name": resolved_name,
    "timestamp": timestamp,
    "encoder_arch": encoder_arch.lower(),
    "epochs": epoch_count,
    "val_subset_size": val_subset_size,
    "device": str(device),
    "train_size": train_size,
    "val_size": val_size,
    "vocab_size": vocab_size,
    "input_dim": INPUT_DIM,
    "enc_emb_dim": ENC_EMB_DIM,
    "dec_emb_dim": DEC_EMB_DIM,
    "hid_dim": HID_DIM,
    "n_layers": N_LAYERS,
    "enc_dropout": ENC_DROPOUT,
    "dec_dropout": DEC_DROPOUT,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE,
    "clip": CLIP,
  }

  with open(run_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(config_payload, f, indent=2)

  return run_dir, resolved_name


def write_metrics_logs(run_dir, history):
  metrics_json = run_dir / "metrics.json"
  metrics_csv = run_dir / "metrics.csv"

  with open(metrics_json, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

  if not history:
    return

  with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
    writer.writeheader()
    writer.writerows(history)


def write_summary(run_dir, summary):
  with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)


def run_training(
    encoder_arch=DEFAULT_ENCODER_ARCH,
    epochs=None,
    run_name=None,
    val_subset_size=VAL_SUBSET_SIZE):
  encoder_arch = encoder_arch.lower()
  if encoder_arch not in {"bilstm", "unilstm"}:
    raise ValueError("encoder_arch must be 'bilstm' or 'unilstm'")

  epoch_count = epochs if epochs is not None else N_EPOCHS
  if epoch_count <= 0:
    raise ValueError("epochs must be a positive integer")

  device = get_device()
  if device.type == "mps":
    torch.mps.empty_cache()
    print("M3 GPU cache cleared.")
  elif device.type == "cuda":
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")

  vocab = SignVocab()
  vocab.load(VOCAB_PATH)

  train_ds = SignLanguageDataset(
    "data/2_processed/train_npy",
    "data/0_metadata/how2sign_realigned_train.csv", vocab)
  val_ds = SignLanguageDataset(
    "data/2_processed/val_npy",
    "data/0_metadata/how2sign_realigned_val.csv", vocab)

  val_count = min(val_subset_size, len(val_ds))
  val_subset = Subset(val_ds, list(range(val_count)))

  train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=sign_language_collate)
  val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=sign_language_collate)

  model, bidirectional = build_model(len(vocab), device, encoder_arch)
  optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-4)
  criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

  run_dir, resolved_run_name = prepare_run_dir(
    run_name,
    encoder_arch,
    epoch_count,
    val_count,
    device,
    len(vocab),
    len(train_ds),
    len(val_ds))
  print(f"Run directory: {run_dir}")
  print(
    f"\nRunning on {len(train_ds)} specimens with {encoder_arch} encoder ({'bi' if bidirectional else 'uni'}-directional)..."
  )

  best_val_tf_loss = float("inf")
  best_epoch = 0
  metrics_history = []
  checkpoint_dir = Path("models/checkpoints")
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  arch_ckpt = checkpoint_dir / f"best_sign_model_{encoder_arch}.pth"
  compatibility_ckpt = checkpoint_dir / "best_sign_model.pth"
  run_best_ckpt = run_dir / "best_sign_model.pth"

  for epoch in range(epoch_count):
    start_time = time.time()
    lr_used = optimizer.param_groups[0]["lr"]

    if epoch < 5:
      current_tf_ratio = 0.80
    else:
      current_tf_ratio = max(0.40, 0.80 - ((epoch - 5) * 0.05))

    train_loss = train_one_epoch(
      model, train_loader, optimizer, criterion, CLIP, device,
      current_tf_ratio)

    val_tf_loss = evaluate_teacher_forced(model, val_loader, criterion, device)
    greedy_metrics = evaluate_greedy(model, val_loader, device, vocab)

    scheduler.step(val_tf_loss)
    next_lr = optimizer.param_groups[0]["lr"]
    duration = time.time() - start_time

    best_updated = val_tf_loss < best_val_tf_loss
    if best_updated:
      best_val_tf_loss = val_tf_loss
      best_epoch = epoch + 1
      torch.save(model.state_dict(), run_best_ckpt)
      torch.save(model.state_dict(), arch_ckpt)
      if encoder_arch == DEFAULT_ENCODER_ARCH:
        torch.save(model.state_dict(), compatibility_ckpt)

    epoch_record = {
      "epoch": epoch + 1,
      "encoder_arch": encoder_arch,
      "bidirectional": bidirectional,
      "train_loss": round(train_loss, 6),
      "val_tf_loss": round(val_tf_loss, 6),
      "greedy_token_acc": round(greedy_metrics["greedy_token_acc"], 6),
      "greedy_exact_match": round(greedy_metrics["greedy_exact_match"], 6),
      "greedy_bleu1": round(greedy_metrics["greedy_bleu1"], 6),
      "greedy_avg_pred_len": round(greedy_metrics["greedy_avg_pred_len"], 4),
      "greedy_avg_ref_len": round(greedy_metrics["greedy_avg_ref_len"], 4),
      "greedy_eos_rate": round(greedy_metrics["greedy_eos_rate"], 6),
      "teacher_forcing_ratio": round(current_tf_ratio, 4),
      "lr_used": round(lr_used, 8),
      "lr_next": round(next_lr, 8),
      "duration_sec": round(duration, 2),
      "best_checkpoint_updated": best_updated,
      "sample_target": greedy_metrics["sample_target"],
      "sample_pred": greedy_metrics["sample_pred"],
    }
    metrics_history.append(epoch_record)
    write_metrics_logs(run_dir, metrics_history)

    print(
      f"\nEpoch: {epoch+1:02} | Time: {duration:.2f}s | Train: {train_loss:.4f} | ValTF: {val_tf_loss:.4f} | "
      f"GreedyTokAcc: {greedy_metrics['greedy_token_acc']:.4f} | GreedyExact: {greedy_metrics['greedy_exact_match']:.4f} | "
      f"GreedyBLEU1: {greedy_metrics['greedy_bleu1']:.4f}"
    )

    if (epoch + 1) % 5 == 0:
      print("\n[Validation Sample Translation]")
      print(f"  Target: {greedy_metrics['sample_target']}")
      print(f"  Pred:   {greedy_metrics['sample_pred']}\n")

    if best_updated:
      print("Best val_tf_loss checkpoint saved.")

    if device.type == "mps":
      torch.mps.empty_cache()

  summary = {
    "run_name": resolved_run_name,
    "encoder_arch": encoder_arch,
    "epochs_completed": epoch_count,
    "best_epoch": best_epoch,
    "best_val_tf_loss": best_val_tf_loss,
    "run_dir": str(run_dir),
    "best_checkpoint_path": str(run_best_ckpt),
    "arch_checkpoint_path": str(arch_ckpt),
    "compatibility_checkpoint_path": str(compatibility_ckpt)
    if encoder_arch == DEFAULT_ENCODER_ARCH else None,
  }
  write_summary(run_dir, summary)

  print("\nRun complete.")
  print(f"Best epoch: {best_epoch} | Best val_tf_loss: {best_val_tf_loss:.4f}")


if __name__ == "__main__":
  run_training()
