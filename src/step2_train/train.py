import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.step1_preprocess.vocab import SignVocab
from src.step1_preprocess.dataset import SignLanguageDataset, sign_language_collate
from src.step2_train.model import Encoder, Decoder, Seq2Seq
from src.step2_train.config import VOCAB_PATH, INPUT_DIM, ENC_EMB_DIM, DEC_EMB_DIM, ENC_DROPOUT, DEC_DROPOUT, HID_DIM, N_LAYERS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, N_EPOCHS, CLIP


def decode_indices(indices, vocab):
  """
  Decodes a tensor of indices into a human-readable string.
  Optimized for JSON-based vocab where keys are stored as strings.
  """
  words = []
  if indices.dim() > 1:
    indices = indices[0]

  mapping = getattr(vocab, 'itos', {})
  for idx in indices:
    idx_val = idx.item()
    if idx_val == 0: continue  # Skip <PAD>

    word = mapping.get(idx_val) or mapping.get(str(idx_val))
    if word == "<EOS>":
      break
    if word is None:
      word = mapping.get("3", f"[{idx_val}]")
    if word not in ["<SOS>", "<PAD>", "<UNK>"]:
      words.append(word)

  return " ".join(words) if words else "..."


# Added teacher_forcing_ratio to the arguments
def train(
    model, loader, optimizer, criterion, clip, device, teacher_forcing_ratio):
  model.train()
  epoch_loss = 0

  for i, (src, trg, _, _) in enumerate(loader):
    src, trg = src.to(device), trg.to(device)

    noise = torch.randn_like(src) * 0.002
    src = src + noise

    optimizer.zero_grad()

    # Use the passed teacher_forcing_ratio instead of a fixed 1.0
    output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

    output_dim = output.shape[-1]
    output_flattened = output[1:].view(-1, output_dim)
    trg_flattened = trg[:, 1:].transpose(0, 1).contiguous().view(-1)

    loss = criterion(output_flattened, trg_flattened)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    epoch_loss += loss.item()

    current_lr = optimizer.param_groups[0]['lr']
    print(
      f"  Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | TF: {teacher_forcing_ratio:.2f}",
      end="\r")

  return epoch_loss / len(loader)


def evaluate(model, loader, criterion, device):
  model.eval()
  epoch_loss = 0

  with torch.no_grad():
    for i, (src, trg, _, _) in enumerate(loader):
      src, trg = src.to(device), trg.to(device)

      # Inference mode (no teacher forcing)
      output = model(src, trg, teacher_forcing_ratio=0)

      output_dim = output.shape[-1]
      output_flattened = output[1:].view(-1, output_dim)
      trg_flattened = trg[:, 1:].transpose(0, 1).contiguous().view(-1)

      loss = criterion(output_flattened, trg_flattened)
      epoch_loss += loss.item()

    return epoch_loss / len(loader)


def run_training():
  DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  if DEVICE.type == "mps":
    torch.mps.empty_cache()
    print("M3 GPU Cache Cleared.")

  vocab = SignVocab()
  vocab.load(VOCAB_PATH)

  train_ds = SignLanguageDataset(
    "data/2_processed/train_npy",
    "data/0_metadata/how2sign_realigned_train.csv", vocab)
  # train_ds = Subset(full_train_ds, list(range(10)))

  val_ds = SignLanguageDataset(
    "data/2_processed/val_npy", "data/0_metadata/how2sign_realigned_val.csv",
    vocab)
  # val_ds = Subset(full_val_ds, list(range(10)))

  train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=sign_language_collate)

  val_subset = Subset(val_ds, list(range(200)))
  val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=sign_language_collate)

  model = Seq2Seq(
    Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT),
    Decoder(len(vocab), DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT),
    DEVICE).to(DEVICE)

  optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-4)
  criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

  print(f"\nRunning on {len(train_ds)} specimens...")

  best_val_loss = float('inf')

  for epoch in range(N_EPOCHS):
    start_time = time.time()

    # --- TEACHER FORCING DECAY LOGIC ---
    # Start at 0.75, decrease by 0.05 each epoch, but never go below 0.4
    # current_tf_ratio = max(0.65, 0.8 - (epoch * 0.01))

    # Stay at 80% for the first 5 epochs, then drop
    if epoch < 5:
      current_tf_ratio = 0.80
    else:
      current_tf_ratio = max(0.40, 0.80 - ((epoch - 5) * 0.05))

    # Pass the ratio into the training function
    train_loss = train(
      model, train_loader, optimizer, criterion, CLIP, DEVICE,
      current_tf_ratio)

    val_loss = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step(val_loss)

    duration = time.time() - start_time

    print(
      f"\nEpoch: {epoch+1:02} | Time: {duration:.2f}s | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
    )

    # --- Sample Translation Check every 5 Epochs ---
    if (epoch + 1) % 5 == 0:
      model.eval()
      with torch.no_grad():
        # Get sample batch (first 5 samples)
        src, trg, _, _ = next(iter(val_loader))
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        output = model(src, trg, teacher_forcing_ratio=0)
        preds = output.argmax(2).transpose(0, 1)

        print(f"\n[Validation Sample Translation]")
        print(f"  Target: {decode_indices(trg[0].cpu(), vocab)}")
        print(f"  Pred:   {decode_indices(preds[0].cpu(), vocab)}\n")

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      os.makedirs("models/checkpoints", exist_ok=True)
      torch.save(model.state_dict(), "models/checkpoints/best_sign_model.pth")
      print("✨ Best Model Saved!")

    if DEVICE.type == "mps":
      torch.mps.empty_cache()

  print("\n✅ Run complete.")


if __name__ == "__main__":
  run_training()
