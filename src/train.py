import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from vocab import SignVocab
from dataset import OriginalDataset, sign_language_collate
from model import Encoder, Decoder, Seq2Seq

def train_one_epoch(model, loader, optimizer, criterion, clip, device):
  model.train()
  epoch_loss = 0
  total_batches = len(loader)
  
  for i, (src, trg, f_len, l_len) in enumerate(loader):
    src, trg = src.to(device), trg.to(device)
    optimizer.zero_grad()
    
    # 1. Forward pass
    output = model(src, trg)
    
    # 2. Reshape for Loss: [trg_len, batch, vocab] -> [(trg_len-1)*batch, vocab]
    output_dim = output.shape[-1]
    # We slice [1:] to skip the <SOS> token
    output_flattened = output[1:].view(-1, output_dim)
    trg_flattened = trg.permute(1, 0)[1:].reshape(-1)
    
    # 3. Calculate Loss (Assigned to 'loss' now!)
    loss = criterion(output_flattened, trg_flattened)
    
    # 4. Backward pass
    loss.backward()
    
    # 5. Clip gradients and Step
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    
    epoch_loss += loss.item()

    if i % 5 == 0:
      print(f"  Batch {i}/{total_batches} | Loss: {loss.item():.4f}", end="\r")
      
  return epoch_loss / total_batches

def evaluate(model, loader, criterion, device):
  model.eval()
  epoch_loss = 0
  total_batches = len(loader)
  with torch.no_grad():
    for i, (src, trg, f_len, l_len) in enumerate(loader):
      src, trg = src.to(device), trg.to(device)
      # No teacher forcing during eval
      output = model(src, trg, teacher_forcing_ratio=0)
      
      output_dim = output.shape[-1]
      output_flattened = output[1:].view(-1, output_dim)
      trg_flattened = trg.permute(1, 0)[1:].reshape(-1)
      
      loss = criterion(output_flattened, trg_flattened)
      epoch_loss += loss.item()
      
      if i % 10 == 0:
        if device.type == "mps":
          torch.mps.empty_cache()
        print(f"  Batch {i}/{total_batches} | Loss: {loss.item():.4f}", end="\r")
      
  return epoch_loss / total_batches

if __name__ == "__main__":
  # Configuration
  DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  INPUT_DIM, ENC_EMB_DIM, DEC_EMB_DIM = 134, 256, 256
  HID_DIM, N_LAYERS = 512, 2
  ENC_DROPOUT, DEC_DROPOUT = 0.5, 0.5
  LEARNING_RATE, BATCH_SIZE, N_EPOCHS, CLIP = 0.0005, 32, 20, 1
  
  # Cleaning cache for mps
  if DEVICE.type == "mps":
    torch.mps.empty_cache()
    print("M3 GPU Cache Cleared. Starting run...")

  vocab = SignVocab()
  vocab.load("models/vocab.json")
  
  # Load Data
  train_ds = OriginalDataset("data/2_processed/train_npy", "data/0_metadata/how2sign_realigned_train.csv", vocab)
  val_ds = OriginalDataset("data/2_processed/val_npy", "data/0_metadata/how2sign_realigned_val.csv", vocab)

  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=sign_language_collate)
  val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=sign_language_collate)

  # Init Model
  model = Seq2Seq(
    Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT), 
    Decoder(len(vocab), DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT), DEVICE
  ).to(DEVICE)

  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  criterion = nn.CrossEntropyLoss(ignore_index=0)

  print(f"\n🚀 Starting Full Run on {len(train_ds)} specimens...")
  print(f"Device: {DEVICE} | Batch Size: {BATCH_SIZE} | Epochs: {N_EPOCHS}")
  
  best_val_loss = float('inf')
  
  for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    # Train
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CLIP, DEVICE)
    # Evaluate
    val_loss = evaluate(model, val_loader, criterion, DEVICE)
    
    end_time = time.time()
    duration = end_time - start_time

    print(f"\nEpoch: {epoch+1:02} | Time: {duration:.2f}s")
    print(f"\tTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save BEST model based on Validation performance
    os.makedirs("models/checkpoints", exist_ok=True)
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), "models/checkpoints/best_sign_model.pth")
      print(f"✨ New Best Model Saved!")
    
    # Optional: Save a "Last" checkpoint regardless of performance
    torch.save(model.state_dict(), "models/checkpoints/last_model_checkpoint.pth")
    
    # Clear M3 Cache before next epoch
    if DEVICE.type == "mps":
      torch.mps.empty_cache()

  print("\n✅ Training Complete! Check 'models/checkpoints/' for your weights.")