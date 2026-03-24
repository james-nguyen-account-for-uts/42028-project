import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

# Import your custom modules
from vocab import SignVocab
from dataset import OriginalDataset, sign_language_collate
from model import Encoder, Decoder, Seq2Seq

def train_one_epoch(model, loader, optimizer, criterion, clip, device):
  model.train()
  epoch_loss = 0
  
  for i, (src, trg, f_len, l_len) in enumerate(loader):
    src, trg = src.to(device), trg.to(device)
    
    optimizer.zero_grad()
    
    # trg shape: [batch, trg_len]
    # output shape: [trg_len, batch, vocab_size]
    output = model(src, trg)
    
    # Reshape for Loss function: ignore the <SOS> token (index 0 of trg)
    output_dim = output.shape[-1]
    output = output[1:].view(-1, output_dim)
    trg = trg.permute(1, 0)[1:].reshape(-1)
    
    loss = criterion(output, trg)
    loss.backward()
    
    # Clip gradients to prevent "Exploding Gradients" in LSTMs
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    
    epoch_loss += loss.item()
    
  return epoch_loss / len(loader)

if __name__ == "__main__":
  # 1. Hyperparameters & Device
  # Change this:
  # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # To this:
  if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
  elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
  else:
    DEVICE = torch.device("cpu")
  
  INPUT_DIM = 134   # Your skeleton keypoints
  OUTPUT_DIM = 0    # Will be set by Vocab
  ENC_EMB_DIM = 256
  DEC_EMB_DIM = 256
  HID_DIM = 512
  N_LAYERS = 2
  ENC_DROPOUT = 0.5
  DEC_DROPOUT = 0.5
  LEARNING_RATE = 0.001
  BATCH_SIZE = 32
  N_EPOCHS = 10
  CLIP = 1

  # 2. Load Vocab
  vocab = SignVocab()
  vocab.load("models/vocab.json")
  OUTPUT_DIM = len(vocab)

  # 3. Initialize Model
  enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
  dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
  model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  # Ignore <PAD> token (ID 0) when calculating loss
  criterion = nn.CrossEntropyLoss(ignore_index=0)

  # 4. Data Loaders (Using Val for a "Dry Run" while Train extracts)
  val_dataset = OriginalDataset("data/2_processed/val_npy", "data/0_metadata/how2sign_realigned_val.csv", vocab)
  val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=sign_language_collate)

  print(f"\n--- Starting Training (Dry Run on Val Set) ---")
  print(f"Device: {DEVICE} | Vocab Size: {OUTPUT_DIM}")

  for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss = train_one_epoch(model, val_loader, optimizer, criterion, CLIP, DEVICE)
    
    end_time = time.time()
    print(f"Epoch: {epoch+1:02} | Time: {end_time - start_time:.2f}s | Loss: {train_loss:.4f}")

    # Save Checkpoint
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"models/checkpoints/sign_model_e{epoch+1}.pth")