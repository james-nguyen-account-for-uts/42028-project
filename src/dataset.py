import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from vocab import SignVocab

class OriginalDataset(Dataset):
  def __init__(self, npy_dir, csv_path, vocab):
    self.npy_dir = npy_dir
    self.vocab = vocab
    
    try:
      df = pd.read_csv(csv_path, sep='\t')
      if 'SENTENCE' not in df.columns:
        df = pd.read_csv(csv_path, sep=',')
    except Exception as e:
      print(f"Error loading CSV {csv_path}: {e}")
      self.data_pairs = []
      return

    self.data_pairs = []
    
    for _, row in df.iterrows():
      file_id = row['SENTENCE_NAME']
      transcript = row['SENTENCE']
      
      npy_path = os.path.join(self.npy_dir, f"{file_id}.npy")
      
      if os.path.exists(npy_path):
        # Migration: Numericalize labels immediately
        numerical_label = self.vocab.numericalize(transcript)
        self.data_pairs.append((npy_path, torch.tensor(numerical_label)))
    
    print(f"--- Dataset Initialized ---")
    print(f"Source: {csv_path}")
    print(f"Mapped Specimens: {len(self.data_pairs)}")

  def __len__(self):
    return len(self.data_pairs)

  def __getitem__(self, idx):
    npy_path, label_tensor = self.data_pairs[idx]
    movements = np.load(npy_path)
    features = torch.tensor(movements, dtype=torch.float32)
    return features, label_tensor

def sign_language_collate(batch):
  """Migration: Now pads both Movement Features AND Numerical Labels."""
  features, labels = zip(*batch)
  
  # Pad Video (Frames, 134)
  padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
  feature_lengths = torch.tensor([f.shape[0] for f in features])
  
  # Pad Labels (Word IDs) - 0 is the <PAD> token in SignVocab
  padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
  label_lengths = torch.tensor([l.shape[0] for l in labels])

  return padded_features, padded_labels, feature_lengths, label_lengths

if __name__ == "__main__":
  # 1. Initialize and Load Vocab
  vocab = SignVocab()
  vocab_path = "models/vocab.json"
  if os.path.exists(vocab_path):
    vocab.load(vocab_path)
  else:
    print(f"⚠️ Warning: {vocab_path} not found. Ensure you ran vocab.py first.")
    # For testing purposes, we'll build a tiny one if missing
    vocab.build_vocabulary(["data/0_metadata/how2sign_realigned_test.csv"])

  # 2. Migration: Verification Tasks
  data_tasks = [
    {
      "name": "TEST",
      "npy": "data/2_processed/test_npy",
      "csv": "data/0_metadata/how2sign_realigned_test.csv"
    },
    {
      "name": "VALIDATION",
      "npy": "data/2_processed/val_npy",
      "csv": "data/0_metadata/how2sign_realigned_val.csv"
    },
    {
      "name": "TRAINING",
      "npy": "data/2_processed/train_npy",
      "csv": "data/0_metadata/how2sign_realigned_train.csv"
    }
  ]

  for task in data_tasks:
    print(f"\n[VERIFYING {task['name']} SET]")
    
    if os.path.exists(task['csv']):
      dataset = OriginalDataset(task['npy'], task['csv'], vocab)
      
      if len(dataset) > 0:
        loader = DataLoader(
          dataset, 
          batch_size=4, 
          shuffle=True, 
          collate_fn=sign_language_collate
        )

        # Test one batch
        feat, lab, f_len, l_len = next(iter(loader))

        print(f"Video Shape:   {feat.shape} (Batch, Time, Feat)")
        print(f"Label Shape:   {lab.shape} (Batch, Words)")
        print(f"Sample Label:  {lab[0].tolist()}")
        print(f"Decoded:       {[vocab.itos[i.item()] for i in lab[0]]}")
        print("Status: READY FOR AI ✅")
      else:
        print("Status: EMPTY (No .npy files found) ❌")
    else:
      print(f"Status: SKIP (CSV not found at {task['csv']}) ⚠️")

  print("\n--- ALL CHECKS FINISHED ---")