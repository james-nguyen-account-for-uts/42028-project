import pandas as pd
import torch
import os
import json

class SignVocab:
  def __init__(self, min_freq=2):
    # Special Tokens
    self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
    self.stoi = {v: k for k, v in self.itos.items()}
    self.min_freq = min_freq

  def __len__(self):
    return len(self.itos)

  def tokenize(self, text):
    # Basic cleaning: lowercase and remove punctuation
    return str(text).lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").split()

  def build_vocabulary(self, csv_list):
    frequencies = {}
    
    for csv_path in csv_list:
      if os.path.exists(csv_path):
        print(f"Reading {csv_path}...")
        try:
          df = pd.read_csv(csv_path, sep='\t')
          if 'SENTENCE' not in df.columns:
            df = pd.read_csv(csv_path, sep=',')
          
          for sentence in df['SENTENCE']:
            for word in self.tokenize(sentence):
              frequencies[word] = frequencies.get(word, 0) + 1
        except Exception as e:
          print(f"Error reading {csv_path}: {e}")
      else:
        print(f"Skipping {csv_path} (File not found)")

    # Assign IDs to words that meet the frequency threshold
    idx = 4
    sorted_freqs = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    
    for word, freq in sorted_freqs:
      if freq >= self.min_freq:
        self.stoi[word] = idx
        self.itos[idx] = word
        idx += 1
    
    print(f"Vocab built! Total unique words: {len(self.itos)}")

  def numericalize(self, text):
    tokenized = self.tokenize(text)
    return [self.stoi["<SOS>"]] + \
           [self.stoi.get(word, self.stoi["<UNK>"]) for word in tokenized] + \
           [self.stoi["<EOS>"]]

  def save(self, file_path):
    with open(file_path, "w") as f:
      json.dump({"itos": self.itos, "stoi": self.stoi}, f)
    print(f"Vocab saved to {file_path}")

  def load(self, file_path):
    with open(file_path, "r") as f:
      data = json.load(f)
      # JSON keys are always strings, convert itos keys back to int
      self.itos = {int(k): v for k, v in data["itos"].items()}
      self.stoi = data["stoi"]
    print(f"Vocab loaded from {file_path}")

if __name__ == "__main__":
  # Configuration: All three datasets
  metadata_dir = "data/0_metadata"
  csv_files = [
    os.path.join(metadata_dir, "how2sign_realigned_test.csv"),
    os.path.join(metadata_dir, "how2sign_realigned_val.csv"),
    os.path.join(metadata_dir, "how2sign_realigned_train.csv") # Will skip if not yet downloaded
  ]

  vocab = SignVocab(min_freq=2)
  vocab.build_vocabulary(csv_files)
  
  # Save the vocabulary so you don't have to rebuild it every time
  os.makedirs("models", exist_ok=True)
  vocab.save("models/vocab.json")

  # Final Test
  sample = "That's good."
  encoded = vocab.numericalize(sample)
  decoded = [vocab.itos[i] for i in encoded]
  
  print(f"\n--- TEST ---")
  print(f"Input text:  {sample}")
  print(f"Numerical:   {encoded}")
  print(f"Decoded back: {decoded}")