import os
import json
import re
import pandas as pd
from src.step1_preprocess.config import VOCAB_PATH


class SignVocab:

  def __init__(self, min_freq=2):
    # Special Tokens
    self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
    self.stoi = {v: k for k, v in self.itos.items()}
    self.min_freq = min_freq

  def __len__(self):
    return len(self.itos)

  def tokenize(self, text):
    """Improved cleaning: Use regex to remove all punctuation except apostrophes (e.g., "don't")"""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9'\s]", "", text)
    return text.split()

  def build_vocabulary(self, csv_list):
    frequencies = {}

    for csv_path in csv_list:
      if os.path.exists(csv_path):
        print(f"Reading {csv_path}...")
        try:
          df = pd.read_csv(csv_path, sep='\t')
          if 'SENTENCE' not in df.columns:
            df = pd.read_csv(csv_path, sep=',')

          if 'SENTENCE' in df.columns:
            for sentence in df['SENTENCE']:
              for word in self.tokenize(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1
          else:
            print(f"⚠️ Column 'SENTENCE' not found in {csv_path}")
        except Exception as e:
          print(f"Error reading {csv_path}: {e}")
      else:
        print(f"Skipping {csv_path} (File not found)")

    # Assign IDs to words that meet the frequency threshold
    # Sort by frequency to keep most common words at lower indices
    idx = 4
    sorted_freqs = sorted(
      frequencies.items(), key=lambda x: x[1], reverse=True)

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
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
      json.dump({"itos": self.itos, "stoi": self.stoi}, f)
    print(f"Vocab saved to {file_path}")

  def load(self, file_path):
    with open(file_path, "r") as f:
      data = json.load(f)
      # CRITICAL: JSON keys are strings; convert itos keys back to int
      # Also ensure all stoi keys stay strings and values stay int
      self.itos = {int(k): str(v) for k, v in data["itos"].items()}
      self.stoi = {str(k): int(v) for k, v in data["stoi"].items()}
    print(f"Vocab loaded from {file_path}")


if __name__ == "__main__":
  metadata_dir = "data/0_metadata"
  csv_files = [
    os.path.join(metadata_dir, "how2sign_realigned_test.csv"),
    os.path.join(metadata_dir, "how2sign_realigned_val.csv"),
    os.path.join(metadata_dir, "how2sign_realigned_train.csv")
  ]

  vocab = SignVocab(min_freq=2)
  vocab.build_vocabulary(csv_files)

  os.makedirs("models", exist_ok=True)
  vocab.save(VOCAB_PATH)

  sample = "That's good."
  encoded = vocab.numericalize(sample)
  decoded = [vocab.itos[i] for i in encoded]

  print(f"\n--- TEST ---")
  print(f"Input text:  {sample}")
  print(f"Numerical:   {encoded}")
  print(f"Decoded back: {decoded}")
