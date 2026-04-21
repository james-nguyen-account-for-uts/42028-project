import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.step1_preprocess.vocab import SignVocab
from src.step1_preprocess.config import TARGET_SHAPE, VOCAB_PATH


class SignLanguageDataset(Dataset):

  def __init__(self, npy_dir, csv_path, vocab):
    self.npy_dir = npy_dir
    self.vocab = vocab
    self.data_pairs = []

    try:
      df = pd.read_csv(csv_path, sep='\t')
      if 'SENTENCE' not in df.columns:
        df = pd.read_csv(csv_path, sep=',')
    except Exception as e:
      print(f"Error loading CSV {csv_path}: {e}")
      return

    for _, row in df.iterrows():
      file_id = row['SENTENCE_NAME']
      transcript = row['SENTENCE']
      npy_path = os.path.join(self.npy_dir, f"{file_id}.npy")

      if os.path.exists(npy_path):
        # Numericalize labels immediately for efficiency
        numerical_label = self.vocab.numericalize(transcript)
        self.data_pairs.append((npy_path, torch.tensor(numerical_label)))

    print(f"Dataset Initialized: {len(self.data_pairs)} specimens found.")

  def __len__(self):
    return len(self.data_pairs)

  def __getitem__(self, idx):
    npy_path, label_tensor = self.data_pairs[idx]
    movements = np.load(npy_path)
    features = torch.tensor(movements, dtype=torch.float32)
    return features, label_tensor


def sign_language_collate(batch):
  """
  Pads already-processed skeleton sequences.

  The training pipeline loads data from data/2_processed, where coordinates have
  already been normalized around the signer. We therefore leave the values
  untouched unless they still look like raw pixel coordinates from an older
  preprocessing path.
  """
  src_list, trg_list = [], []

  for src, trg in batch:
    finite_src = torch.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)

    # Older experiments may have saved pixel-space coordinates directly.
    # Use a robust magnitude check so normalized samples with a few outliers are
    # not incorrectly rescaled a second time.
    if finite_src.abs().median() > 10.0 and finite_src.abs().max() > 100.0:
      repeat_count = src.shape[1] // 2
      norm_vec = torch.tensor([1280.0, 720.0], dtype=src.dtype).repeat(
        repeat_count)
      src = src / norm_vec

    src_list.append(src)
    trg_list.append(trg)

  # Step 1: Padding
  # src_padded: [Batch, Max_Time, 134]
  src_padded = pad_sequence(src_list, batch_first=True, padding_value=0.0)
  # trg_padded: [Batch, Max_Words]
  trg_padded = pad_sequence(trg_list, batch_first=True, padding_value=0)

  # Step 2: Lengths
  src_lens = torch.LongTensor([len(x) for x in src_list])
  trg_lens = torch.LongTensor([len(x) for x in trg_list])

  return src_padded, trg_padded, src_lens, trg_lens


def verify_files(folder_path):
  """Integrates your verifydata.py logic."""
  if not os.path.exists(folder_path):
    print(f"[ERROR] {folder_path} not found.")
    return True

  files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
  print(f"Verifying {len(files)} files in {folder_path}...")

  corrupt, wrong_shape = [], []
  for f_name in tqdm(files):
    try:
      data = np.load(os.path.join(folder_path, f_name))
      if data.shape != TARGET_SHAPE:
        wrong_shape.append((f_name, data.shape))
    except Exception as e:
      corrupt.append((f_name, str(e)))

  if not corrupt and not wrong_shape:
    print(f"[SUCCESS] All files in {folder_path} are valid.")
    return True
  else:
    if corrupt: print(f"[ERROR] Corrupt: {len(corrupt)} files.")
    if wrong_shape:
      print(
        f"[ERROR] Wrong Shape: {len(wrong_shape)} files (Expected {TARGET_SHAPE})."
      )
    return False


if __name__ == "__main__":
  # 1. Setup Vocab
  vocab = SignVocab()
  if os.path.exists(VOCAB_PATH):
    vocab.load(VOCAB_PATH)

  # 2. Define verification tasks
  data_tasks = [
    {
      "name": "TRAIN",
      "npy": "data/2_processed/train_npy",
      "csv": "data/0_metadata/how2sign_realigned_train.csv"
    }, {
      "name": "VAL",
      "npy": "data/2_processed/val_npy",
      "csv": "data/0_metadata/how2sign_realigned_val.csv"
    }, {
      "name": "TEST",
      "npy": "data/2_processed/test_npy",
      "csv": "data/0_metadata/how2sign_realigned_test.csv"
    }
  ]

  all_verified = True
  for task in data_tasks:
    print(f"\n--- {task['name']} VERIFICATION ---")

    # Physical file check
    if not verify_files(task['npy']):
      all_verified = False

    # Loader check
    if os.path.exists(task['csv']):
      ds = SignLanguageDataset(task['npy'], task['csv'], vocab)
      if len(ds) > 0:
        loader = DataLoader(
          ds, batch_size=4, shuffle=True, collate_fn=sign_language_collate)
        feat, lab, f_len, l_len = next(iter(loader))

        print(f"Batch Shapes | Video: {feat.shape} | Labels: {lab.shape}")
        print(
          f"First Translation: {[vocab.itos.get(i.item(), '??') for i in lab[0]]}"
        )
      else:
        print("[ERROR] Dataset is empty.")
    else:
      print(f"[ERROR] CSV missing: {task['csv']}")

  if all_verified:
    print("\n[READY] Data is clean. Proceed to training.")
  else:
    print("\n[STOP] Fix the shape/corruption errors before training.")
