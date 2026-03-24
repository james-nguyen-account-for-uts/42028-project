import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
  def __init__(self, npy_dir, metadata_csv):
    self.npy_dir = npy_dir
    self.metadata = pd.read_csv(metadata_csv) # Assume you have 'file_id' and 'label'
    
    # Convert words/sentences into numbers (Label Encoding)
    self.label_map = {word: i for i, word in enumerate(self.metadata['label'].unique())}

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    file_id = self.metadata.iloc[idx]['file_id']
    label_name = self.metadata.iloc[idx]['label']
    
    # Load the preprocessed movements
    data = np.load(f"{self.npy_dir}/{file_id}.npy")
    
    # Convert to PyTorch Tensor
    x = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(self.label_map[label_name], dtype=torch.long)
    
    return x, y