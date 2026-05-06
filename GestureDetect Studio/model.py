from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from paths import project_path

import config


class ASLWordLSTM(nn.Module):

  def __init__(self, num_classes: int):
    super().__init__()
    self.lstm = nn.LSTM(
      input_size=config.LANDMARK_FEATURES,
      hidden_size=config.HIDDEN_SIZE,
      num_layers=config.NUM_LAYERS,
      batch_first=True,
      bidirectional=True,
      dropout=config.DROPOUT)
    self.fc = nn.Sequential(
      nn.Linear(config.HIDDEN_SIZE * 2, 128),
      nn.ReLU(),
      nn.Dropout(config.DROPOUT),
      nn.Linear(128, num_classes))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    lstm_out, _ = self.lstm(x)
    return self.fc(lstm_out[:, -1, :])


def load_classes() -> list[str]:
  class_path = project_path(config.CLASS_LIST_PATH)
  if not class_path.exists():
    return []
  return np.load(class_path).tolist()


def load_model(classes: list[str], device: torch.device) -> Optional[ASLWordLSTM]:
  if not classes:
    return None

  model_path = project_path(config.MODEL_SAVE_PATH)
  if not model_path.exists():
    return None

  model = ASLWordLSTM(len(classes)).to(device)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()
  return model
