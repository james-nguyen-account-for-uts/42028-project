from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from paths import project_path

import config


@dataclass(frozen=True)
class ModelInfo:
  name: str
  model_path: Path
  class_path: Path

  @property
  def display_name(self) -> str:
    return self.name


class LandmarkLSTMV2(nn.Module):

  def __init__(self, num_classes: int):
    super().__init__()
    self.input_norm = nn.LayerNorm(config.LANDMARK_FEATURES)
    self.lstm = nn.LSTM(
      input_size=config.LANDMARK_FEATURES,
      hidden_size=64,
      num_layers=1,
      batch_first=True,
      bidirectional=True)
    self.classifier = nn.Sequential(
      nn.LayerNorm(128),
      nn.Dropout(0.35),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Dropout(0.25),
      nn.Linear(64, num_classes))

  def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = self.input_norm(x)
    output, _ = self.lstm(x)
    mask_f = mask.float().unsqueeze(-1)
    pooled = (output * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
    return self.classifier(pooled)


def _model_filename() -> str:
  return Path(config.LANDMARK_V2_MODEL_SAVE_PATH).name


def _class_filename() -> str:
  return Path(config.LANDMARK_V2_CLASS_LIST_PATH).name


def _default_model_name() -> str:
  return Path(config.LANDMARK_V2_MODEL_DIR).name


def _class_path_for_model(model_name: str) -> Path:
  if model_name == _default_model_name():
    return project_path(config.LANDMARK_V2_CLASS_LIST_PATH)
  return project_path(config.PROCESSED_DATA_DIR) / model_name / _class_filename()


def discover_models() -> list[ModelInfo]:
  model_root = project_path(config.MODEL_DIR)
  if not model_root.exists():
    return []

  infos: list[ModelInfo] = []
  seen: set[str] = set()
  for model_path in sorted(model_root.rglob("*.pth")):
    if model_path.parent == model_root:
      model_name = model_path.stem
    else:
      model_name = model_path.parent.name
    if model_name in seen:
      continue

    class_path = _class_path_for_model(model_name)
    if not class_path.exists():
      continue
    infos.append(
      ModelInfo(
        name=model_name,
        model_path=model_path,
        class_path=class_path))
    seen.add(model_name)

  default_name = _default_model_name()
  infos.sort(key=lambda info: (info.name != default_name, info.name))
  return infos


def default_model_info() -> Optional[ModelInfo]:
  models = discover_models()
  return models[0] if models else None


def load_classes(model_info: Optional[ModelInfo] = None) -> list[str]:
  if model_info is None:
    model_info = default_model_info()
  if model_info is None:
    return []

  class_path = model_info.class_path
  if not class_path.exists():
    return []
  return np.load(class_path).tolist()


def load_model(classes: list[str],
               device: torch.device,
               model_info: Optional[ModelInfo] = None
               ) -> Optional[LandmarkLSTMV2]:
  if not classes:
    return None
  if model_info is None:
    model_info = default_model_info()
  if model_info is None:
    return None

  model_path = model_info.model_path
  if not model_path.exists():
    return None

  model = LandmarkLSTMV2(len(classes)).to(device)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()
  return model
