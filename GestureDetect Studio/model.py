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
  kind: str
  input_mode: str
  uses_mask: bool = False
  sequence_length: int = config.SEQUENCE_LENGTH

  @property
  def display_name(self) -> str:
    return self.name


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


class TemporalSelfAttention(nn.Module):

  def __init__(self, hidden_size: int, num_heads: int = 4):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    if hidden_size % num_heads != 0:
      raise ValueError("hidden_size must be divisible by num_heads")
    self.q_proj = nn.Linear(hidden_size, hidden_size)
    self.k_proj = nn.Linear(hidden_size, hidden_size)
    self.v_proj = nn.Linear(hidden_size, hidden_size)
    self.out_proj = nn.Linear(hidden_size, hidden_size)
    self.scale = self.head_dim ** -0.5

  def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
    batch, steps, hidden = lstm_out.shape
    q = self.q_proj(lstm_out).view(
      batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(lstm_out).view(
      batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(lstm_out).view(
      batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
    weights = torch.softmax(scores, dim=-1)
    attended = torch.matmul(weights, v)
    attended = attended.transpose(1, 2).contiguous().view(
      batch, steps, hidden)
    return self.out_proj(attended).mean(dim=1)


class ASLWordLSTMAttention(nn.Module):

  def __init__(self, num_classes: int):
    super().__init__()
    self.input_norm = nn.LayerNorm(config.LANDMARK_FEATURES)
    self.lstm = nn.LSTM(
      input_size=config.LANDMARK_FEATURES,
      hidden_size=config.HIDDEN_SIZE,
      num_layers=config.NUM_LAYERS,
      batch_first=True,
      bidirectional=True,
      dropout=config.DROPOUT)
    hidden = config.HIDDEN_SIZE * 2
    self.attention = TemporalSelfAttention(hidden, num_heads=4)
    self.fc = nn.Sequential(
      nn.LayerNorm(hidden),
      nn.Linear(hidden, 256),
      nn.GELU(),
      nn.Dropout(config.DROPOUT),
      nn.Linear(256, 128),
      nn.GELU(),
      nn.Dropout(config.DROPOUT / 2),
      nn.Linear(128, num_classes))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.input_norm(x)
    lstm_out, _ = self.lstm(x)
    return self.fc(self.attention(lstm_out))


class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, max_len: int = config.SEQUENCE_LENGTH):
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
      torch.arange(0, d_model, 2).float() *
      (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer("pe", pe.unsqueeze(0))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.pe[:, :x.size(1), :]


class ASLTransformer(nn.Module):

  def __init__(self,
               num_classes: int,
               d_model: int = 128,
               nhead: int = 4,
               num_layers: int = 2,
               dropout: float = 0.3):
    super().__init__()
    self.input_norm = nn.LayerNorm(config.LANDMARK_FEATURES)
    self.input_proj = nn.Linear(config.LANDMARK_FEATURES, d_model)
    self.pos_encoding = PositionalEncoding(d_model)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=256,
      dropout=dropout,
      batch_first=True,
      norm_first=True)
    self.transformer = nn.TransformerEncoder(
      encoder_layer, num_layers=num_layers)
    self.fc = nn.Sequential(
      nn.LayerNorm(d_model),
      nn.Linear(d_model, 128),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(128, num_classes))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.input_norm(x)
    x = self.input_proj(x)
    x = self.pos_encoding(x)
    x = self.transformer(x)
    return self.fc(x.mean(dim=1))


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


def _info(name: str, model_path: Path, class_path: Path,
          kind: str, input_mode: str, uses_mask: bool = False,
          sequence_length: int = config.SEQUENCE_LENGTH) -> Optional[ModelInfo]:
  model_path = project_path(str(model_path))
  class_path = project_path(str(class_path))
  if not model_path.exists() or not class_path.exists():
    return None
  return ModelInfo(
    name=name,
    model_path=model_path,
    class_path=class_path,
    kind=kind,
    input_mode=input_mode,
    uses_mask=uses_mask,
    sequence_length=sequence_length)


def _known_models() -> list[ModelInfo]:
  candidates = [
    _info(
      "wlasl_lstm",
      Path(config.MODEL_SAVE_PATH),
      Path(config.CLASS_LIST_PATH),
      "bilstm",
      "raw"),
    _info(
      "wlasl_lstm_attention",
      Path(config.MODEL_SAVE_PATH.replace("wlasl_lstm",
                                          "wlasl_lstm_attention")),
      Path(config.CLASS_LIST_PATH),
      "attention",
      "raw"),
    _info(
      "wlasl_transformer",
      Path(config.MODEL_SAVE_PATH.replace("wlasl_lstm",
                                          "wlasl_transformer")),
      Path(config.CLASS_LIST_PATH),
      "transformer",
      "raw"),
    _info(
      "landmark_v2",
      Path(config.LANDMARK_V2_MODEL_SAVE_PATH),
      Path(config.LANDMARK_V2_CLASS_LIST_PATH),
      "landmark_v2",
      "normalized",
      uses_mask=True,
      sequence_length=config.LANDMARK_V2_SEQUENCE_LENGTH),
  ]
  return [candidate for candidate in candidates if candidate is not None]


def _infer_model(model_path: Path) -> Optional[ModelInfo]:
  model_root = project_path(config.MODEL_DIR)
  try:
    relative = model_path.relative_to(model_root)
  except ValueError:
    return None

  class_path = project_path(config.CLASS_LIST_PATH)
  stem = model_path.stem.lower()
  if relative.parts[0] == "landmark_v2":
    return _info(
      "landmark_v2",
      model_path,
      Path(config.LANDMARK_V2_CLASS_LIST_PATH),
      "landmark_v2",
      "normalized",
      uses_mask=True,
      sequence_length=config.LANDMARK_V2_SEQUENCE_LENGTH)
  if "attention" in stem:
    return _info(model_path.stem, model_path, class_path, "attention", "raw")
  if "transformer" in stem:
    return _info(model_path.stem, model_path, class_path, "transformer", "raw")
  if stem == "wlasl_lstm":
    return _info(model_path.stem, model_path, class_path, "bilstm", "raw")
  return None


def discover_models() -> list[ModelInfo]:
  model_root = project_path(config.MODEL_DIR)
  if not model_root.exists():
    return []

  infos = _known_models()
  seen_paths = {info.model_path.resolve() for info in infos}
  seen_names = {info.name for info in infos}
  for model_path in sorted(model_root.rglob("*.pth")):
    if model_path.resolve() in seen_paths:
      continue
    info = _infer_model(model_path)
    if info is None or info.name in seen_names:
      continue
    infos.append(info)
    seen_paths.add(info.model_path.resolve())
    seen_names.add(info.name)

  order = {
    "landmark_v2": 0,
    "wlasl_lstm": 1,
    "wlasl_lstm_attention": 2,
    "wlasl_transformer": 3,
  }
  infos.sort(key=lambda info: (order.get(info.name, 99), info.name))
  return infos


def default_model_info() -> Optional[ModelInfo]:
  models = discover_models()
  return models[0] if models else None


def load_classes(model_info: Optional[ModelInfo] = None) -> list[str]:
  if model_info is None:
    model_info = default_model_info()
  if model_info is None or not model_info.class_path.exists():
    return []
  return np.load(model_info.class_path).tolist()


def load_model(classes: list[str],
               device: torch.device,
               model_info: Optional[ModelInfo] = None) -> Optional[nn.Module]:
  if not classes:
    return None
  if model_info is None:
    model_info = default_model_info()
  if model_info is None or not model_info.model_path.exists():
    return None

  if model_info.kind == "landmark_v2":
    model = LandmarkLSTMV2(len(classes)).to(device)
  elif model_info.kind == "attention":
    model = ASLWordLSTMAttention(len(classes)).to(device)
  elif model_info.kind == "transformer":
    model = ASLTransformer(len(classes)).to(device)
  else:
    model = ASLWordLSTM(len(classes)).to(device)

  model.load_state_dict(torch.load(model_info.model_path, map_location=device))
  model.eval()
  return model
