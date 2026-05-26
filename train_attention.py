import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import config

# ==========================================
# 1. LOAD CLASSES
# ==========================================
try:
  CLASSES = config.get_classes()
  NUM_CLASSES = len(CLASSES)
  if NUM_CLASSES == 0:
    raise ValueError("Class list is empty.")
  print(f"✅ Loaded {NUM_CLASSES} classes from {config.CLASS_LIST_PATH}")
except Exception as e:
  print(f"❌ Error: {e}")
  exit()


# ==========================================
# 2. MULTI-HEAD SELF-ATTENTION MODULE
# ==========================================
class TemporalSelfAttention(nn.Module):
  """
  Multi-head self-attention over the time dimension.
  More expressive than a single linear attention layer.
  """

  def __init__(self, hidden_size, num_heads=4):
    super(TemporalSelfAttention, self).__init__()
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

    self.q_proj = nn.Linear(hidden_size, hidden_size)
    self.k_proj = nn.Linear(hidden_size, hidden_size)
    self.v_proj = nn.Linear(hidden_size, hidden_size)
    self.out_proj = nn.Linear(hidden_size, hidden_size)
    self.scale = self.head_dim ** -0.5

  def forward(self, lstm_out):
    # lstm_out: (batch, seq_len, hidden_size)
    B, T, H = lstm_out.shape

    Q = self.q_proj(lstm_out).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    K = self.k_proj(lstm_out).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    V = self.v_proj(lstm_out).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, heads, T, T)
    weights = torch.softmax(scores, dim=-1)                      # (B, heads, T, T)

    attended = torch.matmul(weights, V)                          # (B, heads, T, head_dim)
    attended = attended.transpose(1, 2).contiguous().view(B, T, H)
    attended = self.out_proj(attended)

    # Aggregate over time using mean-pooling of attended output
    context = attended.mean(dim=1)                               # (B, H)
    avg_weights = weights.mean(dim=1).mean(dim=1)                # (B, T) for visualisation
    return context, avg_weights


# ==========================================
# 3. ATTENTION MODEL
# ==========================================
class ASLWordLSTMAttention(nn.Module):

  def __init__(self, num_classes):
    super(ASLWordLSTMAttention, self).__init__()
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

  def forward(self, x, return_weights=False):
    x = self.input_norm(x)
    lstm_out, _ = self.lstm(x)                         # (batch, 45, hidden*2)
    context, weights = self.attention(lstm_out)         # (batch, hidden*2)
    out = self.fc(context)
    if return_weights:
      return out, weights
    return out


# ==========================================
# 4. DATA AUGMENTATION
# ==========================================
def augment(batch_x):
  """Apply random augmentations to a batch of landmark sequences."""
  # Gaussian noise
  batch_x = batch_x + torch.randn_like(batch_x) * 0.01

  B, T, _ = batch_x.shape

  # Random temporal scaling: stretch/compress by ±10%
  if torch.rand(1).item() > 0.5:
    scale = 0.9 + torch.rand(1).item() * 0.2  # [0.9, 1.1]
    new_T = max(2, int(T * scale))
    batch_x = torch.nn.functional.interpolate(
      batch_x.permute(0, 2, 1),
      size=new_T, mode='linear', align_corners=False)
    batch_x = torch.nn.functional.interpolate(
      batch_x, size=T, mode='linear', align_corners=False)
    batch_x = batch_x.permute(0, 2, 1)

  # Random spatial scaling (simulate different signing sizes)
  scale_factor = 0.9 + torch.rand(B, 1, 1).to(batch_x.device) * 0.2
  batch_x = batch_x * scale_factor

  return batch_x


# ==========================================
# 5. TRAINING
# ==========================================
def train():
  if not os.path.exists(config.DATA_PATH):
    print(f"❌ Data not found at {config.DATA_PATH}")
    return

  X = np.load(config.DATA_PATH)
  Y = np.load(config.LABEL_PATH)

  VAL_DATA_PATH  = config.DATA_PATH.replace('landmarks_data', 'val_landmarks_data')
  VAL_LABEL_PATH = config.LABEL_PATH.replace('labels_data', 'val_labels_data')
  X_val = np.load(VAL_DATA_PATH)
  Y_val = np.load(VAL_LABEL_PATH)

  train_dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(Y))
  val_dataset   = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(Y_val))
  train_size    = len(train_dataset)
  val_size      = len(val_dataset)

  train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,  pin_memory=True)
  val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)

  model     = ASLWordLSTMAttention(NUM_CLASSES).to(config.DEVICE)
  # Label smoothing reduces overconfidence
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
  # Cosine annealing: LR gradually drops to near 0, then restarts
  scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6)

  history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

  best_val_acc     = 0.0
  best_model_state = None

  print(f"🚀 Attention model training on {config.DEVICE} for {config.EPOCHS} epochs...\n")

  for epoch in range(config.EPOCHS):
    model.train()
    train_loss, train_correct = 0, 0

    for batch_x, batch_y in train_loader:
      batch_x = augment(batch_x)
      batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)

      optimizer.zero_grad()
      outputs = model(batch_x)
      loss = criterion(outputs, batch_y)
      loss.backward()
      # Gradient clipping prevents LSTM gradient explosion
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
      optimizer.step()

      train_loss += loss.item()
      _, pred = torch.max(outputs, 1)
      train_correct += (pred == batch_y).sum().item()

    scheduler.step()

    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
      for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        val_correct += (pred == batch_y).sum().item()

    t_loss = train_loss / len(train_loader)
    v_loss = val_loss   / len(val_loader)
    t_acc  = train_correct / train_size
    v_acc  = val_correct   / val_size
    cur_lr = optimizer.param_groups[0]['lr']

    history['train_loss'].append(t_loss)
    history['val_loss'].append(v_loss)
    history['train_acc'].append(t_acc)
    history['val_acc'].append(v_acc)

    print(f"Epoch [{epoch+1}/{config.EPOCHS}] | LR: {cur_lr:.6f} | Loss: {t_loss:.4f} | Acc: {t_acc:.2%} | Val Acc: {v_acc:.2%}", end="")

    # Save best by val accuracy (more meaningful than val loss)
    if v_acc > best_val_acc:
      best_val_acc     = v_acc
      best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
      print(f"  ✅ best val acc → {best_val_acc:.2%}")
    else:
      print()

  # Save best model
  os.makedirs(config.MODEL_DIR, exist_ok=True)
  ATTENTION_MODEL_PATH = config.MODEL_SAVE_PATH.replace('wlasl_lstm', 'wlasl_lstm_attention')
  if best_model_state:
    model.load_state_dict(best_model_state)
  torch.save(model.state_dict(), ATTENTION_MODEL_PATH)
  print(f"\n✅ Attention model saved to {ATTENTION_MODEL_PATH}")

  # ==========================================
  # 6. ATTENTION WEIGHT VISUALISATION
  # ==========================================
  model.eval()
  sample_x = torch.FloatTensor(X_val[:1]).to(config.DEVICE)
  with torch.no_grad():
    _, attn_weights = model(sample_x, return_weights=True)

  attn_weights = attn_weights.squeeze().cpu().numpy()

  # ==========================================
  # 7. PLOT
  # ==========================================
  actual_epochs = len(history['train_loss'])
  epoch_range   = range(1, actual_epochs + 1)

  _, axes = plt.subplots(1, 3, figsize=(18, 5))

  axes[0].plot(epoch_range, history['train_loss'], label='Train Loss', color='blue')
  axes[0].plot(epoch_range, history['val_loss'],   label='Val Loss',   color='red')
  axes[0].set_title('Loss History (Attention Model)')
  axes[0].set_xlabel('Epoch')
  axes[0].legend()

  axes[1].plot(epoch_range, history['train_acc'], label='Train Acc', color='blue')
  axes[1].plot(epoch_range, history['val_acc'],   label='Val Acc',   color='green')
  axes[1].set_title('Accuracy History (Attention Model)')
  axes[1].set_xlabel('Epoch')
  axes[1].legend()

  axes[2].bar(range(len(attn_weights)), attn_weights, color='steelblue')
  axes[2].set_title('Attention Weights — Sample Frame Importance')
  axes[2].set_xlabel('Frame index (0–44)')
  axes[2].set_ylabel('Attention weight')

  plt.tight_layout()
  ATTENTION_PLOT_PATH = config.PERFORMANCE_PLOT_PATH.replace(
    'training_performance', 'training_performance_attention')
  plt.savefig(ATTENTION_PLOT_PATH)
  print(f"📊 Plot saved to {ATTENTION_PLOT_PATH}")
  plt.show()


if __name__ == "__main__":
  train()
