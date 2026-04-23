import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
import config  # Import your central configuration

# ==========================================
# 1. LOAD CLASSES & DETERMINE SHAPE
# ==========================================
try:
  CLASSES = config.get_classes()
  NUM_CLASSES = len(CLASSES)
  if NUM_CLASSES == 0:
    raise ValueError("Class list is empty.")
  print(f"✅ Loaded {NUM_CLASSES} classes from {config.CLASS_LIST_PATH}")
except Exception as e:
  print(f"❌ Error: Could not load class list. {e}")
  print("Double check your preprocessing script finished correctly.")
  exit()


# ==========================================
# 2. MODEL ARCHITECTURE (Uses config values)
# ==========================================
class ASLWordLSTM(nn.Module):

  def __init__(self, num_classes):
    super(ASLWordLSTM, self).__init__()
    # Parameters pulled directly from config.py
    self.lstm = nn.LSTM(
      input_size=config.LANDMARK_FEATURES,
      hidden_size=config.HIDDEN_SIZE,
      num_layers=config.NUM_LAYERS,
      batch_first=True,
      bidirectional=True,
      dropout=config.DROPOUT)
    self.fc = nn.Sequential(
      nn.Linear(config.HIDDEN_SIZE * 2, 128), nn.ReLU(),
      nn.Dropout(config.DROPOUT), nn.Linear(128, num_classes))

  def forward(self, x):
    # x shape: (batch, sequence_length, features)
    lstm_out, _ = self.lstm(x)
    # Take the hidden state of the last time step
    last_step = lstm_out[:, -1, :]
    return self.fc(last_step)


# ==========================================
# 3. TRAINING LOGIC
# ==========================================
def train():
  # 1. Load Data from processed paths in config
  if not os.path.exists(config.DATA_PATH):
    print(f"❌ Data file not found at {config.DATA_PATH}")
    return

  X = np.load(config.DATA_PATH)
  Y = np.load(config.LABEL_PATH)

  X_tensor = torch.FloatTensor(X)
  Y_tensor = torch.LongTensor(Y)

  # 2. Dataset Splitting
  full_dataset = TensorDataset(X_tensor, Y_tensor)
  train_size = int(0.8 * len(full_dataset))
  val_size = len(full_dataset) - train_size
  train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size])

  train_loader = DataLoader(
    train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(
    val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

  # 3. Initialize Model, Loss, and Optimizer
  model = ASLWordLSTM(NUM_CLASSES).to(config.DEVICE)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

  # Metrics history for plotting
  history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

  print(
    f"🚀 Starting training on {config.DEVICE} for {config.EPOCHS} epochs...")

  for epoch in range(config.EPOCHS):
    model.train()
    train_loss, train_correct = 0, 0

    for batch_x, batch_y in train_loader:
      # Add random noise (0.005 is a good start)
      noise = torch.randn_like(batch_x) * 0.005
      batch_x = batch_x + noise

      batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)

      optimizer.zero_grad()
      outputs = model(batch_x)
      loss = criterion(outputs, batch_y)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      _, pred = torch.max(outputs, 1)
      train_correct += (pred == batch_y).sum().item()

    # 4. Validation Phase
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

    # 5. Metrics Recording
    t_loss = train_loss / len(train_loader)
    v_loss = val_loss / len(val_loader)
    t_acc = train_correct / train_size
    v_acc = val_correct / val_size

    history['train_loss'].append(t_loss)
    history['val_loss'].append(v_loss)
    history['train_acc'].append(t_acc)
    history['val_acc'].append(v_acc)

    print(
      f"Epoch [{epoch+1}/{config.EPOCHS}] | Loss: {t_loss:.4f} | Acc: {t_acc:.2%} | Val Acc: {v_acc:.2%}"
    )

  # 6. Final Save & Plotting
  os.makedirs(config.MODEL_DIR, exist_ok=True)
  torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
  print(f"✅ Training Complete. Model saved to {config.MODEL_SAVE_PATH}")

  # Create Performance Visuals
  plt.figure(figsize=(12, 5))

  plt.subplot(1, 2, 1)
  plt.plot(history['train_loss'], label='Train Loss', color='blue')
  plt.plot(history['val_loss'], label='Val Loss', color='red')
  plt.title('Loss History')
  plt.xlabel('Epoch')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(history['train_acc'], label='Train Acc', color='blue')
  plt.plot(history['val_acc'], label='Val Acc', color='green')
  plt.title('Accuracy History')
  plt.xlabel('Epoch')
  plt.legend()

  plt.tight_layout()
  plt.savefig(config.PERFORMANCE_PLOT_PATH)
  print(f"📊 Performance plot saved to {config.PERFORMANCE_PLOT_PATH}")
  plt.show()


if __name__ == "__main__":
  train()
