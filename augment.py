import numpy as np
import os
import config


def augment_time_stretch(sequence, rate=0.8):
  """模拟签手速度变化：随机抽取帧重新采样到45帧"""
  n_frames = sequence.shape[0]
  new_length = int(n_frames * rate)
  indices = np.linspace(0, n_frames - 1, new_length).astype(int)
  stretched = sequence[indices]
  if len(stretched) < config.SEQUENCE_LENGTH:
    pad = np.zeros((config.SEQUENCE_LENGTH - len(stretched), config.LANDMARK_FEATURES))
    stretched = np.vstack([stretched, pad])
  return stretched[:config.SEQUENCE_LENGTH]


def augment_noise(sequence, sigma=0.01):
  """给关键点坐标加轻微抖动，模拟手部颤动"""
  noise = np.random.normal(0, sigma, sequence.shape)
  return sequence + noise


def augment_mirror(sequence):
  """左右镜像：把x坐标翻转，模拟左手签名者"""
  mirrored = sequence.copy()
  for i in range(0, config.LANDMARK_FEATURES, 3):
    mirrored[:, i] = 1.0 - mirrored[:, i]
  return mirrored


def augment_speed_up(sequence, rate=1.25):
  """签得更快：跳帧采样"""
  n_frames = sequence.shape[0]
  new_length = int(n_frames * rate)
  indices = np.linspace(0, n_frames - 1, new_length).astype(int)
  sped_up = sequence[indices]
  return sped_up[:config.SEQUENCE_LENGTH]


def main():
  print("📂 Loading original data...")
  X = np.load(config.DATA_PATH)
  Y = np.load(config.LABEL_PATH)
  n_total = len(X)
  print(f"   Original: {n_total} sequences, {len(np.unique(Y))} classes")

  # ==========================================
  # 先切分，再增强（防止数据泄漏）
  # ==========================================
  indices = np.random.permutation(n_total)
  train_size = int(0.8 * n_total)

  train_idx = indices[:train_size]
  val_idx   = indices[train_size:]

  X_train, Y_train = X[train_idx], Y[train_idx]
  X_val,   Y_val   = X[val_idx],   Y[val_idx]

  print(f"   Split → train: {len(X_train)}, val: {len(X_val)}")

  # ==========================================
  # 只对训练集做增强，验证集保持原始不动
  # ==========================================
  print("🔄 Augmenting training set only...")

  X_aug = [X_train]
  Y_aug = [Y_train]

  X_noisy = np.array([augment_noise(seq) for seq in X_train])
  X_aug.append(X_noisy)
  Y_aug.append(Y_train)
  print(f"   ✅ Noise augmentation:   +{len(X_noisy)} sequences")

  X_slow = np.array([augment_time_stretch(seq, rate=0.8) for seq in X_train])
  X_aug.append(X_slow)
  Y_aug.append(Y_train)
  print(f"   ✅ Slow stretch (0.8x):  +{len(X_slow)} sequences")

  X_mirror = np.array([augment_mirror(seq) for seq in X_train])
  X_aug.append(X_mirror)
  Y_aug.append(Y_train)
  print(f"   ✅ Mirror flip:          +{len(X_mirror)} sequences")

  X_train_aug = np.vstack(X_aug)
  Y_train_aug = np.concatenate(Y_aug)

  # 打乱训练集
  shuf = np.random.permutation(len(X_train_aug))
  X_train_aug = X_train_aug[shuf]
  Y_train_aug = Y_train_aug[shuf]

  # ==========================================
  # 保存：训练集和验证集分开存
  # ==========================================
  np.save(config.DATA_PATH,  X_train_aug)
  np.save(config.LABEL_PATH, Y_train_aug)

  VAL_DATA_PATH  = config.DATA_PATH.replace('landmarks_data', 'val_landmarks_data')
  VAL_LABEL_PATH = config.LABEL_PATH.replace('labels_data',   'val_labels_data')
  np.save(VAL_DATA_PATH,  X_val)
  np.save(VAL_LABEL_PATH, Y_val)

  print(f"\n✅ Done — no data leakage!")
  print(f"   Train (augmented): {len(X_train_aug)} sequences")
  print(f"   Val   (original):  {len(X_val)} sequences")
  print(f"   Saved train → {config.DATA_PATH}")
  print(f"   Saved val   → {VAL_DATA_PATH}")

  # 每类样本数统计
  classes = np.load(config.CLASS_LIST_PATH).tolist()
  print(f"\n📊 Train samples per class:")
  for i, cls in enumerate(classes):
    print(f"   {cls}: {np.sum(Y_train_aug == i)}")


if __name__ == "__main__":
  main()