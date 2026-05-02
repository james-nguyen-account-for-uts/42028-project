import torch
import numpy as np
import os

# ==========================================
# PATHS
# ==========================================
# Where your data lives
RAW_VIDEO_DIR = 'data/1_raw/videos'
RAW_JSON_PATH = 'data/1_raw/WLASL_v0.3.json'

# Where your processed landmarks and model live
PROCESSED_DATA_DIR = 'data/2_processed'
DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'wlasl_landmarks_data.npy')
LABEL_PATH = os.path.join(PROCESSED_DATA_DIR, 'wlasl_labels_data.npy')
CLASS_LIST_PATH = os.path.join(PROCESSED_DATA_DIR, 'wlasl_class_list.npy')

MODEL_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'wlasl_lstm.pth')
PERFORMANCE_PLOT_PATH = os.path.join(MODEL_DIR, 'training_performance.png')
HAND_LANDMARKER_TASK_PATH = os.path.join(MODEL_DIR, 'hand_landmarker.task')

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================
SEQUENCE_LENGTH = 45  # Number of frames per "specimen" (video)
LANDMARK_FEATURES = 126  # 21 points * 3 (xyz) * 2 hands
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.7

# ==========================================
# TRAINING SETTINGS
# ==========================================
DEVICE = torch.device(
  "cuda" if torch.cuda.is_available()
  else "mps" if torch.backends.mps.is_available()
  else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
MIN_SAMPLES_PER_WORD = 10  # Filter out words with fewer than this many videos


# ==========================================
# UTILS
# ==========================================
def get_classes():
  """Helper to load classes in any script."""
  try:
    return np.load(CLASS_LIST_PATH).tolist()
  except:
    return []
