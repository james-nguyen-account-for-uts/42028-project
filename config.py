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

# MediaPipe landmark v2 outputs. These keep the trained classifier from
# scratch, but make the coordinate representation more position/scale robust.
LANDMARK_V2_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'landmark_v2')
LANDMARK_V2_DATA_PATH = os.path.join(LANDMARK_V2_DATA_DIR,
                                    'wlasl_landmarks_v2.npy')
LANDMARK_V2_LABEL_PATH = os.path.join(LANDMARK_V2_DATA_DIR,
                                     'wlasl_landmark_labels_v2.npy')
LANDMARK_V2_MASK_PATH = os.path.join(LANDMARK_V2_DATA_DIR,
                                    'wlasl_landmark_masks_v2.npy')
LANDMARK_V2_SPLIT_PATH = os.path.join(LANDMARK_V2_DATA_DIR,
                                     'wlasl_landmark_splits_v2.npy')
LANDMARK_V2_CLASS_LIST_PATH = os.path.join(LANDMARK_V2_DATA_DIR,
                                          'wlasl_landmark_class_list_v2.npy')
LANDMARK_V2_SPLIT_INDICES_PATH = os.path.join(
  LANDMARK_V2_DATA_DIR, 'wlasl_landmark_split_indices_v2.npz')
LANDMARK_V2_METADATA_PATH = os.path.join(LANDMARK_V2_DATA_DIR,
                                        'wlasl_landmark_metadata_v2.json')

MODEL_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'wlasl_lstm.pth')
PERFORMANCE_PLOT_PATH = os.path.join(MODEL_DIR, 'training_performance.png')
HAND_LANDMARKER_TASK_PATH = os.path.join(MODEL_DIR, 'hand_landmarker.task')
LANDMARK_V2_MODEL_DIR = os.path.join(MODEL_DIR, 'landmark_v2')
LANDMARK_V2_MODEL_SAVE_PATH = os.path.join(LANDMARK_V2_MODEL_DIR,
                                          'wlasl_landmark_lstm_v2.pth')
LANDMARK_V2_PERFORMANCE_PLOT_PATH = os.path.join(
  LANDMARK_V2_MODEL_DIR, 'landmark_v2_training_performance.png')
LANDMARK_V2_CONFUSION_MATRIX_PATH = os.path.join(
  LANDMARK_V2_MODEL_DIR, 'landmark_v2_confusion_matrix.png')
LANDMARK_V2_METRICS_PATH = os.path.join(LANDMARK_V2_MODEL_DIR,
                                       'landmark_v2_training_metrics.json')

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================
SEQUENCE_LENGTH = 45  # Number of frames per "specimen" (video)
LANDMARK_FEATURES = 126  # 21 points * 3 (xyz) * 2 hands
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.7

LANDMARK_V2_SEQUENCE_LENGTH = 45
LANDMARK_V2_TOP_K_CLASSES = 30
LANDMARK_V2_MIN_VALID_FRAME_RATIO = 0.35
LANDMARK_V2_DEFAULT_CLASSES = [
  'yes',
  'no',
  'hello',
  'thank you',
  'please',
  'go',
  'thin',
  'drink',
  'goodbye',
  'help',
]

# ==========================================
# TRAINING SETTINGS
# ==========================================
DEVICE = torch.device(
  "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.
  is_available() else "cpu")
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
EPOCHS = 200
MIN_SAMPLES_PER_WORD = 10  # Filter out words with fewer than this many videos

RANDOM_SEED = 42
LANDMARK_V2_BATCH_SIZE = 8
LANDMARK_V2_LEARNING_RATE = 0.001
LANDMARK_V2_WEIGHT_DECAY = 0.001
LANDMARK_V2_LABEL_SMOOTHING = 0.05
LANDMARK_V2_EPOCHS = 150
LANDMARK_V2_MIN_EPOCHS = 30
LANDMARK_V2_EARLY_STOPPING_PATIENCE = 25


# ==========================================
# UTILS
# ==========================================
def get_classes():
  """Helper to load classes in any script."""
  try:
    return np.load(CLASS_LIST_PATH).tolist()
  except:
    return []
