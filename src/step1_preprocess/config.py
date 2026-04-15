# --- FILE PATHS ---

TRAIN_FILE_PATHS = (
  "data/1_raw/train/openpose_output/json", "data/2_processed/train_npy")

VAL_FILE_PATHS = (
  "data/1_raw/val/openpose_output/json", "data/2_processed/val_npy")

TEST_FILE_PATHS = (
  "data/1_raw/test/openpose_output/json", "data/2_processed/test_npy")

FILE_PATHS = [TRAIN_FILE_PATHS, VAL_FILE_PATHS, TEST_FILE_PATHS]

VOCAB_PATH = "models/vocab.json"

# --- CONFIGURATION ---

TARGET_FRAMES = 60
TARGET_SHAPE = (60, 134)

LEFT_SHOULDER_INDEX = 5
RIGHT_SHOULDER_INDEX = 2
HAND_POINTS = 21
POSE_POINTS = 25
