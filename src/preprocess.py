import os
import numpy as np
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURATION ---
POSE_POINTS = 25
HAND_POINTS = 21
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 2

# Efficient JSON loader
try:
  import orjson
  def load_json(path):
    with open(path, "rb") as f:
      return orjson.loads(f.read())
except ImportError:
  import json
  def load_json(path):
    with open(path, 'r') as f:
      return json.load(f)

# --- UTILITIES ---

def extract_xy_conf(keypoints, n_points):
  """Pulls (x, y) coordinates and ignores confidence scores."""
  kp = np.asarray(keypoints, dtype=np.float32).reshape(n_points, 3)
  return kp[:, :2]

def interpolate_nan(sequence):
  """Fills in gaps if a hand disappears for a few frames."""
  T, D = sequence.shape
  t = np.arange(T)
  for d in range(D):
    col = sequence[:, d]
    mask = ~np.isnan(col)
    if mask.sum() < 2:
      continue
    sequence[:, d] = np.interp(t, t[mask], col[mask])
  return sequence

def normalize_frames(frames):
  """Centers the person and scales them so distance from camera doesn't matter."""
  ls = frames[:, LEFT_SHOULDER]
  rs = frames[:, RIGHT_SHOULDER]
  center = (ls + rs) / 2
  scale = np.linalg.norm(ls - rs, axis=1)
  
  # Avoid division by zero
  valid = scale > 1e-6
  frames[valid] = (frames[valid] - center[valid, None, :]) / scale[valid, None, None]
  return frames

# --- THE WORKER (Processing one video/sentence) ---

def preprocess_sentence(sentence_dir):
  # Find all frame files for this specific video
  json_files = sorted(glob(os.path.join(sentence_dir, "*_keypoints.json")))
  frames = []

  for jf in json_files:
    data = load_json(jf)
    if not data.get("people"):
      continue
    person = data["people"][0]

    # Extract only Pose and Hands (ignore Face to save memory)
    pose = extract_xy_conf(person["pose_keypoints_2d"], POSE_POINTS)
    lh   = extract_xy_conf(person["hand_left_keypoints_2d"], HAND_POINTS)
    rh   = extract_xy_conf(person["hand_right_keypoints_2d"], HAND_POINTS)

    # Stack them vertically into one frame
    frame = np.vstack([pose, lh, rh]) 
    frames.append(frame)

  if len(frames) == 0:
    return None

  # Final math cleanup
  frames = np.stack(frames)  # Resulting shape: (Time, Points, XY)
  frames = normalize_frames(frames)
  
  T, K, _ = frames.shape
  sequence = frames.reshape(T, K * 2) # Flatten Points and XY
  sequence = interpolate_nan(sequence)

  return sequence.astype(np.float32)

def worker(sentence_folder_name, root_dir, output_dir):
    """Helper function for the 'Parallel Processing'."""
    sent_path = os.path.join(root_dir, sentence_folder_name)
    seq = preprocess_sentence(sent_path)
    if seq is not None:
      # Save as a single efficient file
      np.save(os.path.join(output_dir, f"{sentence_folder_name}.npy"), seq)
      return seq.shape[0]
    return 0

# --- MAIN RUNNER ---

def preprocess_dataset(root_dir, output_dir, num_workers=4):
  os.makedirs(output_dir, exist_ok=True)

  # Get list of all video folders in the directory
  sentences = [
    d for d in sorted(os.listdir(root_dir))
    if os.path.isdir(os.path.join(root_dir, d))
  ]

  print(f"Found {len(sentences)} folders to process...")

  total_frames = 0
  # ProcessPoolExecutor runs multiple folders at once using all your CPU cores
  with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(worker, s, root_dir, output_dir) for s in sentences]
    for f in tqdm(as_completed(futures), total=len(futures), desc="Shrinking Data"):
      total_frames += f.result()

  print(f"\n[SUCCESS] Processed {len(sentences)} videos into {total_frames} total frames.")

if __name__ == "__main__":
  tasks = [
    ("data/1_raw/test/openpose_output/json", "data/2_processed/test_npy"),
    ("data/1_raw/val/openpose_output/json",  "data/2_processed/val_npy"),
    ("data/1_raw/train/openpose_output/json", "data/2_processed/train_npy"),
  ]

  for input_path, output_path in tasks:
    if os.path.exists(input_path):
      print(f"\n--- STARTING TASK: {input_path} ---")
      preprocess_dataset(input_path, output_path, num_workers=4)
    else:
      print(f"\n[SKIP] Path not found: {input_path} (Is it still downloading?)")

  print("\n[ALL TASKS COMPLETE] Your specimens are ready in data/processed/")