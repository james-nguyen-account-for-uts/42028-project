import os
import numpy as np
from glob import glob
from scipy.interpolate import interp1d
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from config import HAND_POINTS, POSE_POINTS, TARGET_FRAMES, FILE_PATHS
from utils import extract_xy_conf, interpolate_nan, resample_sequence, normalize_frames

# --- JSON loader ---
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


# --- PREPROCESSING WORKERS ---
def preprocess_sentence(sentence_dir):
  # Find all frame files for this specific video
  json_files = sorted(glob(os.path.join(sentence_dir, "*_keypoints.json")))
  if not json_files:
    return None

  frames = []
  for jf in json_files:
    data = load_json(jf)
    if not data.get("people"):
      continue
    person = data["people"][0]

    # Extract only Pose and Hands (ignore Face to save memory)
    pose = extract_xy_conf(person["pose_keypoints_2d"], POSE_POINTS)
    lh = extract_xy_conf(person["hand_left_keypoints_2d"], HAND_POINTS)
    rh = extract_xy_conf(person["hand_right_keypoints_2d"], HAND_POINTS)

    # Stack them vertically into one frame
    frame = np.vstack([pose, lh, rh])
    frames.append(frame)

  if len(frames) < 5:
    return None

  # Spatial clean (center and scale)
  frames = np.stack(frames)
  frames = normalize_frames(frames)

  # Flatten for RNN
  T, K, _ = frames.shape
  sequence = frames.reshape(T, K * 2)

  # Temporal clean (Fill gaps)
  sequence = interpolate_nan(sequence)

  # Standardize length
  sequence = resample_sequence(sequence, TARGET_FRAMES)

  return sequence.astype(np.float32)


def worker(sentence_folder_name, root_dir, output_dir):
  """Helper function for the 'Parallel Processing', then save to .npy file"""
  sent_path = os.path.join(root_dir, sentence_folder_name)
  seq = preprocess_sentence(sent_path)
  if seq is not None:
    np.save(os.path.join(output_dir, f"{sentence_folder_name}.npy"), seq)
    return seq.shape[0]
  return 0


def _run_with_executor(executor_cls, sentences, root_dir, output_dir, max_workers):
  total_frames = 0
  with executor_cls(max_workers=max_workers) as executor:
    futures = [
      executor.submit(worker, s, root_dir, output_dir) for s in sentences
    ]
    for f in tqdm(as_completed(futures), total=len(futures)):
      total_frames += f.result()
  return total_frames


def _run_sequential(sentences, root_dir, output_dir):
  total_frames = 0
  for sentence in tqdm(sentences):
    total_frames += worker(sentence, root_dir, output_dir)
  return total_frames


# --- MAIN RUNNER ---
def preprocess_dataset(root_dir, output_dir, num_workers=4):
  os.makedirs(output_dir, exist_ok=True)

  sentences = [
    d for d in sorted(os.listdir(root_dir))
    if os.path.isdir(os.path.join(root_dir, d))
  ]
  print(f"Found {len(sentences)} folders to process...")

  if num_workers <= 1:
    total_frames = _run_sequential(sentences, root_dir, output_dir)
  else:
    try:
      # ProcessPoolExecutor runs multiple folders at once using all CPU cores.
      total_frames = _run_with_executor(
        ProcessPoolExecutor, sentences, root_dir, output_dir, num_workers)
    except (PermissionError, OSError) as e:
      print(
        f"[WARN] ProcessPoolExecutor unavailable ({e}). Falling back to threads."
      )
      try:
        total_frames = _run_with_executor(
          ThreadPoolExecutor, sentences, root_dir, output_dir, num_workers)
      except Exception as thread_error:
        print(
          f"[WARN] ThreadPoolExecutor failed ({thread_error}). Falling back to sequential mode."
        )
        total_frames = _run_sequential(sentences, root_dir, output_dir)

  print(
    f"\n[SUCCESS] Processed {len(sentences)} videos into {total_frames} total frames."
  )


if __name__ == "__main__":
  for input_path, output_path in FILE_PATHS:
    if os.path.exists(input_path):
      print(f"\n--- STARTING PREPROCESSING TASK: {input_path} ---")
      preprocess_dataset(input_path, output_path, num_workers=4)
    else:
      print(f"\n[SKIPPED] Path not found: {input_path}")

  print("\nYour data is ready in data/2_processed/")
