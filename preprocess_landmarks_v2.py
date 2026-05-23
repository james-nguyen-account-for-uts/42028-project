from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

import config


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Extract position/scale-normalized MediaPipe hand landmarks for LSTM "
      "training. The classifier is still trained from scratch."))
  parser.add_argument("--raw-json", default=config.RAW_JSON_PATH)
  parser.add_argument("--video-dir", default=config.RAW_VIDEO_DIR)
  parser.add_argument("--output-dir", default=config.LANDMARK_V2_DATA_DIR)
  parser.add_argument(
    "--classes",
    nargs="*",
    default=None,
    help="Override the default fixed Landmark v2 class list.")
  parser.add_argument(
    "--class-list",
    default=None,
    help="Optional text or JSON file containing class names to preprocess.")
  parser.add_argument(
    "--sequence-length",
    type=int,
    default=config.LANDMARK_V2_SEQUENCE_LENGTH)
  parser.add_argument(
    "--min-valid-frame-ratio",
    type=float,
    default=config.LANDMARK_V2_MIN_VALID_FRAME_RATIO)
  return parser.parse_args()


def create_hand_detector():
  if hasattr(mp, "solutions"):
    hands = mp.solutions.hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.5)
    return hands, False

  from mediapipe.tasks import python
  from mediapipe.tasks.python import vision

  task_path = Path(config.HAND_LANDMARKER_TASK_PATH)
  if not task_path.exists():
    raise FileNotFoundError(f"Missing MediaPipe task model: {task_path}")
  options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(task_path.resolve())),
    num_hands=2,
    min_hand_detection_confidence=0.5)
  hands = vision.HandLandmarker.create_from_options(options)
  return hands, True


def load_entries(path: Path) -> list[dict[str, Any]]:
  with path.open("r", encoding="utf-8") as f:
    data = json.load(f)
  if not isinstance(data, list):
    raise ValueError(f"Expected a list in {path}")
  return data


def existing_instances(entry: dict[str, Any],
                       video_dir: Path) -> list[dict[str, Any]]:
  instances: list[dict[str, Any]] = []
  for instance in entry.get("instances", []):
    video_id = str(instance.get("video_id", ""))
    if (video_dir / f"{video_id}.mp4").exists():
      instances.append(instance)
  return instances


def load_class_list(path: Path) -> list[str]:
  text = path.read_text(encoding="utf-8").strip()
  if not text:
    return []
  if path.suffix.lower() == ".json":
    payload = json.loads(text)
    if not isinstance(payload, list):
      raise ValueError(f"Expected a JSON list in {path}")
    return [str(item) for item in payload]

  classes: list[str] = []
  for line in text.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    classes.extend(part.strip() for part in line.split(",") if part.strip())
  return list(dict.fromkeys(classes))


def select_classes(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
  if args.classes:
    classes = list(dict.fromkeys(args.classes))
    return classes, {"mode": "manual_cli", "selected": classes}

  if args.class_list:
    path = Path(args.class_list)
    classes = load_class_list(path)
    return classes, {
      "mode": "manual_file",
      "path": str(path),
      "selected": classes,
    }

  classes = list(config.LANDMARK_V2_DEFAULT_CLASSES)
  return classes, {
    "mode": "default_config",
    "source": "config.LANDMARK_V2_DEFAULT_CLASSES",
    "selected": classes,
  }


def valid_frame_range(instance: dict[str, Any],
                      total_frames: int) -> tuple[int, int]:
  if total_frames <= 0:
    return 0, 0

  frame_start = int(instance.get("frame_start", 1) or 1)
  frame_end = int(instance.get("frame_end", -1) or -1)
  start = max(frame_start - 1, 0)
  end = frame_end if frame_end > 0 else total_frames

  if start < total_frames and start < end <= total_frames:
    return start, end
  return 0, total_frames


def sample_indices(start: int, end: int, sequence_length: int) -> np.ndarray:
  if end <= start:
    return np.zeros(sequence_length, dtype=np.int32)
  return np.linspace(start, end - 1, sequence_length).round().astype(np.int32)


def normalize_hand(points: np.ndarray) -> np.ndarray:
  wrist = points[0].copy()
  centered = points - wrist
  xy_distances = np.linalg.norm(centered[:, :2], axis=1)
  scale = float(np.max(xy_distances))
  if scale < 1e-6:
    scale = 1.0
  normalized = centered / scale
  return normalized.astype(np.float32).flatten()


def extract_frame_landmarks(frame: np.ndarray, hands, use_tasks_api: bool
                            ) -> tuple[np.ndarray, bool, int]:
  img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  if use_tasks_api:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = hands.detect(mp_image)
    detected_hands = results.hand_landmarks
  else:
    results = hands.process(img_rgb)
    detected_hands = results.multi_hand_landmarks

  frame_features = np.zeros(config.LANDMARK_FEATURES, dtype=np.float32)
  if not detected_hands:
    return frame_features, False, 0

  hand_points: list[tuple[float, np.ndarray]] = []
  for hand_lms in detected_hands[:2]:
    landmarks = hand_lms if use_tasks_api else hand_lms.landmark
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks],
                      dtype=np.float32)
    hand_points.append((float(points[0, 0]), normalize_hand(points)))

  hand_points.sort(key=lambda item: item[0])
  for hand_index, (_, flattened) in enumerate(hand_points):
    start = hand_index * 63
    frame_features[start:start + 63] = flattened

  return frame_features, True, len(hand_points)


def extract_sequence(video_path: Path, instance: dict[str, Any], hands,
                     use_tasks_api: bool, sequence_length: int
                     ) -> tuple[np.ndarray | None, np.ndarray | None,
                                dict[str, Any]]:
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    return None, None, {"reason": "unreadable"}

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
  if total_frames <= 0:
    cap.release()
    return None, None, {"reason": "empty_video"}

  start, end = valid_frame_range(instance, total_frames)
  indices = sample_indices(start, end, sequence_length)
  sequence: list[np.ndarray] = []
  mask: list[bool] = []
  hand_counts: list[int] = []

  for index in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok:
      sequence.append(np.zeros(config.LANDMARK_FEATURES, dtype=np.float32))
      mask.append(False)
      hand_counts.append(0)
      continue

    features, valid, hand_count = extract_frame_landmarks(
      frame, hands, use_tasks_api)
    sequence.append(features)
    mask.append(valid)
    hand_counts.append(hand_count)

  cap.release()
  if not sequence:
    return None, None, {"reason": "no_frames_read"}

  metadata = {
    "frame_count": total_frames,
    "sample_start": int(start),
    "sample_end": int(end),
    "sampled_indices": [int(index) for index in indices],
    "valid_frames": int(sum(mask)),
    "valid_ratio": float(sum(mask) / len(mask)),
    "avg_detected_hands": float(np.mean(hand_counts)) if hand_counts else 0.0,
  }
  return np.stack(sequence, axis=0), np.array(mask, dtype=bool), metadata


def collect_target_instances(entries: list[dict[str, Any]], video_dir: Path,
                             classes: list[str]) -> list[dict[str, Any]]:
  class_set = set(classes)
  rows: list[dict[str, Any]] = []
  for entry in entries:
    gloss = str(entry.get("gloss", ""))
    if gloss not in class_set:
      continue
    for instance in existing_instances(entry, video_dir):
      rows.append({"gloss": gloss, "instance": instance})
  return rows


def split_indices(splits: np.ndarray) -> dict[str, np.ndarray]:
  return {
    split: np.where(splits == split)[0].astype(np.int64)
    for split in ("train", "val", "test")
  }


def save_outputs(output_dir: Path, X: np.ndarray, y: np.ndarray,
                 masks: np.ndarray, splits: np.ndarray, classes: list[str],
                 metadata: list[dict[str, Any]], skipped: list[dict[str, Any]],
                 selection_info: dict[str, Any],
                 args: argparse.Namespace) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  data_path = output_dir / Path(config.LANDMARK_V2_DATA_PATH).name
  label_path = output_dir / Path(config.LANDMARK_V2_LABEL_PATH).name
  mask_path = output_dir / Path(config.LANDMARK_V2_MASK_PATH).name
  split_path = output_dir / Path(config.LANDMARK_V2_SPLIT_PATH).name
  class_path = output_dir / Path(config.LANDMARK_V2_CLASS_LIST_PATH).name
  split_indices_path = output_dir / Path(
    config.LANDMARK_V2_SPLIT_INDICES_PATH).name
  metadata_path = output_dir / Path(config.LANDMARK_V2_METADATA_PATH).name

  np.save(data_path, X)
  np.save(label_path, y)
  np.save(mask_path, masks)
  np.save(split_path, splits)
  np.save(class_path, np.array(classes))
  np.savez(split_indices_path, **split_indices(splits))

  payload = {
    "classes": classes,
    "class_to_label": {name: index for index, name in enumerate(classes)},
    "shape": list(X.shape),
    "dtype": str(X.dtype),
    "sequence_length": args.sequence_length,
    "min_valid_frame_ratio": args.min_valid_frame_ratio,
    "selection": selection_info,
    "split_counts": {
      split: int((splits == split).sum())
      for split in ("train", "val", "test")
    },
    "samples": metadata,
    "skipped": skipped,
  }
  metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

  print(f"Saved landmarks: {data_path}")
  print(f"Saved labels: {label_path}")
  print(f"Saved masks: {mask_path}")
  print(f"Saved splits: {split_path}")
  print(f"Saved classes: {class_path}")
  print(f"Saved split indices: {split_indices_path}")
  print(f"Saved metadata: {metadata_path}")


def main() -> None:
  args = parse_args()
  raw_json = Path(args.raw_json)
  video_dir = Path(args.video_dir)
  output_dir = Path(args.output_dir)

  entries = load_entries(raw_json)
  classes, selection_info = select_classes(args)
  hands, use_tasks_api = create_hand_detector()
  class_to_label = {name: index for index, name in enumerate(classes)}
  rows = collect_target_instances(entries, video_dir, classes)

  print(f"Selected classes ({len(classes)}): {classes}")
  print(f"Target local videos: {len(rows)}")
  print("Landmark normalization: per hand wrist-centered and scale-normalized")

  X: list[np.ndarray] = []
  y: list[int] = []
  masks: list[np.ndarray] = []
  splits: list[str] = []
  metadata: list[dict[str, Any]] = []
  skipped: list[dict[str, Any]] = []

  for index, row in enumerate(rows, start=1):
    gloss = row["gloss"]
    instance = row["instance"]
    video_id = str(instance.get("video_id", ""))
    split = str(instance.get("split", "unknown"))
    video_path = video_dir / f"{video_id}.mp4"

    sequence, mask, info = extract_sequence(
      video_path, instance, hands, use_tasks_api, args.sequence_length)
    if sequence is None or mask is None:
      skipped.append({"gloss": gloss, "video_id": video_id, **info})
      continue

    valid_ratio = float(mask.mean())
    if valid_ratio < args.min_valid_frame_ratio:
      skipped.append({
        "gloss": gloss,
        "video_id": video_id,
        "split": split,
        "reason": "low_valid_frame_ratio",
        **info,
      })
      continue

    X.append(sequence)
    y.append(class_to_label[gloss])
    masks.append(mask)
    splits.append(split)
    metadata.append({
      "gloss": gloss,
      "label": class_to_label[gloss],
      "video_id": video_id,
      "split": split,
      "bbox": instance.get("bbox", []),
      "signer_id": instance.get("signer_id"),
      "source": instance.get("source"),
      **info,
    })

    if index % 25 == 0 or index == len(rows):
      print(f"Processed {index}/{len(rows)} videos")

  close = getattr(hands, "close", None)
  if callable(close):
    close()

  if not X:
    raise RuntimeError("No valid landmark sequences were extracted.")

  X_array = np.stack(X, axis=0).astype(np.float32)
  y_array = np.array(y, dtype=np.int64)
  mask_array = np.stack(masks, axis=0).astype(bool)
  splits_array = np.array(splits)

  save_outputs(output_dir, X_array, y_array, mask_array, splits_array,
               classes, metadata, skipped, selection_info, args)

  print(f"Final X shape: {X_array.shape}, dtype={X_array.dtype}")
  print(f"Final mask shape: {mask_array.shape}")
  print(f"Split counts: {Counter(splits)}")
  print(f"Mean valid frame ratio: {float(mask_array.mean()):.2%}")
  print(f"Skipped videos: {len(skipped)}")


if __name__ == "__main__":
  main()
