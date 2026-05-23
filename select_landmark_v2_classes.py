from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config
from preprocess_landmarks_v2 import (
  create_hand_detector,
  existing_instances,
  extract_frame_landmarks,
  load_entries,
  sample_indices,
  valid_frame_range,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Select candidate WLASL classes for the Landmark v2 pipeline. This is "
      "an exploration tool; preprocessing uses the fixed class list in "
      "config.py by default."))
  parser.add_argument("--raw-json", default=config.RAW_JSON_PATH)
  parser.add_argument("--video-dir", default=config.RAW_VIDEO_DIR)
  parser.add_argument("--top-k", type=int, default=config.LANDMARK_V2_TOP_K_CLASSES)
  parser.add_argument("--min-local", type=int, default=10)
  parser.add_argument("--min-train", type=int, default=7)
  parser.add_argument("--min-val", type=int, default=1)
  parser.add_argument("--min-test", type=int, default=1)
  parser.add_argument(
    "--selection-mode",
    choices=("quality", "count"),
    default="quality",
    help="Rank classes by landmark quality audit or by local video count.")
  parser.add_argument(
    "--quality-pool-size",
    type=int,
    default=60,
    help="Number of count-ranked candidate classes to landmark-audit.")
  parser.add_argument(
    "--quality-videos-per-split",
    type=int,
    default=2,
    help="Videos sampled per train/val/test split during quality audit.")
  parser.add_argument(
    "--quality-frames",
    type=int,
    default=12,
    help="Frames sampled per video during quality audit.")
  parser.add_argument(
    "--output-file",
    default="MDFile/LandmarkV2SelectedClasses.txt",
    help="Text file that receives the selected class names.")
  parser.add_argument(
    "--metadata-output",
    default="MDFile/LandmarkV2ClassSelection.json",
    help="JSON file that receives scoring details.")
  return parser.parse_args()


def split_counts(instances: list[dict[str, Any]]) -> Counter:
  counts: Counter = Counter()
  for instance in instances:
    counts[str(instance.get("split", "unknown"))] += 1
  return counts


def collect_candidate_rows(entries: list[dict[str, Any]], video_dir: Path,
                           args: argparse.Namespace) -> list[dict[str, Any]]:
  candidates: list[dict[str, Any]] = []
  for entry in entries:
    local_instances = existing_instances(entry, video_dir)
    counts = split_counts(local_instances)
    row = {
      "gloss": str(entry.get("gloss", "")),
      "local": len(local_instances),
      "train": counts["train"],
      "val": counts["val"],
      "test": counts["test"],
      "instances": local_instances,
    }
    if (
        row["local"] >= args.min_local and
        row["train"] >= args.min_train and
        row["val"] >= args.min_val and
        row["test"] >= args.min_test):
      candidates.append(row)
  return candidates


def public_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
  return {key: value for key, value in row.items() if key != "instances"}


def count_score_candidate(row: dict[str, Any]) -> tuple:
  split_floor = min(row["train"], row["val"], row["test"])
  split_balance = row["train"] + (row["val"] * 2) + (row["test"] * 2)
  return row["local"], split_floor, split_balance, row["gloss"]


def quality_score_candidate(row: dict[str, Any]) -> tuple:
  quality = float(row.get("quality_valid_ratio", 0.0))
  split_quality = float(row.get("quality_min_split_valid_ratio", 0.0))
  hand_score = min(float(row.get("quality_avg_detected_hands", 0.0)), 1.5) / 1.5
  local_score = min(float(row["local"]), 18.0) / 18.0
  split_floor = min(row["train"], row["val"], row["test"])
  split_score = min(float(split_floor), 3.0) / 3.0
  balance_score = min(row["val"], 3) + min(row["test"], 3)
  total_score = (
    quality * 5.0 +
    split_quality * 2.0 +
    hand_score +
    local_score +
    split_score +
    balance_score * 0.1)
  return total_score, quality, split_quality, row["local"], row["gloss"]


def quality_sample_instances(instances: list[dict[str, Any]],
                             per_split: int) -> list[dict[str, Any]]:
  selected: list[dict[str, Any]] = []
  used_ids: set[str] = set()
  for split in ("train", "val", "test"):
    split_instances = [
      instance for instance in instances
      if str(instance.get("split", "unknown")) == split
    ]
    for instance in split_instances[:per_split]:
      selected.append(instance)
      used_ids.add(str(instance.get("video_id", "")))

  target_count = per_split * 3
  if len(selected) < target_count:
    for instance in instances:
      video_id = str(instance.get("video_id", ""))
      if video_id in used_ids:
        continue
      selected.append(instance)
      used_ids.add(video_id)
      if len(selected) >= target_count:
        break
  return selected


def estimate_video_quality(video_path: Path, instance: dict[str, Any], hands,
                           use_tasks_api: bool,
                           quality_frames: int) -> tuple[float, float]:
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    return 0.0, 0.0

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
  if total_frames <= 0:
    cap.release()
    return 0.0, 0.0

  start, end = valid_frame_range(instance, total_frames)
  indices = sample_indices(start, end, quality_frames)
  valid_flags: list[bool] = []
  hand_counts: list[int] = []

  for index in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok:
      valid_flags.append(False)
      hand_counts.append(0)
      continue

    _, valid, hand_count = extract_frame_landmarks(frame, hands, use_tasks_api)
    valid_flags.append(valid)
    hand_counts.append(hand_count)

  cap.release()
  if not valid_flags:
    return 0.0, 0.0
  return float(np.mean(valid_flags)), float(np.mean(hand_counts))


def estimate_candidate_quality(row: dict[str, Any], video_dir: Path, hands,
                               use_tasks_api: bool,
                               args: argparse.Namespace) -> dict[str, Any]:
  instances = quality_sample_instances(
    row["instances"],
    args.quality_videos_per_split)
  valid_ratios: list[float] = []
  hand_counts: list[float] = []
  split_ratios: dict[str, list[float]] = defaultdict(list)

  for instance in instances:
    split = str(instance.get("split", "unknown"))
    video_id = str(instance.get("video_id", ""))
    video_path = video_dir / f"{video_id}.mp4"
    valid_ratio, avg_hands = estimate_video_quality(
      video_path,
      instance,
      hands,
      use_tasks_api,
      args.quality_frames)
    valid_ratios.append(valid_ratio)
    hand_counts.append(avg_hands)
    split_ratios[split].append(valid_ratio)

  split_means = {
    split: float(np.mean(values)) if values else 0.0
    for split, values in split_ratios.items()
  }
  required_split_means = [
    split_means.get(split, 0.0)
    for split in ("train", "val", "test")
  ]
  return {
    "quality_sampled_videos": len(instances),
    "quality_valid_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
    "quality_min_split_valid_ratio": min(required_split_means),
    "quality_avg_detected_hands": (
      float(np.mean(hand_counts)) if hand_counts else 0.0),
    "quality_split_valid_ratio": split_means,
  }


def select_by_count(candidates: list[dict[str, Any]],
                    args: argparse.Namespace) -> tuple[list[str],
                                                       dict[str, Any]]:
  ranked = sorted(candidates, key=count_score_candidate, reverse=True)
  selected = ranked[:args.top_k]
  return [row["gloss"] for row in selected], {
    "mode": "count",
    "candidate_count": len(candidates),
    "selected": [public_candidate_row(row) for row in selected],
  }


def select_by_quality(candidates: list[dict[str, Any]], video_dir: Path,
                      args: argparse.Namespace) -> tuple[list[str],
                                                         dict[str, Any]]:
  count_ranked = sorted(candidates, key=count_score_candidate, reverse=True)
  pool_size = max(args.top_k, args.quality_pool_size)
  pool = count_ranked[:pool_size]
  hands, use_tasks_api = create_hand_detector()
  scored: list[dict[str, Any]] = []

  try:
    print(
      f"Quality-auditing {len(pool)} class candidates "
      f"({args.quality_frames} frames/video)")
    for index, row in enumerate(pool, start=1):
      scored_row = dict(row)
      scored_row.update(
        estimate_candidate_quality(row, video_dir, hands, use_tasks_api, args))
      scored.append(scored_row)
      print(
        f"Quality {index:02d}/{len(pool)} {row['gloss']}: "
        f"valid={scored_row['quality_valid_ratio']:.2%}, "
        f"min_split={scored_row['quality_min_split_valid_ratio']:.2%}, "
        f"hands={scored_row['quality_avg_detected_hands']:.2f}")
  finally:
    close = getattr(hands, "close", None)
    if callable(close):
      close()

  selected = sorted(scored, key=quality_score_candidate, reverse=True)[:args.top_k]
  return [row["gloss"] for row in selected], {
    "mode": "quality",
    "candidate_count": len(candidates),
    "pool_size": len(pool),
    "selected": [public_candidate_row(row) for row in selected],
    "audited_pool": [public_candidate_row(row) for row in scored],
  }


def write_outputs(classes: list[str], metadata: dict[str, Any],
                  args: argparse.Namespace) -> None:
  output_file = Path(args.output_file)
  output_file.parent.mkdir(parents=True, exist_ok=True)
  output_file.write_text("\n".join(classes) + "\n", encoding="utf-8")

  metadata_output = Path(args.metadata_output)
  metadata_output.parent.mkdir(parents=True, exist_ok=True)
  metadata_output.write_text(
    json.dumps(
      {
        "classes": classes,
        **metadata,
      },
      indent=2),
    encoding="utf-8")

  print(f"Selected classes ({len(classes)}): {classes}")
  print(f"Saved class list: {output_file}")
  print(f"Saved selection metadata: {metadata_output}")
  print("To preprocess with this list, run:")
  print(
    ".\\.venv\\Scripts\\python.exe preprocess_landmarks_v2.py "
    f"--class-list {output_file}")


def main() -> None:
  args = parse_args()
  entries = load_entries(Path(args.raw_json))
  video_dir = Path(args.video_dir)
  candidates = collect_candidate_rows(entries, video_dir, args)
  if not candidates:
    raise RuntimeError("No class candidates matched the selection criteria.")

  if args.selection_mode == "count":
    classes, metadata = select_by_count(candidates, args)
  else:
    classes, metadata = select_by_quality(candidates, video_dir, args)

  write_outputs(classes, metadata, args)


if __name__ == "__main__":
  main()
