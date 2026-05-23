from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np

import config


DEFAULT_DEMO_WORDS = ("yes", "no", "hello")


@dataclass(frozen=True)
class VideoInfo:
  video_id: str
  exists: bool
  readable: bool
  frame_count: int
  fps: float
  width: int
  height: int

  @property
  def duration_seconds(self) -> float:
    if self.fps <= 0:
      return 0.0
    return self.frame_count / self.fps


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Audit local WLASL data and recommend trainable class sets.")
  parser.add_argument(
    "--raw-json",
    default=config.RAW_JSON_PATH,
    help="Path to WLASL_v0.3.json.")
  parser.add_argument(
    "--video-dir",
    default=config.RAW_VIDEO_DIR,
    help="Directory containing WLASL mp4 videos.")
  parser.add_argument(
    "--output",
    default="MDFile/DatasetSelectionReport.md",
    help="Markdown report output path.")
  parser.add_argument(
    "--top-k",
    type=int,
    default=10,
    help="Number of recommended classes to select.")
  parser.add_argument(
    "--min-local",
    type=int,
    default=10,
    help="Minimum local videos per selected class.")
  parser.add_argument(
    "--min-train",
    type=int,
    default=7,
    help="Minimum local train videos per selected class.")
  parser.add_argument(
    "--min-val",
    type=int,
    default=1,
    help="Minimum local validation videos per selected class.")
  parser.add_argument(
    "--min-test",
    type=int,
    default=1,
    help="Minimum local test videos per selected class.")
  parser.add_argument(
    "--demo-words",
    nargs="*",
    default=list(DEFAULT_DEMO_WORDS),
    help="Words to report separately for demo/business requirements.")
  parser.add_argument(
    "--inspect-videos",
    action="store_true",
    help=(
      "Open local mp4 files with OpenCV to collect frame counts and validate "
      "readability. This is slower but gives richer quality statistics."))
  return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
  with path.open("r", encoding="utf-8") as f:
    data = json.load(f)
  if not isinstance(data, list):
    raise ValueError(f"Expected a list in {path}")
  return data


def inspect_video(video_dir: Path, video_id: str,
                  inspect_metadata: bool) -> VideoInfo:
  path = video_dir / f"{video_id}.mp4"
  if not path.exists():
    return VideoInfo(video_id, False, False, 0, 0.0, 0, 0)

  if not inspect_metadata:
    return VideoInfo(video_id, True, True, 0, 0.0, 0, 0)

  cap = cv2.VideoCapture(str(path))
  readable = bool(cap.isOpened())
  if not readable:
    cap.release()
    return VideoInfo(video_id, True, False, 0, 0.0, 0, 0)

  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
  fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
  cap.release()

  readable = readable and frame_count > 0 and width > 0 and height > 0
  return VideoInfo(video_id, True, readable, frame_count, fps, width, height)


def local_instances_for_entry(
    entry: dict[str, Any],
    video_dir: Path,
    video_cache: dict[str, VideoInfo],
    inspect_metadata: bool) -> list[dict[str, Any]]:
  local: list[dict[str, Any]] = []
  for instance in entry.get("instances", []):
    video_id = str(instance.get("video_id", ""))
    if video_id not in video_cache:
      video_cache[video_id] = inspect_video(video_dir, video_id,
                                            inspect_metadata)
    if video_cache[video_id].exists:
      local.append(instance)
  return local


def split_counts(instances: list[dict[str, Any]]) -> Counter:
  counts: Counter = Counter()
  for instance in instances:
    counts[str(instance.get("split", "unknown"))] += 1
  return counts


def frame_summary(instances: list[dict[str, Any]],
                  video_cache: dict[str, VideoInfo]) -> dict[str, float | int]:
  infos = [
    video_cache[str(instance.get("video_id", ""))]
    for instance in instances
    if str(instance.get("video_id", "")) in video_cache
  ]
  readable = [info for info in infos if info.readable]
  frames = [info.frame_count for info in readable if info.frame_count > 0]
  durations = [
    info.duration_seconds for info in readable if info.duration_seconds > 0
  ]
  return {
    "readable": len(readable),
    "unreadable": len(infos) - len(readable),
    "avg_frames": mean(frames) if frames else 0.0,
    "min_frames": min(frames) if frames else 0,
    "max_frames": max(frames) if frames else 0,
    "avg_duration": mean(durations) if durations else 0.0,
  }


def build_gloss_stats(
    entries: list[dict[str, Any]], video_dir: Path,
    inspect_metadata: bool
) -> tuple[list[dict[str, Any]], dict[str, VideoInfo]]:
  video_cache: dict[str, VideoInfo] = {}
  stats: list[dict[str, Any]] = []

  for entry in entries:
    local_instances = local_instances_for_entry(entry, video_dir, video_cache,
                                                inspect_metadata)
    counts = split_counts(local_instances)
    frames = frame_summary(local_instances, video_cache)
    total_instances = len(entry.get("instances", []))
    stats.append({
      "gloss": str(entry.get("gloss", "")),
      "total_instances": total_instances,
      "local_instances": len(local_instances),
      "missing_instances": total_instances - len(local_instances),
      "train": counts["train"],
      "val": counts["val"],
      "test": counts["test"],
      "other_split": sum(
        count for split, count in counts.items()
        if split not in {"train", "val", "test"}),
      **frames,
    })

  return stats, video_cache


def class_score(row: dict[str, Any]) -> tuple:
  split_floor = min(row["train"], row["val"], row["test"])
  split_balance = row["train"] + (row["val"] * 2) + (row["test"] * 2)
  return (
    int(row["local_instances"]),
    int(row["readable"]),
    split_floor,
    split_balance,
    -int(row["unreadable"]),
    str(row["gloss"]),
  )


def select_classes(stats: list[dict[str, Any]], args: argparse.Namespace
                   ) -> list[dict[str, Any]]:
  candidates = [
    row for row in stats
    if row["local_instances"] >= args.min_local and
    row["train"] >= args.min_train and
    row["val"] >= args.min_val and
    row["test"] >= args.min_test and
    row["readable"] == row["local_instances"]
  ]
  return sorted(candidates, key=class_score, reverse=True)[:args.top_k]


def load_existing_processed_classes() -> list[str]:
  class_path = Path(config.CLASS_LIST_PATH)
  if not class_path.exists():
    return []
  return np.load(class_path).tolist()


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
  lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
  ]
  for row in rows:
    lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
  return lines


def format_float(value: float, digits: int = 1) -> str:
  return f"{value:.{digits}f}"


def count_distribution(stats: list[dict[str, Any]]) -> list[list[Any]]:
  thresholds = [16, 15, 14, 12, 10, 8, 5, 1]
  rows: list[list[Any]] = []
  for threshold in thresholds:
    rows.append([
      f">= {threshold}",
      sum(1 for row in stats if row["local_instances"] >= threshold),
    ])
  rows.append([
    "0",
    sum(1 for row in stats if row["local_instances"] == 0),
  ])
  return rows


def row_for_report(rank: int | None, row: dict[str, Any],
                   include_video_metrics: bool) -> list[Any]:
  frame_metrics: list[Any]
  if include_video_metrics:
    frame_metrics = [
      format_float(float(row["avg_frames"])),
      row["min_frames"],
      row["max_frames"],
    ]
  else:
    frame_metrics = ["not inspected", "not inspected", "not inspected"]

  base = [
    row["gloss"],
    row["local_instances"],
    row["train"],
    row["val"],
    row["test"],
    row["readable"],
    row["unreadable"],
    *frame_metrics,
  ]
  if rank is None:
    return base
  return [rank, *base]


def write_report(
    output: Path,
    raw_json: Path,
    video_dir: Path,
    entries: list[dict[str, Any]],
    stats: list[dict[str, Any]],
    video_cache: dict[str, VideoInfo],
    selected: list[dict[str, Any]],
    args: argparse.Namespace) -> None:
  output.parent.mkdir(parents=True, exist_ok=True)
  processed_classes = load_existing_processed_classes()
  by_gloss = {row["gloss"]: row for row in stats}

  local_instances = sum(row["local_instances"] for row in stats)
  total_instances = sum(row["total_instances"] for row in stats)
  readable_local = sum(row["readable"] for row in stats)
  unreadable_local = sum(row["unreadable"] for row in stats)
  local_split_counts = Counter()
  for row in stats:
    local_split_counts["train"] += row["train"]
    local_split_counts["val"] += row["val"]
    local_split_counts["test"] += row["test"]
    local_split_counts["other"] += row["other_split"]

  top_by_count = sorted(stats, key=class_score, reverse=True)[:25]

  lines: list[str] = []
  lines.append("# Dataset Selection Report")
  lines.append("")
  lines.append(
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  lines.append("")
  lines.append("## Inputs")
  lines.extend(markdown_table(
    ["Item", "Value"],
    [
      ["Raw JSON", raw_json],
      ["Video directory", video_dir],
      ["Selection top_k", args.top_k],
      ["Minimum local videos", args.min_local],
      ["Minimum train videos", args.min_train],
      ["Minimum validation videos", args.min_val],
      ["Minimum test videos", args.min_test],
      ["Video metadata inspected", "yes" if args.inspect_videos else "no"],
    ]))
  lines.append("")
  lines.append("## Dataset Overview")
  lines.extend(markdown_table(
    ["Metric", "Value"],
    [
      ["Gloss entries", len(entries)],
      ["Metadata instances", total_instances],
      ["Local instances with mp4 file", local_instances],
      ["Readable local videos", readable_local],
      ["Unreadable/invalid local videos", unreadable_local],
      ["Unique metadata video ids checked", len(video_cache)],
      ["Maximum local videos in one gloss",
       max(row["local_instances"] for row in stats)],
    ]))
  lines.append("")
  lines.append("## Local Split Distribution")
  lines.extend(markdown_table(
    ["Split", "Local instances"],
    [
      ["train", local_split_counts["train"]],
      ["val", local_split_counts["val"]],
      ["test", local_split_counts["test"]],
      ["other/unknown", local_split_counts["other"]],
    ]))
  lines.append("")
  lines.append("## Local Count Distribution")
  lines.extend(markdown_table(
    ["Local videos per gloss", "Gloss count"],
    count_distribution(stats)))
  lines.append("")
  lines.append("## Recommended Class Set")
  if selected:
    lines.append(
      "These classes satisfy the current thresholds and are ranked by local "
      "sample count, readable video count, and split coverage.")
    lines.append("")
    headers = [
      "Rank", "Gloss", "Local", "Train", "Val", "Test", "Readable",
      "Invalid", "Avg frames", "Min frames", "Max frames"
    ]
    lines.extend(markdown_table(
      headers,
      [row_for_report(rank, row, args.inspect_videos)
       for rank, row in enumerate(selected, start=1)]))
  else:
    lines.append(
      "No classes satisfy the current thresholds. Lower thresholds or add data.")
  lines.append("")
  lines.append("## Demo Word Status")
  demo_rows: list[list[Any]] = []
  for word in args.demo_words:
    row = by_gloss.get(word)
    if row is None:
      demo_rows.append([word, "missing", 0, 0, 0, 0, 0, "n/a"])
      continue
    recommended = "yes" if row in selected else "no"
    demo_rows.append([
      word,
      row["local_instances"],
      row["train"],
      row["val"],
      row["test"],
      row["readable"],
      row["unreadable"],
      recommended,
    ])
  lines.extend(markdown_table(
    [
      "Gloss", "Local", "Train", "Val", "Test", "Readable", "Invalid",
      "In recommended set"
    ],
    demo_rows))
  lines.append("")
  lines.append("## Current Processed Class Check")
  if processed_classes:
    existing_rows = []
    for gloss in processed_classes:
      row = by_gloss.get(gloss)
      if row is None:
        existing_rows.append([gloss, "missing", 0, 0, 0, 0, 0])
      else:
        existing_rows.append([
          gloss,
          row["local_instances"],
          row["train"],
          row["val"],
          row["test"],
          row["readable"],
          row["unreadable"],
        ])
    lines.extend(markdown_table(
      ["Gloss", "Local", "Train", "Val", "Test", "Readable", "Invalid"],
      existing_rows))
  else:
    lines.append("No processed class list was found.")
  lines.append("")
  lines.append("## Top Local Classes")
  headers = [
    "Rank", "Gloss", "Local", "Train", "Val", "Test", "Readable",
    "Invalid", "Avg frames", "Min frames", "Max frames"
  ]
  lines.extend(markdown_table(
    headers,
    [row_for_report(rank, row, args.inspect_videos)
     for rank, row in enumerate(top_by_count, start=1)]))
  lines.append("")
  lines.append("## Notes")
  lines.append("")
  lines.append(
    "- The current local WLASL copy is broad but shallow: the best classes "
    "have only a small number of local videos.")
  lines.append(
    "- Classes with low validation/test counts should not be used for strong "
    "claims about generalization.")
  lines.append(
    "- If demo words are required but have too few videos, collect extra "
    "project-specific recordings instead of forcing them into the main "
    "training set.")
  lines.append(
    "- Re-run this script after adding data or changing selection thresholds.")
  lines.append("")

  output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
  args = parse_args()
  raw_json = Path(args.raw_json)
  video_dir = Path(args.video_dir)
  output = Path(args.output)

  entries = load_dataset(raw_json)
  stats, video_cache = build_gloss_stats(entries, video_dir,
                                         args.inspect_videos)
  selected = select_classes(stats, args)
  write_report(output, raw_json, video_dir, entries, stats, video_cache,
               selected, args)

  print(f"Gloss entries: {len(entries)}")
  print(f"Metadata video ids checked: {len(video_cache)}")
  print(f"Video metadata inspected: {args.inspect_videos}")
  print(f"Recommended classes: {[row['gloss'] for row in selected]}")
  print(f"Report written to: {output}")


if __name__ == "__main__":
  main()
