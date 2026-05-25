from __future__ import annotations

import platform
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch

from paths import project_path

import config


@dataclass
class Prediction:
  word: str
  confidence: float
  emitted: bool
  status: str


class HandLandmarkExtractor:

  def __init__(self):
    self.use_tasks_api = not hasattr(mp, "solutions")
    if self.use_tasks_api:
      from mediapipe.tasks import python
      from mediapipe.tasks.python import vision

      task_path = project_path(config.HAND_LANDMARKER_TASK_PATH)
      if not task_path.exists():
        raise FileNotFoundError(
          f"Missing MediaPipe task model: {task_path}")
      options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(task_path)),
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5)
      self.hands = vision.HandLandmarker.create_from_options(options)
    else:
      self.hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5)

  def extract(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if self.use_tasks_api:
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
      results = self.hands.detect(mp_image)
      detected_hands = results.hand_landmarks
    else:
      results = self.hands.process(img_rgb)
      detected_hands = results.multi_hand_landmarks

    frame_lms = np.zeros(config.LANDMARK_FEATURES)
    if not detected_hands:
      return frame_lms, False

    for i, hand_lms in enumerate(detected_hands):
      if i > 1:
        break
      landmarks = hand_lms if self.use_tasks_api else hand_lms.landmark
      points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
      start_idx = i * 63
      frame_lms[start_idx:start_idx + len(points)] = points
    return frame_lms, True

  def close(self) -> None:
    close = getattr(self.hands, "close", None)
    if callable(close):
      close()


class CameraController:
  CONFIDENCE_THRESHOLD = 0.5
  PREDICT_EVERY_FRAMES = 4
  WORD_COOLDOWN_SECONDS = 1.2
  REPEAT_COOLDOWN_SECONDS = 2.0

  def __init__(self, model, classes: list[str], device: torch.device):
    self.model = model
    self.classes = classes
    self.device = device
    self.extractor = HandLandmarkExtractor()
    self.cap: Optional[cv2.VideoCapture] = None
    self.sequence_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    self.frame_count = 0
    self.last_output_word = ""
    self.last_output_at = 0.0

  def start(self) -> bool:
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
    self.cap = cv2.VideoCapture(0, backend)
    if not self.cap.isOpened():
      self.cap.release()
      self.cap = None
      return False

    self.sequence_buffer.clear()
    self.frame_count = 0
    return True

  def stop(self) -> None:
    if self.cap is not None:
      self.cap.release()
      self.cap = None

  def read(self) -> Optional[np.ndarray]:
    if self.cap is None:
      return None
    ok, frame = self.cap.read()
    if not ok:
      return None
    return cv2.flip(frame, 1)

  def process_frame(self, frame: np.ndarray) -> Optional[Prediction]:
    landmarks, valid = self.extractor.extract(frame)
    if not valid:
      self.sequence_buffer.clear()
      self.frame_count = 0
      return Prediction(
        word="--",
        confidence=0.0,
        emitted=False,
        status="no_hand")

    self.sequence_buffer.append(landmarks)
    self.frame_count += 1

    if len(self.sequence_buffer) != config.SEQUENCE_LENGTH:
      return None
    if self.frame_count % self.PREDICT_EVERY_FRAMES != 0:
      return None

    input_data = torch.FloatTensor(
      np.array(self.sequence_buffer)).unsqueeze(0).to(self.device)
    with torch.no_grad():
      outputs = self.model(input_data)
      probs = torch.nn.functional.softmax(outputs, dim=1)
      confidence, predicted = torch.max(probs, 1)

    confidence_value = float(confidence.item())
    word = self.classes[int(predicted.item())]
    emitted, status = self._emit_status(word, confidence_value)
    return Prediction(
      word=word,
      confidence=confidence_value,
      emitted=emitted,
      status=status)

  def _emit_status(self, word: str, confidence: float) -> tuple[bool, str]:
    if confidence < self.CONFIDENCE_THRESHOLD:
      return False, "low_confidence"

    now = time.time()
    word_too_soon = now - self.last_output_at < self.WORD_COOLDOWN_SECONDS
    if word_too_soon:
      return False, "cooldown"

    repeated_too_soon = (
      word == self.last_output_word and
      now - self.last_output_at < self.REPEAT_COOLDOWN_SECONDS)
    if repeated_too_soon:
      return False, "cooldown"

    self.last_output_word = word
    self.last_output_at = now
    return True, "emitted"

  def reset_output_state(self) -> None:
    self.last_output_word = ""
    self.last_output_at = 0.0

  def close(self) -> None:
    self.stop()
    self.extractor.close()
