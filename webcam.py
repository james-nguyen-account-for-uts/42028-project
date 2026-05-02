import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from collections import deque
import os
import config

# Settings match train.py
DEVICE = config.DEVICE
SEQUENCE_LENGTH = config.SEQUENCE_LENGTH
MODEL_PATH = config.MODEL_SAVE_PATH
CLASS_LIST_PATH = config.CLASS_LIST_PATH

# Load Classes
try:
  CLASSES = np.load(CLASS_LIST_PATH).tolist()
  NUM_CLASSES = len(CLASSES)
except:
  print("❌ Error: Class list not found.")
  exit()


class ASLWordLSTM(nn.Module):

  def __init__(self, num_classes):
    super(ASLWordLSTM, self).__init__()
    self.lstm = nn.LSTM(
      input_size=config.LANDMARK_FEATURES,
      hidden_size=config.HIDDEN_SIZE,
      num_layers=config.NUM_LAYERS,
      batch_first=True,
      bidirectional=True,
      dropout=config.DROPOUT)
    self.fc = nn.Sequential(
      nn.Linear(config.HIDDEN_SIZE * 2, 128), nn.ReLU(),
      nn.Dropout(config.DROPOUT),
      nn.Linear(128, num_classes))

  def forward(self, x):
    lstm_out, _ = self.lstm(x)
    last_step = lstm_out[:, -1, :]
    return self.fc(last_step)


if hasattr(mp, 'solutions'):
  mp_hands = mp.solutions.hands
  hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)
  USE_TASKS_API = False
else:
  from mediapipe.tasks import python
  from mediapipe.tasks.python import vision

  if not os.path.exists(config.HAND_LANDMARKER_TASK_PATH):
    raise FileNotFoundError(
      f"Missing MediaPipe task model: {config.HAND_LANDMARKER_TASK_PATH}")
  options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
      model_asset_path=os.path.abspath(config.HAND_LANDMARKER_TASK_PATH)),
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5)
  hands = vision.HandLandmarker.create_from_options(options)
  USE_TASKS_API = True


def extract_landmarks(frame):
  img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  if USE_TASKS_API:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = hands.detect(mp_image)
    detected_hands = results.hand_landmarks
  else:
    results = hands.process(img_rgb)
    detected_hands = results.multi_hand_landmarks

  frame_lms = np.zeros(config.LANDMARK_FEATURES)
  if detected_hands:
    for i, hand_lms in enumerate(detected_hands):
      if i > 1: break
      landmarks = hand_lms if USE_TASKS_API else hand_lms.landmark
      points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
      frame_lms[i * 63:i * 63 + len(points)] = points
  return frame_lms


def run():
  model = ASLWordLSTM(NUM_CLASSES).to(DEVICE)
  if not os.path.exists(MODEL_PATH):
    print("❌ Trained model not found. Run train.py first.")
    return

  model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
  model.eval()

  cap = cv2.VideoCapture(0)
  sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
  current_word = "..."

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    lms = extract_landmarks(frame)
    sequence_buffer.append(lms)

    if len(sequence_buffer) == SEQUENCE_LENGTH:
      input_data = torch.FloatTensor(
        np.array(sequence_buffer)).unsqueeze(0).to(DEVICE)
      with torch.no_grad():
        outputs = model(input_data)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

        if confidence.item() > 0.65:
          current_word = CLASSES[predicted.item()]
        else:
          current_word = "..."

    cv2.rectangle(frame, (0, 0), (500, 80), (255, 255, 255), -1)
    cv2.putText(
      frame, f"WORD: {current_word.upper()}", (20, 55),
      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.imshow('WLASL Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break
  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  run()
