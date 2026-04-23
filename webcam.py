import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from collections import deque
import os

# Settings match train.py
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEQUENCE_LENGTH = 30
MODEL_PATH = 'models/wlasl_lstm.pth'
CLASS_LIST_PATH = 'data/2_processed/wlasl_class_list.npy'

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
      input_size=126,
      hidden_size=256,
      num_layers=3,
      batch_first=True,
      bidirectional=True,
      dropout=0.5)
    self.fc = nn.Sequential(
      nn.Linear(256 * 2, 128), nn.ReLU(), nn.Dropout(0.5),
      nn.Linear(128, num_classes))

  def forward(self, x):
    lstm_out, _ = self.lstm(x)
    last_step = lstm_out[:, -1, :]
    return self.fc(last_step)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
  min_detection_confidence=0.7, min_tracking_confidence=0.5)


def extract_landmarks(frame):
  results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  frame_lms = np.zeros(126)
  if results.multi_hand_landmarks:
    for i, hand_lms in enumerate(results.multi_hand_landmarks):
      if i > 1: break
      points = np.array([[lm.x, lm.y, lm.z]
                         for lm in hand_lms.landmark]).flatten()
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
