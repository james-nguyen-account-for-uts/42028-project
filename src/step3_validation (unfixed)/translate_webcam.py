import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

# --- THE M3 DIRECT IMPORT FIX ---
# Instead of mp.solutions.holistic, we go to the source:

from mediapipe.tasks.python.vision import holistic_landmarker as mp_holistic

from mediapipe.tasks.python.vision import drawing_utils as mp_drawing

# --------------------------------

from src.preprocess import normalize_frames, resample_sequence, interpolate_nan
from src.vocab import SignVocab
from src.model import Seq2Seq, Encoder, Decoder


class RealTimeTranslator:

  def __init__(self, model, vocab, device):
    self.model = model
    self.vocab = vocab
    self.device = device
    self.model.eval()

    self.recording_buffer = []

    # Initialize Holistic using the direct import
    self.holistic_engine = mp_holistic.Holistic(
      static_image_mode=False,
      model_complexity=1,
      smooth_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5)

  def extract_landmarks(self, results):
    """Extracts 134 features: 25 Pose (x,y), 21 L-Hand (x,y), 21 R-Hand (x,y)"""
    if results.pose_landmarks:
      # First 25 landmarks capture head, shoulders, and arms to get 134 features
      pose = np.array(
        [[res.x, res.y]
         for res in results.pose_landmarks.landmark[:25]]).flatten()
    else:
      pose = np.zeros(50)

    lh = np.array(
      [[res.x, res.y] for res in results.left_hand_landmarks.landmark
       ]).flatten() if results.left_hand_landmarks else np.zeros(42)
    rh = np.array(
      [[res.x, res.y] for res in results.right_hand_landmarks.landmark
       ]).flatten() if results.right_hand_landmarks else np.zeros(42)

    return np.concatenate([pose, lh, rh])

  def predict(self):
    if len(self.recording_buffer) < 10:
      return "Hold 'S' longer to sign!"

    raw_data = np.array(self.recording_buffer)
    T = raw_data.shape[0]
    sequence = raw_data.reshape(T, -1, 2)

    # Apply normalization and resampling
    sequence = normalize_frames(sequence)
    sequence = sequence.reshape(T, -1)
    sequence = interpolate_nan(sequence)
    sequence = resample_sequence(sequence, target_len=60)

    src = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

    with torch.no_grad():
      _, (hidden, cell) = self.model.encoder(src)
      input_token = torch.LongTensor([1]).to(self.device)  # <SOS>
      predicted_sentence = []

      for _ in range(20):
        output, hidden, cell = self.model.decoder(input_token, hidden, cell)
        idx = output.argmax(1).item()
        if idx == 2: break  # <EOS>

        word = self.vocab.itos.get(idx, self.vocab.itos.get(str(idx), "<UNK>"))
        if word not in ["<SOS>", "<PAD>", "<EOS>"]:
          predicted_sentence.append(word)
        input_token = torch.LongTensor([idx]).to(self.device)

    return " ".join(
      predicted_sentence) if predicted_sentence else "[No translation]"

  def run(self):
    cap = cv2.VideoCapture(0)
    print("🚀 Webcam Live. [S] Start/Stop Record | [C] Clear | [Q] Quit")

    recording = False
    prediction = ""

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret: break

      frame = cv2.flip(frame, 1)
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = self.holistic_engine.process(rgb_frame)

      # Use direct mp_drawing import for visualization
      if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
          frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
      if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
          frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

      if recording:
        self.recording_buffer.append(self.extract_landmarks(results))
        cv2.circle(frame, (30, 40), 10, (0, 0, 255), -1)

      # UI Overlays
      cv2.putText(
        frame, f"STATUS: {'RECORDING' if recording else 'IDLE'}", (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
      cv2.rectangle(
        frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]),
        (0, 0, 0), -1)
      cv2.putText(
        frame, f"PRED: {prediction}", (20, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

      cv2.imshow('ASL Real-Time Translator', frame)

      key = cv2.waitKey(1) & 0xFF
      if key == ord('s'):
        if recording:
          prediction = self.predict()
          recording = False
        else:
          self.recording_buffer = []
          prediction = "Listening..."
          recording = True
      elif key == ord('c'):
        self.recording_buffer = []
        prediction = ""
      elif key == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

  vocab = SignVocab()
  vocab.load("models/vocab.json")

  # Matching your 1024 HID / 3 LAYER setup
  enc = Encoder(134, 256, 1024, 3, 0.4)
  dec = Decoder(len(vocab), 256, 1024, 3, 0.15)
  model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

  model.load_state_dict(
    torch.load("models/checkpoints/best_sign_model.pth", map_location=DEVICE))

  app = RealTimeTranslator(model, vocab, DEVICE)
  app.run()
