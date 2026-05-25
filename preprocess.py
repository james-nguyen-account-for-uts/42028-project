import os
import json
import cv2
import numpy as np
import mediapipe as mp
import config  # Import our central config file

# ==========================================
# INITIALIZE MEDIAPIPE
# ==========================================
if hasattr(mp, 'solutions'):
  mp_hands = mp.solutions.hands
  hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
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
    min_hand_detection_confidence=0.5)
  hands = vision.HandLandmarker.create_from_options(options)
  USE_TASKS_API = True


def extract_landmarks(video_path):
  cap = cv2.VideoCapture(video_path)
  sequence = []

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

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
        start_idx = i * 63
        frame_lms[start_idx:start_idx + len(points)] = points

    sequence.append(frame_lms)

  cap.release()

  if len(sequence) == 0:
    return None

  # Standardize length using the config value
  if len(sequence) < config.SEQUENCE_LENGTH:
    padding = [np.zeros(config.LANDMARK_FEATURES)
               ] * (config.SEQUENCE_LENGTH - len(sequence))
    sequence.extend(padding)
  else:
    sequence = sequence[:config.SEQUENCE_LENGTH]

  return np.array(sequence)


def main():
  # Ensure output directory exists
  os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

  with open(config.RAW_JSON_PATH, 'r') as f:
    dataset_json = json.load(f)

  # ==========================================
  # 1. DYNAMIC DATA MINING (Forcing Yes/No)
  # ==========================================
  REQUIRED_WORDS = ['yes', 'no', 'hello']
  other_word_counts = []
  required_entries = []

  for entry in dataset_json:
    gloss = entry['gloss']
    # Count how many of this word's instances actually exist in your folder
    existing_instances = [
      inst for inst in entry['instances'] if os.path.exists(
        os.path.join(config.RAW_VIDEO_DIR, f"{inst['video_id']}.mp4"))
    ]

    # If it's Yes or No, we save it immediately
    if gloss in REQUIRED_WORDS:
      required_entries.append(
        {
          'gloss': gloss,
          'count': len(existing_instances),
          'instances': existing_instances
        })
    # Otherwise, we put it in the pool for the top 8
    elif len(existing_instances) >= 10:
      other_word_counts.append(
        {
          'gloss': gloss,
          'count': len(existing_instances),
          'instances': existing_instances
        })

  # Sort the pool by popularity and take the top 8 to fill the 10-word limit
  other_word_counts.sort(key=lambda x: x['count'], reverse=True)
  target_entries = required_entries + other_word_counts[:7]

  # Sort alphabetically so the indexes stay consistent
  target_entries.sort(key=lambda x: x['gloss'])

  all_glosses = [e['gloss'] for e in target_entries]
  word_to_int = {word: i for i, word in enumerate(all_glosses)}

  # Save class list
  np.save(config.CLASS_LIST_PATH, np.array(all_glosses))
  print(f"📊 Final 10 words (Forced Hello/Yes/No + Top 7 Data-Rich):")
  for e in target_entries:
    print(f"   - {e['gloss']}: {e['count']} videos found")

  X, Y = [], []

  # ==========================================
  # 2. PROCESS THE DATA
  # ==========================================
  for entry in target_entries:
    gloss = entry['gloss']
    print(f"\nProcessing Landmarks: {gloss}")

    for instance in entry['instances']:
      video_path = os.path.join(
        config.RAW_VIDEO_DIR, f"{instance['video_id']}.mp4")
      landmarks = extract_landmarks(video_path)
      if landmarks is not None:
        X.append(landmarks)
        Y.append(word_to_int[gloss])

  # ==========================================
  # 3. SAVE
  # ==========================================
  if len(X) > 0:
    X = np.array(X)
    Y = np.array(Y)
    np.save(config.DATA_PATH, X)
    np.save(config.LABEL_PATH, Y)
    print(f"\n✅ Preprocessing complete!")
    print(f"Total sequences captured: {len(X)}")
  else:
    print("❌ No videos found. Check config.RAW_VIDEO_DIR!")


if __name__ == "__main__":
  main()
