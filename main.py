import os

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
# import random
# import glob
import argparse
import torch
# import numpy as np
# import pandas as pd
# from src.step2_train.vocab import SignVocab
# from src.step2_train.model import Seq2Seq, Encoder, Decoder
# from src.translate_webcam import RealTimeTranslator
# from src.inference import SignTranslator

# --- HYPERPARAMETERS (Must match your training run) ---
INPUT_DIM = 134
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 1024
N_LAYERS = 3
ENC_DROPOUT = 0.4
DEC_DROPOUT = 0.15

# def load_system(model_path, vocab_path, device):
#   """Helper to initialize the model and vocab together."""
#   vocab = SignVocab()
#   vocab.load(vocab_path)
#   print(f"📊 Vocab loaded. Size: {len(vocab)}")

#   enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
#   dec = Decoder(len(vocab), DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
#   model = Seq2Seq(enc, dec, device).to(device)

#   if os.path.exists(model_path):
#     print(f"📦 Loading weights from {model_path}...")
#     model.load_state_dict(torch.load(model_path, map_location=device))
#   else:
#     print(f"⚠️ Warning: Model weights not found at {model_path}")

#   return model, vocab

# def get_ground_truth(
#     file_path, csv_path="data/0_metadata/how2sign_realigned_test.csv"):
#   try:
#     # 1. Load the CSV (How2Sign is tab-separated)
#     df = pd.read_csv(csv_path, sep='\t')

#     # 2. Clean the filename to match the CSV "SENTENCE_NAME"
#     # Example: _G0RrDVpOZ4_11-5-rgb_front.npy -> _G0RrDVpOZ4_11-5
#     file_id = os.path.basename(file_path).replace(".npy", "")
#     # Remove common suffixes that aren't in the metadata ID
#     clean_id = file_id.replace("-rgb_front", "").replace("_rgb_front", "")

#     # 3. Search using 'SENTENCE_NAME'
#     match = df[df['SENTENCE_NAME'] == clean_id]

#     if not match.empty:
#       return match.iloc[0]['SENTENCE']

#     # 4. Final Fallback: Search for the ID anywhere in the column
#     partial = df[df['SENTENCE_NAME'].str.contains(clean_id, na=False)]
#     if not partial.empty:
#       return partial.iloc[0]['SENTENCE']

#     return f"ID [{clean_id}] not found in CSV (Check 'SENTENCE_NAME' column)"
#   except Exception as e:
#     return f"Metadata Error: {e}"


def main():
  parser = argparse.ArgumentParser(description="Sign-to-Text Command Center")
  parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["train", "webcam", "test"],
    help="Mode to run: train, webcam, or test")
  parser.add_argument(
    "--file", type=str, help="Path to .npy file (only for --mode test)")
  parser.add_argument(
    "--random",
    action="store_true",
    help="Pick a random file from test_npy folder")
  args = parser.parse_args()

  DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
  MODEL_PATH = "models/checkpoints/best_sign_model.pth"
  VOCAB_PATH = "models/vocab.json"
  TEST_DATA_DIR = "data/2_processed/test_npy"

  # --- TRAIN FUNCTION ---
  if args.mode == "train":
    print("🚀 Starting Training Process...")
    from src.step2_train import train
    train.run_training()

  # elif args.mode == "webcam":
  #   print("📷 Launching Real-Time Translator...")
  #   model, vocab = load_system(MODEL_PATH, VOCAB_PATH, DEVICE)
  #   app = RealTimeTranslator(model, vocab, DEVICE)
  #   app.run()

  # elif args.mode == "test":
  #   target_file = args.file

  #   if args.random:
  #     # Look for all .npy files in your processed test folder
  #     files = glob.glob(os.path.join(TEST_DATA_DIR, "*.npy"))
  #     if not files:
  #       print(f"❌ Error: No .npy files found in {TEST_DATA_DIR}")
  #       return
  #     target_file = random.choice(files)
  #     print(f"🎲 Randomly selected: {os.path.basename(target_file)}")

  #   if not target_file:
  #     print("❌ Error: Please provide a file path using --file [path]")
  #     return

  #   print(f"🔍 Testing file: {target_file}")
  #   model, vocab = load_system(MODEL_PATH, VOCAB_PATH, DEVICE)
  #   translator = SignTranslator(model, vocab, DEVICE)

  #   sequence = np.load(target_file)
  #   result, confidence = translator.translate_sequence(sequence)
  #   print("-" * 30)
  #   print(f"✨ Predicted Sentence: {result if result else '[BLANK]'}")
  #   print(f"🎯 Confidence: {confidence:.4f}%")
  #   ground_truth = get_ground_truth(target_file)
  #   print(f"📖 Real Sentence: {ground_truth}")
  #   print("-" * 30)


if __name__ == "__main__":
  main()

#   How to use your new Command Center:

#     To run your Webcam Demo:
#     python3 main.py --mode webcam

#     To test a specific processed file:
#     python3 main.py --mode test --file data/2_processed/test_npy/g1HvmBOR7Y4_3-3-rgb_front.npy

#     To test a random processed file:
#     python3 main.py --mode test --random

#     To start a fresh training run (if you want to restart later):
#     python3 main.py --mode train
