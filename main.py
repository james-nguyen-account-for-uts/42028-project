import argparse
import os

import torch

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def get_device():
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


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
    help="Pick a random file from the processed test folder.")
  parser.add_argument(
    "--encoder-arch",
    type=str,
    default="bilstm",
    choices=["bilstm", "unilstm"],
    help="Encoder baseline to use during training.")
  parser.add_argument(
    "--epochs",
    type=int,
    help="Override the default epoch count for training runs.")
  parser.add_argument(
    "--run-name",
    type=str,
    help="Optional name for the training run directory.")
  parser.add_argument(
    "--val-subset-size",
    type=int,
    default=200,
    help="Number of validation samples to score during training.")
  args = parser.parse_args()

  device = get_device()
  model_path = "models/checkpoints/best_sign_model.pth"
  vocab_path = "models/vocab.json"
  test_data_dir = "data/2_processed/test_npy"

  if args.mode == "train":
    print("Starting training process...")
    from src.step2_train import train
    train.run_training(
      encoder_arch=args.encoder_arch,
      epochs=args.epochs,
      run_name=args.run_name,
      val_subset_size=args.val_subset_size)

  # Webcam/test entry points are still intentionally disabled until the
  # validation and inference scripts are brought back in sync with training.
  _ = device, model_path, vocab_path, test_data_dir


if __name__ == "__main__":
  main()
