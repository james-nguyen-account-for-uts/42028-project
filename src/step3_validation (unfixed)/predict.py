import torch
import json
import numpy as np
from src.model import Seq2Seq, Encoder, Decoder  # Ensure these match your class names

# --- Configuration (Must match your latest training run) ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_PATH = "models/checkpoints/best_sign_model.pth"
VOCAB_PATH = "models/vocab.json"

INPUT_DIM = 134   # Total skeletal features
HID_DIM = 256     # The "New" hidden dim we switched to
ENC_LAYERS = 2
DEC_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

def load_inference_model():
  # 1. Load Vocabulary
  with open(VOCAB_PATH, 'r') as f:
      vocab = json.load(f)
  
  # Invert vocab for ID -> Word lookup
  id_to_word = {int(v): k for k, v in vocab.items()}
  output_dim = len(vocab)
  
  # 2. Initialize Model Architecture
  enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_DROPOUT)
  dec = Decoder(output_dim, HID_DIM, DEC_LAYERS, DEC_DROPOUT)
  model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
  
  # 3. Load Trained Weights
  model.load_state_dict(torch.load(CHECKPOINT_PATH, map_map=DEVICE))
  model.eval()
  
  return model, vocab, id_to_word

def predict(model, npy_path, vocab, id_to_word, max_len=20):
  # Load and prepare the skeletal data
  data = np.load(npy_path)
  src_tensor = torch.FloatTensor(data).unsqueeze(1).to(DEVICE) # [Len, Batch=1, Feat]
  
  with torch.no_grad():
    # Get hidden/cell states from Encoder
    hidden, cell = model.encoder(src_tensor)
    
    # Initial input to Decoder is <sos> token (usually ID 2)
    input_token = torch.LongTensor([vocab.get("<sos>", 2)]).to(DEVICE)
    
    translated_sentence = []
    
    for _ in range(max_len):
      output, hidden, cell = model.decoder(input_token, hidden, cell)
      
      # Get the word with the highest probability
      top_token = output.argmax(1)
      predicted_word = id_to_word.get(top_token.item(), "<UNK>")
      
      if predicted_word == "<EOS>":
          break
          
      translated_sentence.append(predicted_word)
      input_token = top_token # Feed current prediction as next input
          
  return " ".join(translated_sentence)

if __name__ == "__main__":
  print(f"Loading model from {CHECKPOINT_PATH}...")
  model, vocab, id_to_word = load_inference_model()
  
  # Test on a specific validation file
  # Replace this path with an actual .npy file in your data folder
  test_file = "data/1_raw/val_npy/some_sample_file.npy" 
  
  try:
    translation = predict(model, test_file, vocab, id_to_word)
    print(f"\n🎥 Source File: {test_file}")
    print(f"📝 Predicted Translation: {translation}")
  except FileNotFoundError:
    print(f"Error: Could not find {test_file}. Please check the path!")