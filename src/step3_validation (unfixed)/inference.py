import torch
import numpy as np
import os
# Ensure these imports match your project structure
from src.preprocess import normalize_frames, interpolate_nan, resample_sequence
from src.vocab import SignVocab
from src.model import Seq2Seq, Encoder, Decoder 

class SignTranslator:
  def __init__(self, model, vocab, device):
    self.model = model
    self.vocab = vocab
    self.device = device
    self.model.eval()

  def translate_sequence(self, sequence, max_len=20):
    """
    Fixed Translation: Adds a Repetition Penalty to stop the 'and and and' loop.
    """
    # 1. Preprocessing Pipeline
    T = sequence.shape[0]
    # Ensure landmarks are treated as (Time, Landmarks, XY)
    frames = sequence.reshape(T, -1, 2)
    frames = normalize_frames(frames)
    sequence = frames.reshape(T, -1)
    sequence = interpolate_nan(sequence)
    sequence = resample_sequence(sequence, target_len=60)
    
    # 2. Prepare Tensor: [Batch=1, Time=60, Feat=134]
    src = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
    
    # 3. Encoder Pass
    with torch.no_grad():
      # Get the context vector from the encoder
      outputs, (hidden, cell) = self.model.encoder(src)
      
      # Starting token <SOS> is usually index 1
      input_token = torch.LongTensor([1]).to(self.device)
      predicted_indices = []
      total_probs = []

      print(f"\n--- 🔍 Top-5 Candidates Per Step ---")
      
      # 4. Decoder Loop
      for step in range(max_len):
        output, hidden, cell = self.model.decoder(input_token, hidden, cell)
        
        if output.dim() == 1:
          output = output.unsqueeze(0)

        # Calculate raw probabilities
        probs = torch.softmax(output, dim=1)
        
        # # --- FIX 1: REPETITION PENALTY ---
        # # If we just said a word, heavily penalize saying it again immediately
        # if len(predicted_indices) > 0:
        #   last_idx = predicted_indices[-1]
        #   probs[0, last_idx] *= 0.1 # Reduce probability of the last word by 90%
        
        # # --- FIX 2: MIN LENGTH CONSTRAINT ---
        # # Prevent the model from cutting off with <EOS> (index 2) too early
        # if step < 3:
        #   probs[0, 2] = 0 
        
        # --- AGGRESSIVE REPETITION PENALTY ---
        if len(predicted_indices) > 0:
          # Penalize the last word
          probs[0, predicted_indices[-1]] *= 0.01 
          
          # Penalize the word before that (stops "and is and is")
          if len(predicted_indices) > 1:
            probs[0, predicted_indices[-2]] *= 0.05
            
        # --- EOS BOOST ---
        # If we've reached 10 words, start making <EOS> more likely
        if step > 8:
          probs[0, 2] *= (1.0 + (step * 0.2))
          
        # Re-normalize after adjustments
        probs = probs / probs.sum()
        
        # Get Top 5 for debugging
        top_probs, top_indices = torch.topk(probs, 5, dim=1)
        
        idx = top_indices[0][0].item()
        max_prob = top_probs[0][0].item()

        # --- DEBUG PRINT BLOCK (Fixed for your SignVocab) ---
        print(f"Step {step:02}: ", end="")
        for i in range(5):
          p = top_probs[0][i].item() * 100
          t_idx = top_indices[0][i].item()
          # Use your vocab's itos mapping
          word = self.vocab.itos.get(t_idx, self.vocab.itos.get(str(t_idx), f"[{t_idx}]"))
          marker = "⭐" if i == 0 else "  "
          print(f"{marker} {word}({p:.2f}%) ", end=" | ")
        print() 

        if idx == 2: # <EOS>
          break
          
        predicted_indices.append(idx)
        total_probs.append(max_prob)
        input_token = torch.LongTensor([idx]).to(self.device)

      # 5. Result Construction
      avg_confidence = (sum(total_probs) / len(total_probs)) * 100 if total_probs else 0
      # Final string conversion
      words = [self.vocab.itos.get(i, self.vocab.itos.get(str(i), "<UNK>")) for i in predicted_indices]
      sentence = " ".join(words)
      
      return sentence, avg_confidence