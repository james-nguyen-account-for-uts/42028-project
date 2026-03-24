import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
  def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
    super().__init__()
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    # Linear layer to project 134 skeleton features to the embedding dimension
    self.input_projection = nn.Linear(input_dim, emb_dim)
    self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
    self.dropout = nn.Dropout(dropout)
    # Combine forward and backward hidden states
    self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
    self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)

  def forward(self, src):
    projected = self.dropout(torch.relu(self.input_projection(src)))
    projected = projected.permute(1, 0, 2)
    
    outputs, (hidden, cell) = self.rnn(projected)
    
    # hidden shape: [n_layers * 2, batch_size, hid_dim] (2 because it's bidirectional)
    # We need to collapse the bidirectional states into the number of layers the decoder expects
    
    # Take the last forward and backward states for EACH layer
    # This loop handles N_LAYERS correctly
    h_combined = []
    c_combined = []
    for i in range(self.n_layers):
      h_combined.append(torch.tanh(self.fc_hidden(torch.cat((hidden[i*2,:,:], hidden[i*2+1,:,:]), dim=1))))
      c_combined.append(torch.tanh(self.fc_cell(torch.cat((cell[i*2,:,:], cell[i*2+1,:,:]), dim=1))))
    
    new_hidden = torch.stack(h_combined) # [n_layers, batch, hid_dim]
    new_cell = torch.stack(c_combined)   # [n_layers, batch, hid_dim]
    
    return outputs, (new_hidden, new_cell)

class Decoder(nn.Module):
  def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
    super().__init__()
    self.output_dim = output_dim
    self.embedding = nn.Embedding(output_dim, emb_dim)
    self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
    self.fc_out = nn.Linear(hid_dim, output_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input, hidden, cell):
    # input shape: [batch_size] -> predict one word at a time
    input = input.unsqueeze(0)
    embedded = self.dropout(self.embedding(input))
    output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
    prediction = self.fc_out(output.squeeze(0))
    return prediction, hidden, cell

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, device):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = device

  def forward(self, src, trg, teacher_forcing_ratio=0.5):
    batch_size = src.shape[0]
    trg_len = trg.shape[1]
    trg_vocab_size = self.decoder.output_dim
    
    # Initialize outputs on the correct device
    outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
    
    # Encoder now returns [n_layers, batch, hid_dim]
    _, (hidden, cell) = self.encoder(src)
    
    input = trg[:, 0]
    for t in range(1, trg_len):
      output, hidden, cell = self.decoder(input, hidden, cell)
      outputs[t] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.argmax(1)
      input = trg[:, t] if teacher_force else top1
      
    return outputs