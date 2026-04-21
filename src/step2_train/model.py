import random
import torch
import torch.nn as nn


class Encoder(nn.Module):

  def __init__(
      self,
      input_dim,
      emb_dim,
      hid_dim,
      n_layers,
      dropout,
      bidirectional=True):
    super().__init__()
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    self.bidirectional = bidirectional
    self.num_directions = 2 if bidirectional else 1

    self.embedding = nn.Sequential(
      nn.Linear(input_dim, emb_dim), nn.LayerNorm(emb_dim), nn.ReLU(),
      nn.Dropout(dropout), nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim))

    self.rnn = nn.LSTM(
      emb_dim,
      hid_dim,
      n_layers,
      dropout=dropout,
      bidirectional=bidirectional)
    self.dropout = nn.Dropout(dropout)

    if bidirectional:
      # Projection to collapse [n_layers*2, batch, hid_dim] -> [n_layers, batch, hid_dim]
      self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
      self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)
    else:
      self.fc_hidden = None
      self.fc_cell = None

  def forward(self, src):
    # src: [batch, seq, input_dim]
    embedded = self.embedding(src).permute(1, 0, 2)  # [seq, batch, emb_dim]
    embedded = self.dropout(embedded)

    outputs, (hidden, cell) = self.rnn(embedded)

    if not self.bidirectional:
      return outputs, (hidden, cell)

    # hidden is [n_layers * 2, batch, hid_dim]
    # Reshape to [n_layers, 2, batch, hid_dim] then concat the directions
    batch_size = src.shape[0]

    # Reshape to [n_layers, 2, batch, hid_dim]
    # index 0 is forward, index 1 is backward
    hidden = hidden.view(self.n_layers, 2, batch_size, self.hid_dim)
    cell = cell.view(self.n_layers, 2, batch_size, self.hid_dim)

    # Concatenate the two directions on the last dimension.
    h_cat = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
    c_cat = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)

    # Project back to [n_layers, batch, hid_dim].
    new_hidden = torch.tanh(self.fc_hidden(h_cat))
    new_cell = torch.tanh(self.fc_cell(c_cat))

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
    """input: [batch] -> [1, batch]; output: [1, batch, hid_dim] -> [batch, hid_dim]"""
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

    outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

    # Get context from encoder
    _, (hidden, cell) = self.encoder(src)

    # First input is the <SOS> token from every sentence in the batch
    input = trg[:, 0]

    for t in range(1, trg_len):
      output, hidden, cell = self.decoder(input, hidden, cell)
      outputs[t] = output

      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.argmax(1)

      # Use actual target if teacher forcing, else use model's own prediction
      input = trg[:, t] if teacher_force else top1

    return outputs
