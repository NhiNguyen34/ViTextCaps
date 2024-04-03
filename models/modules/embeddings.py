import torch
from torch import nn
from torch.nn import functional as F
import math

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=512, dropout_prob=0.1):
    super().__init__()

    self.dropout = nn.Dropout(dropout_prob)

    position_ids = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(size=(1, max_len, d_model))
    pe[0, :, 0::2] = torch.sin(position_ids / div_term)
    pe[0, :, 1::2] = torch.cos(position_ids / div_term)
    self.register_buffer('pe', pe)

  def forward(self, x):
    # x shape (batch_size, seq_length, d_model)
    return x + self.pe[:, :x.size(1), :]

def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    results = F.embedding(batch_offsets + inds, x.view(batch_size * length, dim)) # batch_size, T, hidden_size
    return results
