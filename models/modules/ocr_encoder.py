import torch
from torch import nn
from torch.nn import functional as F

class OCREncoder(nn.Module):
  def __init__(self, ocr_in_dim, hidden_size, dropout_prob=0.1):
    super().__init__()

    # 300 (FastText) + 256 (rec_features) + 256 (det_features) = 812
    self.linear_ocr_feat_to_mmt_in = nn.Linear(ocr_in_dim, hidden_size)

    # OCR location feature
    self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, hidden_size)

    self.ocr_feat_layer_norm = nn.LayerNorm(hidden_size)
    self.ocr_bbox_layer_norm = nn.LayerNorm(hidden_size)
    self.dropout = nn.Dropout(dropout_prob)

  def forward(self, ocr_boxes, ocr_token_embeddings, ocr_rec_features, ocr_det_features):

    # Normalize input
    ocr_token_embeddings = F.normalize(ocr_token_embeddings, dim=-1)
    ocr_rec_features = F.normalize(ocr_rec_features, dim=-1)
    ocr_det_features = F.normalize(ocr_det_features, dim=-1)

    # get OCR combine features
    ocr_combine_features = torch.cat([ocr_token_embeddings, ocr_rec_features, ocr_det_features], dim=-1)
    ocr_combine_features = self.ocr_feat_layer_norm(self.linear_ocr_feat_to_mmt_in(ocr_combine_features))

    # Get OCR bbox features
    ocr_bbox_features = self.ocr_bbox_layer_norm(self.linear_ocr_bbox_to_mmt_in(ocr_boxes))

    return self.dropout(ocr_combine_features + ocr_bbox_features) # batch_size, seq_length, hidden_size