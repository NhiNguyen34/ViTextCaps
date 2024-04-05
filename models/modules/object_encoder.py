from torch import nn
from torch.nn import functional as F

class ObjectEncoder(nn.Module):
  def __init__(self, obj_in_dim, hidden_size, dropout_prob=0.1):
    super().__init__()

    # 2048 (FasterRCNN)
    self.linear_obj_feat_to_mmt_in = nn.Linear(obj_in_dim, hidden_size)

    # OBJ location feature
    self.linear_obj_bbox_to_mmt_in = nn.Linear(4, hidden_size)

    self.obj_feat_layer_norm = nn.LayerNorm(hidden_size)
    self.obj_bbox_layer_norm = nn.LayerNorm(hidden_size)
    self.dropout = nn.Dropout(dropout_prob)

  def forward(self, obj_boxes, obj_features):

    # Features to hidden size
    obj_features = F.normalize(obj_features, dim=-1)

    # Get obj features
    obj_features = self.obj_feat_layer_norm(self.linear_obj_feat_to_mmt_in(obj_features))
    obj_bbox_features = self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(obj_boxes))

    return self.dropout(obj_features + obj_bbox_features) # batch_size, seq_length, hidden_size