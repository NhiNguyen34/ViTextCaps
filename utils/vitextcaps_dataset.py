from glob import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence

from utils import normalize_sentence

class ViTextCapsDataset(Dataset):
  def __init__(self, tokenizer, data_folder_path=None, mask_value=64000.0):

    self.data_paths=glob(data_folder_path+'/*')
    self.dummy_tensor = torch.ones((1, 300))
    self.mask_value = mask_value
    self.tokenizer = tokenizer

  def __getitem__(self, idx):
    sample = np.load(self.data_paths[idx], allow_pickle=True).item()
    return {
            "image_id": sample["image_id"],
            'captions': sample['captions'],
            'obj_boxes': torch.tensor(sample['obj']['boxes']),
            'obj_features': torch.tensor(sample['obj']['features']),
            'ocr_texts': sample['ocr']['texts'],
            'ocr_boxes': torch.tensor(sample['ocr']['boxes']),
            'ocr_token_embeddings': torch.tensor(sample['ocr']['fasttext_token']) if len(sample['ocr']['fasttext_token']) > 0 else self.dummy_tensor,
            'ocr_rec_features': torch.tensor(sample['ocr']['rec_features']),
            'ocr_det_features': torch.tensor(sample['ocr']['det_features'])
        }

  def __len__(self):
    return len(self.data_paths)

  def collate_fn(self, batch):
    image_ids = []
    raw_captions = []
    captions_ = []
    obj_boxes_tensor = []
    obj_features_tensor = []
    ocr_boxes_tensor = []
    ocr_token_embeddings_tensor = []
    ocr_rec_features_tensor = []
    ocr_det_features_tensor = []
    texts_ = []

    for each in batch:
      image_ids.append(each["image_id"])
      captions = each['captions']
      for i in range(len(each['captions'])):
        raw_captions.append(captions)
        caption = normalize_sentence(each['captions'][i])
        captions_.append(caption)

      obj_boxes_tensor.extend([each['obj_boxes']]*len(captions))
      obj_features_tensor.extend([each['obj_features']]*len(captions))

      ocr_boxes_tensor.extend([each['ocr_boxes']]*len(captions))
      ocr_token_embeddings_tensor.extend([each['ocr_token_embeddings']]*len(captions))
      ocr_rec_features_tensor.extend([each['ocr_rec_features']]*len(captions))
      ocr_det_features_tensor.extend([each['ocr_det_features']]*len(captions))

      texts_.extend([each['ocr_texts']]*len(captions))

    # Convert obj list to tensor
    obj_boxes_tensor = torch.stack(obj_boxes_tensor)
    obj_features_tensor = torch.stack(obj_features_tensor)

    # Convert ocr list to tensor
    ocr_boxes_tensor = pad_sequence(ocr_boxes_tensor, batch_first=True, padding_value=self.mask_value)
    ocr_token_embeddings_tensor = pad_sequence(ocr_token_embeddings_tensor, batch_first=True, padding_value=1)
    ocr_rec_features_tensor = pad_sequence(ocr_rec_features_tensor, batch_first=True, padding_value=1)
    ocr_det_features_tensor = pad_sequence(ocr_det_features_tensor, batch_first=True, padding_value=1)

    vs = self.tokenizer.vocab_size + 1
    labels_= []

    # Captions to token
    for i, caption in enumerate(captions_):
      label_ = []

      for token in word_tokenize(caption):

          if token in texts_[i] and token not in self.tokenizer.get_vocab():
            label_.append(texts_[i].index(token) + vs)
          else:
            label_ += self.tokenizer(token)['input_ids'][1: -1]

      label_.append(2) # 2 is <eos> in tokenizer
      labels_.append(torch.tensor(label_))

    # Convert labels_ 2 tensor
    labels_ = pad_sequence(labels_, batch_first=True, padding_value=1)

    dec_mask = torch.ones_like(labels_)
    dec_mask = dec_mask.masked_fill(labels_ == 1, 0) # batch_size, seq_length

    # Get the ocr_attention_mask
    ocr_attn_mask = torch.ones_like(ocr_boxes_tensor)
    ocr_attn_mask = ocr_attn_mask.masked_fill(ocr_boxes_tensor == self.mask_value, 0)[:, :, 0] # batch_size, seq_length
    ocr_boxes_tensor = ocr_boxes_tensor.masked_fill(ocr_boxes_tensor == self.mask_value, 1)

    # Join attention_mask
    obj_attn_mask = torch.ones(size=(obj_boxes_tensor.size(0), obj_boxes_tensor.size(1))) # batch_size, seq_length
    join_attn_mask = torch.cat([obj_attn_mask, ocr_attn_mask, dec_mask], dim=-1)

    return {
		'image_ids': image_ids,
		'obj_boxes': obj_boxes_tensor,
		'obj_features': obj_features_tensor,
		'ocr_boxes': ocr_boxes_tensor,
		'ocr_token_embeddings': ocr_token_embeddings_tensor,
		'ocr_rec_features': ocr_rec_features_tensor,
		'ocr_det_features': ocr_det_features_tensor,
		'join_attn_mask': join_attn_mask,
		'labels': labels_,
		'texts': texts_,
		'raw_captions': raw_captions
    }
