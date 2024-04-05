import json
import torch
from torch.utils.data import Dataset

from utils import normalize_sentence
from utils.instances import Instance, InstanceList

from typing import *
import os
import numpy as np

def collate_fn(samples: list) -> InstanceList:
    return InstanceList(samples)

class ViTextCapsDataset(Dataset):
    def __init__(self, tokenizer, path, config):

        self.annotations = self.load_annotations(path)
        self.tokenizer = tokenizer
        
        self.max_ocr = config.max_ocr
        self.max_object = config.max_object
        self.masked_vision_value = config.masked_vision_value
        self.ocr_path = config.ocr_path
        self.object_path = config.obj_path

    def load_annotations(self, path: str) -> list:
        annotations = json.load(open(path))
        anns = []
        for ann_id in annotations:
            annotation = annotations[ann_id]
            anns.append({
                "id": ann_id,
                "image_id": annotation["image_id"],
                "caption": annotation["caption"]
            })

        return anns
    
    def load_ocr_features(self, image_id):
        npy_file = os.path.join(self.ocr_path, f"{image_id}.npy")
        features = np.load(npy_file, allow_pickle=True)[()]

        return features
    
    def load_object_features(self, image_id):
        npy_file = os.path.join(self.object_path, f"{image_id}.npy")
        features = np.load(npy_file, allow_pickle=True)[()]

        return features
    
    def process_ocr_tokens(self, ocr_tokens: str) -> torch.Tensor:
        ocr_tokens = normalize_sentence(ocr_tokens)
        ocr_tokens = ocr_tokens.split()
        ocr_tokens = self.tokenizer(ocr_tokens, 
                                    padding="max_length",
                                    add_special_tokens=False,
                                    return_tensors="pt").input_ids

        return ocr_tokens

    def process_object_tags(self, object_tags: str) -> torch.Tensor:
        object_tags = normalize_sentence(object_tags)
        object_tags = object_tags.split()
        object_tags = self.tokenizer(object_tags,
                                    padding="max_length",
                                    add_special_tokens=False,
                                    return_tensors="pt").input_ids

        return object_tags
    
    def process_caption(self, caption: str) -> Tuple[torch.Tensor, torch.Tensor]:
        caption = normalize_sentence(caption)
        encoded_item = self.tokenizer(caption,
                                        padding="max_length",
                                        add_special_tokens=True,
                                        return_tensors="pt")
        
        return encoded_item.input_ids, encoded_item.attention_mask

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        ocr_features = self.load_ocr_features(annotation["image_id"])
        object_features = self.load_object_features(annotation["image_id"])

        ocr_tokens = self.process_ocr_tokens(ocr_features["ocr_texts"])
        object_tags = self.process_object_tags(object_features["object_tags"])

        caption_tokens, caption_mask = self.process_caption(annotation["caption"])

        return Instance(
                **annotation,
                **ocr_features,
                **object_features,
                ocr_tokens = ocr_tokens,
                object_tags = object_tags,
                caption_tokens = caption_tokens,
                caption_mask = caption_mask
            )

    def __len__(self):
        return len(self.annotations)
