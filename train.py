import yaml
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import PhobertTokenizer, AutoModel

from utils.vitextcaps_dataset import ViTextCapsDataset
from utils.training_utils import train, evaluate
from models.m4c import M4C

def start_training(config):
    tokenizer = PhobertTokenizer.from_pretrained(config.pretrained_name)

    train_dataset = ViTextCapsDataset(tokenizer, config.train_path)
    dev_dataset = ViTextCapsDataset(tokenizer, config.dev_path)
    test_dataset = ViTextCapsDataset(tokenizer, config.test_path)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=test_dataset.collate_fn)

    device = torch.device(config.device)
    phobert_model = AutoModel.from_pretrained(config.pretrained_name)
    phobert_model.embeddings.word_embeddings.requires_grad = False
    fixed_ans_emb = phobert_model.embeddings.word_embeddings.weight
    model = M4C(obj_in_dim=1024,
                ocr_in_dim=812,
                hidden_size=768,
                n_heads=12,
                d_k=64,
                n_layers=4,
                vocab_size=tokenizer.vocab_size + 1,
                fixed_ans_emb=fixed_ans_emb).to(device)


    train(model, train_dataloader, dev_dataloader, tokenizer, config)
    evaluate(model, test_dataloader, tokenizer, config)

    print("Task completed.")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    with open(args.config) as conf_file:
        config = yaml.safe_load(conf_file)
    
    start_training(config)
