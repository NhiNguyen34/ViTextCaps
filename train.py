import os
import yaml
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

from utils.vitextcaps_dataset import ViTextCapsDataset, collate_fn
from utils.training_utils import train, evaluate
from models.m4c import M4C

def start_training(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_name'])
    train_dataset = ViTextCapsDataset(tokenizer, config['dataset']['train_path'], config)
    dev_dataset = ViTextCapsDataset(tokenizer, config['dataset']['dev_path'], config)
    test_dataset = ViTextCapsDataset(tokenizer, config['dataset']['test_path'], config)

    train_dataloader = DataLoader(train_dataset, batch_size=config.model.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.model.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    device = torch.device(config.model.device)
    pretrained_config = AutoConfig.from_pretrained(config.model.pretrained_name)
    model = M4C(config.model, pretrained_config).to(device)
    
    if not os.path.isdir(config.model.checkpoint_path):
        os.makedirs(config.model.checkpoint_path)

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
