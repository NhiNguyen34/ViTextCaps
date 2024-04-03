from torch import nn
import torch
from tqdm import tqdm
import os
import json

from evaluation import compute_metrics

def loss_fn(outputs, labels, padding_value):
    vocab_ocr_size = outputs.size(-1)
    outputs_ = outputs.reshape(-1, vocab_ocr_size) # batch_size * seq_l, vocab + ocr size
    labels_ = labels.reshape(-1) # batch_size * seq_l

    mask = labels_ != padding_value

    outputs_non_pad = outputs_[mask] # batch_size * seq_l - pad, vocab + ocr size
    labels_non_pad = labels_[mask] # batch_size * seq_l - pad

    # Use the original tensor as indices to select rows from the identity matrix
    converted_labels = torch.zeros_like(outputs_non_pad) # batch_size * seq_l - pad, vocab + ocr size
    converted_labels[torch.arange(len(labels_non_pad)), labels_non_pad] = 1

    pos_weight = (converted_labels==0.).sum()/converted_labels.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')

    return criterion(outputs_non_pad, converted_labels)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# Learning rate warm-up function
def warmup_lr(current_epoch):
    warmup = 400
    model_size = 768
    factor = 0.1
    current_epoch += 1

    return factor * (model_size ** (-0.5) * min(current_epoch ** (-0.5), current_epoch * warmup ** (-1.5)))

def train(model, train_dataloader, dev_dataloader, tokenizer, config):
    # Create a LambdaLR scheduler for learning rate warm-up
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    base_lr = 1e-4
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.Adam(model.get_optimizer_parameters(base_lr), lr=base_lr)

    if os.path.isfile(os.path.join(config.checkpoint_path, "last_model.pth")):
        checkpoint = torch.load(os.path.join(config.checkpoint_path, "last_model.pth"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_score = checkpoint["best_score"]
        patient = checkpoint["patient"]
        epoch = checkpoint["epoch"]
    else:
        best_score = 0
        patient = 0
        epoch = 0
        model.apply(initialize_weights)

    # Send model to device
    device = torch.device(config.device)
    model = model.to(device)
    while True:
        epoch += 1
        print(f"Epoch {epoch}")
        step = 0
        k = len(train_dataloader)
        total_loss = 0
        model.train()
        with tqdm(train_dataloader, desc="Training") as pb:
                for sample in pb:
                    # Forwad pass
                    outputs = model(sample, device) # batch_size, T, vocab + ocr_tokens
                    loss = criterion(outputs.mT, sample['labels'].to(device))
                    optimizer.zero_grad() # Xóa cái gradient ở vòng lặp trước

                    # Calculate loss per batch
                    total_loss += loss.item()
                    step += 1
                    pb.set_postfix({
                        "loss": total_loss / step
                    })

                    # Optimizer & Backward
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                    optimizer.step() # update weight
                    # scheduler.step()
                    if step == round(k * 0.8):
                        for g in optimizer.param_groups:
                            g['lr'] *= 0.95

        ### Evaluate ###
        model.eval()
        gt_caps = []
        pred_caps = []
        with tqdm(dev_dataloader) as pb:
            for sample in pb:
                # Forward pass
                with torch.inference_mode():
                    outputs = model(sample, device) # batch_size, T, vocab
                pred_cap = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                gt_cap = sample["raw_captions"]
                pred_caps.extend(pred_cap)
                gt_caps.extend(gt_cap)

        scores = compute_metrics(pred_caps, gt_caps)
        for score in scores:
            print(f"\t{score}: {scores[score]}")

        score = scores[config.metric]
        # Save checkpoints
        if score > best_score:
            best_score = score
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
                "patient": patient
            }, os.path.join(config.checkpoint_path, "best_model.pth"))
        else:
            patient += 1
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
                "patient": patient
            }, os.path.join(config.checkpoint_path, "last_model.pth"))
            if patient > config.max_patient:
                print("Patient reached. Training completed.")
                break

def evaluate(model, test_dataloader, tokenizer, config):
    if not os.path.isfile(os.path.join(config.checkpoint_path, "best_model.pth")):
        raise FileNotFoundError("Cannot file the best_model.path. Please ensure the training was completed.")
    
    checkpoint = torch.load(os.path.join(config.checkpoint_path, "best_model.pth"))
    model.load_state_dict(checkpoint["model"])
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()
    gt_caps = []
    pred_caps = []
    results = []
    with tqdm(test_dataloader) as pb:
        for sample in pb:
            # Forward pass
            with torch.inference_mode():
                outputs = model(sample, device) # batch_size, T, vocab
            pred_cap = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            gt_cap = sample["raw_captions"]
            pred_caps.extend(pred_cap)
            gt_caps.extend(gt_cap)
            results.append({
                "image_id": sample["image_id"][0],
                "prediction": pred_cap[0],
                "ground_truths": gt_cap[0]
            })

    scores = compute_metrics(pred_caps, gt_caps)
    for score in scores:
        print(f"\t{score}: {scores[score]}")

    json.dump({
        "results": results,
        "scores": scores
    }, open(os.path.join(config.checkpoint_path, "results.json"), "w+"), ensure_ascii=False, indent=4)
