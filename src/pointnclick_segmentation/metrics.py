from __future__ import annotations

import math

import numpy as np
import torch


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def batch_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (intersection / union).mean().item()


def batch_dice_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def variation_of_information_binary(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = pred_mask.astype(np.uint8).reshape(-1)
    true = true_mask.astype(np.uint8).reshape(-1)
    counts = np.zeros((2, 2), dtype=np.float64)
    for true_label in (0, 1):
        true_idx = true == true_label
        if not np.any(true_idx):
            continue
        for pred_label in (0, 1):
            counts[true_label, pred_label] = np.sum(true_idx & (pred == pred_label))
    total = counts.sum()
    if total <= 0:
        return 0.0
    joint = counts / total
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mutual_info = 0.0
    for i in range(2):
        for j in range(2):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mutual_info += joint[i, j] * math.log(joint[i, j] / (px[i] * py[j]), 2)
    hx = -sum(prob * math.log(prob, 2) for prob in px if prob > 0)
    hy = -sum(prob * math.log(prob, 2) for prob in py if prob > 0)
    return float(hx + hy - 2.0 * mutual_info)


def batch_vi_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    vis: list[float] = []
    for pred_prob, target in zip(probs, target_np):
        pred_mask = (pred_prob[0] >= threshold).astype(np.uint8)
        true_mask = (target[0] >= 0.5).astype(np.uint8)
        vis.append(variation_of_information_binary(pred_mask, true_mask))
    return float(np.mean(vis)) if vis else 0.0
