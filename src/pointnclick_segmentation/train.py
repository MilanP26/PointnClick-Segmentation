from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from pointnclick_segmentation.config import TrainConfig
from pointnclick_segmentation.data.dataset import build_dataloader
from pointnclick_segmentation.model import ClickUNet
from pointnclick_segmentation.utils import ensure_dir, resolve_device, save_json, set_seed


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


def train_model(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    config.save(output_dir / "config.json")

    device = resolve_device(config.device)
    train_loader = build_dataloader(
        root_dir=config.train_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        augment=True,
        extra_dir=config.feedback_dir,
    )
    val_loader = build_dataloader(
        root_dir=config.val_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        augment=False,
    )

    model = ClickUNet().to(device)
    if config.resume_checkpoint:
        state = torch.load(config.resume_checkpoint, map_location=device)
        model.load_state_dict(state["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    best_val_iou = -1.0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_iou": [],
        "val_loss": [],
        "val_iou": [],
    }

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_steps = 0

        for batch in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False):
            inputs = batch["input"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = bce(logits, masks) + dice_loss_from_logits(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += batch_iou_from_logits(logits.detach(), masks)
            train_steps += 1

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val epoch {epoch}", leave=False):
                inputs = batch["input"].to(device)
                masks = batch["mask"].to(device)
                logits = model(inputs)
                loss = bce(logits, masks) + dice_loss_from_logits(logits, masks)

                val_loss += loss.item()
                val_iou += batch_iou_from_logits(logits, masks)
                val_steps += 1

        epoch_train_loss = train_loss / max(train_steps, 1)
        epoch_train_iou = train_iou / max(train_steps, 1)
        epoch_val_loss = val_loss / max(val_steps, 1)
        epoch_val_iou = val_iou / max(val_steps, 1)

        history["train_loss"].append(epoch_train_loss)
        history["train_iou"].append(epoch_train_iou)
        history["val_loss"].append(epoch_val_loss)
        history["val_iou"].append(epoch_val_iou)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": {
                "train_loss": epoch_train_loss,
                "train_iou": epoch_train_iou,
                "val_loss": epoch_val_loss,
                "val_iou": epoch_val_iou,
            },
        }
        torch.save(checkpoint, output_dir / "last_model.pt")

        if epoch_val_iou > best_val_iou:
            best_val_iou = epoch_val_iou
            torch.save(checkpoint, output_dir / "best_model.pt")

        save_json(
            {
                "best_val_iou": best_val_iou,
                "history": history,
            },
            output_dir / "metrics.json",
        )

    return {
        "best_val_iou": best_val_iou,
        "output_dir": str(output_dir),
    }
