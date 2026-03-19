from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from pointnclick_segmentation.config import TrainConfig
from pointnclick_segmentation.data.dataset import build_dataloader
from pointnclick_segmentation.metrics import (
    batch_dice_from_logits,
    batch_iou_from_logits,
    batch_vi_from_logits,
)
from pointnclick_segmentation.model import UNet2D
from pointnclick_segmentation.utils import ensure_dir, resolve_device, save_json, set_seed


def _build_model(config: TrainConfig, device: torch.device) -> nn.Module:
    model = UNet2D(in_channels=2, base_channels=config.base_channels).to(device)
    if config.resume_checkpoint:
        state = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    return model


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    bce = nn.BCEWithLogitsLoss()
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    totals = {
        "loss": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "vi": 0.0,
    }
    steps = 0

    grad_context = torch.enable_grad() if train_mode else torch.no_grad()
    with grad_context:
        for batch in loader:
            inputs = batch["input"].to(device)
            masks = batch["mask"].to(device)
            logits = model(inputs)
            loss = bce(logits, masks)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            totals["loss"] += loss.item()
            totals["iou"] += batch_iou_from_logits(logits.detach(), masks)
            totals["dice"] += batch_dice_from_logits(logits.detach(), masks)
            totals["vi"] += batch_vi_from_logits(logits.detach(), masks)
            steps += 1

    if steps == 0:
        return {key: 0.0 for key in totals}
    return {key: value / steps for key, value in totals.items()}


def _is_better(selection_metric: str, current: dict[str, float], best: dict[str, float] | None) -> bool:
    if best is None:
        return True
    if selection_metric == "vi":
        if current["vi"] < best["vi"] - 1e-6:
            return True
        if abs(current["vi"] - best["vi"]) <= 1e-6:
            return current["iou"] > best["iou"]
        return False
    return current["iou"] > best["iou"] + 1e-6


def train_model(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    config.save(output_dir / "config.json")

    device = resolve_device(config.device)
    train_loader = build_dataloader(
        root_dir=config.train_dir,
        image_size=config.image_size,
        crop_size=config.crop_size,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        augment=True,
        extra_dir=config.feedback_dir,
    )
    val_loader = build_dataloader(
        root_dir=config.val_dir,
        image_size=config.image_size,
        crop_size=config.crop_size,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        augment=False,
    )

    model = _build_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_iou": [],
        "train_dice": [],
        "train_vi": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
        "val_vi": [],
    }

    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=tqdm(train_loader, desc=f"train epoch {epoch}", leave=False),
            device=device,
            optimizer=optimizer,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=tqdm(val_loader, desc=f"val epoch {epoch}", leave=False),
            device=device,
            optimizer=None,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["train_vi"].append(train_metrics["vi"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_vi"].append(val_metrics["vi"])

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": {
                "train_loss": train_metrics["loss"],
                "train_iou": train_metrics["iou"],
                "train_dice": train_metrics["dice"],
                "train_vi": train_metrics["vi"],
                "val_loss": val_metrics["loss"],
                "val_iou": val_metrics["iou"],
                "val_dice": val_metrics["dice"],
                "val_vi": val_metrics["vi"],
                "loss_function": config.loss_function,
            },
        }
        torch.save(checkpoint, output_dir / "last_model.pt")

        if _is_better(config.selection_metric, val_metrics, best_metrics):
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        save_json(
            {
                "selection_metric": config.selection_metric,
                "loss_function": config.loss_function,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "history": history,
            },
            output_dir / "metrics.json",
        )

        if epoch >= config.min_epochs and epochs_without_improvement >= config.early_stopping_patience:
            break

    return {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "output_dir": str(output_dir),
    }


def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    device_name: str = "cuda",
) -> dict[str, float]:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    config = TrainConfig(
        train_dir=data_dir,
        val_dir=data_dir,
        output_dir=".",
        image_size=int(checkpoint_config.get("image_size", 512)),
        crop_size=checkpoint_config.get("crop_size"),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device_name,
        base_channels=int(checkpoint_config.get("base_channels", 32)),
        loss_function=str(checkpoint_config.get("loss_function", "bce")),
    )
    loader = build_dataloader(
        root_dir=data_dir,
        image_size=config.image_size,
        crop_size=config.crop_size,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        augment=False,
    )
    model = _build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    return _run_epoch(model=model, loader=tqdm(loader, desc="evaluate", leave=False), device=device, optimizer=None)
