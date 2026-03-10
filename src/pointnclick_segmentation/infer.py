from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from pointnclick_segmentation.model import ClickUNet
from pointnclick_segmentation.utils import ensure_dir, load_grayscale_image, resolve_device, save_mask, save_overlay


def make_click_map(shape: tuple[int, int], x: int, y: int, sigma: float = 10.0) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist_sq = (xx - x) ** 2 + (yy - y) ** 2
    return np.exp(-dist_sq / (2 * sigma ** 2)).astype(np.float32)


def _resize_for_model(image: np.ndarray, image_size: int) -> np.ndarray:
    pil_image = Image.fromarray(image, mode="L")
    pil_image = pil_image.resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(pil_image, dtype=np.uint8)


def _resize_mask_back(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil_mask = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    pil_mask = pil_mask.resize(size, resample=Image.NEAREST)
    return (np.asarray(pil_mask, dtype=np.uint8) > 0).astype(np.uint8)


def predict_mask(
    checkpoint_path: str | Path,
    image_path: str | Path,
    x: int,
    y: int,
    output_mask: str | Path,
    output_overlay: str | Path | None = None,
    image_size: int | None = None,
    threshold: float = 0.5,
    device_name: str = "cuda",
) -> np.ndarray:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = checkpoint.get("config", {})
    model_image_size = image_size or int(train_config.get("image_size", 512))

    image = load_grayscale_image(image_path)
    original_h, original_w = image.shape

    resized_image = _resize_for_model(image, model_image_size)
    scale_x = model_image_size / original_w
    scale_y = model_image_size / original_h
    model_x = min(max(int(round(x * scale_x)), 0), model_image_size - 1)
    model_y = min(max(int(round(y * scale_y)), 0), model_image_size - 1)

    click_map = make_click_map(resized_image.shape, model_x, model_y)
    input_tensor = torch.from_numpy(resized_image.astype(np.float32) / 255.0).unsqueeze(0)
    click_tensor = torch.from_numpy(click_map).unsqueeze(0)
    input_tensor = torch.cat([input_tensor, click_tensor], dim=0).unsqueeze(0).to(device)

    model = ClickUNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    pred_mask_small = (probs >= threshold).astype(np.uint8)
    pred_mask = _resize_mask_back(pred_mask_small, (original_w, original_h))

    ensure_dir(Path(output_mask).parent)
    save_mask(pred_mask, output_mask)
    if output_overlay:
        ensure_dir(Path(output_overlay).parent)
        save_overlay(image, pred_mask, output_overlay)
    return pred_mask
