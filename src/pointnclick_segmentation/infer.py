from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from pointnclick_segmentation.model import UNet2D
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


def _extract_with_padding(array: np.ndarray, x0: int, y0: int, x1: int, y1: int, fill_value: int) -> np.ndarray:
    h, w = array.shape
    out = np.full((y1 - y0, x1 - x0), fill_value, dtype=array.dtype)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return out


def predict_mask_from_array(
    checkpoint_path: str | Path,
    image: np.ndarray,
    x: int,
    y: int,
    image_size: int | None = None,
    threshold: float = 0.5,
    device_name: str = "cuda",
) -> np.ndarray:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = checkpoint.get("config", {})
    model_image_size = image_size or int(train_config.get("image_size", 512))
    crop_size = int(train_config.get("crop_size") or model_image_size)

    if image.ndim != 2:
        raise ValueError("predict_mask_from_array expects a 2D grayscale image")

    original_h, original_w = image.shape
    crop_half = crop_size // 2
    x0 = x - crop_half
    y0 = y - crop_half
    x1 = x0 + crop_size
    y1 = y0 + crop_size
    crop = _extract_with_padding(image.astype(np.uint8), x0, y0, x1, y1, fill_value=int(image.mean()))

    resized_image = _resize_for_model(crop, model_image_size)
    model_x = min(max(int(round((x - x0) * model_image_size / crop_size)), 0), model_image_size - 1)
    model_y = min(max(int(round((y - y0) * model_image_size / crop_size)), 0), model_image_size - 1)

    click_map = make_click_map(resized_image.shape, model_x, model_y)
    input_tensor = torch.from_numpy(resized_image.astype(np.float32) / 255.0).unsqueeze(0)
    click_tensor = torch.from_numpy(click_map).unsqueeze(0)
    input_tensor = torch.cat([input_tensor, click_tensor], dim=0).unsqueeze(0).to(device)

    model = UNet2D(
        in_channels=2,
        base_channels=int(train_config.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    pred_mask_small = (probs >= threshold).astype(np.uint8)
    pred_crop = _resize_mask_back(pred_mask_small, (crop_size, crop_size))

    full_mask = np.zeros((original_h, original_w), dtype=np.uint8)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(original_w, x1)
    src_y1 = min(original_h, y1)
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    full_mask[src_y0:src_y1, src_x0:src_x1] = pred_crop[dst_y0:dst_y1, dst_x0:dst_x1]
    return full_mask


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
    image = load_grayscale_image(image_path)
    pred_mask = predict_mask_from_array(
        checkpoint_path=checkpoint_path,
        image=image,
        x=x,
        y=y,
        image_size=image_size,
        threshold=threshold,
        device_name=device_name,
    )

    ensure_dir(Path(output_mask).parent)
    save_mask(pred_mask, output_mask)
    if output_overlay:
        ensure_dir(Path(output_overlay).parent)
        save_overlay(image, pred_mask, output_overlay)
    return pred_mask
