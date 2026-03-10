from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_grayscale_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    array = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(array, mode="L").save(path)


def save_overlay(image: np.ndarray, mask: np.ndarray, path: str | Path) -> None:
    base = np.stack([image, image, image], axis=-1).astype(np.uint8)
    overlay = base.copy()
    overlay[mask > 0] = np.array([255, 64, 64], dtype=np.uint8)
    blended = (0.65 * base + 0.35 * overlay).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended, mode="RGB").save(path)
