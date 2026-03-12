from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pointnclick_segmentation.infer import predict_mask
from pointnclick_segmentation.utils import ensure_dir, load_grayscale_image, save_overlay


def segment_id_to_rgb(segment_id: int) -> tuple[int, int, int]:
    if segment_id < 0 or segment_id > 0xFFFFFF:
        raise ValueError("VAST segment ids must fit in 24 bits")
    red = (segment_id >> 16) & 0xFF
    green = (segment_id >> 8) & 0xFF
    blue = segment_id & 0xFF
    return red, green, blue


def encode_vast_segmentation(mask: np.ndarray, segment_id: int) -> np.ndarray:
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color = np.array(segment_id_to_rgb(segment_id), dtype=np.uint8)
    rgb[mask > 0] = color
    return rgb


def save_vast_segmentation_image(mask: np.ndarray, segment_id: int, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    encoded = encode_vast_segmentation(mask, segment_id)
    Image.fromarray(encoded, mode="RGB").save(path)
    return path


def predict_vast_import_image(
    checkpoint_path: str | Path,
    image_path: str | Path,
    x: int,
    y: int,
    segment_id: int,
    z_index: int,
    output_dir: str | Path,
    output_stem: str | None = None,
    threshold: float = 0.5,
    image_size: int | None = None,
    device_name: str = "cuda",
) -> dict[str, str | int]:
    output_dir = ensure_dir(output_dir)
    image_path = Path(image_path)
    stem = output_stem or image_path.stem

    binary_mask_path = output_dir / f"{stem}_mask_binary.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    vast_import_path = output_dir / f"{stem}_vast_import.png"
    metadata_path = output_dir / f"{stem}_vast_import.json"

    mask = predict_mask(
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        x=x,
        y=y,
        output_mask=binary_mask_path,
        output_overlay=overlay_path,
        image_size=image_size,
        threshold=threshold,
        device_name=device_name,
    )
    save_vast_segmentation_image(mask, segment_id=segment_id, path=vast_import_path)

    em_image = load_grayscale_image(image_path)
    colored_overlay = np.asarray(Image.open(vast_import_path).convert("RGB"), dtype=np.uint8)
    preview = np.clip(0.7 * np.stack([em_image, em_image, em_image], axis=-1) + 0.3 * colored_overlay, 0, 255).astype(
        np.uint8
    )
    preview_path = output_dir / f"{stem}_vast_preview.png"
    Image.fromarray(preview, mode="RGB").save(preview_path)

    metadata = {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "click_x": x,
        "click_y": y,
        "segment_id": segment_id,
        "z_index": z_index,
        "start_x": 0,
        "start_y": 0,
        "start_z": z_index,
        "binary_mask_path": str(binary_mask_path),
        "overlay_path": str(overlay_path),
        "vast_import_path": str(vast_import_path),
        "vast_preview_path": str(preview_path),
        "import_notes": [
            "In VAST Lite, use File > Import Segmentations from Images.",
            "Choose the generated RGB image.",
            "Use start coordinates X=0, Y=0, Z=<z_index> for a full-slice import.",
            "Assign the import to the segmentation layer you want to update.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "binary_mask_path": str(binary_mask_path),
        "overlay_path": str(overlay_path),
        "vast_import_path": str(vast_import_path),
        "vast_preview_path": str(preview_path),
        "metadata_path": str(metadata_path),
        "segment_id": segment_id,
        "z_index": z_index,
    }
