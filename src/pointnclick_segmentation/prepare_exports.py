from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from pointnclick_segmentation.utils import IMAGE_EXTENSIONS, ensure_dir


SLICE_PATTERN = re.compile(r"_s(\d+)\.", re.IGNORECASE)
BOUTON_PATTERN = re.compile(r"^Bouton\s+(\d+)$", re.IGNORECASE)


def _extract_slice_id(path: Path) -> str | None:
    match = SLICE_PATTERN.search(path.name)
    if not match:
        return None
    return match.group(1)


def _extract_bouton_index(path: Path) -> int | None:
    match = BOUTON_PATTERN.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def _iter_image_files(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _binarize_mask(mask_path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return (mask > 0).astype(np.uint8) * 255


def prepare_exports_dataset(
    exports_dir: str | Path,
    output_dir: str | Path,
    val_boutons: list[int] | None = None,
    resize_masks_to_em: bool = True,
) -> dict[str, int | list[int]]:
    exports_dir = Path(exports_dir)
    boutons_root = exports_dir / "Boutons"
    em_root = exports_dir / "EM"
    if not boutons_root.exists() or not em_root.exists():
        raise FileNotFoundError(f"Expected {boutons_root} and {em_root}")

    bouton_dirs = [p for p in boutons_root.iterdir() if p.is_dir() and _extract_bouton_index(p) is not None]
    bouton_indices = sorted(_extract_bouton_index(p) for p in bouton_dirs if _extract_bouton_index(p) is not None)
    if not bouton_indices:
        raise ValueError(f"No bouton folders found under {boutons_root}")

    if val_boutons is None:
        val_boutons = bouton_indices[-4:] if len(bouton_indices) >= 4 else bouton_indices[-1:]

    output_dir = Path(output_dir)
    train_image_dir = ensure_dir(output_dir / "train" / "images")
    train_mask_dir = ensure_dir(output_dir / "train" / "masks")
    val_image_dir = ensure_dir(output_dir / "val" / "images")
    val_mask_dir = ensure_dir(output_dir / "val" / "masks")

    manifest: list[dict[str, object]] = []
    num_train = 0
    num_val = 0

    for bouton_dir in sorted(bouton_dirs, key=lambda p: _extract_bouton_index(p) or -1):
        bouton_idx = _extract_bouton_index(bouton_dir)
        if bouton_idx is None:
            continue
        em_dir = em_root / bouton_dir.name
        if not em_dir.exists():
            raise FileNotFoundError(f"Missing EM folder for {bouton_dir.name}: {em_dir}")

        masks_by_slice = {
            slice_id: path
            for path in _iter_image_files(bouton_dir)
            if (slice_id := _extract_slice_id(path)) is not None
        }
        images_by_slice = {
            slice_id: path
            for path in _iter_image_files(em_dir)
            if (slice_id := _extract_slice_id(path)) is not None
        }

        common_slice_ids = sorted(set(masks_by_slice) & set(images_by_slice), key=lambda x: int(x))
        if not common_slice_ids:
            continue

        is_val = bouton_idx in val_boutons
        dst_image_dir = val_image_dir if is_val else train_image_dir
        dst_mask_dir = val_mask_dir if is_val else train_mask_dir

        for slice_id in common_slice_ids:
            image_path = images_by_slice[slice_id]
            mask_path = masks_by_slice[slice_id]
            sample_id = f"bouton_{bouton_idx:02d}_s{int(slice_id):03d}"

            image = Image.open(image_path).convert("L")
            mask = _binarize_mask(mask_path)
            resized_mask = False
            if image.size != (mask.shape[1], mask.shape[0]):
                if not resize_masks_to_em:
                    raise ValueError(
                        f"Image/mask size mismatch for {sample_id}: image={image.size}, mask={(mask.shape[1], mask.shape[0])}"
                    )
                mask = np.asarray(
                    Image.fromarray(mask, mode="L").resize(image.size, resample=Image.NEAREST),
                    dtype=np.uint8,
                )
                resized_mask = True

            image.save(dst_image_dir / f"{sample_id}.png")
            Image.fromarray(mask, mode="L").save(dst_mask_dir / f"{sample_id}.png")

            manifest.append(
                {
                    "sample_id": sample_id,
                    "bouton": bouton_idx,
                    "slice": int(slice_id),
                    "split": "val" if is_val else "train",
                    "image_src": str(image_path),
                    "mask_src": str(mask_path),
                    "resized_mask_to_match_em": resized_mask,
                }
            )
            if is_val:
                num_val += 1
            else:
                num_train += 1

    summary = {
        "num_train": num_train,
        "num_val": num_val,
        "val_boutons": sorted(val_boutons),
        "resize_masks_to_em": resize_masks_to_em,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
