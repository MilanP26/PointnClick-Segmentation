from __future__ import annotations

import json
from pathlib import Path
import shutil

from pointnclick_segmentation.utils import ensure_dir


def add_feedback_sample(
    image_path: str | Path,
    mask_path: str | Path,
    feedback_dir: str | Path,
    sample_id: str,
) -> dict[str, str]:
    feedback_dir = Path(feedback_dir)
    image_dir = ensure_dir(feedback_dir / "images")
    mask_dir = ensure_dir(feedback_dir / "masks")

    image_src = Path(image_path)
    mask_src = Path(mask_path)
    image_dst = image_dir / f"{sample_id}{image_src.suffix.lower()}"
    mask_dst = mask_dir / f"{sample_id}.png"

    shutil.copy2(image_src, image_dst)
    shutil.copy2(mask_src, mask_dst)
    manifest_path = feedback_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "image": str(image_dst),
                    "mask": str(mask_dst),
                    "source_image": str(image_src),
                    "source_mask": str(mask_src),
                }
            )
            + "\n"
        )

    return {
        "image": str(image_dst),
        "mask": str(mask_dst),
    }
