from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from pointnclick_segmentation.utils import IMAGE_EXTENSIONS


@dataclass
class SampleRecord:
    image_path: Path
    mask_path: Path
    sample_id: str


def _list_records(split_dir: str | Path) -> list[SampleRecord]:
    split_dir = Path(split_dir)
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Expected 'images' and 'masks' under {split_dir}")

    image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    records: list[SampleRecord] = []
    for image_path in sorted(image_paths):
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            candidates = list(mask_dir.glob(f"{image_path.stem}.*"))
            if not candidates:
                raise FileNotFoundError(f"No matching mask found for {image_path.name}")
            mask_path = candidates[0]
        records.append(SampleRecord(image_path=image_path, mask_path=mask_path, sample_id=image_path.stem))
    if not records:
        raise ValueError(f"No image/mask pairs found in {split_dir}")
    return records


class ClickSegmentationDataset(Dataset):
    def __init__(self, root_dir: str | Path, image_size: int = 512, augment: bool = True) -> None:
        self.records = _list_records(root_dir)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("L")
        mask = Image.open(record.mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                image_arr = np.fliplr(image_arr).copy()
                mask_arr = np.fliplr(mask_arr).copy()
            if random.random() < 0.5:
                image_arr = np.flipud(image_arr).copy()
                mask_arr = np.flipud(mask_arr).copy()

        click_y, click_x = self._sample_positive_click(mask_arr)
        click_map = self._make_click_map(mask_arr.shape, click_x, click_y)

        image_tensor = torch.from_numpy(image_arr).unsqueeze(0)
        click_tensor = torch.from_numpy(click_map).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        input_tensor = torch.cat([image_tensor, click_tensor], dim=0)

        return {
            "input": input_tensor,
            "mask": mask_tensor,
            "click": torch.tensor([click_x, click_y], dtype=torch.float32),
            "sample_id": record.sample_id,
        }

    @staticmethod
    def _sample_positive_click(mask: np.ndarray) -> tuple[int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            h, w = mask.shape
            return h // 2, w // 2
        idx = random.randrange(len(xs))
        return int(ys[idx]), int(xs[idx])

    @staticmethod
    def _make_click_map(shape: tuple[int, int], click_x: int, click_y: int, sigma: float = 10.0) -> np.ndarray:
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        dist_sq = (xx - click_x) ** 2 + (yy - click_y) ** 2
        click_map = np.exp(-dist_sq / (2 * sigma ** 2)).astype(np.float32)
        return click_map


def build_dataloader(
    root_dir: str | Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    augment: bool,
    extra_dir: str | Path | None = None,
) -> DataLoader:
    datasets: list[Dataset] = [ClickSegmentationDataset(root_dir, image_size=image_size, augment=augment)]
    if extra_dir:
        extra_path = Path(extra_dir)
        if extra_path.exists():
            datasets.append(ClickSegmentationDataset(extra_path, image_size=image_size, augment=augment))
    dataset: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
