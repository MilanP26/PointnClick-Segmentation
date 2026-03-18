from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from pointnclick_segmentation.utils import IMAGE_EXTENSIONS


@dataclass
class SampleRecord:
    image_path: Path
    mask_path: Path
    sample_id: str
    label_id: int
    centroid_x: float
    centroid_y: float


def _find_mask_for_image(mask_dir: Path, image_path: Path) -> Path:
    direct = mask_dir / f"{image_path.stem}.png"
    if direct.exists():
        return direct
    candidates = list(mask_dir.glob(f"{image_path.stem}.*"))
    if candidates:
        return candidates[0]
    image_suffix = image_path.stem.split("_s")[-1]
    suffix_matches = sorted(mask_dir.glob(f"*_s{image_suffix}.*"))
    if suffix_matches:
        return suffix_matches[0]
    raise FileNotFoundError(f"No matching mask found for {image_path.name}")


def _mask_label_records(mask_path: Path) -> list[tuple[int, float, float]]:
    mask = np.asarray(Image.open(mask_path), dtype=np.int32)
    labels = [int(v) for v in np.unique(mask) if int(v) > 0]
    records: list[tuple[int, float, float]] = []
    for label in labels:
        ys, xs = np.where(mask == label)
        if len(xs) == 0:
            continue
        records.append((label, float(xs.mean()), float(ys.mean())))
    return records


def _list_records(split_dir: str | Path) -> list[SampleRecord]:
    split_dir = Path(split_dir)
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Expected 'images' and 'masks' under {split_dir}")

    image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    records: list[SampleRecord] = []
    for image_path in sorted(image_paths):
        mask_path = _find_mask_for_image(mask_dir, image_path)
        label_records = _mask_label_records(mask_path)
        for label_id, centroid_x, centroid_y in label_records:
            records.append(
                SampleRecord(
                    image_path=image_path,
                    mask_path=mask_path,
                    sample_id=f"{image_path.stem}_label{label_id}",
                    label_id=label_id,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                )
            )
    if not records:
        raise ValueError(f"No image/mask pairs with foreground labels found in {split_dir}")
    return records


class ClickSegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 512,
        crop_size: int | None = None,
        augment: bool = True,
    ) -> None:
        self.records = _list_records(root_dir)
        self.image_size = image_size
        self.crop_size = crop_size or image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image = np.asarray(Image.open(record.image_path).convert("L"), dtype=np.uint8)
        dense_mask = np.asarray(Image.open(record.mask_path), dtype=np.int32)
        target_mask = (dense_mask == record.label_id).astype(np.uint8)

        click_y, click_x = self._sample_click(target_mask, record)
        crop_image, crop_mask = self._crop_around_click(image, target_mask, click_x, click_y)

        image_tensor = torch.from_numpy(crop_image.astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy(crop_mask.astype(np.float32)).unsqueeze(0)

        if self.augment:
            image_tensor, mask_tensor = self._augment_pair(image_tensor, mask_tensor)

        if self.crop_size != self.image_size:
            image_tensor = TF.resize(image_tensor, [self.image_size, self.image_size], antialias=True)
            mask_tensor = TF.resize(mask_tensor, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        final_mask = (mask_tensor[0].numpy() > 0.5).astype(np.float32)
        final_click_y, final_click_x = self._sample_click_from_binary_mask(final_mask, deterministic=not self.augment)
        click_map = self._make_click_map(final_mask.shape, final_click_x, final_click_y)

        input_tensor = torch.cat(
            [image_tensor, torch.from_numpy(click_map).unsqueeze(0)],
            dim=0,
        )

        return {
            "input": input_tensor,
            "mask": torch.from_numpy(final_mask).unsqueeze(0),
            "click": torch.tensor([final_click_x, final_click_y], dtype=torch.float32),
            "sample_id": record.sample_id,
            "label_id": torch.tensor(record.label_id, dtype=torch.int64),
        }

    def _sample_click(self, mask: np.ndarray, record: SampleRecord) -> tuple[int, int]:
        return self._sample_click_from_binary_mask(mask, deterministic=not self.augment, fallback=(record.centroid_y, record.centroid_x))

    @staticmethod
    def _sample_click_from_binary_mask(
        mask: np.ndarray,
        deterministic: bool,
        fallback: tuple[float, float] | None = None,
    ) -> tuple[int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            h, w = mask.shape
            fy = int(round(fallback[0])) if fallback else h // 2
            fx = int(round(fallback[1])) if fallback else w // 2
            return max(0, min(h - 1, fy)), max(0, min(w - 1, fx))
        if deterministic:
            target_y = float(ys.mean())
            target_x = float(xs.mean())
            distances = (ys - target_y) ** 2 + (xs - target_x) ** 2
            idx = int(np.argmin(distances))
        else:
            idx = random.randrange(len(xs))
        return int(ys[idx]), int(xs[idx])

    def _crop_around_click(self, image: np.ndarray, mask: np.ndarray, click_x: int, click_y: int) -> tuple[np.ndarray, np.ndarray]:
        half = self.crop_size // 2
        if self.augment:
            jitter = self.crop_size // 6
            click_x += random.randint(-jitter, jitter)
            click_y += random.randint(-jitter, jitter)
        x0 = click_x - half
        y0 = click_y - half
        x1 = x0 + self.crop_size
        y1 = y0 + self.crop_size
        image_crop = self._extract_with_padding(image, x0, y0, x1, y1, fill_value=int(image.mean()))
        mask_crop = self._extract_with_padding(mask, x0, y0, x1, y1, fill_value=0)
        return image_crop, mask_crop

    @staticmethod
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

    def _augment_pair(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        rotations = random.randint(0, 3)
        if rotations:
            angle = 90 * rotations
            image = TF.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST)

        if random.random() < 0.8:
            angle = random.uniform(-20.0, 20.0)
            max_translate = int(0.08 * self.crop_size)
            translate = [random.randint(-max_translate, max_translate), random.randint(-max_translate, max_translate)]
            scale = random.uniform(0.9, 1.1)
            shear = [random.uniform(-8.0, 8.0), random.uniform(-8.0, 8.0)]
            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.BILINEAR,
                fill=float(image.mean().item()),
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )

        if random.random() < 0.3:
            image, mask = self._elastic_deform(image, mask)

        image = self._augment_intensity(image)
        return image.clamp(0.0, 1.0), (mask > 0.5).float()

    @staticmethod
    def _augment_intensity(image: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.8:
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.85, 1.15))
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            image = TF.adjust_gamma(image, gamma=gamma)
        if random.random() < 0.3:
            noise = torch.randn_like(image) * random.uniform(0.01, 0.05)
            image = image + noise
        if random.random() < 0.2:
            kernel = random.choice([3, 5])
            image = TF.gaussian_blur(image, kernel_size=[kernel, kernel], sigma=random.uniform(0.1, 1.2))
        if random.random() < 0.4:
            h, w = image.shape[-2:]
            for _ in range(random.randint(1, 3)):
                box_h = random.randint(max(8, h // 20), max(12, h // 6))
                box_w = random.randint(max(8, w // 20), max(12, w // 6))
                y0 = random.randint(0, max(0, h - box_h))
                x0 = random.randint(0, max(0, w - box_w))
                fill = random.uniform(0.0, 1.0)
                image[:, y0 : y0 + box_h, x0 : x0 + box_w] = fill
        return image

    @staticmethod
    def _elastic_deform(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h, w = image.shape
        device = image.device
        displacement = torch.randn(1, 2, h, w, device=device)
        for _ in range(3):
            displacement = F.avg_pool2d(displacement, kernel_size=7, stride=1, padding=3)
        alpha = random.uniform(4.0, 10.0) / max(h, w)
        displacement = displacement * alpha

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing="ij",
        )
        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        flow = displacement.permute(0, 2, 3, 1)
        grid = (base_grid + flow).clamp(-1.0, 1.0)

        image_out = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
        mask_out = F.grid_sample(mask.unsqueeze(0), grid, mode="nearest", padding_mode="zeros", align_corners=True)
        return image_out[0], mask_out[0]

    @staticmethod
    def _make_click_map(shape: tuple[int, int], click_x: int, click_y: int, sigma: float = 10.0) -> np.ndarray:
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        dist_sq = (xx - click_x) ** 2 + (yy - click_y) ** 2
        return np.exp(-dist_sq / (2 * sigma ** 2)).astype(np.float32)


def build_dataloader(
    root_dir: str | Path,
    image_size: int,
    crop_size: int | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    augment: bool,
    extra_dir: str | Path | None = None,
) -> DataLoader:
    datasets: list[Dataset] = [
        ClickSegmentationDataset(root_dir, image_size=image_size, crop_size=crop_size, augment=augment)
    ]
    if extra_dir:
        extra_path = Path(extra_dir)
        if extra_path.exists():
            datasets.append(ClickSegmentationDataset(extra_path, image_size=image_size, crop_size=crop_size, augment=augment))
    dataset: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
