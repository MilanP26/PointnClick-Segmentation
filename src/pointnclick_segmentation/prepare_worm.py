from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import shutil
import zipfile

from PIL import Image

from pointnclick_segmentation.utils import ensure_dir, save_json


SLICE_PATTERN = re.compile(r"_s(\d+)", re.IGNORECASE)


def _slice_nr(path_like: str) -> int:
    match = SLICE_PATTERN.search(path_like)
    if not match:
        raise ValueError(f"Could not extract slice number from {path_like}")
    return int(match.group(1))


def _list_pngs_in_dir(directory: Path) -> dict[int, Path]:
    return {
        _slice_nr(path.name): path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() == ".png"
    }


def _list_pngs_in_zip(zip_path: Path) -> dict[int, str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return {
            _slice_nr(name): name
            for name in sorted(zf.namelist())
            if name.lower().endswith(".png")
        }


def _read_png_from_zip(zip_path: Path, member: str) -> bytes:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.read(member)


def _copy_png(source: Path, destination: Path) -> None:
    ensure_dir(destination.parent)
    shutil.copy2(source, destination)


def _write_png_from_bytes(payload: bytes, destination: Path) -> None:
    ensure_dir(destination.parent)
    image = Image.open(BytesIO(payload))
    image.save(destination)


def _resolve_source_maps(data_dir: str | Path) -> tuple[dict[int, Path | str], dict[int, Path | str], Path | None, Path | None]:
    data_dir = Path(data_dir)
    em_dir = data_dir / "em"
    mask_dir = data_dir / "mask"
    em_zip = data_dir / "em.zip"
    mask_zip = data_dir / "mask.zip"

    if em_dir.exists() and mask_dir.exists():
        return _list_pngs_in_dir(em_dir), _list_pngs_in_dir(mask_dir), None, None
    if em_zip.exists() and mask_zip.exists():
        return _list_pngs_in_zip(em_zip), _list_pngs_in_zip(mask_zip), em_zip, mask_zip
    raise FileNotFoundError(
        f"Expected either extracted 'em'/'mask' folders or 'em.zip'/'mask.zip' under {data_dir}"
    )


def prepare_worm_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    em_map, mask_map, em_zip, mask_zip = _resolve_source_maps(data_dir)
    common_slices = sorted(set(em_map) & set(mask_map))
    if not common_slices:
        raise ValueError("No matching EM/mask slices found in the worm dataset")

    num_total = len(common_slices)
    num_test = max(1, int(round(num_total * test_fraction)))
    num_val = max(1, int(round(num_total * val_fraction)))
    if num_val + num_test >= num_total:
        raise ValueError("Validation and test fractions are too large for the number of slices")

    train_slices = common_slices[: num_total - num_val - num_test]
    val_slices = common_slices[num_total - num_val - num_test : num_total - num_test]
    test_slices = common_slices[num_total - num_test :]
    split_map = {
        "train": train_slices,
        "val": val_slices,
        "test": test_slices,
    }

    for split, slices in split_map.items():
        image_dir = ensure_dir(output_dir / split / "images")
        mask_dir = ensure_dir(output_dir / split / "masks")
        for slice_nr in slices:
            image_name = f"worm_s{slice_nr:04d}.png"
            mask_name = f"worm_s{slice_nr:04d}.png"
            if em_zip is None or mask_zip is None:
                _copy_png(Path(em_map[slice_nr]), image_dir / image_name)
                _copy_png(Path(mask_map[slice_nr]), mask_dir / mask_name)
            else:
                _write_png_from_bytes(_read_png_from_zip(em_zip, str(em_map[slice_nr])), image_dir / image_name)
                _write_png_from_bytes(_read_png_from_zip(mask_zip, str(mask_map[slice_nr])), mask_dir / mask_name)

    summary = {
        "num_total": num_total,
        "num_train": len(train_slices),
        "num_val": len(val_slices),
        "num_test": len(test_slices),
        "train_slices": train_slices,
        "val_slices": val_slices,
        "test_slices": test_slices,
    }
    save_json(summary, output_dir / "split_summary.json")
    return summary
