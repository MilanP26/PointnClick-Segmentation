from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pointnclick_segmentation.data.dataset import _list_records
from pointnclick_segmentation.infer import predict_mask_from_array
from pointnclick_segmentation.utils import ensure_dir, load_grayscale_image, save_mask, save_overlay


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _plot_loss_curves(
    history: dict[str, list[float]],
    output_path: Path,
    log_scale: bool,
    loss_label: str,
) -> None:
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if not train_loss and not val_loss:
        raise ValueError("metrics.json does not contain train_loss or val_loss history")

    width = 1000
    height = 640
    margin_left = 90
    margin_right = 30
    margin_top = 55
    margin_bottom = 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    raw_values = [float(v) for v in train_loss + val_loss if v is not None]
    if log_scale:
        values = [np.log10(max(v, 1e-8)) for v in raw_values]
    else:
        values = raw_values
    y_min = min(values)
    y_max = max(values)
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    num_epochs = max(len(train_loss), len(val_loss))

    def to_xy(epoch_idx: int, value: float) -> tuple[int, int]:
        px = margin_left + int(round(epoch_idx * plot_w / max(num_epochs - 1, 1)))
        mapped_value = np.log10(max(value, 1e-8)) if log_scale else value
        py = margin_top + int(round((y_max - mapped_value) * plot_h / (y_max - y_min)))
        return px, py

    grid_color = (225, 225, 225)
    axis_color = (40, 40, 40)
    for frac in np.linspace(0.0, 1.0, 6):
        y = margin_top + int(round(frac * plot_h))
        draw.line([(margin_left, y), (margin_left + plot_w, y)], fill=grid_color, width=1)
    for frac in np.linspace(0.0, 1.0, 6):
        x = margin_left + int(round(frac * plot_w))
        draw.line([(x, margin_top), (x, margin_top + plot_h)], fill=grid_color, width=1)

    draw.line([(margin_left, margin_top), (margin_left, margin_top + plot_h)], fill=axis_color, width=2)
    draw.line(
        [(margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h)],
        fill=axis_color,
        width=2,
    )

    def draw_series(series: list[float], color: tuple[int, int, int]) -> None:
        if not series:
            return
        points = [to_xy(i, float(value)) for i, value in enumerate(series)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            return
        draw.line(points, fill=color, width=3)

    draw_series([float(v) for v in train_loss], (34, 94, 168))
    draw_series([float(v) for v in val_loss], (196, 68, 54))

    draw.text((margin_left, 16), "Train vs Eval Loss", fill=axis_color)
    draw.text((margin_left, height - 28), "Epoch", fill=axis_color)
    draw.text((12, margin_top - 4), loss_label + (" (log10)" if log_scale else ""), fill=axis_color)

    draw.rectangle((width - 190, 18, width - 18, 76), outline=(180, 180, 180), width=1)
    draw.line([(width - 176, 38), (width - 146, 38)], fill=(34, 94, 168), width=3)
    draw.text((width - 136, 30), "Train", fill=axis_color)
    draw.line([(width - 176, 60), (width - 146, 60)], fill=(196, 68, 54), width=3)
    draw.text((width - 136, 52), "Eval", fill=axis_color)

    for frac in np.linspace(0.0, 1.0, 6):
        y = margin_top + int(round(frac * plot_h))
        value = y_max - frac * (y_max - y_min)
        label = f"{(10 ** value if log_scale else value):.3g}"
        draw.text((18, y - 7), label, fill=(90, 90, 90))
    for frac in np.linspace(0.0, 1.0, 6):
        x = margin_left + int(round(frac * plot_w))
        epoch = 1 + int(round(frac * max(num_epochs - 1, 0)))
        draw.text((x - 8, margin_top + plot_h + 8), str(epoch), fill=(90, 90, 90))

    image.save(output_path)


def _mask_panel(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = np.stack([image, image, image], axis=-1).astype(np.uint8)
    overlay = base.copy()
    overlay[mask > 0] = np.array(color, dtype=np.uint8)
    return (0.65 * base + 0.35 * overlay).clip(0, 255).astype(np.uint8)


def _add_label(image: np.ndarray, text: str) -> np.ndarray:
    panel = Image.fromarray(image, mode="RGB")
    banner_h = 26
    out = Image.new("RGB", (panel.width, panel.height + banner_h), color=(245, 245, 245))
    out.paste(panel, (0, banner_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 6), text, fill=(20, 20, 20))
    return np.asarray(out, dtype=np.uint8)


def _deterministic_click(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return w // 2, h // 2
    target_y = float(ys.mean())
    target_x = float(xs.mean())
    distances = (ys - target_y) ** 2 + (xs - target_x) ** 2
    idx = int(np.argmin(distances))
    return int(xs[idx]), int(ys[idx])


def build_training_report(
    metrics_path: str | Path,
    checkpoint_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    max_examples: int = 6,
    device_name: str = "cuda",
    log_scale: bool = False,
    loss_label: str | None = None,
) -> dict[str, str]:
    metrics = _load_json(metrics_path)
    history = metrics.get("history", {})
    detected_loss_label = loss_label
    if detected_loss_label is None:
        loss_name = str(metrics.get("loss_function", "")).strip().lower()
        detected_loss_label = "Cross-entropy loss" if loss_name == "bce" else "Recorded loss"

    output_dir = ensure_dir(output_dir)
    examples_dir = ensure_dir(output_dir / "eval_examples")
    loss_plot_path = output_dir / ("loss_curve_log.png" if log_scale else "loss_curve.png")
    _plot_loss_curves(history=history, output_path=loss_plot_path, log_scale=log_scale, loss_label=detected_loss_label)

    rows: list[np.ndarray] = []
    for record in _list_records(data_dir)[:max_examples]:
        image = load_grayscale_image(record.image_path)
        dense_mask = np.asarray(Image.open(record.mask_path), dtype=np.int32)
        gt_mask = (dense_mask == record.label_id).astype(np.uint8)
        click_x, click_y = _deterministic_click(gt_mask)
        pred_mask = predict_mask_from_array(
            checkpoint_path=checkpoint_path,
            image=image,
            x=click_x,
            y=click_y,
            device_name=device_name,
        )

        stem = record.sample_id
        save_mask(pred_mask, examples_dir / f"{stem}_pred.png")
        save_overlay(image, pred_mask, examples_dir / f"{stem}_pred_overlay.png")
        save_mask(gt_mask, examples_dir / f"{stem}_gt.png")
        save_overlay(image, gt_mask, examples_dir / f"{stem}_gt_overlay.png")

        click_panel = np.stack([image, image, image], axis=-1).astype(np.uint8)
        rr = 6
        y0 = max(0, click_y - rr)
        y1 = min(click_panel.shape[0], click_y + rr + 1)
        x0 = max(0, click_x - rr)
        x1 = min(click_panel.shape[1], click_x + rr + 1)
        click_panel[y0:y1, click_x : click_x + 1] = np.array([0, 255, 255], dtype=np.uint8)
        click_panel[click_y : click_y + 1, x0:x1] = np.array([0, 255, 255], dtype=np.uint8)

        row = np.concatenate(
            [
                _add_label(np.stack([image, image, image], axis=-1).astype(np.uint8), f"{stem}: image"),
                _add_label(click_panel, f"{stem}: click"),
                _add_label(_mask_panel(image, gt_mask, (64, 255, 64)), f"{stem}: ground truth"),
                _add_label(_mask_panel(image, pred_mask, (255, 64, 64)), f"{stem}: prediction"),
            ],
            axis=1,
        )
        rows.append(row)

    if not rows:
        raise ValueError(f"No labeled evaluation examples found in {data_dir}")

    example_grid_path = output_dir / "eval_examples_grid.png"
    Image.fromarray(np.concatenate(rows, axis=0), mode="RGB").save(example_grid_path)

    summary = {
        "metrics_path": str(Path(metrics_path)),
        "checkpoint_path": str(Path(checkpoint_path)),
        "data_dir": str(Path(data_dir)),
        "loss_plot": str(loss_plot_path),
        "example_grid": str(example_grid_path),
        "examples_dir": str(examples_dir),
        "loss_label": detected_loss_label,
    }
    (output_dir / "report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
