from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class TrainConfig:
    train_dir: str
    val_dir: str
    output_dir: str
    feedback_dir: str | None = None
    image_size: int = 512
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    resume_checkpoint: str | None = None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
