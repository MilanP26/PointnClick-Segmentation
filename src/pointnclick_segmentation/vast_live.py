from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from pointnclick_segmentation.feedback import add_feedback_array_sample
from pointnclick_segmentation.infer import predict_mask_from_array
from pointnclick_segmentation.utils import ensure_dir, save_json, save_mask, save_overlay
from pointnclick_segmentation.vast_client import VastClient, VastProtocolError


@dataclass
class VastLiveConfig:
    checkpoint_path: str
    host: str = "127.0.0.1"
    port: int = 22081
    poll_interval_s: float = 0.1
    crop_size: int = 512
    threshold: float = 0.5
    image_size: int | None = None
    device_name: str = "cpu"
    output_dir: str = "outputs\\vast_live"
    allowed_uimode: int | None = None
    auto_segment_key: str = "P"
    feedback_capture_key: str = "I"
    feedback_dir: str = "data\\feedback_vast"
    online_learning: bool = True
    online_learning_output_dir: str = "runs\\live_feedback"
    online_learning_epochs: int = 1
    online_learning_rate: float = 1e-4


def run_vast_live_bridge(config: VastLiveConfig) -> None:
    output_dir = ensure_dir(config.output_dir)
    print("Connecting to VAST Remote Control API server...")
    with VastClient(host=config.host, port=config.port) as client:
        try:
            info = client.get_info()
            data_x, data_y, data_z = (int(info["uints"][0]), int(info["uints"][1]), int(info["uints"][2]))
            client.set_api_layers_enabled(True)
            _, em_layer_nr, seg_layer_nr = client.get_selected_layer_info()
            client.set_api_layers_enabled(False)
            if em_layer_nr < 0 or seg_layer_nr < 0:
                raise RuntimeError("VAST did not report valid EM and segmentation layers")
            print(f"Connected. EM layer={em_layer_nr}, segmentation layer={seg_layer_nr}, volume=({data_x}, {data_y}, {data_z})")
            print(
                f"Watching for clicks. Hold {config.auto_segment_key.upper()} while clicking to auto-segment. "
                f"Hold {config.feedback_capture_key.upper()} while clicking corrected masks to capture feedback."
            )
            last_signature: tuple[int, int, int, int] | None = None
            prev_lbuttondown = 0
            while True:
                state = client.get_current_ui_state()
                current_lbuttondown = int(state["lbuttondown"])
                signature = (
                    int(state["lastleftclickx"]),
                    int(state["lastleftclicky"]),
                    int(state["lastleftreleasex"]),
                    int(state["lastleftreleasey"]),
                )
                if int(state["deletepressed"]) == 1:
                    last_signature = signature
                    prev_lbuttondown = current_lbuttondown
                    time.sleep(config.poll_interval_s)
                    continue
                if not (prev_lbuttondown == 1 and current_lbuttondown == 0):
                    prev_lbuttondown = current_lbuttondown
                    time.sleep(config.poll_interval_s)
                    continue
                prev_lbuttondown = current_lbuttondown
                if signature == last_signature:
                    time.sleep(config.poll_interval_s)
                    continue
                last_signature = signature
                if state["lastleftclickx"] < 0 or state["lastleftclicky"] < 0:
                    time.sleep(config.poll_interval_s)
                    continue
                if config.allowed_uimode is not None and state["uimode"] != config.allowed_uimode:
                    time.sleep(config.poll_interval_s)
                    continue
                wants_feedback = _is_key_pressed(config.feedback_capture_key)
                wants_auto = _is_key_pressed(config.auto_segment_key)
                if not wants_feedback and not wants_auto:
                    time.sleep(config.poll_interval_s)
                    continue
                try:
                    client.set_api_layers_enabled(True)
                    selected_segment = client.get_selected_segment_nr()
                except VastProtocolError as exc:
                    if exc.error_code == 21:
                        client.set_api_layers_enabled(False)
                        time.sleep(config.poll_interval_s)
                        continue
                    raise
                if selected_segment <= 0:
                    client.set_api_layers_enabled(False)
                    time.sleep(config.poll_interval_s)
                    continue
                if wants_feedback:
                    _capture_feedback_click(
                        client=client,
                        config=config,
                        em_layer_nr=em_layer_nr,
                        seg_layer_nr=seg_layer_nr,
                        selected_segment=selected_segment,
                        data_x=data_x,
                        data_y=data_y,
                        data_z=data_z,
                        state=state,
                    )
                    client.set_api_layers_enabled(False)
                    time.sleep(config.poll_interval_s)
                    continue
                _process_click(
                    client=client,
                    config=config,
                    output_dir=output_dir,
                    em_layer_nr=em_layer_nr,
                    seg_layer_nr=seg_layer_nr,
                    selected_segment=selected_segment,
                    data_x=data_x,
                    data_y=data_y,
                    data_z=data_z,
                    state=state,
                )
                client.set_api_layers_enabled(False)
                time.sleep(config.poll_interval_s)
        finally:
            try:
                client.set_api_layers_enabled(False)
            except Exception:
                pass


def _process_click(
    client: VastClient,
    config: VastLiveConfig,
    output_dir: Path,
    em_layer_nr: int,
    seg_layer_nr: int,
    selected_segment: int,
    data_x: int,
    data_y: int,
    data_z: int,
    state: dict[str, int],
) -> None:
    if selected_segment > 65535:
        raise RuntimeError(
            f"Selected segment {selected_segment} exceeds uint16 storage. "
            "This bridge currently supports segment ids up to 65535."
        )
    click_x = int(state["mousecoordsx"])
    click_y = int(state["mousecoordsy"])
    click_z = int(state["mousecoordsz"])
    if not (0 <= click_x < data_x and 0 <= click_y < data_y and 0 <= click_z < data_z):
        return

    half = config.crop_size // 2
    minx = max(click_x - half, 0)
    miny = max(click_y - half, 0)
    maxx = min(minx + config.crop_size - 1, data_x - 1)
    maxy = min(miny + config.crop_size - 1, data_y - 1)
    minx = max(maxx - config.crop_size + 1, 0)
    miny = max(maxy - config.crop_size + 1, 0)
    width = maxx - minx + 1
    height = maxy - miny + 1

    em_image = client.get_em_image(
        layer_nr=em_layer_nr,
        miplevel=0,
        minx=minx,
        maxx=maxx,
        miny=miny,
        maxy=maxy,
        minz=click_z,
        maxz=click_z,
    )
    client.set_selected_api_layer_nr(seg_layer_nr)
    seg_image = client.get_seg_image(
        miplevel=0,
        minx=minx,
        maxx=maxx,
        miny=miny,
        maxy=maxy,
        minz=click_z,
        maxz=click_z,
    )
    local_x = click_x - minx
    local_y = click_y - miny
    pred_mask = predict_mask_from_array(
        checkpoint_path=config.checkpoint_path,
        image=em_image,
        x=local_x,
        y=local_y,
        image_size=config.image_size,
        threshold=config.threshold,
        device_name=config.device_name,
    )
    updated = np.array(seg_image, copy=True)
    updated[pred_mask > 0] = np.uint16(selected_segment)
    client.set_selected_api_layer_nr(seg_layer_nr)
    client.set_seg_image_rle(
        miplevel=0,
        minx=minx,
        maxx=maxx,
        miny=miny,
        maxy=maxy,
        minz=click_z,
        maxz=click_z,
        segimage=updated,
    )
    try:
        client.refresh_layer_region(
            layer_nr=seg_layer_nr,
            minx=minx,
            maxx=maxx,
            miny=miny,
            maxy=maxy,
            minz=click_z,
            maxz=click_z,
        )
    except VastProtocolError as exc:
        print(f"Refresh warning: {exc}")

    stamp = int(time.time() * 1000)
    stem = f"z{click_z}_x{click_x}_y{click_y}_seg{selected_segment}_{stamp}"
    save_mask(pred_mask, output_dir / f"{stem}_mask.png")
    save_overlay(em_image, pred_mask, output_dir / f"{stem}_overlay.png")
    Image.fromarray(em_image, mode="L").save(output_dir / f"{stem}_crop.png")
    save_json(
        {
            "selected_segment": int(selected_segment),
            "uimode": int(state["uimode"]),
            "global_click_xyz": [click_x, click_y, click_z],
            "crop_bounds_xyz": [minx, maxx, miny, maxy, click_z, click_z],
            "local_click_xy": [local_x, local_y],
            "crop_shape_hw": [height, width],
            "note": "Click coordinates come from VAST mouse voxel state at poll time. Keep the cursor on the target when clicking.",
        },
        output_dir / f"{stem}_event.json",
    )
    print(
        f"Wrote segment {selected_segment} at z={click_z} "
        f"for crop x={minx}:{maxx}, y={miny}:{maxy}"
    )


def _capture_feedback_click(
    client: VastClient,
    config: VastLiveConfig,
    em_layer_nr: int,
    seg_layer_nr: int,
    selected_segment: int,
    data_x: int,
    data_y: int,
    data_z: int,
    state: dict[str, int],
) -> None:
    click_x = int(state["mousecoordsx"])
    click_y = int(state["mousecoordsy"])
    click_z = int(state["mousecoordsz"])
    if not (0 <= click_x < data_x and 0 <= click_y < data_y and 0 <= click_z < data_z):
        return
    half = config.crop_size // 2
    minx = max(click_x - half, 0)
    miny = max(click_y - half, 0)
    maxx = min(minx + config.crop_size - 1, data_x - 1)
    maxy = min(miny + config.crop_size - 1, data_y - 1)
    minx = max(maxx - config.crop_size + 1, 0)
    miny = max(maxy - config.crop_size + 1, 0)

    em_image = client.get_em_image(
        layer_nr=em_layer_nr,
        miplevel=0,
        minx=minx,
        maxx=maxx,
        miny=miny,
        maxy=maxy,
        minz=click_z,
        maxz=click_z,
    )
    client.set_selected_api_layer_nr(seg_layer_nr)
    seg_image = client.get_seg_image(
        miplevel=0,
        minx=minx,
        maxx=maxx,
        miny=miny,
        maxy=maxy,
        minz=click_z,
        maxz=click_z,
    )
    mask = (seg_image == np.uint16(selected_segment)).astype(np.uint8)
    if mask.sum() == 0:
        print("Feedback capture skipped: selected segment has no pixels in the current crop.")
        return
    stamp = int(time.time() * 1000)
    sample_id = f"z{click_z}_x{click_x}_y{click_y}_seg{selected_segment}_{stamp}"
    result = add_feedback_array_sample(
        image=em_image,
        mask=mask,
        feedback_dir=config.feedback_dir,
        sample_id=sample_id,
        metadata={
            "selected_segment": int(selected_segment),
            "global_click_xyz": [click_x, click_y, click_z],
            "crop_bounds_xyz": [minx, maxx, miny, maxy, click_z, click_z],
        },
    )
    print(f"Captured feedback sample: {result['mask']}")
    if config.online_learning:
        _run_online_learning(config)


def _run_online_learning(config: VastLiveConfig) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    run_cli = repo_root / "run_cli.py"
    output_dir = Path(config.online_learning_output_dir)
    command = [
        sys.executable,
        str(run_cli),
        "finetune",
        "--checkpoint",
        config.checkpoint_path,
        "--train-dir",
        config.feedback_dir,
        "--val-dir",
        config.feedback_dir,
        "--feedback-dir",
        config.feedback_dir,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(config.online_learning_epochs),
        "--batch-size",
        "1",
        "--learning-rate",
        str(config.online_learning_rate),
        "--device",
        config.device_name,
    ]
    print("Running online feedback update...")
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        print("Online feedback update failed.")
        if result.stderr.strip():
            print(result.stderr.strip())
        return
    new_checkpoint = output_dir / "best_model.pt"
    if new_checkpoint.exists():
        config.checkpoint_path = str(new_checkpoint)
        print(f"Updated live model checkpoint to: {config.checkpoint_path}")


def _is_key_pressed(key: str) -> bool:
    if not key:
        return False
    vk = ord(key.upper())
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
