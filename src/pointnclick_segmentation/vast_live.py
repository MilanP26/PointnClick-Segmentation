from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from pointnclick_segmentation.feedback import add_feedback_array_sample
from pointnclick_segmentation.infer import LoadedPredictor
from pointnclick_segmentation.utils import ensure_dir, save_json, save_mask, save_overlay
from pointnclick_segmentation.vast_client import VastClient, VastProtocolError


@dataclass
class VastLiveConfig:
    checkpoint_path: str
    host: str = "127.0.0.1"
    port: int = 22081
    api_timeout_s: float = 30.0
    poll_interval_s: float = 0.1
    crop_size: int = 512
    threshold: float = 0.5
    image_size: int | None = None
    device_name: str = "cuda"
    output_dir: str = "outputs\\vast_live"
    allowed_uimode: int | None = None
    auto_segment_key: str = "P"
    feedback_capture_key: str = "I"
    feedback_dir: str = "data\\feedback_vast"
    online_learning: bool = True
    online_learning_output_dir: str = "runs\\live_feedback"
    online_learning_epochs: int = 1
    online_learning_rate: float = 1e-4
    debug_timings: bool = False
    save_click_artifacts: bool = False


class TimingRecorder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = ensure_dir(output_dir / "debug_timings")
        self.events_path = self.output_dir / "timing_events.jsonl"
        self.summary_path = self.output_dir / "timing_summary.json"
        self.records: list[dict[str, object]] = []

    @staticmethod
    def _bottleneck(timings_ms: dict[str, float]) -> tuple[str, float] | None:
        positive = [(key, value) for key, value in timings_ms.items() if value >= 0.0]
        if not positive:
            return None
        return max(positive, key=lambda item: item[1])

    def record(self, event_type: str, timings_ms: dict[str, float], metadata: dict[str, object]) -> None:
        bottleneck = self._bottleneck(timings_ms)
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "timings_ms": timings_ms,
            "metadata": metadata,
            "bottleneck_stage": bottleneck[0] if bottleneck else None,
            "bottleneck_ms": bottleneck[1] if bottleneck else None,
        }
        self.records.append(record)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self._write_summary()

    def _write_summary(self) -> None:
        if not self.records:
            return
        stage_totals: dict[str, float] = {}
        stage_max: dict[str, float] = {}
        stage_counts: dict[str, int] = {}
        bottleneck_counts: dict[str, int] = {}
        for record in self.records:
            timings_ms = record["timings_ms"]
            assert isinstance(timings_ms, dict)
            for stage, value in timings_ms.items():
                stage_totals[stage] = stage_totals.get(stage, 0.0) + float(value)
                stage_max[stage] = max(stage_max.get(stage, 0.0), float(value))
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
            bottleneck_stage = record.get("bottleneck_stage")
            if isinstance(bottleneck_stage, str):
                bottleneck_counts[bottleneck_stage] = bottleneck_counts.get(bottleneck_stage, 0) + 1

        average_timings_ms = {
            stage: stage_totals[stage] / max(stage_counts[stage], 1)
            for stage in sorted(stage_totals)
        }
        dominant_stage = max(average_timings_ms.items(), key=lambda item: item[1])
        summary = {
            "num_records": len(self.records),
            "average_timings_ms": average_timings_ms,
            "max_timings_ms": {stage: stage_max[stage] for stage in sorted(stage_max)},
            "most_common_bottlenecks": bottleneck_counts,
            "dominant_average_stage": dominant_stage[0],
            "dominant_average_stage_ms": dominant_stage[1],
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_vast_live_bridge(config: VastLiveConfig) -> None:
    output_dir = ensure_dir(config.output_dir)
    timing_recorder = TimingRecorder(output_dir) if config.debug_timings else None
    startup_t0 = time.perf_counter()
    predictor = LoadedPredictor(
        checkpoint_path=config.checkpoint_path,
        image_size=config.image_size,
        device_name=config.device_name,
    )
    predictor_load_ms = (time.perf_counter() - startup_t0) * 1000.0
    print("Connecting to VAST Remote Control API server...")
    with VastClient(host=config.host, port=config.port, timeout_s=config.api_timeout_s) as client:
        try:
            connect_t0 = time.perf_counter()
            info = client.get_info()
            get_info_ms = (time.perf_counter() - connect_t0) * 1000.0
            data_x, data_y, data_z = (int(info["uints"][0]), int(info["uints"][1]), int(info["uints"][2]))
            layer_t0 = time.perf_counter()
            client.set_api_layers_enabled(True)
            _, em_layer_nr, seg_layer_nr = client.get_selected_layer_info()
            client.set_api_layers_enabled(False)
            layer_info_ms = (time.perf_counter() - layer_t0) * 1000.0
            if em_layer_nr < 0 or seg_layer_nr < 0:
                raise RuntimeError("VAST did not report valid EM and segmentation layers")
            if timing_recorder is not None:
                timing_recorder.record(
                    event_type="startup",
                    timings_ms={
                        "predictor_load": predictor_load_ms,
                        "get_info": get_info_ms,
                        "get_layer_info": layer_info_ms,
                        "startup_total": predictor_load_ms + get_info_ms + layer_info_ms,
                    },
                    metadata={
                        "checkpoint_path": config.checkpoint_path,
                        "device": config.device_name,
                        "crop_size": config.crop_size,
                        "volume_xyz": [data_x, data_y, data_z],
                    },
                )
            print(f"Connected. EM layer={em_layer_nr}, segmentation layer={seg_layer_nr}, volume=({data_x}, {data_y}, {data_z})")
            print(
                f"Watching for clicks. Hold {config.auto_segment_key.upper()} while clicking to auto-segment. "
                f"Hold {config.feedback_capture_key.upper()} while clicking corrected masks to capture feedback."
            )
            last_signature: tuple[int, int, int, int] | None = None
            prev_lbuttondown = 0
            armed_mode: str | None = None
            while True:
                wants_feedback = _is_key_pressed(config.feedback_capture_key)
                wants_auto = _is_key_pressed(config.auto_segment_key)
                current_lbuttondown = int(_is_left_button_down())

                # Arm only when the user is explicitly holding one of the bridge keys.
                if armed_mode is None:
                    if wants_feedback:
                        armed_mode = "feedback"
                    elif wants_auto:
                        armed_mode = "auto"

                # Stay completely out of the way unless we are currently armed for an explicit bridge action.
                if armed_mode is None:
                    prev_lbuttondown = current_lbuttondown
                    time.sleep(config.poll_interval_s)
                    continue

                # If the user let go of the modifier before clicking, cancel the armed action.
                if armed_mode == "feedback" and not wants_feedback and current_lbuttondown == 0:
                    armed_mode = None
                    prev_lbuttondown = current_lbuttondown
                    time.sleep(config.poll_interval_s)
                    continue
                if armed_mode == "auto" and not wants_auto and current_lbuttondown == 0:
                    armed_mode = None
                    prev_lbuttondown = current_lbuttondown
                    time.sleep(config.poll_interval_s)
                    continue

                # Wait for a real left-button release from the local OS before querying VAST.
                if not (prev_lbuttondown == 1 and current_lbuttondown == 0):
                    prev_lbuttondown = current_lbuttondown
                    time.sleep(config.poll_interval_s)
                    continue
                prev_lbuttondown = current_lbuttondown

                ui_state_t0 = time.perf_counter()
                state = client.get_current_ui_state()
                get_ui_state_ms = (time.perf_counter() - ui_state_t0) * 1000.0
                signature = (
                    int(state["lastleftclickx"]),
                    int(state["lastleftclicky"]),
                    int(state["lastleftreleasex"]),
                    int(state["lastleftreleasey"]),
                )
                if int(state["deletepressed"]) == 1:
                    last_signature = signature
                    armed_mode = None
                    time.sleep(config.poll_interval_s)
                    continue
                if signature == last_signature:
                    armed_mode = None
                    time.sleep(config.poll_interval_s)
                    continue
                last_signature = signature
                if state["lastleftclickx"] < 0 or state["lastleftclicky"] < 0:
                    armed_mode = None
                    time.sleep(config.poll_interval_s)
                    continue
                if config.allowed_uimode is not None and state["uimode"] != config.allowed_uimode:
                    armed_mode = None
                    time.sleep(config.poll_interval_s)
                    continue
                try:
                    selected_segment_t0 = time.perf_counter()
                    client.set_api_layers_enabled(True)
                    selected_segment = client.get_selected_segment_nr()
                    selected_segment_ms = (time.perf_counter() - selected_segment_t0) * 1000.0
                except VastProtocolError as exc:
                    if exc.error_code == 21:
                        client.set_api_layers_enabled(False)
                        armed_mode = None
                        time.sleep(config.poll_interval_s)
                        continue
                    raise
                if selected_segment <= 0:
                    client.set_api_layers_enabled(False)
                    armed_mode = None
                    time.sleep(config.poll_interval_s)
                    continue
                if armed_mode == "feedback":
                    _capture_feedback_click(
                        client=client,
                        config=config,
                        timing_recorder=timing_recorder,
                        initial_timings_ms={
                            "get_ui_state": get_ui_state_ms,
                            "get_selected_segment": selected_segment_ms,
                        },
                        em_layer_nr=em_layer_nr,
                        seg_layer_nr=seg_layer_nr,
                        selected_segment=selected_segment,
                        data_x=data_x,
                        data_y=data_y,
                        data_z=data_z,
                        state=state,
                    )
                    client.set_api_layers_enabled(False)
                    armed_mode = None
                    time.sleep(config.poll_interval_s)
                    continue
                _process_click(
                    client=client,
                    config=config,
                    predictor=predictor,
                    timing_recorder=timing_recorder,
                    initial_timings_ms={
                        "get_ui_state": get_ui_state_ms,
                        "get_selected_segment": selected_segment_ms,
                    },
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
                armed_mode = None
                time.sleep(config.poll_interval_s)
        finally:
            try:
                client.set_api_layers_enabled(False)
            except Exception:
                pass


def _process_click(
    client: VastClient,
    config: VastLiveConfig,
    predictor: LoadedPredictor,
    timing_recorder: TimingRecorder | None,
    initial_timings_ms: dict[str, float],
    output_dir: Path,
    em_layer_nr: int,
    seg_layer_nr: int,
    selected_segment: int,
    data_x: int,
    data_y: int,
    data_z: int,
    state: dict[str, int],
) -> None:
    click_t0 = time.perf_counter()
    timings_ms = dict(initial_timings_ms)
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

    em_t0 = time.perf_counter()
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
    timings_ms["get_em_image"] = (time.perf_counter() - em_t0) * 1000.0
    select_seg_read_t0 = time.perf_counter()
    client.set_selected_api_layer_nr(seg_layer_nr)
    timings_ms["select_seg_layer_for_read"] = (time.perf_counter() - select_seg_read_t0) * 1000.0
    seg_t0 = time.perf_counter()
    seg_image = client.get_seg_image(
        miplevel=0,
        minx=minx,
        maxx=maxx,
        miny=miny,
        maxy=maxy,
        minz=click_z,
        maxz=click_z,
    )
    timings_ms["get_seg_image"] = (time.perf_counter() - seg_t0) * 1000.0
    local_x = click_x - minx
    local_y = click_y - miny
    inference_t0 = time.perf_counter()
    pred_mask = predictor.predict(
        image=em_image,
        x=local_x,
        y=local_y,
        threshold=config.threshold,
    )
    timings_ms["predict_mask"] = (time.perf_counter() - inference_t0) * 1000.0
    updated = np.array(seg_image, copy=True)
    updated[pred_mask > 0] = np.uint16(selected_segment)
    select_seg_write_t0 = time.perf_counter()
    client.set_selected_api_layer_nr(seg_layer_nr)
    timings_ms["select_seg_layer_for_write"] = (time.perf_counter() - select_seg_write_t0) * 1000.0
    write_t0 = time.perf_counter()
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
    timings_ms["set_seg_image_rle"] = (time.perf_counter() - write_t0) * 1000.0
    try:
        refresh_t0 = time.perf_counter()
        client.refresh_layer_region(
            layer_nr=seg_layer_nr,
            minx=minx,
            maxx=maxx,
            miny=miny,
            maxy=maxy,
            minz=click_z,
            maxz=click_z,
        )
        timings_ms["refresh_layer_region"] = (time.perf_counter() - refresh_t0) * 1000.0
    except VastProtocolError as exc:
        timings_ms["refresh_layer_region"] = -1.0
        print(f"Refresh warning: {exc}")

    stamp = int(time.time() * 1000)
    stem = f"z{click_z}_x{click_x}_y{click_y}_seg{selected_segment}_{stamp}"
    event_payload = {
        "selected_segment": int(selected_segment),
        "uimode": int(state["uimode"]),
        "global_click_xyz": [click_x, click_y, click_z],
        "crop_bounds_xyz": [minx, maxx, miny, maxy, click_z, click_z],
        "local_click_xy": [local_x, local_y],
        "crop_shape_hw": [height, width],
        "note": "Click coordinates come from VAST mouse voxel state at poll time. Keep the cursor on the target when clicking.",
    }
    save_t0 = time.perf_counter()
    if config.save_click_artifacts:
        save_mask(pred_mask, output_dir / f"{stem}_mask.png")
        save_overlay(em_image, pred_mask, output_dir / f"{stem}_overlay.png")
        Image.fromarray(em_image, mode="L").save(output_dir / f"{stem}_crop.png")
        save_json(event_payload, output_dir / f"{stem}_event.json")
    timings_ms["save_outputs"] = (time.perf_counter() - save_t0) * 1000.0
    timings_ms["click_total"] = (time.perf_counter() - click_t0) * 1000.0
    if timing_recorder is not None:
        timing_recorder.record(
            event_type="auto_click",
            timings_ms=timings_ms,
            metadata={
                "stem": stem,
                **event_payload,
            },
        )
        save_json(
            {
                "timings_ms": timings_ms,
                "event": event_payload,
            },
            output_dir / f"{stem}_timing.json",
        )
    print(
        f"Wrote segment {selected_segment} at z={click_z} "
        f"for crop x={minx}:{maxx}, y={miny}:{maxy}"
    )


def _capture_feedback_click(
    client: VastClient,
    config: VastLiveConfig,
    timing_recorder: TimingRecorder | None,
    initial_timings_ms: dict[str, float],
    em_layer_nr: int,
    seg_layer_nr: int,
    selected_segment: int,
    data_x: int,
    data_y: int,
    data_z: int,
    state: dict[str, int],
) -> None:
    feedback_t0 = time.perf_counter()
    timings_ms = dict(initial_timings_ms)
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

    em_t0 = time.perf_counter()
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
    timings_ms["get_em_image"] = (time.perf_counter() - em_t0) * 1000.0
    select_seg_read_t0 = time.perf_counter()
    client.set_selected_api_layer_nr(seg_layer_nr)
    timings_ms["select_seg_layer_for_read"] = (time.perf_counter() - select_seg_read_t0) * 1000.0
    seg_t0 = time.perf_counter()
    seg_image = client.get_seg_image(
        miplevel=0,
        minx=minx,
        maxx=maxx,
        miny=miny,
        maxy=maxy,
        minz=click_z,
        maxz=click_z,
    )
    timings_ms["get_seg_image"] = (time.perf_counter() - seg_t0) * 1000.0
    mask = (seg_image == np.uint16(selected_segment)).astype(np.uint8)
    if mask.sum() == 0:
        print("Feedback capture skipped: selected segment has no pixels in the current crop.")
        return
    stamp = int(time.time() * 1000)
    sample_id = f"z{click_z}_x{click_x}_y{click_y}_seg{selected_segment}_{stamp}"
    save_feedback_t0 = time.perf_counter()
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
    timings_ms["save_feedback_sample"] = (time.perf_counter() - save_feedback_t0) * 1000.0
    print(f"Captured feedback sample: {result['mask']}")
    if config.online_learning:
        online_t0 = time.perf_counter()
        _run_online_learning(config)
        timings_ms["online_learning"] = (time.perf_counter() - online_t0) * 1000.0
    timings_ms["feedback_total"] = (time.perf_counter() - feedback_t0) * 1000.0
    if timing_recorder is not None:
        timing_recorder.record(
            event_type="feedback_click",
            timings_ms=timings_ms,
            metadata={
                "sample_id": sample_id,
                "selected_segment": int(selected_segment),
                "global_click_xyz": [click_x, click_y, click_z],
                "crop_bounds_xyz": [minx, maxx, miny, maxy, click_z, click_z],
            },
        )
        save_json(
            {
                "timings_ms": timings_ms,
                "sample_id": sample_id,
            },
            Path(result["image"]).parents[1] / f"{sample_id}_timing.json",
        )


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


def _is_left_button_down() -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000)
