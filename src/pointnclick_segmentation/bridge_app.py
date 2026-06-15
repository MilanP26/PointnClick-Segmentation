from __future__ import annotations

import json
import os
import queue
import socket
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, fields
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pointnclick_segmentation.model_store import (
    DEFAULT_MODEL_FILENAME,
    default_app_dir,
    default_config_path,
    default_model_dir,
    default_model_url,
    download_model,
    filename_from_url,
)


def default_log_path() -> Path:
    return default_app_dir() / "bridge_app.log"


def append_log_file(message: str) -> None:
    path = default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message.rstrip()}\n")


class LogFileStream:
    def write(self, message: str) -> int:
        if message and message.strip():
            append_log_file(message)
        return len(message)

    def flush(self) -> None:
        return


def install_windowed_stdio_redirect() -> None:
    stream = LogFileStream()
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream

if TYPE_CHECKING:
    from pointnclick_segmentation.webknossos_bridge import WebKnossosBridgeConfig


@dataclass
class BridgeAppConfig:
    checkpoint_path: str = ""
    checkpoint_url: str = field(default_factory=default_model_url)
    checkpoint_sha256: str = ""
    dataset: str = ""
    organization_id: str = ""
    annotation: str = ""
    sharing_token: str = ""
    webknossos_url: str = "https://webknossos.org"
    token: str = ""
    remember_token: bool = False
    color_layer: str = "color"
    mag: str = "1"
    host: str = "127.0.0.1"
    port: int = 8765
    crop_size: int = 512
    threshold: float = 0.5
    image_size: str = ""
    device_name: str = "cuda"
    timeout_s: int = 120
    output_dir: str = ""
    client_key: str = "p"

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(default_app_dir() / "webknossos_bridge")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BridgeAppConfig":
        config_path = Path(path) if path else default_config_path()
        if not config_path.exists():
            return cls()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        valid_names = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in raw.items() if key in valid_names}
        return cls(**filtered)

    def save(self, path: str | Path | None = None) -> Path:
        config_path = Path(path) if path else default_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        if not self.remember_token:
            payload["token"] = ""
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return config_path

    def bridge_url(self) -> str:
        return f"http://{self.host}:{int(self.port)}"

    def to_bridge_config(self, checkpoint_path: str) -> WebKnossosBridgeConfig:
        from pointnclick_segmentation.webknossos_bridge import WebKnossosBridgeConfig

        image_size = int(self.image_size) if str(self.image_size).strip() else None
        return WebKnossosBridgeConfig(
            checkpoint_path=checkpoint_path,
            dataset=self.dataset.strip(),
            organization_id=_none_if_blank(self.organization_id),
            annotation=_none_if_blank(self.annotation),
            sharing_token=_none_if_blank(self.sharing_token),
            webknossos_url=self.webknossos_url.strip() or "https://webknossos.org",
            token=_none_if_blank(self.token) or os.environ.get("WEBKNOSSOS_TOKEN"),
            color_layer=self.color_layer.strip() or "color",
            mag=self.mag.strip() or "1",
            host=self.host.strip() or "127.0.0.1",
            port=int(self.port),
            crop_size=int(self.crop_size),
            threshold=float(self.threshold),
            image_size=image_size,
            device_name=self.device_name.strip() or "cuda",
            timeout_s=int(self.timeout_s),
            output_dir=self.output_dir.strip() or str(default_app_dir() / "webknossos_bridge"),
            client_key=(self.client_key.strip() or "p").lower(),
        )


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class LocalBridgeRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._server: ReusableThreadingHTTPServer | None = None
        self._bridge: WebKnossosBridge | None = None
        self._thread: threading.Thread | None = None

    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, config: WebKnossosBridgeConfig, callback: Callable[[str, dict[str, Any]], None]) -> None:
        if self.is_running():
            raise RuntimeError("The bridge is already running")

        def worker() -> None:
            bridge: Any | None = None
            server: ReusableThreadingHTTPServer | None = None
            try:
                from pointnclick_segmentation.webknossos_bridge import WebKnossosBridge, make_handler

                callback("log", {"message": "Loading model and opening WebKnossos dataset..."})
                bridge = WebKnossosBridge(config)
                handler = make_handler(bridge)
                server = ReusableThreadingHTTPServer((config.host, config.port), handler)
                with self._lock:
                    self._bridge = bridge
                    self._server = server
                callback(
                    "ready",
                    {
                        "url": f"http://{config.host}:{config.port}",
                        "dataset": config.dataset,
                        "layer_names": bridge.layer_names,
                    },
                )
                server.serve_forever()
            except Exception as exc:
                callback(
                    "error",
                    {
                        "message": friendly_error_message(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            finally:
                if server is not None:
                    server.server_close()
                if bridge is not None:
                    bridge.close()
                with self._lock:
                    if self._server is server:
                        self._server = None
                    if self._bridge is bridge:
                        self._bridge = None
                callback("stopped", {})

        self._thread = threading.Thread(target=worker, name="PointnClickBridge", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            server = self._server
        if server is not None:
            server.shutdown()


def _none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def friendly_error_message(exc: BaseException) -> str:
    text = str(exc)
    lowered = text.lower()
    if isinstance(exc, OSError) and "address already in use" in lowered:
        return "Port 8765 is already in use. Close the other PointnClick Bridge window, or change the port in Advanced settings."
    if "could not open webknossos color layer" in lowered or "could not open webknossos layer" in lowered:
        return text + "\n\nOpen Advanced settings and check Raw layer. Common values are color, em, image, or raw."
    if "could not open magnification" in lowered:
        return text + "\n\nOpen Advanced settings and check Magnification. Start with 1."
    if "401" in lowered or "unauthorized" in lowered or "forbidden" in lowered or "403" in lowered:
        return text + "\n\nCheck the WebKnossos token and dataset permissions."
    if "checkpoint" in lowered and ("no such file" in lowered or "cannot find" in lowered or "does not exist" in lowered):
        return text + "\n\nClick Download model, or choose a local best_model.pt file."
    if "dynamic link library" in lowered or "c10.dll" in lowered or "torch_cuda" in lowered or "torch" in lowered and "failed" in lowered:
        return text + "\n\nTorch failed to load inside the packaged app. Rebuild the bridge app on the CUDA/PyTorch environment that works on this machine, or use a machine with a compatible NVIDIA driver."
    if "timed out" in lowered:
        return text + "\n\nThe WebKnossos request timed out. Check internet/VPN access and confirm the dataset URL opens in the browser."
    return text or repr(exc)


def check_port_available(host: str, port: int) -> tuple[bool, str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True, f"Port {host}:{port} is available."
    except OSError as exc:
        return False, f"Port {host}:{port} is not available: {exc}"


def build_diagnostics_report(config: BridgeAppConfig) -> list[str]:
    lines: list[str] = []
    lines.append("Diagnostics report")
    lines.append(f"App data folder: {default_app_dir()}")
    lines.append(f"Config file: {default_config_path()}")
    lines.append(f"Log file: {default_log_path()}")
    lines.append(f"Bridge URL: {config.bridge_url()}")

    if config.dataset.strip():
        lines.append("OK: Dataset/view URL is filled.")
    else:
        lines.append("FAIL: Dataset/view URL is empty.")

    if config.token.strip() or os.environ.get("WEBKNOSSOS_TOKEN"):
        lines.append("OK: WebKnossos token is present.")
    else:
        lines.append("WARN: WebKnossos token is empty. This only works for public/shared datasets.")

    checkpoint_path = Path(config.checkpoint_path).expanduser() if config.checkpoint_path.strip() else None
    if checkpoint_path and checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        lines.append(f"OK: Model checkpoint exists: {checkpoint_path} ({size_mb:.1f} MB)")
    elif checkpoint_path:
        lines.append(f"FAIL: Model checkpoint path does not exist: {checkpoint_path}")
    elif config.checkpoint_url.strip():
        lines.append("OK: Model checkpoint is blank, but Model download URL is filled.")
    else:
        lines.append("FAIL: No model checkpoint and no Model download URL.")

    if config.checkpoint_url.strip():
        lines.append(f"Model download URL: {config.checkpoint_url.strip()}")

    try:
        port_ok, port_message = check_port_available(config.host.strip() or "127.0.0.1", int(config.port))
        lines.append(("OK: " if port_ok else "FAIL: ") + port_message)
    except Exception as exc:
        lines.append(f"FAIL: Could not check port: {exc}")

    try:
        import torch

        lines.append(f"OK: torch imports. Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        lines.append(f"CUDA available: {cuda_available}")
        if cuda_available:
            try:
                lines.append(f"CUDA device count: {torch.cuda.device_count()}")
                lines.append(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
            except Exception as exc:
                lines.append(f"WARN: CUDA is available, but device name check failed: {exc}")
        elif config.device_name.strip().lower() == "cuda":
            lines.append("FAIL: Device is set to cuda, but torch.cuda.is_available() is False on this computer.")
    except Exception as exc:
        lines.append("FAIL: torch import failed.")
        lines.append(friendly_error_message(exc))
        lines.append(traceback.format_exc())

    try:
        import webknossos as wk

        lines.append(f"OK: webknossos imports. Version: {getattr(wk, '__version__', 'unknown')}")
    except Exception as exc:
        lines.append("FAIL: webknossos import failed.")
        lines.append(friendly_error_message(exc))
        lines.append(traceback.format_exc())

    lines.append("Diagnostics complete.")
    return lines


class BridgeApp:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import scrolledtext, ttk

        self.tk = tk
        self.ttk = ttk
        self.scrolledtext = scrolledtext
        self.root = tk.Tk()
        self.root.title("PointnClick WebKnossos Bridge")
        self.root.geometry("820x660")
        self.root.minsize(760, 560)
        self.events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self.runtime = LocalBridgeRuntime()
        self.config = BridgeAppConfig.load()
        self.vars: dict[str, Any] = {}
        self.status_var = tk.StringVar(value="Stopped")
        self.bridge_url_var = tk.StringVar(value=self.config.bridge_url())
        self._build_ui()
        self._load_vars_from_config()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(150, self._pump_events)

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, padding=(14, 12, 14, 6))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="PointnClick WebKnossos Bridge", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status_var).grid(row=0, column=1, sticky="e")
        ttk.Label(header, textvariable=self.bridge_url_var, foreground="#555555").grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        notebook = ttk.Notebook(root)
        notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        self.main_tab = ttk.Frame(notebook, padding=12)
        self.advanced_tab = ttk.Frame(notebook, padding=12)
        self.log_tab = ttk.Frame(notebook, padding=8)
        notebook.add(self.main_tab, text="Bridge")
        notebook.add(self.advanced_tab, text="Advanced")
        notebook.add(self.log_tab, text="Log")

        self._build_main_tab()
        self._build_advanced_tab()

        self.log_text = self.scrolledtext.ScrolledText(self.log_tab, height=18, wrap=tk.WORD, state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def _build_main_tab(self) -> None:
        ttk = self.ttk
        tab = self.main_tab
        tab.columnconfigure(1, weight=1)

        row = 0
        self._add_entry(tab, "Dataset/view URL", "dataset", row)
        row += 1
        self._add_entry(tab, "WebKnossos token", "token", row, show="*")
        row += 1
        self._add_check(tab, "Remember token in local config", "remember_token", row)
        row += 1
        self._add_entry(tab, "Model checkpoint", "checkpoint_path", row, browse_file=True)
        row += 1
        self._add_entry(tab, "Model download URL", "checkpoint_url", row)
        row += 1
        self._add_entry(tab, "Model SHA256 optional", "checkpoint_sha256", row)
        row += 1

        controls = ttk.Frame(tab)
        controls.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(14, 4))
        controls.columnconfigure(4, weight=1)
        self.start_button = ttk.Button(controls, text="Start bridge", command=self._start_bridge)
        self.start_button.grid(row=0, column=0, padx=(0, 8))
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop_bridge, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 8))
        ttk.Button(controls, text="Download model", command=self._download_model_now).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(controls, text="Diagnostics", command=self._run_diagnostics).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(controls, text="Save settings", command=self._save_settings).grid(row=0, column=4, padx=(0, 8))
        row += 1

        note = (
            "In WebKnossos, select the segment/color you want, place the crosshair, then press the extension shortcut. "
            "The bridge writes normal volume annotation voxels, so eraser and brush edits still work afterward."
        )
        ttk.Label(tab, text=note, wraplength=720, foreground="#444444").grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))

    def _build_advanced_tab(self) -> None:
        ttk = self.ttk
        tab = self.advanced_tab
        tab.columnconfigure(1, weight=1)

        row = 0
        self._add_entry(tab, "WebKnossos URL", "webknossos_url", row)
        row += 1
        self._add_entry(tab, "Organization ID", "organization_id", row)
        row += 1
        self._add_entry(tab, "Annotation URL/ID", "annotation", row)
        row += 1
        self._add_entry(tab, "Sharing token", "sharing_token", row)
        row += 1
        self._add_entry(tab, "Raw layer", "color_layer", row)
        row += 1
        self._add_entry(tab, "Magnification", "mag", row)
        row += 1
        self._add_entry(tab, "Crop size", "crop_size", row)
        row += 1
        self._add_entry(tab, "Threshold", "threshold", row)
        row += 1
        self._add_entry(tab, "Model image size", "image_size", row)
        row += 1
        self._add_combobox(tab, "Device", "device_name", row, values=("cuda", "cpu"))
        row += 1
        self._add_entry(tab, "Host", "host", row)
        row += 1
        self._add_entry(tab, "Port", "port", row)
        row += 1
        self._add_entry(tab, "Timeout seconds", "timeout_s", row)
        row += 1
        self._add_entry(tab, "Output folder", "output_dir", row, browse_dir=True)
        row += 1
        self._add_entry(tab, "Extension shortcut", "client_key", row)
        row += 1

        ttk.Label(
            tab,
            text="The extension reads its own shortcut and bridge URL. Keep these values matched in the extension options.",
            wraplength=720,
            foreground="#555555",
        ).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))

    def _add_entry(
        self,
        parent: Any,
        label: str,
        name: str,
        row: int,
        show: str | None = None,
        browse_file: bool = False,
        browse_dir: bool = False,
    ) -> None:
        ttk = self.ttk
        tk = self.tk
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 10))
        var = tk.StringVar()
        self.vars[name] = var
        entry = ttk.Entry(parent, textvariable=var, show=show)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        if name in {"host", "port"}:
            var.trace_add("write", lambda *_args: self._refresh_bridge_url_var())
        if browse_file:
            ttk.Button(parent, text="Browse", command=lambda: self._browse_file(name)).grid(row=row, column=2, sticky="e", padx=(8, 0))
        elif browse_dir:
            ttk.Button(parent, text="Browse", command=lambda: self._browse_dir(name)).grid(row=row, column=2, sticky="e", padx=(8, 0))

    def _add_combobox(self, parent: Any, label: str, name: str, row: int, values: tuple[str, ...]) -> None:
        ttk = self.ttk
        tk = self.tk
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 10))
        var = tk.StringVar()
        self.vars[name] = var
        combo = ttk.Combobox(parent, textvariable=var, values=values)
        combo.grid(row=row, column=1, sticky="ew", pady=4)

    def _add_check(self, parent: Any, label: str, name: str, row: int) -> None:
        ttk = self.ttk
        tk = self.tk
        var = tk.BooleanVar()
        self.vars[name] = var
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=1, columnspan=2, sticky="w", pady=4)

    def _load_vars_from_config(self) -> None:
        values = asdict(self.config)
        for name, var in self.vars.items():
            if name in values:
                var.set(values[name])
        self._refresh_bridge_url_var()

    def _config_from_vars(self) -> BridgeAppConfig:
        raw: dict[str, Any] = {}
        for field in fields(BridgeAppConfig):
            var = self.vars.get(field.name)
            if var is not None:
                raw[field.name] = var.get()
            else:
                raw[field.name] = getattr(self.config, field.name)
        return BridgeAppConfig(
            checkpoint_path=str(raw["checkpoint_path"]),
            checkpoint_url=str(raw["checkpoint_url"]),
            checkpoint_sha256=str(raw["checkpoint_sha256"]),
            dataset=str(raw["dataset"]),
            organization_id=str(raw["organization_id"]),
            annotation=str(raw["annotation"]),
            sharing_token=str(raw["sharing_token"]),
            webknossos_url=str(raw["webknossos_url"]),
            token=str(raw["token"]),
            remember_token=bool(raw["remember_token"]),
            color_layer=str(raw["color_layer"]),
            mag=str(raw["mag"]),
            host=str(raw["host"]),
            port=int(raw["port"]),
            crop_size=int(raw["crop_size"]),
            threshold=float(raw["threshold"]),
            image_size=str(raw["image_size"]),
            device_name=str(raw["device_name"]),
            timeout_s=int(raw["timeout_s"]),
            output_dir=str(raw["output_dir"]),
            client_key=str(raw["client_key"]),
        )

    def _browse_file(self, name: str) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Choose model checkpoint",
            filetypes=(("PyTorch checkpoints", "*.pt *.pth"), ("All files", "*.*")),
        )
        if path:
            self.vars[name].set(path)

    def _browse_dir(self, name: str) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.vars[name].set(path)

    def _refresh_bridge_url_var(self) -> None:
        host = self.vars.get("host").get() if "host" in self.vars else self.config.host
        port = self.vars.get("port").get() if "port" in self.vars else self.config.port
        self.bridge_url_var.set(f"Bridge URL: http://{host or '127.0.0.1'}:{port or '8765'}")

    def _save_settings(self) -> None:
        try:
            self.config = self._config_from_vars()
            path = self.config.save()
            self._log(f"Saved settings to {path}")
        except Exception as exc:
            self._log(f"Could not save settings: {exc}")

    def _resolve_checkpoint_for_start(self, config: BridgeAppConfig) -> str:
        checkpoint_path = Path(config.checkpoint_path).expanduser() if config.checkpoint_path.strip() else None
        if checkpoint_path and checkpoint_path.exists():
            return str(checkpoint_path)
        if not config.checkpoint_url.strip():
            raise ValueError("Choose a model checkpoint or provide a model download URL.")

        return str(self._download_checkpoint_from_url(config))

    def _download_checkpoint_from_url(self, config: BridgeAppConfig) -> Path:
        if not config.checkpoint_url.strip():
            raise ValueError("Model download URL is required.")

        filename = filename_from_url(config.checkpoint_url.strip(), fallback=DEFAULT_MODEL_FILENAME)
        destination = default_model_dir() / filename
        self._queue_event("log", {"message": f"Downloading model to {destination}..."})

        def progress(received: int, total: int | None) -> None:
            if total:
                pct = 100.0 * received / max(total, 1)
                self._queue_event("status", {"message": f"Downloading model {pct:.1f}%"})
            else:
                self._queue_event("status", {"message": f"Downloading model {received / (1024 * 1024):.1f} MB"})

        path = download_model(
            url=config.checkpoint_url.strip(),
            destination=destination,
            expected_sha256=config.checkpoint_sha256.strip() or None,
            progress_callback=progress,
        )
        self._queue_event("log", {"message": f"Downloaded model to {path}"})
        self._queue_event("model_downloaded", {"path": str(path)})
        return path

    def _start_bridge(self) -> None:
        try:
            config = self._config_from_vars()
            config.save()
        except Exception as exc:
            self._log(f"Invalid settings: {exc}")
            return
        if not config.dataset.strip():
            self._log("Dataset/view URL is required.")
            return
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Starting")
        self._log("Starting bridge...")

        def worker() -> None:
            try:
                checkpoint = self._resolve_checkpoint_for_start(config)
                bridge_config = config.to_bridge_config(checkpoint)
                self.runtime.start(bridge_config, self._queue_event)
            except Exception as exc:
                self._queue_event("error", {"message": friendly_error_message(exc), "traceback": traceback.format_exc()})

        threading.Thread(target=worker, name="PointnClickBridgeStartup", daemon=True).start()

    def _stop_bridge(self) -> None:
        self.status_var.set("Stopping")
        self._log("Stopping bridge...")
        self.runtime.stop()

    def _download_model_now(self) -> None:
        try:
            config = self._config_from_vars()
        except Exception as exc:
            self._log(f"Invalid settings: {exc}")
            return
        if not config.checkpoint_url.strip():
            self._log("Model download URL is required.")
            return
        self.status_var.set("Downloading model")

        def worker() -> None:
            try:
                checkpoint = self._download_checkpoint_from_url(config)
                self._queue_event("status", {"message": "Model ready"})
                self._queue_event("log", {"message": f"Model ready: {checkpoint}"})
            except Exception as exc:
                self._queue_event("error", {"message": friendly_error_message(exc), "traceback": traceback.format_exc()})

        threading.Thread(target=worker, name="PointnClickModelDownload", daemon=True).start()

    def _run_diagnostics(self) -> None:
        try:
            config = self._config_from_vars()
        except Exception as exc:
            self._log(f"Invalid settings: {exc}")
            return
        self.status_var.set("Running diagnostics")
        self._log("Running diagnostics...")

        def worker() -> None:
            lines = build_diagnostics_report(config)
            self._queue_event("diagnostics", {"lines": lines})

        threading.Thread(target=worker, name="PointnClickDiagnostics", daemon=True).start()

    def _queue_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.put((event_type, payload))

    def _pump_events(self) -> None:
        while True:
            try:
                event_type, payload = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event_type, payload)
        self.root.after(150, self._pump_events)

    def _handle_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "log":
            self._log(str(payload.get("message", "")))
        elif event_type == "status":
            self.status_var.set(str(payload.get("message", "")))
        elif event_type == "set_var":
            name = str(payload.get("name", ""))
            if name in self.vars:
                self.vars[name].set(str(payload.get("value", "")))
        elif event_type == "model_downloaded":
            path = str(payload.get("path", ""))
            if path and "checkpoint_path" in self.vars:
                self.vars["checkpoint_path"].set(path)
                try:
                    self.config = self._config_from_vars()
                    self.config.save()
                except Exception as exc:
                    self._log(f"Downloaded model, but could not save settings: {exc}")
        elif event_type == "diagnostics":
            lines = [str(line) for line in payload.get("lines", [])]
            for line in lines:
                self._log(line)
            self.status_var.set("Diagnostics complete")
        elif event_type == "ready":
            url = str(payload.get("url", ""))
            self.status_var.set(f"Running at {url}")
            self._log(f"Bridge running at {url}")
            layer_names = payload.get("layer_names")
            if layer_names:
                self._log(f"Available layers: {', '.join(str(name) for name in layer_names)}")
        elif event_type == "error":
            from tkinter import messagebox

            self.status_var.set("Error")
            message = str(payload.get("message", ""))
            self._log(f"Error: {message}")
            tb = payload.get("traceback")
            if tb:
                self._log(str(tb))
            messagebox.showerror(
                "PointnClick Bridge failed",
                f"{message}\n\nTroubleshooting log:\n{default_log_path()}",
            )
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
        elif event_type == "stopped":
            if self.status_var.get() != "Error":
                self.status_var.set("Stopped")
            self._log("Bridge stopped.")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def _log(self, message: str) -> None:
        append_log_file(message)
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        self.runtime.stop()
        self.root.after(250, self.root.destroy)


def main() -> None:
    install_windowed_stdio_redirect()
    BridgeApp().run()


if __name__ == "__main__":
    main()
