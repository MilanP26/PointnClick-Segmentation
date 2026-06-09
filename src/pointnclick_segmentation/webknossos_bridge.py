from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from pointnclick_segmentation.infer import LoadedPredictor
from pointnclick_segmentation.utils import ensure_dir


@dataclass
class WebKnossosBridgeConfig:
    checkpoint_path: str
    dataset: str
    organization_id: str | None = None
    annotation: str | None = None
    sharing_token: str | None = None
    webknossos_url: str = "https://webknossos.org"
    token: str | None = None
    color_layer: str = "color"
    mag: str = "1"
    host: str = "127.0.0.1"
    port: int = 8765
    crop_size: int = 512
    threshold: float = 0.5
    image_size: int | None = None
    device_name: str = "cuda"
    timeout_s: int = 120
    output_dir: str = "outputs\\webknossos_bridge"
    client_key: str = "p"


class WebKnossosBridge:
    def __init__(self, config: WebKnossosBridgeConfig) -> None:
        try:
            import webknossos as wk
        except ImportError as exc:
            raise RuntimeError(
                "The WebKnossos bridge needs the Python package named 'webknossos'. "
                "Install it in your Anaconda env with: pip install webknossos"
            ) from exc

        self.config = config
        self.wk = wk
        self.output_dir = ensure_dir(config.output_dir)
        self.events_path = self.output_dir / "events.jsonl"
        self.predict_lock = threading.Lock()
        self.context = None
        if config.token:
            self.context = wk.webknossos_context(
                url=config.webknossos_url,
                token=config.token,
                timeout=config.timeout_s,
            )
            self.context.__enter__()
        self.predictor = LoadedPredictor(
            checkpoint_path=config.checkpoint_path,
            image_size=config.image_size,
            crop_size=config.crop_size,
            device_name=config.device_name,
        )
        self.dataset = self._open_dataset()
        self.layer = self._get_layer()
        self.mag_view = self._get_mag_view()
        self.layer_names = _layer_names(self.dataset)

    def close(self) -> None:
        if self.context is not None:
            self.context.__exit__(None, None, None)

    def _open_dataset(self) -> Any:
        dataset = self.config.dataset
        parsed = urlparse(dataset)
        is_url = bool(parsed.scheme and parsed.netloc)
        open_kwargs: dict[str, Any] = {
            "dataset_name_or_url": dataset,
            "annotation_id_or_url": self.config.annotation,
        }
        if not is_url:
            open_kwargs.update(
                {
                    "organization_id": self.config.organization_id,
                    "sharing_token": self.config.sharing_token,
                    "webknossos_url": self.config.webknossos_url,
                }
            )
        return self.wk.RemoteDataset.open(**open_kwargs)

    def _get_layer(self) -> Any:
        if hasattr(self.dataset, "get_layer"):
            try:
                return self.dataset.get_layer(self.config.color_layer)
            except Exception as exc:
                names = _layer_names(self.dataset)
                if len(names) == 1 and self.config.color_layer == "color":
                    return self.dataset.get_layer(names[0])
                raise RuntimeError(
                    f"Could not open WebKnossos color layer '{self.config.color_layer}'. "
                    f"Available layers: {', '.join(names) or 'unknown'}"
                ) from exc

        layers = getattr(self.dataset, "layers", {})
        try:
            return layers[self.config.color_layer]
        except Exception as exc:
            raise RuntimeError(f"Could not open WebKnossos layer '{self.config.color_layer}'") from exc

    def _get_mag_view(self) -> Any:
        attempts: list[Any] = [self.config.mag]
        try:
            attempts.append(int(self.config.mag))
        except ValueError:
            pass
        try:
            attempts.append(self.wk.Mag(self.config.mag))
        except Exception:
            pass

        last_error: Exception | None = None
        for mag in attempts:
            try:
                return self.layer.get_mag(mag)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not open magnification '{self.config.mag}' for layer '{self.config.color_layer}'") from last_error

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "dataset": self.config.dataset,
            "color_layer": self.config.color_layer,
            "available_layers": self.layer_names,
            "mag": self.config.mag,
            "device": self.config.device_name,
            "crop_size": self.config.crop_size,
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_t0 = time.perf_counter()
        position = _as_int_triplet(payload.get("position"), "position")
        segment_id = int(payload.get("segment_id", 0))
        if segment_id <= 0:
            raise ValueError("segment_id must be a positive integer")

        click_x, click_y, click_z = position
        half = self.config.crop_size // 2
        minx = click_x - half
        miny = click_y - half
        width = self.config.crop_size
        height = self.config.crop_size
        timings_ms: dict[str, float] = {}

        with self.predict_lock:
            read_t0 = time.perf_counter()
            image, read_bounds = read_padded_grayscale_crop(
                mag_view=self.mag_view,
                minx=minx,
                miny=miny,
                z=click_z,
                width=width,
                height=height,
            )
            timings_ms["read_em"] = (time.perf_counter() - read_t0) * 1000.0

            predict_t0 = time.perf_counter()
            pred_mask = self.predictor.predict(
                image=image,
                x=click_x - minx,
                y=click_y - miny,
                threshold=self.config.threshold,
            )
            timings_ms["predict_mask"] = (time.perf_counter() - predict_t0) * 1000.0

        runs_t0 = time.perf_counter()
        runs = mask_to_row_runs(pred_mask, minx=minx, miny=miny)
        num_pixels = int(sum(x1 - x0 for _y, x0, x1 in runs))
        timings_ms["encode_runs"] = (time.perf_counter() - runs_t0) * 1000.0
        timings_ms["request_total"] = (time.perf_counter() - request_t0) * 1000.0

        response = {
            "status": "ok",
            "segment_id": segment_id,
            "position": [click_x, click_y, click_z],
            "z": click_z,
            "bbox": [minx, minx + width - 1, miny, miny + height - 1, click_z, click_z],
            "read_bbox": read_bounds,
            "runs": runs,
            "num_pixels": num_pixels,
            "timings_ms": timings_ms,
        }
        self._record_event(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "request": {
                    "position": [click_x, click_y, click_z],
                    "segment_id": segment_id,
                },
                "response": {
                    "num_pixels": num_pixels,
                    "num_runs": len(runs),
                    "bbox": response["bbox"],
                    "read_bbox": response["read_bbox"],
                },
                "timings_ms": timings_ms,
            }
        )
        return response

    def _record_event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")


def run_webknossos_bridge(config: WebKnossosBridgeConfig) -> None:
    token_from_env = os.environ.get("WEBKNOSSOS_TOKEN")
    if config.token is None and token_from_env:
        config.token = token_from_env

    bridge = WebKnossosBridge(config)
    handler = make_handler(bridge)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    url = f"http://{config.host}:{config.port}"
    print("WebKnossos bridge is running.")
    print(f"Bridge health: {url}/health")
    print(f"Browser client script: {url}/client.js")
    print(f"Open WebKnossos, paste/run that script, then press {config.client_key.upper()} at the crosshair to segment.")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        bridge.close()


def make_handler(bridge: WebKnossosBridge) -> type[BaseHTTPRequestHandler]:
    class WebKnossosBridgeHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self._send_bytes(b"", status=204)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._send_json(bridge.health())
                return
            if path == "/client.js":
                script = build_client_script(
                    bridge_url=f"http://{bridge.config.host}:{bridge.config.port}",
                    key=bridge.config.client_key,
                )
                self._send_bytes(script.encode("utf-8"), content_type="text/javascript; charset=utf-8")
                return
            if path == "/userscript.user.js":
                script = build_client_script(
                    bridge_url=f"http://{bridge.config.host}:{bridge.config.port}",
                    key=bridge.config.client_key,
                    userscript=True,
                )
                self._send_bytes(script.encode("utf-8"), content_type="text/javascript; charset=utf-8")
                return
            if path == "/diagnostic.user.js":
                script = build_diagnostic_userscript()
                self._send_bytes(script.encode("utf-8"), content_type="text/javascript; charset=utf-8")
                return
            self._send_json({"status": "error", "message": "Not found"}, status=404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/predict":
                self._send_json({"status": "error", "message": "Not found"}, status=404)
                return
            try:
                payload = self._read_json()
                response = bridge.predict(payload)
                self._send_json(response)
            except Exception as exc:
                self._send_json({"status": "error", "message": str(exc)}, status=500)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
            self._send_bytes(json.dumps(data).encode("utf-8"), status=status, content_type="application/json")

        def _send_bytes(self, data: bytes, status: int = 200, content_type: str = "text/plain") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            if data:
                self.wfile.write(data)

    return WebKnossosBridgeHandler


def read_padded_grayscale_crop(
    mag_view: Any,
    minx: int,
    miny: int,
    z: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, list[int]]:
    read_x0 = max(minx, 0)
    read_y0 = max(miny, 0)
    read_x1 = max(minx + width, read_x0)
    read_y1 = max(miny + height, read_y0)
    read_width = read_x1 - read_x0
    read_height = read_y1 - read_y0
    if read_width <= 0 or read_height <= 0:
        fill = np.zeros((height, width), dtype=np.uint8)
        return fill, [read_x0, read_x0 - 1, read_y0, read_y0 - 1, z, z]

    raw = mag_view.read(
        absolute_offset=(read_x0, read_y0, z),
        size=(read_width, read_height, 1),
    )
    valid = webknossos_array_to_grayscale(raw, width=read_width, height=read_height)
    fill_value = int(valid.mean()) if valid.size else 0
    crop = np.full((height, width), fill_value, dtype=np.uint8)
    dst_x0 = read_x0 - minx
    dst_y0 = read_y0 - miny
    crop[dst_y0:dst_y0 + read_height, dst_x0:dst_x0 + read_width] = valid
    return crop, [read_x0, read_x1 - 1, read_y0, read_y1 - 1, z, z]


def webknossos_array_to_grayscale(data: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 4 and arr.shape[1] == width and arr.shape[2] == height:
        plane = arr[:, :, :, 0]
        xy = plane[0] if plane.shape[0] == 1 else plane[:3].mean(axis=0)
        return _to_uint8(xy.T)

    squeezed = np.squeeze(arr)
    if squeezed.ndim == 2:
        if squeezed.shape == (height, width):
            return _to_uint8(squeezed)
        if squeezed.shape == (width, height):
            return _to_uint8(squeezed.T)

    if squeezed.ndim == 3:
        if squeezed.shape[0] in {1, 3, 4} and squeezed.shape[1] == width and squeezed.shape[2] == height:
            xy = squeezed[0] if squeezed.shape[0] == 1 else squeezed[:3].mean(axis=0)
            return _to_uint8(xy.T)
        if squeezed.shape[-1] in {1, 3, 4} and squeezed.shape[0] == width and squeezed.shape[1] == height:
            xy = squeezed[:, :, 0] if squeezed.shape[-1] == 1 else squeezed[:, :, :3].mean(axis=-1)
            return _to_uint8(xy.T)
        if squeezed.shape == (width, height, 1):
            return _to_uint8(squeezed[:, :, 0].T)

    raise ValueError(f"Could not convert WebKnossos array shape {arr.shape} into a {height}x{width} grayscale image")


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return np.asarray(array, dtype=np.uint8)

    values = np.asarray(array, dtype=np.float32)
    if values.size == 0:
        return values.astype(np.uint8)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)

    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def mask_to_row_runs(mask: np.ndarray, minx: int, miny: int, min_clip_x: int = 0, min_clip_y: int = 0) -> list[list[int]]:
    runs: list[list[int]] = []
    positive = mask > 0
    for local_y, row in enumerate(positive):
        global_y = miny + local_y
        if global_y < min_clip_y:
            continue
        xs = np.flatnonzero(row)
        if xs.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(xs) > 1) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [xs.size]))
        for start, end in zip(starts, ends):
            x0 = max(int(minx + xs[start]), min_clip_x)
            x1 = int(minx + xs[end - 1] + 1)
            if x1 <= min_clip_x:
                continue
            runs.append([global_y, x0, x1])
    return runs


def _as_int_triplet(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be [x, y, z]")
    return int(round(float(value[0]))), int(round(float(value[1]))), int(round(float(value[2])))


def _layer_names(dataset: Any) -> list[str]:
    layers = getattr(dataset, "layers", {})
    if isinstance(layers, dict):
        return [str(name) for name in layers.keys()]
    try:
        return [str(name) for name in layers]
    except TypeError:
        return []


def build_client_script(bridge_url: str, key: str = "p", userscript: bool = False) -> str:
    metadata = ""
    if userscript:
        metadata = f"""// ==UserScript==
// @name         PointnClick WebKnossos Bridge
// @namespace    pointnclick-segmentation
// @version      0.2
// @description  Paint PointnClick model predictions into WebKnossos volume annotations.
// @match        *://*/*
// @match        https://*.webknossos.org/*
// @match        https://demo.wk1.connectomics.hpccloud.mpg.de/*
// @connect      {urlparse(bridge_url).hostname}
// @connect      127.0.0.1
// @connect      localhost
// @grant        GM_xmlhttpRequest
// @grant        unsafeWindow
// @run-at       document-idle
// ==/UserScript==
"""


def build_diagnostic_userscript() -> str:
    return """// ==UserScript==
// @name         PointnClick Tampermonkey Diagnostic
// @namespace    pointnclick-segmentation
// @version      0.1
// @description  Confirms whether Tampermonkey can inject scripts into the current page.
// @match        *://*/*
// @grant        unsafeWindow
// @run-at       document-start
// ==/UserScript==
(function () {
  const pageWindow = typeof unsafeWindow !== "undefined" ? unsafeWindow : window;
  pageWindow.pointnclickTampermonkeyDiagnostic = {
    loaded: true,
    href: pageWindow.location.href,
    loadedAt: new Date().toISOString(),
    message: "Tampermonkey diagnostic script executed.",
  };
  console.log("[PointnClick Diagnostic] Tampermonkey diagnostic script executed.", pageWindow.pointnclickTampermonkeyDiagnostic);
  setTimeout(() => {
    try {
      pageWindow.alert("PointnClick Tampermonkey diagnostic loaded on this page.");
    } catch (error) {
      console.log("[PointnClick Diagnostic] Alert failed:", error);
    }
  }, 1000);
})();
"""
    return metadata + f"""(() => {{
  const PAGE_WINDOW = typeof unsafeWindow !== "undefined" ? unsafeWindow : window;
  const BRIDGE_URL = {json.dumps(bridge_url)};
  const KEY = {json.dumps(key.lower())};
  const CHUNK_SIZE = 5000;
  const MAX_API_WAIT_MS = 120000;
  const API_POLL_MS = 1000;
  let busy = false;
  PAGE_WINDOW.pointnclickWebknossosStatus = {{
    loaded: false,
    message: "PointnClick userscript injected; waiting for WebKnossos API.",
    bridgeUrl: BRIDGE_URL,
    startedAt: new Date().toISOString(),
  }};

  function toast(api, type, message, timeout = 4000) {{
    if (api.utils && api.utils.showToast) {{
      api.utils.showToast(type, message, timeout);
    }} else {{
      console.log(`[PointnClick] ${{type}}: ${{message}}`);
    }}
  }}

  function expandRuns(runs, z) {{
    const chunks = [];
    let chunk = [];
    for (const [y, x0, x1] of runs) {{
      for (let x = x0; x < x1; x += 1) {{
        chunk.push([x, y, z]);
        if (chunk.length >= CHUNK_SIZE) {{
          chunks.push(chunk);
          chunk = [];
        }}
      }}
    }}
    if (chunk.length > 0) chunks.push(chunk);
    return chunks;
  }}

  function findWebKnossosApiHost() {{
    if (PAGE_WINDOW.webknossos && PAGE_WINDOW.webknossos.apiReady) return PAGE_WINDOW.webknossos;
    if (PAGE_WINDOW.parent && PAGE_WINDOW.parent.webknossos && PAGE_WINDOW.parent.webknossos.apiReady) return PAGE_WINDOW.parent.webknossos;
    if (PAGE_WINDOW.opener && PAGE_WINDOW.opener.webknossos && PAGE_WINDOW.opener.webknossos.apiReady) return PAGE_WINDOW.opener.webknossos;
    return null;
  }}

  function postToBridge(path, payload) {{
    const url = `${{BRIDGE_URL}}${{path}}`;
    if (typeof GM_xmlhttpRequest === "function") {{
      return new Promise((resolve, reject) => {{
        GM_xmlhttpRequest({{
          method: "POST",
          url,
          headers: {{"Content-Type": "application/json"}},
          data: JSON.stringify(payload),
          onload: (response) => {{
            try {{
              resolve({{
                ok: response.status >= 200 && response.status < 300,
                status: response.status,
                json: async () => JSON.parse(response.responseText),
              }});
            }} catch (error) {{
              reject(error);
            }}
          }},
          onerror: reject,
          ontimeout: () => reject(new Error("Bridge request timed out")),
        }});
      }});
    }}
    return fetch(url, {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(payload),
    }});
  }}

  function installKeyHandler(api) {{
    if (PAGE_WINDOW.pointnclickWebknossosStatus.keyHandlerInstalled) return;
    PAGE_WINDOW.pointnclickWebknossosStatus.keyHandlerInstalled = true;
    document.addEventListener("keydown", (event) => {{
      if (event.key.toLowerCase() !== KEY) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.repeat) return;
      event.preventDefault();
      event.stopPropagation();
      run(api);
    }}, true);
  }}

  function waitForWebKnossosApi() {{
    const startedAt = Date.now();
    return new Promise((resolve, reject) => {{
      const tick = () => {{
        const host = findWebKnossosApiHost();
        if (host) {{
          resolve(host);
          return;
        }}
        PAGE_WINDOW.pointnclickWebknossosStatus.message = "Waiting for WebKnossos API.";
        PAGE_WINDOW.pointnclickWebknossosStatus.lastCheckedAt = new Date().toISOString();
        if (Date.now() - startedAt >= MAX_API_WAIT_MS) {{
          reject(new Error("WebKnossos API was not found after waiting. Confirm this script is enabled on the annotation page."));
          return;
        }}
        setTimeout(tick, API_POLL_MS);
      }};
      tick();
    }});
  }}

  async function run(api) {{
    if (busy) {{
      toast(api, "warning", "PointnClick is still working on the last seed.");
      return;
    }}
    busy = true;
    try {{
      const position = api.tracing.getCameraPosition().map((value) => Math.round(value));
      const segmentId = api.tracing.getActiveCellId();
      if (!segmentId || segmentId <= 0) {{
        throw new Error("Select/create an active segment first.");
      }}
      toast(api, "info", `PointnClick seed at ${{position.join(", ")}}`, 2500);
      const response = await postToBridge("/predict", {{
          position,
          segment_id: segmentId,
          volume_layer_name: api.data.getVolumeTracingLayerName ? api.data.getVolumeTracingLayerName() : null,
      }});
      const result = await response.json();
      if (!response.ok || result.status !== "ok") {{
        throw new Error(result.message || `Bridge returned HTTP ${{response.status}}`);
      }}
      const chunks = expandRuns(result.runs, result.z);
      for (const voxels of chunks) {{
        api.data.labelVoxels(voxels, result.segment_id);
      }}
      toast(api, "success", `Painted ${{result.num_pixels}} voxels in ${{Math.round(result.timings_ms.request_total)}} ms.`);
    }} catch (error) {{
      console.error("[PointnClick]", error);
      toast(api, "error", error.message || String(error), 8000);
    }} finally {{
      busy = false;
    }}
  }}

  waitForWebKnossosApi().then((webknossosHost) => webknossosHost.apiReady(3)).then((api) => {{
    installKeyHandler(api);
    PAGE_WINDOW.pointnclickWebknossos = {{run: () => run(api), bridgeUrl: BRIDGE_URL}};
    PAGE_WINDOW.pointnclickWebknossosStatus.loaded = true;
    PAGE_WINDOW.pointnclickWebknossosStatus.message = "PointnClick ready.";
    PAGE_WINDOW.pointnclickWebknossosStatus.readyAt = new Date().toISOString();
    toast(api, "success", `PointnClick ready. Press ${{KEY.toUpperCase()}} at the crosshair to segment.`);
  }}).catch((error) => {{
    PAGE_WINDOW.pointnclickWebknossosStatus.loaded = false;
    PAGE_WINDOW.pointnclickWebknossosStatus.error = error.message || String(error);
    console.error("[PointnClick]", error);
  }});
}})();
"""
