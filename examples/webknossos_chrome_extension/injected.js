(() => {
  const PAGE_WINDOW = window;
  const KEY = "p";
  const CHUNK_SIZE = 5000;
  const MAX_API_WAIT_MS = 120000;
  const API_POLL_MS = 1000;
  let busy = false;
  let nextRequestId = 1;
  const pendingRequests = new Map();

  PAGE_WINDOW.pointnclickWebknossosStatus = {
    loaded: false,
    message: "PointnClick Chrome extension injected; waiting for WebKnossos API.",
    bridgeUrl: "http://127.0.0.1:8765",
    startedAt: new Date().toISOString(),
  };

  function toast(api, type, message, timeout = 4000) {
    if (api.utils && api.utils.showToast) {
      api.utils.showToast(type, message, timeout);
    } else {
      console.log(`[PointnClick] ${type}: ${message}`);
    }
  }

  function expandRuns(runs, z) {
    const chunks = [];
    let chunk = [];
    for (const [y, x0, x1] of runs) {
      for (let x = x0; x < x1; x += 1) {
        chunk.push([x, y, z]);
        if (chunk.length >= CHUNK_SIZE) {
          chunks.push(chunk);
          chunk = [];
        }
      }
    }
    if (chunk.length > 0) chunks.push(chunk);
    return chunks;
  }

  function findWebKnossosApiHost() {
    if (PAGE_WINDOW.webknossos && PAGE_WINDOW.webknossos.apiReady) return PAGE_WINDOW.webknossos;
    if (PAGE_WINDOW.parent && PAGE_WINDOW.parent.webknossos && PAGE_WINDOW.parent.webknossos.apiReady) return PAGE_WINDOW.parent.webknossos;
    if (PAGE_WINDOW.opener && PAGE_WINDOW.opener.webknossos && PAGE_WINDOW.opener.webknossos.apiReady) return PAGE_WINDOW.opener.webknossos;
    return null;
  }

  function waitForWebKnossosApi() {
    const startedAt = Date.now();
    return new Promise((resolve, reject) => {
      const tick = () => {
        const host = findWebKnossosApiHost();
        if (host) {
          resolve(host);
          return;
        }
        PAGE_WINDOW.pointnclickWebknossosStatus.message = "Waiting for WebKnossos API.";
        PAGE_WINDOW.pointnclickWebknossosStatus.lastCheckedAt = new Date().toISOString();
        if (Date.now() - startedAt >= MAX_API_WAIT_MS) {
          reject(new Error("WebKnossos API was not found after waiting."));
          return;
        }
        setTimeout(tick, API_POLL_MS);
      };
      tick();
    });
  }

  function postToBridge(payload) {
    const requestId = nextRequestId++;
    return new Promise((resolve, reject) => {
      pendingRequests.set(requestId, {resolve, reject});
      PAGE_WINDOW.postMessage(
        {
          type: "POINTNCLICK_PREDICT_REQUEST",
          requestId,
          payload,
        },
        "*",
      );
      setTimeout(() => {
        if (!pendingRequests.has(requestId)) return;
        pendingRequests.delete(requestId);
        reject(new Error("PointnClick bridge request timed out."));
      }, 120000);
    });
  }

  PAGE_WINDOW.addEventListener("message", (event) => {
    if (event.source !== PAGE_WINDOW) return;
    const message = event.data;
    if (!message || message.type !== "POINTNCLICK_PREDICT_RESPONSE") return;
    const pending = pendingRequests.get(message.requestId);
    if (!pending) return;
    pendingRequests.delete(message.requestId);
    if (!message.response || !message.response.ok) {
      const errorMessage = message.response && message.response.data
        ? message.response.data.message
        : "PointnClick bridge request failed.";
      pending.reject(new Error(errorMessage));
      return;
    }
    pending.resolve(message.response.data);
  });

  function installKeyHandler(api) {
    if (PAGE_WINDOW.pointnclickWebknossosStatus.keyHandlerInstalled) return;
    PAGE_WINDOW.pointnclickWebknossosStatus.keyHandlerInstalled = true;
    document.addEventListener("keydown", (event) => {
      if (event.key.toLowerCase() !== KEY) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.repeat) return;
      event.preventDefault();
      event.stopPropagation();
      run(api);
    }, true);
  }

  async function run(api) {
    if (busy) {
      toast(api, "warning", "PointnClick is still working on the last seed.");
      return;
    }
    busy = true;
    try {
      const position = api.tracing.getCameraPosition().map((value) => Math.round(value));
      const segmentId = api.tracing.getActiveCellId();
      if (!segmentId || segmentId <= 0) {
        throw new Error("Select/create an active segment first.");
      }
      toast(api, "info", `PointnClick seed at ${position.join(", ")}`, 2500);
      const result = await postToBridge({
        position,
        segment_id: segmentId,
        volume_layer_name: api.data.getVolumeTracingLayerName ? api.data.getVolumeTracingLayerName() : null,
      });
      if (result.status !== "ok") {
        throw new Error(result.message || "Bridge returned an error.");
      }
      const chunks = expandRuns(result.runs, result.z);
      for (const voxels of chunks) {
        api.data.labelVoxels(voxels, result.segment_id);
      }
      toast(api, "success", `Painted ${result.num_pixels} voxels in ${Math.round(result.timings_ms.request_total)} ms.`);
    } catch (error) {
      console.error("[PointnClick]", error);
      toast(api, "error", error.message || String(error), 8000);
    } finally {
      busy = false;
    }
  }

  waitForWebKnossosApi().then((webknossosHost) => webknossosHost.apiReady(3)).then((api) => {
    installKeyHandler(api);
    PAGE_WINDOW.pointnclickWebknossos = {run: () => run(api), bridgeUrl: "http://127.0.0.1:8765"};
    PAGE_WINDOW.pointnclickWebknossosStatus.loaded = true;
    PAGE_WINDOW.pointnclickWebknossosStatus.message = "PointnClick ready.";
    PAGE_WINDOW.pointnclickWebknossosStatus.readyAt = new Date().toISOString();
    toast(api, "success", `PointnClick ready. Press ${KEY.toUpperCase()} at the crosshair to segment.`);
  }).catch((error) => {
    PAGE_WINDOW.pointnclickWebknossosStatus.loaded = false;
    PAGE_WINDOW.pointnclickWebknossosStatus.error = error.message || String(error);
    console.error("[PointnClick]", error);
  });
})();
