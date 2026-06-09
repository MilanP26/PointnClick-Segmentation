// Browser-side client for the PointnClick WebKnossos bridge.
// The Python bridge also serves a configured copy at http://127.0.0.1:8765/client.js.
(() => {
  const PAGE_WINDOW = typeof unsafeWindow !== "undefined" ? unsafeWindow : window;
  const BRIDGE_URL = "http://127.0.0.1:8765";
  const KEY = "p";
  const CHUNK_SIZE = 5000;
  const MAX_API_WAIT_MS = 120000;
  const API_POLL_MS = 1000;
  let busy = false;
  PAGE_WINDOW.pointnclickWebknossosStatus = {
    loaded: false,
    message: "PointnClick userscript injected; waiting for WebKnossos API.",
    bridgeUrl: BRIDGE_URL,
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

  function postToBridge(path, payload) {
    const url = `${BRIDGE_URL}${path}`;
    if (typeof GM_xmlhttpRequest === "function") {
      return new Promise((resolve, reject) => {
        GM_xmlhttpRequest({
          method: "POST",
          url,
          headers: {"Content-Type": "application/json"},
          data: JSON.stringify(payload),
          onload: (response) => {
            try {
              resolve({
                ok: response.status >= 200 && response.status < 300,
                status: response.status,
                json: async () => JSON.parse(response.responseText),
              });
            } catch (error) {
              reject(error);
            }
          },
          onerror: reject,
          ontimeout: () => reject(new Error("Bridge request timed out")),
        });
      });
    }
    return fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
  }

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
          reject(new Error("WebKnossos API was not found after waiting. Confirm this script is enabled on the annotation page."));
          return;
        }
        setTimeout(tick, API_POLL_MS);
      };
      tick();
    });
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
      const response = await postToBridge("/predict", {
          position,
          segment_id: segmentId,
          volume_layer_name: api.data.getVolumeTracingLayerName ? api.data.getVolumeTracingLayerName() : null,
      });
      const result = await response.json();
      if (!response.ok || result.status !== "ok") {
        throw new Error(result.message || `Bridge returned HTTP ${response.status}`);
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
    PAGE_WINDOW.pointnclickWebknossos = {run: () => run(api), bridgeUrl: BRIDGE_URL};
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
