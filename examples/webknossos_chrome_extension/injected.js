(() => {
  const PAGE_WINDOW = window;
  if (PAGE_WINDOW.__pointnclickWebknossosInjected) return;
  PAGE_WINDOW.__pointnclickWebknossosInjected = true;

  const DEFAULT_CONFIG = {
    bridgeUrl: "http://127.0.0.1:8765",
    shortcutKey: "p",
    chunkSize: 5000,
    timeoutMs: 120000,
  };
  const MAX_API_WAIT_MS = 120000;
  const API_POLL_MS = 1000;

  let config = {...DEFAULT_CONFIG};
  let busy = false;
  let nextRequestId = 1;
  const pendingRequests = new Map();

  PAGE_WINDOW.pointnclickWebknossosStatus = {
    loaded: false,
    message: "PointnClick extension injected; loading configuration.",
    bridgeUrl: config.bridgeUrl,
    shortcutKey: config.shortcutKey,
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
    const chunkSize = Math.max(1, Number(config.chunkSize || DEFAULT_CONFIG.chunkSize));
    for (const [y, x0, x1] of runs) {
      for (let x = x0; x < x1; x += 1) {
        chunk.push([x, y, z]);
        if (chunk.length >= chunkSize) {
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

  function requestExtension(action, payload = null, timeoutMs = 15000) {
    const requestId = nextRequestId++;
    return new Promise((resolve, reject) => {
      pendingRequests.set(requestId, {resolve, reject});
      PAGE_WINDOW.postMessage(
        {
          type: "POINTNCLICK_EXTENSION_REQUEST",
          requestId,
          action,
          payload,
        },
        "*",
      );
      setTimeout(() => {
        if (!pendingRequests.has(requestId)) return;
        pendingRequests.delete(requestId);
        reject(new Error("PointnClick extension request timed out."));
      }, timeoutMs);
    });
  }

  PAGE_WINDOW.addEventListener("message", (event) => {
    if (event.source !== PAGE_WINDOW) return;
    const message = event.data;
    if (!message || message.type !== "POINTNCLICK_EXTENSION_RESPONSE") return;
    const pending = pendingRequests.get(message.requestId);
    if (!pending) return;
    pendingRequests.delete(message.requestId);
    if (!message.response) {
      pending.reject(new Error("PointnClick extension returned no response."));
      return;
    }
    pending.resolve(message.response);
  });

  async function loadConfig() {
    const response = await requestExtension("POINTNCLICK_GET_CONFIG");
    if (!response.ok) {
      const message = response.data && response.data.message ? response.data.message : "Could not load PointnClick extension settings.";
      throw new Error(message);
    }
    config = {...DEFAULT_CONFIG, ...response.data};
    PAGE_WINDOW.pointnclickWebknossosStatus.bridgeUrl = config.bridgeUrl;
    PAGE_WINDOW.pointnclickWebknossosStatus.shortcutKey = config.shortcutKey;
    PAGE_WINDOW.pointnclickWebknossosStatus.message = "Configuration loaded; waiting for WebKnossos API.";
    return config;
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

  function installKeyHandler(api) {
    if (PAGE_WINDOW.pointnclickWebknossosStatus.keyHandlerInstalled) return;
    PAGE_WINDOW.pointnclickWebknossosStatus.keyHandlerInstalled = true;
    document.addEventListener("keydown", (event) => {
      const key = String(config.shortcutKey || DEFAULT_CONFIG.shortcutKey).toLowerCase();
      if (event.key.toLowerCase() !== key) return;
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
      const response = await requestExtension(
        "POINTNCLICK_PREDICT",
        {
          position,
          segment_id: segmentId,
          volume_layer_name: api.data.getVolumeTracingLayerName ? api.data.getVolumeTracingLayerName() : null,
        },
        Number(config.timeoutMs || DEFAULT_CONFIG.timeoutMs),
      );
      if (!response.ok) {
        const errorMessage = response.data && response.data.message
          ? response.data.message
          : `Bridge returned HTTP ${response.status || 0}`;
        throw new Error(errorMessage);
      }
      const result = response.data;
      if (result.status !== "ok") {
        throw new Error(result.message || "Bridge returned an error.");
      }
      const chunks = expandRuns(result.runs, result.z);
      for (const voxels of chunks) {
        api.data.labelVoxels(voxels, result.segment_id);
      }
      toast(api, "success", `Painted ${result.num_pixels} voxels into segment ${result.segment_id} in ${Math.round(result.timings_ms.request_total)} ms.`);
    } catch (error) {
      console.error("[PointnClick]", error);
      toast(api, "error", error.message || String(error), 8000);
    } finally {
      busy = false;
    }
  }

  async function initialize() {
    await loadConfig();
    const webknossosHost = await waitForWebKnossosApi();
    const api = await webknossosHost.apiReady(3);
    installKeyHandler(api);
    PAGE_WINDOW.pointnclickWebknossos = {
      run: () => run(api),
      refreshConfig: async () => {
        await loadConfig();
        toast(api, "success", `PointnClick settings refreshed. Shortcut: ${config.shortcutKey.toUpperCase()}.`);
      },
      getConfig: () => ({...config}),
    };
    PAGE_WINDOW.pointnclickWebknossosStatus.loaded = true;
    PAGE_WINDOW.pointnclickWebknossosStatus.message = "PointnClick ready.";
    PAGE_WINDOW.pointnclickWebknossosStatus.readyAt = new Date().toISOString();
    toast(api, "success", `PointnClick ready. Press ${config.shortcutKey.toUpperCase()} at the crosshair to segment.`);
  }

  initialize().catch((error) => {
    PAGE_WINDOW.pointnclickWebknossosStatus.loaded = false;
    PAGE_WINDOW.pointnclickWebknossosStatus.error = error.message || String(error);
    console.error("[PointnClick]", error);
  });
})();
