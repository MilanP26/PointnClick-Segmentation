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
  let autoMaskEnabled = false;
  let webknossosApi = null;
  let nextRequestId = 1;
  let pointerStart = null;
  const pendingRequests = new Map();

  PAGE_WINDOW.pointnclickWebknossosStatus = {
    loaded: false,
    message: "PointnClick extension injected; loading configuration.",
    bridgeUrl: config.bridgeUrl,
    shortcutKey: config.shortcutKey,
    autoMaskEnabled,
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

  PAGE_WINDOW.addEventListener("message", (event) => {
    if (event.source !== PAGE_WINDOW) return;
    const message = event.data;
    if (!message || message.type !== "POINTNCLICK_PAGE_COMMAND") return;

    handlePageCommand(message.action)
      .then((response) => {
        PAGE_WINDOW.postMessage(
          {
            type: "POINTNCLICK_PAGE_COMMAND_RESPONSE",
            requestId: message.requestId,
            response,
          },
          "*",
        );
      })
      .catch((error) => {
        PAGE_WINDOW.postMessage(
          {
            type: "POINTNCLICK_PAGE_COMMAND_RESPONSE",
            requestId: message.requestId,
            response: {
              ok: false,
              message: error.message || String(error),
              status: {...PAGE_WINDOW.pointnclickWebknossosStatus},
            },
          },
          "*",
        );
      });
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

  function updateFloatingControl() {
    const root = document.getElementById("pointnclick-floating-control");
    if (!root) return;
    root.dataset.enabled = autoMaskEnabled ? "true" : "false";
    root.dataset.busy = busy ? "true" : "false";
    const status = root.querySelector("[data-pointnclick-status]");
    if (status) {
      status.textContent = busy ? "Working" : (autoMaskEnabled ? "On" : "Off");
    }
    const button = root.querySelector("button");
    if (button) {
      button.setAttribute("aria-pressed", autoMaskEnabled ? "true" : "false");
    }
  }

  function installFloatingControl(api) {
    if (document.getElementById("pointnclick-floating-control")) return;
    const style = document.createElement("style");
    style.textContent = `
      #pointnclick-floating-control {
        position: fixed;
        right: max(14px, env(safe-area-inset-right));
        bottom: max(16px, env(safe-area-inset-bottom));
        z-index: 2147483647;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        touch-action: manipulation;
      }
      #pointnclick-floating-control button {
        display: grid;
        grid-template-columns: 11px auto;
        gap: 7px;
        align-items: center;
        min-width: 112px;
        border: 1px solid rgba(16, 24, 40, 0.22);
        border-radius: 8px;
        padding: 10px 12px;
        background: rgba(246, 248, 251, 0.96);
        color: #111827;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.22);
        font-size: 13px;
        font-weight: 750;
      }
      #pointnclick-floating-control .pointnclick-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: #9ca3af;
      }
      #pointnclick-floating-control[data-enabled="true"] button {
        background: rgba(11, 95, 255, 0.96);
        color: #ffffff;
        border-color: rgba(11, 95, 255, 0.85);
      }
      #pointnclick-floating-control[data-enabled="true"] .pointnclick-dot {
        background: #34d399;
      }
      #pointnclick-floating-control[data-busy="true"] .pointnclick-dot {
        background: #f59e0b;
      }
      #pointnclick-floating-control [data-pointnclick-status] {
        font-weight: 650;
        opacity: 0.82;
      }
    `;
    document.documentElement.appendChild(style);

    const root = document.createElement("div");
    root.id = "pointnclick-floating-control";
    root.className = "pointnclick-floating-control";
    root.dataset.enabled = "false";
    root.dataset.busy = "false";
    root.innerHTML = `
      <button type="button" aria-pressed="false" aria-label="Toggle PointnClick Auto Mask">
        <span class="pointnclick-dot" aria-hidden="true"></span>
        <span>Auto Mask <span data-pointnclick-status>Off</span></span>
      </button>
    `;
    root.querySelector("button").addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      autoMaskEnabled = !autoMaskEnabled;
      PAGE_WINDOW.pointnclickWebknossosStatus.autoMaskEnabled = autoMaskEnabled;
      updateFloatingControl();
      toast(api, autoMaskEnabled ? "success" : "info", `Auto Mask ${autoMaskEnabled ? "on" : "off"}.`);
    }, true);
    document.documentElement.appendChild(root);
  }

  function isIgnoredPointerTarget(target) {
    if (!target || !target.closest) return false;
    return Boolean(target.closest(
      "#pointnclick-floating-control, button, input, textarea, select, a, [role='button'], [contenteditable='true']",
    ));
  }

  function installAutoMaskHandler(api) {
    if (PAGE_WINDOW.pointnclickWebknossosStatus.autoMaskHandlerInstalled) return;
    PAGE_WINDOW.pointnclickWebknossosStatus.autoMaskHandlerInstalled = true;
    document.addEventListener("pointerdown", (event) => {
      if (!autoMaskEnabled || isIgnoredPointerTarget(event.target)) {
        pointerStart = null;
        return;
      }
      pointerStart = {
        x: event.clientX,
        y: event.clientY,
        pointerId: event.pointerId,
        time: Date.now(),
        target: event.target,
      };
    }, true);
    document.addEventListener("pointerup", (event) => {
      if (!autoMaskEnabled || !pointerStart || pointerStart.pointerId !== event.pointerId) {
        pointerStart = null;
        return;
      }
      const dx = Math.abs(event.clientX - pointerStart.x);
      const dy = Math.abs(event.clientY - pointerStart.y);
      const elapsed = Date.now() - pointerStart.time;
      const targetIgnored = isIgnoredPointerTarget(event.target) || isIgnoredPointerTarget(pointerStart.target);
      pointerStart = null;
      if (targetIgnored || dx > 12 || dy > 12 || elapsed > 1500) return;
      setTimeout(() => run(api), 160);
    }, true);
  }

  async function run(api) {
    if (busy) {
      toast(api, "warning", "PointnClick is still working on the last seed.");
      return;
    }
    busy = true;
    updateFloatingControl();
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
          dataset_url: PAGE_WINDOW.location.href,
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
      updateFloatingControl();
    }
  }

  async function handlePageCommand(action) {
    if (action === "status") {
      return {
        ok: true,
        status: {
          ...PAGE_WINDOW.pointnclickWebknossosStatus,
          hasApi: Boolean(webknossosApi),
          busy,
          autoMaskEnabled,
        },
      };
    }
    if (action === "refreshConfig") {
      await loadConfig();
      return {
        ok: true,
        status: {
          ...PAGE_WINDOW.pointnclickWebknossosStatus,
          hasApi: Boolean(webknossosApi),
          busy,
          autoMaskEnabled,
        },
      };
    }
    if (action === "run") {
      if (!webknossosApi) {
        throw new Error("PointnClick is injected, but the WebKnossos API is not ready yet.");
      }
      await run(webknossosApi);
      return {
        ok: true,
        status: {
          ...PAGE_WINDOW.pointnclickWebknossosStatus,
          hasApi: Boolean(webknossosApi),
          busy,
          autoMaskEnabled,
        },
      };
    }
    throw new Error(`Unknown PointnClick page command: ${action}`);
  }

  async function initialize() {
    await loadConfig();
    const webknossosHost = await waitForWebKnossosApi();
    const api = await webknossosHost.apiReady(3);
    webknossosApi = api;
    installKeyHandler(api);
    installFloatingControl(api);
    installAutoMaskHandler(api);
    PAGE_WINDOW.pointnclickWebknossos = {
      run: () => run(api),
      toggleAutoMask: () => {
        autoMaskEnabled = !autoMaskEnabled;
        PAGE_WINDOW.pointnclickWebknossosStatus.autoMaskEnabled = autoMaskEnabled;
        updateFloatingControl();
      },
      refreshConfig: async () => {
        await loadConfig();
        toast(api, "success", `PointnClick settings refreshed. Shortcut: ${config.shortcutKey.toUpperCase()}.`);
      },
      getConfig: () => ({...config}),
    };
    PAGE_WINDOW.pointnclickWebknossosStatus.loaded = true;
    PAGE_WINDOW.pointnclickWebknossosStatus.autoMaskEnabled = autoMaskEnabled;
    PAGE_WINDOW.pointnclickWebknossosStatus.message = "PointnClick ready.";
    PAGE_WINDOW.pointnclickWebknossosStatus.readyAt = new Date().toISOString();
    toast(api, "success", "PointnClick ready.");
  }

  initialize().catch((error) => {
    PAGE_WINDOW.pointnclickWebknossosStatus.loaded = false;
    PAGE_WINDOW.pointnclickWebknossosStatus.error = error.message || String(error);
    console.error("[PointnClick]", error);
  });
})();
