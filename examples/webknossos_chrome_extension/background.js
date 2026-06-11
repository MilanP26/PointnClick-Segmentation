const DEFAULT_CONFIG = {
  bridgeUrl: "http://127.0.0.1:8765",
  shortcutKey: "p",
  chunkSize: 5000,
  timeoutMs: 120000,
};

function normalizeBridgeUrl(value) {
  const url = String(value || DEFAULT_CONFIG.bridgeUrl).trim() || DEFAULT_CONFIG.bridgeUrl;
  return url.replace(/\/+$/, "");
}

function getConfig() {
  return new Promise((resolve) => {
    chrome.storage.local.get(DEFAULT_CONFIG, (stored) => {
      resolve({
        bridgeUrl: normalizeBridgeUrl(stored.bridgeUrl),
        shortcutKey: String(stored.shortcutKey || DEFAULT_CONFIG.shortcutKey).trim().toLowerCase() || "p",
        chunkSize: Number(stored.chunkSize || DEFAULT_CONFIG.chunkSize),
        timeoutMs: Number(stored.timeoutMs || DEFAULT_CONFIG.timeoutMs),
      });
    });
  });
}

async function postJson(url, payload, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const data = await response.json();
    return {
      ok: response.ok,
      status: response.status,
      data,
    };
  } finally {
    clearTimeout(timer);
  }
}

async function getJson(url, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {signal: controller.signal});
    const data = await response.json();
    return {
      ok: response.ok,
      status: response.status,
      data,
    };
  } finally {
    clearTimeout(timer);
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || !message.type) {
    return false;
  }

  if (message.type === "POINTNCLICK_GET_CONFIG") {
    getConfig()
      .then((config) => sendResponse({ok: true, status: 200, data: config}))
      .catch((error) => sendResponse({ok: false, status: 0, data: {status: "error", message: error.message || String(error)}}));
    return true;
  }

  if (message.type === "POINTNCLICK_HEALTH") {
    getConfig()
      .then((config) => getJson(`${config.bridgeUrl}/health`, Math.min(config.timeoutMs, 10000)))
      .then(sendResponse)
      .catch((error) => sendResponse({ok: false, status: 0, data: {status: "error", message: error.message || String(error)}}));
    return true;
  }

  if (message.type === "POINTNCLICK_PREDICT") {
    getConfig()
      .then((config) => postJson(`${config.bridgeUrl}/predict`, message.payload, config.timeoutMs))
      .then(sendResponse)
      .catch((error) => sendResponse({ok: false, status: 0, data: {status: "error", message: error.message || String(error)}}));
    return true;
  }

  return false;
});
