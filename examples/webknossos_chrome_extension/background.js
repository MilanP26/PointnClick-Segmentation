const DEFAULT_CONFIG = {
  bridgeUrl: "http://127.0.0.1:8765",
  shortcutKey: "p",
  chunkSize: 5000,
  timeoutMs: 120000,
  sessionToken: "",
  username: "",
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
        sessionToken: String(stored.sessionToken || ""),
        username: String(stored.username || ""),
      });
    });
  });
}

function saveAuth(username, sessionToken) {
  return new Promise((resolve) => {
    chrome.storage.local.set({username, sessionToken}, () => resolve());
  });
}

function authHeaders(config) {
  const headers = {"Content-Type": "application/json"};
  if (config.sessionToken) {
    headers.Authorization = `Bearer ${config.sessionToken}`;
  }
  return headers;
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

async function postJsonWithHeaders(url, payload, timeoutMs, headers) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers,
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

async function getJsonWithHeaders(url, timeoutMs, headers) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {headers, signal: controller.signal});
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
      .then((config) => sendResponse({
        ok: true,
        status: 200,
        data: {
          bridgeUrl: config.bridgeUrl,
          shortcutKey: config.shortcutKey,
          chunkSize: config.chunkSize,
          timeoutMs: config.timeoutMs,
          username: config.username,
          signedIn: Boolean(config.sessionToken),
        },
      }))
      .catch((error) => sendResponse({ok: false, status: 0, data: {status: "error", message: error.message || String(error)}}));
    return true;
  }

  if (message.type === "POINTNCLICK_HEALTH") {
    getConfig()
      .then((config) => getJsonWithHeaders(`${config.bridgeUrl}/health`, Math.min(config.timeoutMs, 10000), authHeaders(config)))
      .then(sendResponse)
      .catch((error) => sendResponse({ok: false, status: 0, data: {status: "error", message: error.message || String(error)}}));
    return true;
  }

  if (message.type === "POINTNCLICK_LOGIN") {
    getConfig()
      .then((config) => postJson(`${config.bridgeUrl}/api/auth/login`, message.payload, Math.min(config.timeoutMs, 15000)))
      .then(async (response) => {
        if (response.ok && response.data && response.data.session_token) {
          await saveAuth(String(response.data.username || message.payload.username || ""), String(response.data.session_token));
        }
        sendResponse(response);
      })
      .catch((error) => sendResponse({ok: false, status: 0, data: {status: "error", message: error.message || String(error)}}));
    return true;
  }

  if (message.type === "POINTNCLICK_LOGOUT") {
    saveAuth("", "").then(() => sendResponse({ok: true, status: 200, data: {status: "ok"}}));
    return true;
  }

  if (message.type === "POINTNCLICK_ME") {
    getConfig()
      .then((config) => getJsonWithHeaders(`${config.bridgeUrl}/api/me`, Math.min(config.timeoutMs, 10000), authHeaders(config)))
      .then(sendResponse)
      .catch((error) => sendResponse({ok: false, status: 0, data: {status: "error", message: error.message || String(error)}}));
    return true;
  }

  if (message.type === "POINTNCLICK_PREDICT") {
    getConfig()
      .then((config) => {
        const remotePath = config.sessionToken ? "/api/predict" : "/predict";
        return postJsonWithHeaders(`${config.bridgeUrl}${remotePath}`, message.payload, config.timeoutMs, authHeaders(config));
      })
      .then(sendResponse)
      .catch((error) => sendResponse({ok: false, status: 0, data: {status: "error", message: error.message || String(error)}}));
    return true;
  }

  return false;
});
