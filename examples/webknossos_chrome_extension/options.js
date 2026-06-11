const DEFAULT_CONFIG = {
  bridgeUrl: "http://127.0.0.1:8765",
  shortcutKey: "p",
  chunkSize: 5000,
  timeoutMs: 120000,
};

const fields = {
  bridgeUrl: document.getElementById("bridge-url"),
  shortcutKey: document.getElementById("shortcut-key"),
  chunkSize: document.getElementById("chunk-size"),
  timeoutMs: document.getElementById("timeout-ms"),
};
const statusEl = document.getElementById("status");
const saveButton = document.getElementById("save");
const testButton = document.getElementById("test");

function normalizeBridgeUrl(value) {
  return String(value || DEFAULT_CONFIG.bridgeUrl).trim().replace(/\/+$/, "") || DEFAULT_CONFIG.bridgeUrl;
}

function setStatus(message) {
  statusEl.textContent = message;
}

function loadSettings() {
  chrome.storage.local.get(DEFAULT_CONFIG, (stored) => {
    fields.bridgeUrl.value = normalizeBridgeUrl(stored.bridgeUrl);
    fields.shortcutKey.value = String(stored.shortcutKey || DEFAULT_CONFIG.shortcutKey).slice(0, 1).toLowerCase();
    fields.chunkSize.value = Number(stored.chunkSize || DEFAULT_CONFIG.chunkSize);
    fields.timeoutMs.value = Number(stored.timeoutMs || DEFAULT_CONFIG.timeoutMs);
    setStatus("Settings loaded.");
  });
}

function saveSettings() {
  const payload = {
    bridgeUrl: normalizeBridgeUrl(fields.bridgeUrl.value),
    shortcutKey: String(fields.shortcutKey.value || DEFAULT_CONFIG.shortcutKey).slice(0, 1).toLowerCase(),
    chunkSize: Number(fields.chunkSize.value || DEFAULT_CONFIG.chunkSize),
    timeoutMs: Number(fields.timeoutMs.value || DEFAULT_CONFIG.timeoutMs),
  };
  chrome.storage.local.set(payload, () => {
    setStatus("Saved.");
  });
}

function testBridge() {
  setStatus("Testing bridge...");
  chrome.runtime.sendMessage({type: "POINTNCLICK_HEALTH"}, (response) => {
    const runtimeError = chrome.runtime.lastError;
    if (runtimeError) {
      setStatus(`Bridge test failed:\n${runtimeError.message}`);
      return;
    }
    if (!response || !response.ok) {
      const message = response && response.data && response.data.message
        ? response.data.message
        : "No response from bridge.";
      setStatus(`Bridge test failed:\n${message}`);
      return;
    }
    const data = response.data;
    setStatus(`Connected.\nDataset: ${data.dataset}\nLayer: ${data.color_layer}\nDevice: ${data.device}\nCrop: ${data.crop_size}`);
  });
}

saveButton.addEventListener("click", saveSettings);
testButton.addEventListener("click", testBridge);

loadSettings();
