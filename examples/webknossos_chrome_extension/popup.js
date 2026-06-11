const DEFAULT_CONFIG = {
  bridgeUrl: "http://127.0.0.1:8765",
  shortcutKey: "p",
  chunkSize: 5000,
  timeoutMs: 120000,
};

const bridgeUrlInput = document.getElementById("bridge-url");
const shortcutInput = document.getElementById("shortcut-key");
const statusEl = document.getElementById("status");
const saveButton = document.getElementById("save");
const testButton = document.getElementById("test");
const optionsButton = document.getElementById("options");

function normalizeBridgeUrl(value) {
  return String(value || DEFAULT_CONFIG.bridgeUrl).trim().replace(/\/+$/, "") || DEFAULT_CONFIG.bridgeUrl;
}

function setStatus(message) {
  statusEl.textContent = message;
}

function loadSettings() {
  chrome.storage.local.get(DEFAULT_CONFIG, (stored) => {
    bridgeUrlInput.value = normalizeBridgeUrl(stored.bridgeUrl);
    shortcutInput.value = String(stored.shortcutKey || DEFAULT_CONFIG.shortcutKey).slice(0, 1).toLowerCase();
    setStatus("Ready. Start the local bridge app before segmenting.");
  });
}

function saveSettings() {
  const bridgeUrl = normalizeBridgeUrl(bridgeUrlInput.value);
  const shortcutKey = String(shortcutInput.value || DEFAULT_CONFIG.shortcutKey).slice(0, 1).toLowerCase();
  chrome.storage.local.set({bridgeUrl, shortcutKey}, () => {
    setStatus("Saved. Refresh WebKnossos or run window.pointnclickWebknossos.refreshConfig().");
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
    setStatus(`Connected.\nDataset: ${data.dataset}\nLayer: ${data.color_layer}\nDevice: ${data.device}`);
  });
}

saveButton.addEventListener("click", saveSettings);
testButton.addEventListener("click", testBridge);
optionsButton.addEventListener("click", () => chrome.runtime.openOptionsPage());

loadSettings();
