const DEFAULT_CONFIG = {
  bridgeUrl: "http://127.0.0.1:8765",
  shortcutKey: "p",
  chunkSize: 5000,
  timeoutMs: 120000,
  sessionToken: "",
  username: "",
};

const bridgeUrlInput = document.getElementById("bridge-url");
const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const shortcutInput = document.getElementById("shortcut-key");
const statusEl = document.getElementById("status");
const saveButton = document.getElementById("save");
const loginButton = document.getElementById("login");
const logoutButton = document.getElementById("logout");
const testButton = document.getElementById("test");
const pageStatusButton = document.getElementById("page-status");
const runButton = document.getElementById("run");
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
    usernameInput.value = String(stored.username || "");
    shortcutInput.value = String(stored.shortcutKey || DEFAULT_CONFIG.shortcutKey).slice(0, 1).toLowerCase();
    const signedIn = stored.sessionToken ? `Signed in as ${stored.username || "saved user"}.` : "Not signed in for remote server mode.";
    setStatus(`${signedIn}\nRefresh WebKnossos after changing settings.`);
  });
}

function saveSettings() {
  const bridgeUrl = normalizeBridgeUrl(bridgeUrlInput.value);
  const shortcutKey = String(shortcutInput.value || DEFAULT_CONFIG.shortcutKey).slice(0, 1).toLowerCase();
  return new Promise((resolve) => {
    chrome.storage.local.set({bridgeUrl, shortcutKey}, () => {
      setStatus("Saved. Refresh WebKnossos or run window.pointnclickWebknossos.refreshConfig().");
      resolve();
    });
  });
}

function sendRuntimeMessage(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (response) => {
      const runtimeError = chrome.runtime.lastError;
      if (runtimeError) {
        resolve({ok: false, status: 0, data: {message: runtimeError.message}});
        return;
      }
      resolve(response || {ok: false, status: 0, data: {message: "No response."}});
    });
  });
}

async function login() {
  await saveSettings();
  setStatus("Signing in...");
  const response = await sendRuntimeMessage({
    type: "POINTNCLICK_LOGIN",
    payload: {
      username: usernameInput.value.trim(),
      password: passwordInput.value,
    },
  });
  if (!response.ok) {
    const message = response.data && response.data.message ? response.data.message : "Sign in failed.";
    setStatus(`Sign in failed:\n${message}`);
    return;
  }
  passwordInput.value = "";
  setStatus(`Signed in as ${response.data.username}.\nToken saved: ${response.data.has_webknossos_token ? "yes" : "no"}`);
}

async function logout() {
  const response = await sendRuntimeMessage({type: "POINTNCLICK_LOGOUT"});
  if (!response.ok) {
    setStatus("Sign out failed.");
    return;
  }
  setStatus("Signed out.");
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
    setStatus(`Connected.\nMode: ${data.mode || "local bridge"}\nLayer: ${data.color_layer}\nDevice: ${data.device}`);
  });
}

function sendPageCommand(action) {
  return new Promise((resolve) => {
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
      const tab = tabs && tabs[0];
      if (!tab || !tab.id) {
        resolve({ok: false, message: "No active tab found."});
        return;
      }
      chrome.tabs.sendMessage(tab.id, {type: "POINTNCLICK_PAGE_COMMAND", action}, (response) => {
        const runtimeError = chrome.runtime.lastError;
        if (runtimeError) {
          resolve({ok: false, message: runtimeError.message});
          return;
        }
        resolve(response || {ok: false, message: "No response from the page script."});
      });
    });
  });
}

async function checkPageStatus() {
  setStatus("Checking current tab...");
  const response = await sendPageCommand("status");
  if (!response.ok) {
    setStatus(`Page script not ready:\n${response.message}`);
    return;
  }
  const status = response.status || {};
  setStatus(`Page script: ${status.loaded ? "loaded" : "not loaded"}\nWebKnossos API: ${status.hasApi ? "ready" : "not ready"}\nShortcut: ${(status.shortcutKey || shortcutInput.value || "p").toUpperCase()}\nMessage: ${status.message || ""}`);
}

async function runOnCurrentTab() {
  setStatus("Triggering PointnClick on current tab...");
  const response = await sendPageCommand("run");
  if (!response.ok) {
    setStatus(`Run failed:\n${response.message}`);
    return;
  }
  setStatus("Run command sent. Check WebKnossos for the painted segment or toast message.");
}

saveButton.addEventListener("click", saveSettings);
loginButton.addEventListener("click", login);
logoutButton.addEventListener("click", logout);
testButton.addEventListener("click", testBridge);
pageStatusButton.addEventListener("click", checkPageStatus);
runButton.addEventListener("click", runOnCurrentTab);
optionsButton.addEventListener("click", () => chrome.runtime.openOptionsPage());

loadSettings();
