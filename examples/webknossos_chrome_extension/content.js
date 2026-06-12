(() => {
  let nextPageRequestId = 1;
  const pendingPageRequests = new Map();

  const script = document.createElement("script");
  script.src = chrome.runtime.getURL("injected.js");
  script.onload = () => script.remove();
  (document.head || document.documentElement).appendChild(script);

  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const message = event.data;
    if (!message) return;

    if (message.type === "POINTNCLICK_PAGE_COMMAND_RESPONSE") {
      const pending = pendingPageRequests.get(message.requestId);
      if (!pending) return;
      pendingPageRequests.delete(message.requestId);
      pending.resolve(message.response);
      return;
    }

    if (message.type !== "POINTNCLICK_EXTENSION_REQUEST") return;

    chrome.runtime.sendMessage(
      {
        type: message.action,
        payload: message.payload,
      },
      (response) => {
        const runtimeError = chrome.runtime.lastError;
        window.postMessage(
          {
            type: "POINTNCLICK_EXTENSION_RESPONSE",
            requestId: message.requestId,
            response: runtimeError
              ? {
                  ok: false,
                  status: 0,
                  data: {
                    status: "error",
                    message: runtimeError.message,
                  },
                }
              : response,
          },
          "*",
        );
      },
    );
  });

  function requestPage(action) {
    const requestId = nextPageRequestId++;
    return new Promise((resolve, reject) => {
      pendingPageRequests.set(requestId, {resolve, reject});
      window.postMessage(
        {
          type: "POINTNCLICK_PAGE_COMMAND",
          requestId,
          action,
        },
        "*",
      );
      setTimeout(() => {
        if (!pendingPageRequests.has(requestId)) return;
        pendingPageRequests.delete(requestId);
        reject(new Error("No response from the WebKnossos page script. Refresh the WebKnossos tab and confirm the extension is enabled."));
      }, 5000);
    });
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!message || message.type !== "POINTNCLICK_PAGE_COMMAND") {
      return false;
    }
    requestPage(message.action)
      .then((response) => sendResponse(response))
      .catch((error) => sendResponse({
        ok: false,
        message: error.message || String(error),
      }));
    return true;
  });
})();
