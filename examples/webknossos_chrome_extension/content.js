(() => {
  const script = document.createElement("script");
  script.src = chrome.runtime.getURL("injected.js");
  script.onload = () => script.remove();
  (document.head || document.documentElement).appendChild(script);

  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const message = event.data;
    if (!message || message.type !== "POINTNCLICK_EXTENSION_REQUEST") return;

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
})();
