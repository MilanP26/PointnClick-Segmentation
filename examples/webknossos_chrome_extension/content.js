(() => {
  const script = document.createElement("script");
  script.src = chrome.runtime.getURL("injected.js");
  script.onload = () => script.remove();
  (document.head || document.documentElement).appendChild(script);

  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const message = event.data;
    if (!message || message.type !== "POINTNCLICK_PREDICT_REQUEST") return;

    chrome.runtime.sendMessage(
      {
        type: "POINTNCLICK_PREDICT",
        payload: message.payload,
      },
      (response) => {
        window.postMessage(
          {
            type: "POINTNCLICK_PREDICT_RESPONSE",
            requestId: message.requestId,
            response,
          },
          "*",
        );
      },
    );
  });
})();
