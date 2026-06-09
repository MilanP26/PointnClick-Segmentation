chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.type !== "POINTNCLICK_PREDICT") {
    return false;
  }

  fetch("http://127.0.0.1:8765/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(message.payload),
  })
    .then(async (response) => {
      const data = await response.json();
      sendResponse({
        ok: response.ok,
        status: response.status,
        data,
      });
    })
    .catch((error) => {
      sendResponse({
        ok: false,
        status: 0,
        data: {
          status: "error",
          message: error.message || String(error),
        },
      });
    });

  return true;
});
