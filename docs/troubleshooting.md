# Troubleshooting PointnClick Bridge

If the extension loads but painting does nothing, first check the bridge app. The extension cannot paint unless the bridge is running.

## Bridge App Says Error Or Failed

1. Open the `Log` tab in `PointnClickBridge.exe`.
2. Click `Diagnostics`.
3. Try `Start bridge` again.
4. Send the log file to the developer:

```text
%LOCALAPPDATA%\PointnClick\bridge_app.log
```

You can paste that path into File Explorer.

## Common Diagnostics Results

### Device is set to cuda, but CUDA is unavailable

The packaged app can use CUDA only if the computer has a compatible NVIDIA GPU and driver.

Fix:

- Install/update the NVIDIA driver.
- Confirm the machine has an NVIDIA GPU.
- Run the bridge on a GPU workstation.

### Torch import failed

The packaged PyTorch runtime could not load on that computer.

Fix:

- Rebuild the bridge app from the CUDA/PyTorch environment that works on your project machines.
- Make sure users extract the full `PointnClickBridge.zip` folder, not only the `.exe`.
- Do not move `PointnClickBridge.exe` out of its folder.

### Model checkpoint path does not exist

The model was not downloaded or the local path points to a file from another computer.

Fix:

- Click `Download model`.
- Confirm `Model checkpoint` updates to a path under `%LOCALAPPDATA%\PointnClick\models`.

### WebKnossos token or permission error

Fix:

- Paste a valid WebKnossos token.
- Confirm the dataset URL opens in the browser on that same computer.
- Confirm the user has dataset access.

### Raw layer or magnification error

Fix:

- Set `Raw layer` to `color` first.
- Set `Magnification` to `1`.
- If it still fails, use the bridge log's available layer names or ask the developer to inspect the dataset.

### Port 8765 already in use

Fix:

- Close any other PointnClickBridge windows.
- Or change the port in Advanced settings and update the extension popup to match.

## Extension Looks Fine But Nothing Paints

In the extension popup:

1. Click `Test`.
2. Click `Page`.
3. If both are OK, click `Run`.

If `Run` works but `P` does not, change the shortcut or click inside the WebKnossos viewport before pressing `P`.
