# Internal Source Install

Use this route if the packaged `PointnClickBridge.exe` fails on another computer. It runs the same bridge app from a local conda environment, so normal `pip install` fixes work.

## What To Share

Share these with the user:

- The repository folder as a zip, or a GitHub clone link
- `pointnclick-webknossos-extension.zip`

The model is still downloaded from the GitHub Release URL in `model_url.txt`.

## One-Time Setup On The User Computer

Install Miniconda or Anaconda first.

Open Anaconda PowerShell Prompt and run:

```powershell
cd "PATH\TO\PointnClick-Segmentation"
powershell -ExecutionPolicy Bypass -File .\scripts\install_pointnclick_env.ps1
```

This creates a conda environment named `pointnclick`, installs CUDA PyTorch, installs `webknossos`, and validates imports.

## Start The Bridge

```powershell
cd "PATH\TO\PointnClick-Segmentation"
powershell -ExecutionPolicy Bypass -File .\scripts\run_pointnclick_bridge.ps1
```

Then in the bridge app:

1. Paste the WebKnossos dataset/view URL.
2. Paste the WebKnossos token.
3. Click `Download model`.
4. Click `Start bridge`.

## Install The Extension

1. Extract `pointnclick-webknossos-extension.zip`.
2. Open Chrome at `chrome://extensions`.
3. Enable `Developer mode`.
4. Click `Load unpacked`.
5. Select the extracted folder that directly contains `manifest.json`.

## Use

Open WebKnossos, select the active segment/color, place the crosshair, and press `P`.

## Troubleshooting

In the bridge app, click `Diagnostics`. The log file is:

```text
%LOCALAPPDATA%\PointnClick\bridge_app.log
```
