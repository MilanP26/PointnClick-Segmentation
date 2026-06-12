# PointnClick WebKnossos Bridge App

This is the recommended scalable WebKnossos workflow:

1. A local desktop app runs the WebKnossos bridge and PyTorch model inference.
2. A small Chrome extension injects into WebKnossos, reads the active segment and crosshair, and paints the returned mask.

The extension does not run the model. This keeps inference local and fast while avoiding a browser-only ONNX/WebGPU rewrite.

## User Install

### 1. Install the bridge app

Download the `PointnClickBridge` release zip, extract it, and run:

```text
PointnClickBridge.exe
```

Fill in:

- `Dataset/view URL`: the WebKnossos dataset or annotation view URL
- `WebKnossos token`: your account token, unless using a public/shared dataset
- `Model checkpoint`: an existing local `.pt` file, or leave blank and use `Model download URL`
- `Model download URL`: a direct URL to the trained checkpoint
- `Raw layer`: usually `color`
- `Magnification`: usually `1`
- `Device`: `cuda` if the user's machine has a working CUDA PyTorch build, otherwise `cpu`

If `Model download URL` is filled, click `Download model`. The app downloads the checkpoint into the user's local PointnClick model cache and updates `Model checkpoint` automatically.

Click `Start bridge`. The bridge should show:

```text
Running at http://127.0.0.1:8765
```

### 2. Install the Chrome extension

For development/unpacked install:

1. Open `chrome://extensions`.
2. Enable `Developer mode`.
3. Click `Load unpacked`.
4. Select:

```text
examples\webknossos_chrome_extension
```

Open the extension popup and confirm:

- Bridge URL: `http://127.0.0.1:8765`
- Shortcut key: `p`

Click `Test`. It should report the dataset, layer, and device.

### 3. Annotate in WebKnossos

1. Open the WebKnossos annotation.
2. Select or create the segment/color you want to paint into.
3. Put the crosshair on the target object.
4. Press the extension shortcut, default `P`.
5. The model prediction is painted into the currently active WebKnossos segment.
6. Switch to eraser/brush tools and correct the result as normal.

The mask is painted as normal volume annotation voxels. It is not an overlay, so WebKnossos editing behavior stays unchanged.

## Model Weights Without Git LFS

Do not commit `.pt` files. This repo ignores model binaries.

Use GitHub Releases or another direct-download file host:

1. Train the model locally.
2. Choose the checkpoint to distribute, for example:

```text
runs\runs\worm_unet\best_model.pt
```

3. Compute a checksum:

```powershell
Get-FileHash .\runs\runs\worm_unet\best_model.pt -Algorithm SHA256
```

4. Create a GitHub Release.
5. Upload the checkpoint as a release asset, for example:

```text
pointnclick-worm-unet-v1.pt
```

6. Use the release asset URL in the bridge app:

```text
https://github.com/<owner>/<repo>/releases/download/<tag>/pointnclick-worm-unet-v1.pt
```

7. Paste the SHA256 into the app's optional checksum field.

The app downloads the model into the user's local app-data cache and then starts the bridge from that local file.

For public releases, you can place the release asset URL in a `model_url.txt` file beside `PointnClickBridge.exe`. The app reads that file on startup and pre-fills `Model download URL`, so users only need to click `Download model`. See [public_release_checklist.md](public_release_checklist.md).

## Developer Run From Source

From the repo root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_bridge_app.py
```

Equivalent CLI entry:

```powershell
python run_cli.py webknossos-app
```

The old direct CLI bridge still works:

```powershell
python run_cli.py webknossos-serve `
  --checkpoint "C:\path\to\best_model.pt" `
  --dataset "PASTE_WEBKNOSSOS_VIEW_URL" `
  --color-layer color `
  --crop-size 512 `
  --threshold 0.5 `
  --device cuda
```

## Build the Windows App

Build from a Windows environment that already has the desired PyTorch runtime installed. If you build with a CPU-only PyTorch wheel, the app will be CPU-only. If you build with a CUDA PyTorch wheel, the app can use CUDA but the release folder will be much larger.

```powershell
.\packaging\build_bridge_app.ps1
```

Output:

```text
dist\PointnClickBridge\PointnClickBridge.exe
```

Zip the full `dist\PointnClickBridge` folder and attach it to a GitHub Release. Do not zip only the `.exe`; PyInstaller onedir builds need the surrounding DLLs and support files.

## Extension Release

For a simple release artifact, zip the contents of:

```text
examples\webknossos_chrome_extension
```

Users can extract it and load the extracted folder as an unpacked extension.

For a Chrome Web Store release later, audit the extension for store policy, narrow the content-script matches if you know the WebKnossos hostnames, and package the extension through the Chrome developer dashboard.
