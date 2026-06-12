# Public Release Checklist

This is the practical path for distributing PointnClick to other annotators without Git LFS.

## 1. Publish The Model Checkpoint

Do not commit `best_model.pt` to git.

1. Pick the checkpoint you want everyone to use:

```powershell
$model = ".\runs\runs\worm_unet\best_model.pt"
```

2. Compute its checksum:

```powershell
Get-FileHash $model -Algorithm SHA256
```

3. In GitHub, open your repository and go to `Releases`.
4. Draft a new release, for example `v0.1.0`.
5. Upload the checkpoint as a release asset. Use a stable name:

```text
pointnclick-worm-unet-v1.pt
```

6. Copy the release asset URL. It should look like:

```text
https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v0.1.0/pointnclick-worm-unet-v1.pt
```

## 2. Prefill The Bridge App Download URL

Create this file in the repo root:

```powershell
Copy-Item .\model_url.example.txt .\model_url.txt
notepad .\model_url.txt
```

Replace the example text with only the direct release asset URL:

```text
https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v0.1.0/pointnclick-worm-unet-v1.pt
```

When the bridge app starts, it reads `model_url.txt` and pre-fills `Model download URL`. Users can click `Download model`, and the app downloads the checkpoint into their local PointnClick model cache.

## 3. Build The Windows Bridge App

Build from a Python environment where `torch` imports successfully.

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pip install -r requirements.txt
python -m pip install -r requirements-build.txt
.\packaging\build_bridge_app.ps1
```

The build script copies `model_url.txt` into:

```text
dist\PointnClickBridge\model_url.txt
```

Zip the full folder:

```text
dist\PointnClickBridge
```

Do not distribute only `PointnClickBridge.exe`; the folder contains required DLLs and support files.

## 4. Package The Extension

Zip the extension folder:

```text
examples\webknossos_chrome_extension
```

For the first public version, users can install it as an unpacked extension:

1. Open `chrome://extensions`.
2. Enable `Developer mode`.
3. Click `Load unpacked`.
4. Select the extracted `webknossos_chrome_extension` folder.

Later, package the same extension through the Chrome Web Store if you want one-click browser installation.

## 5. Attach Release Assets

Attach these files to the same GitHub Release:

```text
PointnClickBridge.zip
pointnclick-webknossos-extension.zip
pointnclick-worm-unet-v1.pt
```

In the release notes, include:

```text
1. Download and extract PointnClickBridge.zip.
2. Run PointnClickBridge.exe.
3. Paste your WebKnossos dataset/view URL and token.
4. Click Download model.
5. Click Start bridge.
6. Install the Chrome extension folder with chrome://extensions.
7. Open WebKnossos, select a segment/color, place the crosshair, and press P.
```

## 6. What Users Should Leave Blank

Users can leave these blank:

- `Model checkpoint` before clicking `Download model`
- `Model SHA256 optional`, unless you publish the checksum and want users to verify it
- `Organization ID`, unless they are opening a dataset by name instead of a URL
- `Annotation URL/ID`, unless they need a specific annotation-linked data source
- `Sharing token`, unless the dataset uses a sharing token

Users should fill:

- `Dataset/view URL`
- `WebKnossos token`
- `Model download URL`, if it was not prefilled from `model_url.txt`
- `Raw layer`, usually `color`
- `Magnification`, usually `1`
- `Device`, usually `cuda` or `cpu`
