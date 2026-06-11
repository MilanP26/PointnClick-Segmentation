# WebKnossos PointnClick Workflow

This keeps the VAST bridge unchanged and runs a separate local bridge for WebKnossos. Python reads the EM crop from WebKnossos, runs the same trained model checkpoint, and the browser script paints the prediction into the active WebKnossos volume segment.

Recommended user-facing flow: run the desktop bridge app, load the Chrome extension, select the active WebKnossos segment/color, then press the extension shortcut at the crosshair. The prediction is painted as normal WebKnossos volume annotation, so brush and eraser correction still work. See `docs/webknossos_bridge_app.md` for the app, extension, packaging, and model-weight release workflow.

Developer shortcut:

```powershell
python run_cli.py webknossos-app
```

## 1. Start Anaconda and enter the project

```powershell
conda activate wormseg
cd "C:\Ishaan\Segmentation\milan_experiments_with_2d_unets_worm\PointnClick-Segmentation\PointnClick-Segmentation"
```

Note: `cd /d ...` is for `cmd.exe`. In PowerShell, use `cd "path"` instead.

## 2. Install the WebKnossos Python client

```bat
pip install webknossos
```

If you are refreshing the full project environment, use:

```bat
pip install -r requirements.txt
```

## 3. Set your WebKnossos token

Get a token from your WebKnossos account page, then set it for this terminal:

```powershell
$env:WEBKNOSSOS_TOKEN="PASTE_YOUR_TOKEN_HERE"
```

For `cmd.exe`, use:

```bat
set WEBKNOSSOS_TOKEN=PASTE_YOUR_TOKEN_HERE
```

## 4. Start the local bridge

Use a dataset view URL if you have one. The most common raw EM layer name is `color`, but if WebKnossos reports a different layer name, pass that with `--color-layer`.

PowerShell uses a backtick, not `^`, for multi-line commands:

```powershell
python run_cli.py webknossos-serve `
  --checkpoint "C:\Ishaan\Segmentation\milan_experiments_with_2d_unets_worm\PointnClick-Segmentation\PointnClick-Segmentation\runs\runs\worm_unet\best_model.pt" `
  --dataset "PASTE_WEBKNOSSOS_DATASET_VIEW_URL_HERE" `
  --color-layer color `
  --crop-size 512 `
  --threshold 0.5 `
  --device cuda
```

Or use the single-line version, which is the least fussy to paste:

```powershell
python run_cli.py webknossos-serve --checkpoint "C:\Ishaan\Segmentation\milan_experiments_with_2d_unets_worm\PointnClick-Segmentation\PointnClick-Segmentation\runs\runs\worm_unet\best_model.pt" --dataset "PASTE_WEBKNOSSOS_DATASET_VIEW_URL_HERE" --color-layer color --crop-size 512 --threshold 0.5 --device cuda
```

If you are in `cmd.exe` instead of PowerShell, use `^`:

```bat
python run_cli.py webknossos-serve ^
  --checkpoint "C:\Ishaan\Segmentation\milan_experiments_with_2d_unets_worm\PointnClick-Segmentation\PointnClick-Segmentation\runs\runs\worm_unet\best_model.pt" ^
  --dataset "PASTE_WEBKNOSSOS_DATASET_VIEW_URL_HERE" ^
  --color-layer color ^
  --crop-size 512 ^
  --threshold 0.5 ^
  --device cuda
```

If your dataset is referenced by name instead of URL:

```powershell
python run_cli.py webknossos-serve `
  --checkpoint "C:\Ishaan\Segmentation\milan_experiments_with_2d_unets_worm\PointnClick-Segmentation\PointnClick-Segmentation\runs\runs\worm_unet\best_model.pt" `
  --dataset "DATASET_NAME_HERE" `
  --organization-id "ORG_ID_HERE" `
  --color-layer color `
  --crop-size 512 `
  --threshold 0.5 `
  --device cuda
```

The bridge prints browser script URLs. WebKnossos blocks normal page `fetch()` calls to localhost with Content Security Policy, so Chrome should use the included unpacked extension.

## 5. Load the browser script in WebKnossos

Chrome's current extension rules can prevent Tampermonkey from injecting reliably. The most reliable Chrome path is to load the local unpacked extension that is included in this repo:

1. Open Chrome and go to `chrome://extensions`.
2. Turn on `Developer mode`.
3. Click `Load unpacked`.
4. Select this folder:

```text
C:\Ishaan\Segmentation\milan_experiments_with_2d_unets_worm\PointnClick-Segmentation\PointnClick-Segmentation\examples\webknossos_chrome_extension
```

5. Refresh the WebKnossos annotation page.
6. In the WebKnossos console, check:

```js
window.pointnclickWebknossosStatus
```

If it says `loaded: true`, the tool is ready. You can press `P` at the crosshair or manually run:

```js
window.pointnclickWebknossos.run()
```

The Tampermonkey route can also work in some browsers. If you want to try it, install Tampermonkey and open this URL while the bridge is running:

```text
http://127.0.0.1:8765/userscript.user.js
```

If the browser downloads `userscript.user.js` instead of opening a Tampermonkey install page:

1. Open Tampermonkey from the browser extensions menu.
2. Click `Dashboard`.
3. Click the `+` button to create a new script.
4. Open the downloaded `userscript.user.js` file in Notepad.
5. Select all of the script text and copy it.
6. In Tampermonkey, select all of the starter template text and replace it with the copied script.
7. Press `Ctrl+S` to save.
8. Refresh the WebKnossos annotation page.

The older console-paste script at `http://127.0.0.1:8765/client.js` can load, but WebKnossos usually blocks its localhost request with Content Security Policy. Use the unpacked extension for real annotation in Chrome.

## 6. Annotate

Select or create the active volume segment in WebKnossos. Put the crosshair over the target cell, then press `P`.

The bridge will:

1. Read a 512 x 512 EM crop around the crosshair.
2. Run the trained PointnClick model once.
3. Send the predicted mask back as compact row-runs.
4. Paint the mask into the active WebKnossos segment with `labelVoxels`.

Timing/debug events are written to:

```text
outputs\webknossos_bridge\events.jsonl
```

## Notes

If `P` conflicts with a WebKnossos shortcut, restart the bridge with another key, for example:

```bat
python run_cli.py webknossos-serve ^
  --checkpoint "C:\path\to\best_model.pt" ^
  --dataset "PASTE_WEBKNOSSOS_DATASET_VIEW_URL_HERE" ^
  --client-key g
```

If the model paints in the wrong place, confirm that the WebKnossos raw layer coordinates are full-resolution Mag(1) coordinates and keep `--mag 1`.
