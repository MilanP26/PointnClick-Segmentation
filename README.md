# PointnClick Segmentation

Interactive click-to-mask segmentation for electron microscopy imagery, with a correction loop that turns user fixes into new training data.

This repository provides:

- A click-conditioned 2D U-Net for EM segmentation
- Training code for binary masks and dense instance-label masks
- Strong paired augmentations for EM training crops
- Inference code that predicts a full mask from one positive click
- Held-out evaluation with IoU, Dice, and variation of information (VI)
- A feedback pipeline that ingests corrected masks and schedules them for future fine-tuning
- Windows-friendly CLI workflows for use alongside VAST Lite

## Why this is not pure reinforcement learning

For this problem, reinforcement learning is usually the wrong first tool. You already have ground-truth segmentations, and when a prediction is wrong, the user can correct it. That is a supervised learning signal, not a sparse reward problem.

The practical system is:

1. Train a click-conditioned segmentation model on existing masks.
2. Run inference from a user click.
3. Let the user correct the mask in VAST.
4. Save the corrected mask back into the training set.
5. Fine-tune the model on the new corrections.

This gives you the "learn from mistakes" behavior you want, without the instability and complexity of RL.

## Supported dataset formats

```text
data/
|-- train/
|   |-- images/
|   |   |-- sample_0001.png
|   |   `-- sample_0002.png
|   `-- masks/
|       |-- sample_0001.png
|       `-- sample_0002.png
|-- val/
|   |-- images/
|   `-- masks/
`-- feedback/
    |-- images/
    `-- masks/
```

Rules:

- Image and mask filenames must match.
- Images can be `.png`, `.tif`, `.tiff`, `.jpg`, or `.jpeg`.
- Masks should be single-object binary masks where object pixels are nonzero.

For dense worm data, masks can also be integer-labeled instance masks where:

- `0` is background
- each cell/object uses its own nonzero integer id

The training pipeline will automatically turn each labeled object into a click-conditioned binary training target.

## Install

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use `run_cli.py` from the repo root. You do not need `pip install -e .`.

## Prepare bouton exports

You already have:

- `exports\EM\Bouton X\...`
- `exports\Boutons\Bouton X\...`

Use this to create a training dataset directly from those exports:

```powershell
.\.venv\Scripts\python.exe .\run_cli.py prepare-exports `
  --exports-dir exports `
  --output-dir data\prepared `
  --val-boutons 16,17,18,19
```

This will:

- Pair EM slices with bouton segmentation masks by slice number
- Ignore `exports\Vesicles`
- Convert masks to binary
- Build:
  - `data\prepared\train\images`
  - `data\prepared\train\masks`
  - `data\prepared\val\images`
  - `data\prepared\val\masks`

## Prepare dense worm data

If your data lives under `data\Training Round 2` as either:

- `em.zip` and `mask.zip`, or
- extracted `em\` and `mask\` folders

run:

```powershell
.\.venv\Scripts\python.exe .\run_cli.py prepare-worm `
  --data-dir "data\Training Round 2" `
  --output-dir data\worm_round2 `
  --val-fraction 0.2 `
  --test-fraction 0.1
```

This creates contiguous train/validation/test splits:

- `data\worm_round2\train\images`
- `data\worm_round2\train\masks`
- `data\worm_round2\val\images`
- `data\worm_round2\val\masks`
- `data\worm_round2\test\images`
- `data\worm_round2\test\masks`

The split summary is saved to `data\worm_round2\split_summary.json`.

## Train on worm data with the 2D U-Net

The model is still click-conditioned so it stays compatible with the VAST bridge, but the backbone is a 2D U-Net and the dense worm masks are expanded into one object-per-sample targets during training.

Recommended first run:

```powershell
.\.venv\Scripts\python.exe .\run_cli.py train `
  --train-dir data\worm_round2\train `
  --val-dir data\worm_round2\val `
  --output-dir runs\worm_unet `
  --image-size 512 `
  --crop-size 512 `
  --batch-size 4 `
  --epochs 60 `
  --learning-rate 3e-4 `
  --weight-decay 1e-4 `
  --selection-metric vi `
  --early-stopping-patience 12 `
  --min-epochs 20 `
  --device cpu
```

What this training loop uses:

- 2D U-Net
- BCE-with-logits + Dice loss
- AdamW optimizer
- held-out validation every epoch
- checkpoint selection by lowest validation VI by default
- early stopping to reduce overfitting
- augmentations:
  - crop jitter
  - flips
  - 90 degree rotations
  - random affine transforms
  - elastic deformation
  - brightness/contrast/gamma perturbations
  - blur/noise
  - random artifact cutouts

## Evaluate on held-out test data

```powershell
.\.venv\Scripts\python.exe .\run_cli.py evaluate `
  --checkpoint runs\worm_unet\best_model.pt `
  --data-dir data\worm_round2\test `
  --batch-size 4 `
  --device cpu
```

This prints:

- loss
- IoU
- Dice
- VI

The training curves and best epoch are saved to `runs\worm_unet\metrics.json`.

## Train on bouton exports

The older bouton workflow still works:

```powershell
.\.venv\Scripts\python.exe .\run_cli.py train `
  --train-dir data\prepared\train `
  --val-dir data\prepared\val `
  --output-dir runs\baseline `
  --epochs 50 `
  --batch-size 8 `
  --image-size 512
```

## Predict from one click

```powershell
.\.venv\Scripts\python.exe .\run_cli.py predict `
  --checkpoint runs\baseline\best_model.pt `
  --image path\to\slice_0142.png `
  --x 841 `
  --y 612 `
  --output-mask outputs\slice_0142_cellA_mask.png `
  --output-overlay outputs\slice_0142_cellA_overlay.png
```

## Predict and create a VAST-importable segmentation image

Use this when you want to bring the predicted mask back into VAST as a real segmentation image.

```powershell
.\.venv\Scripts\python.exe .\run_cli.py predict-vast-import `
  --checkpoint runs\baseline\best_model.pt `
  --image "exports\EM\Bouton 1\Bouton 1 Export_s20.png" `
  --x 150 `
  --y 150 `
  --segment-id 123 `
  --z-index 20 `
  --output-dir outputs\vast_import `
  --device cpu
```

This writes:

- a binary mask
- a regular overlay preview
- an RGB segmentation image encoded for VAST import
- a JSON metadata file with the import coordinates

Then in VAST Lite:

1. Use `File > Import Segmentations from Images`
2. Choose the generated `*_vast_import.png`
3. Set start coordinates to `X=0`, `Y=0`, `Z=<z-index>`
4. Import into the desired segmentation layer

## Add corrected masks and fine-tune

```powershell
.\.venv\Scripts\python.exe .\run_cli.py add-feedback `
  --image path\to\slice_0142.png `
  --mask path\to\corrected_mask.png `
  --feedback-dir data\feedback `
  --sample-id slice_0142_cellA
```

```powershell
.\.venv\Scripts\python.exe .\run_cli.py finetune `
  --checkpoint runs\baseline\best_model.pt `
  --train-dir data\prepared\train `
  --val-dir data\prepared\val `
  --feedback-dir data\feedback `
  --output-dir runs\finetune_001 `
  --epochs 10 `
  --batch-size 4 `
  --learning-rate 1e-4
```

## VAST Lite workflow

Working today:

1. Export a 2D slice image from VAST Lite.
2. Record the click location on the target cell.
3. Run `predict` or `predict-vast-import`.
4. If using `predict-vast-import`, import the generated VAST segmentation image.
5. Review and correct the mask in VAST.
6. Export the corrected mask.
7. Run `add-feedback`.
8. Periodically run `finetune`.

## End-to-end worm workflow on Windows PowerShell

From the repo root:

```powershell
cd C:\Users\lefty\OneDrive\Documents\GitHub\PointnClick-Segmentation
.\.venv\Scripts\Activate.ps1
```

1. Prepare the dense worm dataset:

```powershell
.\.venv\Scripts\python.exe .\run_cli.py prepare-worm `
  --data-dir "data\Training Round 2" `
  --output-dir data\worm_round2 `
  --val-fraction 0.2 `
  --test-fraction 0.1
```

2. Train the 2D U-Net:

```powershell
.\.venv\Scripts\python.exe .\run_cli.py train `
  --train-dir data\worm_round2\train `
  --val-dir data\worm_round2\val `
  --output-dir runs\worm_unet `
  --image-size 512 `
  --crop-size 512 `
  --batch-size 4 `
  --epochs 60 `
  --learning-rate 3e-4 `
  --weight-decay 1e-4 `
  --selection-metric vi `
  --early-stopping-patience 12 `
  --min-epochs 20 `
  --device cpu
```

3. Evaluate the best checkpoint:

```powershell
.\.venv\Scripts\python.exe .\run_cli.py evaluate `
  --checkpoint runs\worm_unet\best_model.pt `
  --data-dir data\worm_round2\test `
  --batch-size 4 `
  --device cpu
```

4. Test a click offline:

```powershell
.\.venv\Scripts\python.exe .\run_cli.py predict `
  --checkpoint runs\worm_unet\best_model.pt `
  --image "data\worm_round2\test\images\worm_s0646.png" `
  --x 320 `
  --y 280 `
  --output-mask outputs\worm_test_mask.png `
  --output-overlay outputs\worm_test_overlay.png `
  --device cpu
```

5. Run the live VAST bridge:

```powershell
.\.venv\Scripts\python.exe .\run_cli.py vast-live `
  --checkpoint runs\worm_unet\best_model.pt `
  --device cpu `
  --crop-size 512 `
  --output-dir outputs\vast_live
```

## Read current VAST state

First make sure VAST Lite is open and `Window > Remote Control API Server` is enabled.

```powershell
.\.venv\Scripts\python.exe .\run_cli.py vast-state
```

That should print:

- dataset size
- selected layer
- selected EM layer
- selected segmentation layer
- selected segment
- current mouse voxel

## Live VAST bridge

This is the direct bridge. It watches VAST for new left-clicks, reads the current EM crop through the VAST API, runs the model, and writes the predicted mask back into the currently selected segmentation layer.

```powershell
.\.venv\Scripts\python.exe .\run_cli.py vast-live `
  --checkpoint runs\baseline\best_model.pt `
  --device cpu `
  --crop-size 512 `
  --output-dir outputs\vast_live
```

Use it like this:

1. Open your dataset in VAST Lite.
2. Enable `Window > Remote Control API Server`.
3. Select the segmentation layer you want to write into.
4. Select the target segment color/id in VAST.
5. Start `vast-live` from PowerShell.
6. In VAST, move the cursor over the target cell and left-click near its middle.
7. The bridge will write the predicted mask into the selected segmentation layer.

Important limitation in this first live version:

- VAST exposes the last click in window coordinates, but not the clicked voxel directly in the API state payload.
- The bridge therefore uses the current mouse voxel from VAST when it sees the new click event.
- In practice, keep the cursor over the target voxel when you click and do not move it immediately after the click.

Debug outputs are saved in `outputs\vast_live`:

- `*_crop.png`
- `*_mask.png`
- `*_overlay.png`
- `*_event.json`

See [examples/vast_workflow.md](/c:/Users/lefty/OneDrive/Documents/GitHub/PointnClick-Segmentation/examples/vast_workflow.md).
