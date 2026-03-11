# PointnClick Segmentation

Interactive click-to-mask segmentation for electron microscopy imagery, with a correction loop that turns user fixes into new training data.

This repository provides:

- A click-conditioned segmentation model for 2D EM images
- Training code for previously segmented cells
- Inference code that predicts a full mask from one positive click
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

## Expected dataset format

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

## Install

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use `run_cli.py` from the repo root. You do not need `pip install -e .`.

## Prepare your current exports

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

## Train

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

This first version uses a sidecar workflow:

1. Export a 2D slice image from VAST Lite.
2. Record the click location on the target cell.
3. Run `predict`.
4. Review and correct the mask in VAST.
5. Export the corrected mask.
6. Run `add-feedback`.
7. Periodically run `finetune`.

See [examples/vast_workflow.md](/c:/Users/lefty/OneDrive/Documents/GitHub/PointnClick-Segmentation/examples/vast_workflow.md).
