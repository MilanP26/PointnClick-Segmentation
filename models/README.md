# Model Weights

Do not commit trained model binaries here. The repository ignores `*.pt`, `*.pth`, and `*.onnx` files in this folder.

Recommended release flow:

1. Train locally and keep the checkpoint under `runs/`.
2. Upload the selected checkpoint, for example `best_model.pt`, to a GitHub Release or another direct-download host.
3. Paste that asset URL into the PointnClick Bridge app's `Model download URL` field.
4. Optionally paste the SHA256 checksum so users can verify the downloaded file.

Example checksum command on Windows:

```powershell
Get-FileHash .\runs\runs\worm_unet\best_model.pt -Algorithm SHA256
```
