param(
  [string]$Name = "PointnClickBridge"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = (Get-Command python -ErrorAction Stop).Source
Write-Host "Using Python: $Python"
& $Python -c "import sys; print(sys.executable); print(sys.version)"
if ($LASTEXITCODE -ne 0) {
  throw "Could not run the active python executable."
}

Write-Host "Checking torch..."
& $Python -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
  throw "Torch does not import in this Python environment. Activate the environment that runs PointnClick first."
}

Write-Host "Installing runtime requirements..."
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
  throw "Runtime dependency install failed."
}

Write-Host "Installing build requirements..."
& $Python -m pip install -r requirements-build.txt
if ($LASTEXITCODE -ne 0) {
  throw "Build dependency install failed."
}

Write-Host "Building $Name..."
& $Python -m PyInstaller `
  --clean `
  --noconfirm `
  --onedir `
  --windowed `
  --name $Name `
  --paths src `
  --exclude-module matplotlib `
  --exclude-module matplotlib_inline `
  --exclude-module IPython `
  --exclude-module notebook `
  --exclude-module jupyter `
  --hidden-import webknossos `
  --collect-all webknossos `
  run_bridge_app.py
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller build failed."
}

$DistDir = Join-Path $RepoRoot "dist\$Name"
if (-not (Test-Path -LiteralPath $DistDir)) {
  throw "Expected build output directory does not exist: $DistDir"
}

$ModelUrlFile = Join-Path $RepoRoot "model_url.txt"
if (Test-Path -LiteralPath $ModelUrlFile) {
  Copy-Item -LiteralPath $ModelUrlFile -Destination (Join-Path $DistDir "model_url.txt") -Force
  Write-Host "Copied model_url.txt into dist\$Name"
}

Write-Host "Build complete: dist\$Name\$Name.exe"
