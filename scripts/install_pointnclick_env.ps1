param(
  [string]$EnvName = "pointnclick",
  [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu121"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "Checking conda..."
conda --version
if ($LASTEXITCODE -ne 0) {
  throw "Conda was not found. Install Miniconda or Anaconda first."
}

$envExists = conda env list | Select-String -Pattern "^\s*$EnvName\s+"
if (-not $envExists) {
  Write-Host "Creating conda environment: $EnvName"
  conda create -n $EnvName python=3.11 -y
  if ($LASTEXITCODE -ne 0) {
    throw "Could not create conda environment $EnvName."
  }
}

Write-Host "Installing PointnClick dependencies into $EnvName..."
conda run -n $EnvName python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
  throw "pip upgrade failed."
}

conda run -n $EnvName python -m pip install torch torchvision --index-url $TorchIndexUrl
if ($LASTEXITCODE -ne 0) {
  throw "Torch install failed. Check the CUDA wheel URL or internet access."
}

conda run -n $EnvName python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
  throw "Runtime dependency install failed."
}

conda run -n $EnvName python -c "import torch, webknossos; print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available()); print('webknossos ok')"
if ($LASTEXITCODE -ne 0) {
  throw "Validation import failed."
}

Write-Host "PointnClick environment is ready."
Write-Host "Run: .\scripts\run_pointnclick_bridge.ps1"
