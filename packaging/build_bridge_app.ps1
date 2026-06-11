param(
  [string]$Name = "PointnClickBridge"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "Installing runtime requirements..."
py -3 -m pip install -r requirements.txt

Write-Host "Installing build requirements..."
py -3 -m pip install -r requirements-build.txt

Write-Host "Building $Name..."
py -3 -m PyInstaller `
  --clean `
  --noconfirm `
  --onedir `
  --windowed `
  --name $Name `
  --paths src `
  --collect-all webknossos `
  run_bridge_app.py

Write-Host "Build complete: dist\$Name\$Name.exe"
