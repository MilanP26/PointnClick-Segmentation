param(
  [string]$EnvName = "pointnclick"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "Starting PointnClick Bridge from conda environment: $EnvName"
conda run -n $EnvName python run_bridge_app.py
