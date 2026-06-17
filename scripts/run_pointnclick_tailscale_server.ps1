param(
  [string]$EnvName = "pointnclick",
  [string]$Checkpoint = ".\runs\runs\worm_unet\best_model.pt",
  [string]$HostAddress = "0.0.0.0",
  [int]$Port = 8765,
  [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

conda run -n $EnvName python run_cli.py webknossos-remote-server `
  --checkpoint $Checkpoint `
  --host $HostAddress `
  --port $Port `
  --device $Device
