$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

if (-not $env:LTX_HF_USE_XET) { $env:LTX_HF_USE_XET = "0" }
if (-not $env:HF_HUB_DISABLE_XET) { $env:HF_HUB_DISABLE_XET = "1" }
if (-not $env:HF_HUB_DISABLE_PROGRESS_BARS) { $env:HF_HUB_DISABLE_PROGRESS_BARS = "1" }
if (-not $env:LTX_HF_DOWNLOAD_RETRIES) { $env:LTX_HF_DOWNLOAD_RETRIES = "6" }

Set-Location (Join-Path $ProjectDir "backend")
uv run --python 3.13.7 python ../scripts/setup-models.py @args
