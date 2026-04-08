$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Set-Location (Join-Path $ProjectDir "backend")
uv run --python 3.13.7 python ../scripts/setup-models.py @args
