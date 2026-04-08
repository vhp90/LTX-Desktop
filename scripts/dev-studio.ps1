$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

if (-not $env:LTX_BROWSER_ONLY) { $env:LTX_BROWSER_ONLY = "1" }
if (-not $env:LTX_APP_DATA_DIR) { $env:LTX_APP_DATA_DIR = (Join-Path $ProjectDir ".ltx-data") }
if (-not $env:LTX_BACKEND_BIND_HOST) { $env:LTX_BACKEND_BIND_HOST = "0.0.0.0" }
if (-not $env:LTX_PORT) { $env:LTX_PORT = "18000" }
if (-not $env:LTX_ALLOWED_ORIGIN_REGEX) { $env:LTX_ALLOWED_ORIGIN_REGEX = "^https://(lightning\.ai|.*\.cloudspaces\.litng\.ai)$" }

Set-Location $ProjectDir
corepack pnpm exec concurrently -k `
  "cd backend; uv run --python 3.13.7 python ltx2_server.py" `
  "vite"
