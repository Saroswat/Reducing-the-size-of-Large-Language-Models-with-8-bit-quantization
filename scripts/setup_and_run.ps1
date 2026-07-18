[CmdletBinding()]
param(
    [ValidateSet("transformers", "openvino", "all")]
    [string]$Backend = "transformers",

    [switch]$SkipInstall,
    [switch]$NoBrowser,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Assert-LastCommand {
    param([string]$Message)
    if ($LASTEXITCODE -ne 0) {
        throw "$Message (exit code $LASTEXITCODE)"
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$webRoot = Join-Path $projectRoot "web"
$venvRoot = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

Write-Host @"

  QuantLab local setup
  Project: $projectRoot
  Backend: $Backend

"@ -ForegroundColor Green

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot "pyproject.toml"))) {
    throw "pyproject.toml was not found at $projectRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $webRoot "package.json"))) {
    throw "web/package.json was not found at $webRoot"
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Step "Creating the Python virtual environment"
    $pyLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        & $pyLauncher.Source -m venv $venvRoot
    }
    else {
        $pythonLauncher = Get-Command "python.exe" -ErrorAction SilentlyContinue
        if ($null -eq $pythonLauncher) {
            throw "Python 3.10 or newer is required. Install Python, then rerun this script."
        }
        & $pythonLauncher.Source -m venv $venvRoot
    }
    Assert-LastCommand "Virtual-environment creation failed"
}

$pythonVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Assert-LastCommand "Unable to inspect the virtual-environment Python"
$versionParts = $pythonVersion.Trim().Split(".")
if ([int]$versionParts[0] -ne 3 -or [int]$versionParts[1] -lt 10) {
    throw "QuantLab requires Python 3.10 or newer; this environment uses $pythonVersion."
}
Write-Host "Python ${pythonVersion}: $venvPython" -ForegroundColor DarkGray

$extras = switch ($Backend) {
    "transformers" { "web,transformers" }
    "openvino" { "web,openvino" }
    "all" { "web,transformers,bitsandbytes,openvino" }
}

if (-not $SkipInstall) {
    Write-Step "Installing/upgrading QuantLab Python dependencies [$extras]"
    & $venvPython -m pip install --upgrade pip
    Assert-LastCommand "pip upgrade failed"
    Push-Location $projectRoot
    try {
        & $venvPython -m pip install --upgrade -e ".[$extras]"
        Assert-LastCommand "QuantLab dependency installation failed"
    }
    finally {
        Pop-Location
    }
}

Write-Step "Validating the Python runtime"
& $venvPython -c "import fastapi, numpy, uvicorn; print(f'NumPy {numpy.__version__} | FastAPI {fastapi.__version__} | Uvicorn {uvicorn.__version__}')"
Assert-LastCommand "Core Python imports failed. Rerun without -SkipInstall to repair the environment"

if ($Backend -in @("transformers", "all")) {
    & $venvPython -c "import torch, transformers; print(f'PyTorch {torch.__version__} | Transformers {transformers.__version__}')"
    Assert-LastCommand "Transformers backend validation failed"
}
if ($Backend -in @("openvino", "all")) {
    & $venvPython -c "import openvino; from optimum.intel import OVModelForCausalLM; print(f'OpenVINO {openvino.__version__} | Optimum Intel ready')"
    Assert-LastCommand "OpenVINO backend validation failed"
}

$npmCommand = Get-Command "npm.cmd" -ErrorAction SilentlyContinue
if ($null -eq $npmCommand) {
    throw "npm.cmd was not found. Install Node.js, reopen PowerShell, and rerun this script."
}

Push-Location $webRoot
try {
    if (-not $SkipInstall) {
        Write-Step "Installing locked Node/React dependencies"
        if (Test-Path -LiteralPath (Join-Path $webRoot "package-lock.json")) {
            & $npmCommand.Source ci
        }
        else {
            & $npmCommand.Source install
        }
        Assert-LastCommand "Node dependency installation failed"
    }

    if ($CheckOnly) {
        Write-Step "Running frontend verification"
        & $npmCommand.Source run lint
        Assert-LastCommand "ESLint failed"
        & $npmCommand.Source run build
        Assert-LastCommand "Vite production build failed"
        Write-Host "`nQuantLab verification passed." -ForegroundColor Green
        exit 0
    }

    $env:QUANTLAB_PYTHON = $venvPython
    $env:QUANTLAB_OPEN_BROWSER = if ($NoBrowser) { "0" } else { "1" }

    Write-Step "Starting QuantLab"
    Write-Host "Dashboard: http://127.0.0.1:5173" -ForegroundColor Green
    Write-Host "API docs:  http://127.0.0.1:8000/docs" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop both services.`n" -ForegroundColor DarkGray
    & $npmCommand.Source run dev
    Assert-LastCommand "QuantLab stopped with an error"
}
finally {
    Pop-Location
}
