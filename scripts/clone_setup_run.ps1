[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $HOME "QuantLab"),

    [ValidateSet("transformers", "openvino", "all")]
    [string]$Backend = "transformers",

    [switch]$SkipUpdate,
    [switch]$NoBrowser,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repository = "https://github.com/Saroswat/Reducing-the-size-of-Large-Language-Models-with-8-bit-quantization.git"
$expectedOrigin = $repository.ToLowerInvariant().TrimEnd(".git")
$destinationPath = [System.IO.Path]::GetFullPath($Destination)

function Assert-LastCommand {
    param([string]$Message)
    if ($LASTEXITCODE -ne 0) {
        throw "$Message (exit code $LASTEXITCODE)"
    }
}

function Require-Command {
    param(
        [string]$Name,
        [string]$InstallHint
    )
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "$Name is required. $InstallHint"
    }
    return $command.Source
}

Write-Host @"

  QuantLab clone, setup, and run
  Repository:  $repository
  Destination: $destinationPath
  Backend:     $Backend

"@ -ForegroundColor Green

$git = Require-Command "git.exe" "Install Git for Windows from https://git-scm.com/download/win, reopen PowerShell, and retry."
[void](Require-Command "npm.cmd" "Install the Node.js LTS release from https://nodejs.org, reopen PowerShell, and retry.")
if ($null -eq (Get-Command "py.exe" -ErrorAction SilentlyContinue) -and
    $null -eq (Get-Command "python.exe" -ErrorAction SilentlyContinue)) {
    throw "Python 3.10 or newer is required. Install it from https://python.org, reopen PowerShell, and retry."
}

if (Test-Path -LiteralPath $destinationPath) {
    if (-not (Test-Path -LiteralPath (Join-Path $destinationPath ".git"))) {
        $existing = @(Get-ChildItem -LiteralPath $destinationPath -Force -ErrorAction SilentlyContinue)
        if ($existing.Count -gt 0) {
            throw "Destination exists and is not a QuantLab Git checkout: $destinationPath"
        }
        Write-Host "Cloning QuantLab into the existing empty directory..." -ForegroundColor Cyan
        & $git clone $repository $destinationPath
        Assert-LastCommand "Git clone failed"
    }
    else {
        $origin = (& $git -C $destinationPath remote get-url origin).Trim()
        Assert-LastCommand "Unable to read the existing Git origin"
        $normalizedOrigin = $origin.ToLowerInvariant().TrimEnd(".git")
        if ($normalizedOrigin -ne $expectedOrigin) {
            throw "Destination points to a different Git repository: $origin"
        }

        if (-not $SkipUpdate) {
            $changes = & $git -C $destinationPath status --porcelain
            Assert-LastCommand "Unable to inspect the existing checkout"
            if ($changes) {
                throw "The existing QuantLab checkout has local changes. Commit or stash them, or rerun with -SkipUpdate."
            }
            Write-Host "Updating the existing QuantLab checkout..." -ForegroundColor Cyan
            & $git -C $destinationPath pull --ff-only
            Assert-LastCommand "Git update failed; the checkout was left unchanged"
        }
    }
}
else {
    $parent = Split-Path -Parent $destinationPath
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Write-Host "Cloning QuantLab..." -ForegroundColor Cyan
    & $git clone $repository $destinationPath
    Assert-LastCommand "Git clone failed"
}

$setupScript = Join-Path $destinationPath "scripts\setup_and_run.ps1"
if (-not (Test-Path -LiteralPath $setupScript)) {
    throw "The cloned repository does not contain scripts/setup_and_run.ps1"
}

$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $setupScript,
    "-Backend", $Backend
)
if ($NoBrowser) { $arguments += "-NoBrowser" }
if ($CheckOnly) { $arguments += "-CheckOnly" }

Write-Host "`nHanding off to the QuantLab environment installer..." -ForegroundColor Cyan
& powershell.exe @arguments
Assert-LastCommand "QuantLab setup or startup failed"
