$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Project virtual environment not found at $VenvPath. Create it and install dependencies first."
}

$env:VIRTUAL_ENV = $VenvPath
$env:PATH = (Join-Path $VenvPath "Scripts") + [System.IO.Path]::PathSeparator + $env:PATH

Set-Location $ProjectRoot
& $PythonExe -m uvicorn main:app --reload
