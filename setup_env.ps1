# ============================================================
#  XANG_DAU_FORECAST environment setup (Python 3.9) - PowerShell
#  Usage:
#    .\setup_env.ps1
#    .\setup_env.ps1 -Python "C:\Path\to\python.exe"
#  If blocked: run once ->  Set-ExecutionPolicy -Scope Process Bypass
# ============================================================
param([string]$Python = "")
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv39"

if ($Python -ne "") { $pyExe = $Python; $pyArgs = @() }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $pyExe = "py"; $pyArgs = @("-3.9") }
else { $pyExe = "python"; $pyArgs = @() }

Write-Host "=== Using Python: $pyExe $pyArgs ===" -ForegroundColor Cyan
& $pyExe @pyArgs --version

Write-Host "=== Creating virtual env: $Venv ===" -ForegroundColor Cyan
& $pyExe @pyArgs -m venv $Venv

. (Join-Path $Venv "Scripts\Activate.ps1")
Write-Host "=== Installing packages (several minutes) ===" -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r (Join-Path $Root "requirements-py39.txt")

Write-Host "=== Registering Jupyter kernel ===" -ForegroundColor Cyan
python -m ipykernel install --user --name xangdau-py39 --display-name "Python 3.9 (xangdau)"

Write-Host "=== Verifying environment ===" -ForegroundColor Cyan
python (Join-Path $Root "verify_env.py")

Write-Host "`nDONE. In Jupyter pick kernel: 'Python 3.9 (xangdau)'" -ForegroundColor Green
Write-Host "Activate later: $Venv\Scripts\Activate.ps1"
