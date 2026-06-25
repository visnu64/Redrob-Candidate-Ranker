# setup_env.ps1
# PowerShell script to set up a clean Python virtual environment and install requirements.

Write-Host "=== Creating Python Virtual Environment (.venv) ===" -ForegroundColor Cyan
python -m venv .venv

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create virtual environment. Make sure python is installed and on your PATH."
    exit 1
}

Write-Host "=== Upgrading pip ===" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "=== Installing dependencies from requirements.txt ===" -ForegroundColor Cyan
.\.venv\Scripts\pip.exe install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "=== Setup Complete! ===" -ForegroundColor Green
    Write-Host "To activate the environment in PowerShell, run:" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "Then you can run precompute.py and rank.py safely." -ForegroundColor Yellow
} else {
    Write-Error "Failed to install dependencies."
}
