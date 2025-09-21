# PowerShell helper to create venv, activate, and install deps
$proj = Split-Path -Parent $MyInvocation.MyCommand.Definition
Write-Host "Project dir: $proj"
python -m venv $proj\.venv
Write-Host "Created venv at $proj\\.venv"
Write-Host "To activate the venv in PowerShell run: . $proj\\.venv\\Scripts\\Activate.ps1"
Write-Host "Installing requirements... (may take a few minutes)"
& $proj\.venv\Scripts\python.exe -m pip install --upgrade pip
& $proj\.venv\Scripts\python.exe -m pip install -r $proj\requirements.txt
Write-Host "Done. Activate the venv and run: uvicorn app.main:app --reload"