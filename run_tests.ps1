Write-Host "Creating venv if missing..."
if (-Not (Test-Path '.\.venv')) { python -m venv .venv }
Write-Host "Activating venv and installing requirements..."
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Write-Host "Running pytest..."
python -m pytest -q
