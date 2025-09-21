# Starts the server, runs the smoke test, then stops the server.
$proj = Split-Path -Parent $MyInvocation.MyCommand.Definition
$python = Join-Path $proj ".venv\Scripts\python.exe"

# Start server
$proc = Start-Process -FilePath $python -ArgumentList '-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000' -PassThru
Write-Host "Started server (PID $($proc.Id)). Waiting for startup..."
Start-Sleep -Seconds 2

# Run smoke test
Write-Host "Running smoke test"
& $python (Join-Path $proj 'tests\smoke_test.py')

# Stop server
Write-Host "Stopping server"
Stop-Process -Id $proc.Id -Force
Write-Host "Done"