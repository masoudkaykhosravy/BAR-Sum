# BAR-Sum Quick Execution Script for Windows PowerShell
Write-Host "Activating Virtual Environment and Starting Pipeline..." -ForegroundColor Cyan
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& "$PSScriptRoot\venv\Scripts\Activate.ps1"
python run_all.py
