Set-Location -Path $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=========================================================" -ForegroundColor Green
Write-Host "  MaizeSight - UAV Maize Yield Intelligence Web Server" -ForegroundColor Yellow
Write-Host "=========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Starting web server..." -ForegroundColor Cyan

$pyExe = $null
$localPy313 = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
$localPy312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$venvPy = ".\.venv\Scripts\python.exe"

if (Test-Path $localPy313) {
    $pyExe = $localPy313
} elseif (Test-Path $localPy312) {
    $pyExe = $localPy312
} elseif (Test-Path $venvPy) {
    $pyExe = $venvPy
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pyExe = "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pyExe = "python"
}

if ($pyExe) {
    & $pyExe app.py
} else {
    Write-Host "[ERROR] Python was not found." -ForegroundColor Red
    pause
}


