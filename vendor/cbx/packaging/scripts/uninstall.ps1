$ErrorActionPreference = "Stop"

<# 
cbx uninstaller for Windows (removes the installed binary; caches are managed separately).

Usage (PowerShell):
  irm https://raw.githubusercontent.com/srv1n/cbx/main/packaging/scripts/uninstall.ps1 | iex

Optional:
  $env:INSTALL_DIR="$env:LOCALAPPDATA\\cbx\\bin"
#>

$installDir = $env:INSTALL_DIR
if ([string]::IsNullOrWhiteSpace($installDir)) { $installDir = "$env:LOCALAPPDATA\\cbx\\bin" }

$bin = Join-Path $installDir "cbx.exe"

Write-Host "cbx uninstaller"
Write-Host "  uninstall: $bin"

if (Test-Path $bin) {
  Remove-Item -Force $bin
  Write-Host "Removed: $bin"
} else {
  Write-Host "Not found: $bin"
}

Write-Host ""
Write-Host "Note: model + voice caches are NOT removed by this script."
Write-Host "To remove caches, run:"
Write-Host "  cbx clean --all --voices --yes"

