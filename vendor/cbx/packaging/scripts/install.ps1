$ErrorActionPreference = "Stop"

<# 
cbx installer (GitHub Releases) for Windows.

Usage (PowerShell):
  irm https://raw.githubusercontent.com/srv1n/cbx/main/packaging/scripts/install.ps1 | iex

Optional:
  $env:REPO="srv1n/cbx"
  $env:VERSION="v0.0.1"
  $env:INSTALL_DIR="$env:LOCALAPPDATA\\cbx\\bin"

Notes:
  - Installs the binary only. Models download at runtime via Hugging Face Hub.
#>

$repo = $env:REPO
if ([string]::IsNullOrWhiteSpace($repo)) { $repo = "srv1n/cbx" }

$installDir = $env:INSTALL_DIR
if ([string]::IsNullOrWhiteSpace($installDir)) { $installDir = "$env:LOCALAPPDATA\\cbx\\bin" }

function Get-LatestTag($repo) {
  $url = "https://api.github.com/repos/$repo/releases/latest"
  $headers = @{ "Accept" = "application/vnd.github+json"; "User-Agent" = "cbx-installer" }
  $resp = Invoke-RestMethod -Uri $url -Headers $headers -Method Get
  if (-not $resp.tag_name) { throw "Could not determine latest release tag" }
  return $resp.tag_name
}

$version = $env:VERSION
if ([string]::IsNullOrWhiteSpace($version)) { $version = Get-LatestTag $repo }

$binName = "cbx.exe"
$target = "x86_64-pc-windows-msvc"
$archive = "cbx-$version-$target.zip"
$baseUrl = "https://github.com/$repo/releases/download/$version"
$url = "$baseUrl/$archive"
$shaUrl = "$url.sha256"

$tmpDir = Join-Path $env:TEMP ("cbx-" + [Guid]::NewGuid().ToString("n"))
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

try {
  Write-Host "Installing cbx"
  Write-Host "  repo:     $repo"
  Write-Host "  version:  $version"
  Write-Host "  target:   $target"
  Write-Host "  install:  $installDir\\$binName"
  Write-Host "  download: $url"

  $zipPath = Join-Path $tmpDir $archive
  $shaPath = Join-Path $tmpDir ($archive + ".sha256")

  Invoke-WebRequest -Uri $url -OutFile $zipPath
  Invoke-WebRequest -Uri $shaUrl -OutFile $shaPath

  $expected = (Get-Content $shaPath).Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)[0].ToLower()
  $actual = (Get-FileHash $zipPath -Algorithm SHA256).Hash.ToLower()
  if ($expected -ne $actual) { throw "SHA256 mismatch: expected $expected got $actual" }

  Expand-Archive -Path $zipPath -DestinationPath $tmpDir -Force

  New-Item -ItemType Directory -Force -Path $installDir | Out-Null
  Copy-Item -Force (Join-Path $tmpDir $binName) (Join-Path $installDir $binName)

  Write-Host "Done."
  Write-Host "Run: $installDir\\$binName --help"
  Write-Host "If cbx isn't on PATH, add $installDir to your user PATH."
} finally {
  Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
}

