$ErrorActionPreference = "Stop"

<# 
cbx default voice installer for Windows (downloads from GitHub Releases; verifies SHA256).

Usage (PowerShell):
  irm https://raw.githubusercontent.com/srv1n/cbx/main/packaging/scripts/install_default_voice.ps1 | iex

Optional:
  $env:REPO="srv1n/cbx"
  $env:VERSION="v0.0.4"
  $env:HF_HOME="C:\\path\\to\\huggingface"

Notes:
  - Installs two profiles:
    - default.cbxvoice      (dtype=fp32)
    - default-fp16.cbxvoice (dtype=fp16)
#>

$repo = $env:REPO
if ([string]::IsNullOrWhiteSpace($repo)) { $repo = "srv1n/cbx" }

$version = $env:VERSION
if ([string]::IsNullOrWhiteSpace($version)) {
  $headers = @{ "Accept" = "application/vnd.github+json"; "User-Agent" = "cbx-voice-installer" }
  $resp = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/latest" -Headers $headers -Method Get
  if (-not $resp.tag_name) { throw "Could not determine latest release tag" }
  $version = $resp.tag_name
}

$hfHome = $env:HF_HOME
if ([string]::IsNullOrWhiteSpace($hfHome)) {
  # Common HF default on Windows is %LOCALAPPDATA%\huggingface
  $hfHome = Join-Path $env:LOCALAPPDATA "huggingface"
}

$voiceDir = Join-Path (Join-Path $hfHome "cbx") "voices"
$baseUrl = "https://github.com/$repo/releases/download/$version"

Write-Host "cbx default voice installer"
Write-Host "  repo:     $repo"
Write-Host "  version:  $version"
Write-Host "  hf_home:  $hfHome"
Write-Host "  install:  $voiceDir"
Write-Host ""
Write-Host "This installs two profiles:"
Write-Host "  - default.cbxvoice      (dtype=fp16)"
Write-Host "  - default-fp32.cbxvoice (dtype=fp32)"

New-Item -ItemType Directory -Force -Path $voiceDir | Out-Null

function Install-VoiceAsset($assetName, $destName) {
  $url = "$baseUrl/$assetName"
  $shaUrl = "$url.sha256"

  $tmp = Join-Path $env:TEMP ("cbx-voice-" + [Guid]::NewGuid().ToString("n"))
  New-Item -ItemType Directory -Force -Path $tmp | Out-Null
  try {
    $file = Join-Path $tmp $assetName
    $shaFile = Join-Path $tmp ($assetName + ".sha256")

    Invoke-WebRequest -Uri $url -OutFile $file
    Invoke-WebRequest -Uri $shaUrl -OutFile $shaFile

    $expected = (Get-Content $shaFile).Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)[0].ToLower()
    $actual = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
    if ($expected -ne $actual) { throw "SHA256 mismatch for $assetName: expected $expected got $actual" }

    Copy-Item -Force $file (Join-Path $voiceDir $destName)
  } finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
  }
}

Install-VoiceAsset "cbx-voice-default-fp16.cbxvoice" "default.cbxvoice"
Install-VoiceAsset "cbx-voice-default-fp32.cbxvoice" "default-fp32.cbxvoice"

Write-Host ""
Write-Host "Installed:"
Write-Host "  $(Join-Path $voiceDir 'default.cbxvoice')"
Write-Host "  $(Join-Path $voiceDir 'default-fp32.cbxvoice')"
Write-Host ""
Write-Host "Try:"
Write-Host "  cbx speak --text \"Hello\" --out-wav out.wav"
Write-Host "  cbx speak --dtype fp32 --voice default-fp32 --text \"Hello\" --out-wav out.wav"
