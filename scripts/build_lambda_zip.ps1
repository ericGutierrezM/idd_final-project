$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $projectRoot "build"
$stagingDir = Join-Path $buildRoot "lambda_package"
$zipPath = Join-Path $buildRoot "lambda_deployment.zip"

if (Test-Path $stagingDir) {
    Remove-Item -LiteralPath $stagingDir -Recurse -Force
}

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

New-Item -ItemType Directory -Path $stagingDir | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot "src") -Destination $stagingDir -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot "check_output_format.py") -Destination $stagingDir

$pycacheDirs = Get-ChildItem -Path $stagingDir -Recurse -Directory -Filter "__pycache__"
foreach ($dir in $pycacheDirs) {
    Remove-Item -LiteralPath $dir.FullName -Recurse -Force
}

Compress-Archive -Path (Join-Path $stagingDir "*") -DestinationPath $zipPath
Write-Output "Created Lambda zip at $zipPath"
