param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$FashionRaw,
    [string]$DependencyPath = '',
    [string]$OutputRoot = (Join-Path $PSScriptRoot 'replication')
)
$ErrorActionPreference = 'Stop'
if ($DependencyPath) { $env:PYTHONPATH = $DependencyPath }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'

& $Python (Join-Path $PSScriptRoot 'run_sfl.py') --data $FashionRaw --out (Join-Path $OutputRoot 'fashion20') --seeds '311,312,313' --rounds 80 --snr 20
if ($LASTEXITCODE -ne 0) { throw 'fashion20 failed' }

& $Python (Join-Path $PSScriptRoot 'run_sfl.py') --data $FashionRaw --out (Join-Path $OutputRoot 'fashion10') --seeds '311,312,313' --rounds 80 --snr 10 --methods 'Latent4-Q16-W8,OTA-Q16-W8,OTA-Stein-Q16-W8,OTA-Reg-Q16-W8'
if ($LASTEXITCODE -ne 0) { throw 'fashion10 failed' }

& $Python (Join-Path $PSScriptRoot 'run_channel_width.py') --data $FashionRaw --out (Join-Path $OutputRoot 'width20') --seeds '311,312,313' --rounds 80 --snr 20 --methods 'Latent4-Q16-W8,OTA-Q16-W8,OTA-Stein-Q16-W8,OTA-Reg-Q16-W8,OTA-Q16-W4,Latent4-Q16-W4'
if ($LASTEXITCODE -ne 0) { throw 'width20 failed' }

& $Python (Join-Path $PSScriptRoot 'run_digital_control.py') --data $FashionRaw --out (Join-Path $OutputRoot 'fashion20') --seeds '311,312,313' --rounds 80 --snr 20
if ($LASTEXITCODE -ne 0) { throw 'digital controls failed' }

& $Python (Join-Path $PSScriptRoot 'run_digital_control.py') --data $FashionRaw --out (Join-Path $OutputRoot 'width20') --layout width --bits 4 --seeds '311,312,313' --rounds 80 --snr 20
if ($LASTEXITCODE -ne 0) { throw 'width digital control failed' }

& $Python (Join-Path $PSScriptRoot 'analyze.py') --root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw 'result audit failed' }
