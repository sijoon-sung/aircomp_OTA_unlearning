param([ValidateSet('all','1','2','3')][string]$Stage='all')
$ErrorActionPreference='Stop'
$taskRoot=$PSScriptRoot
$taskPython='C:/Users/DISLAB/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$env:PYTHONPATH='C:/Users/DISLAB/Desktop/split_learning/ota_ful/.analysis;C:/Users/DISLAB/Desktop/split_learning/ota_ful/.deps_cuda'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
function Invoke-ResearchScript([string]$Name,[string[]]$TaskArgs=@()) {
    & $taskPython (Join-Path $taskRoot $Name) @TaskArgs
    if ($LASTEXITCODE -ne 0) { throw "Research step failed: $Name" }
}
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'results/math_validation.json'))) {
    Invoke-ResearchScript 'math_validation.py'
}
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'results/calibration.json'))) {
    Invoke-ResearchScript 'calibrate.py'
}
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'results/integration_checks.json'))) {
    Invoke-ResearchScript 'integration_checks.py'
}
Invoke-ResearchScript -Name 'run_experiments.py' -TaskArgs @('--stage',$Stage)
# Existing completed seed results are skipped by the Python runner.
# Reference audit is intentionally separate: Invoke-ResearchScript 'audit_references.py'
