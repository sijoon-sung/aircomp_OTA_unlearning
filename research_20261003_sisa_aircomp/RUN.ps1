param([string]$Mode='preflight',[string]$Out='preflight_v1')
$ErrorActionPreference='Stop'
$env:PYTHONPATH='C:/Users/DISLAB/Desktop/split_learning/ota_ful/.analysis;C:/Users/DISLAB/Desktop/split_learning/ota_ful/.deps_cuda'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='1'
$taskPython='C:/Users/DISLAB/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $taskPython (Join-Path $PSScriptRoot 'experiment.py') --mode $Mode --out $Out
if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $LASTEXITCODE" }
