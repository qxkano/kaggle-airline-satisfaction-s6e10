$ErrorActionPreference='Stop'
$env:TEMP='E:/codex-download/tmp'
$env:TMP='E:/codex-download/tmp'
$env:MPLCONFIGDIR='E:/codex-download/tmp/matplotlib-s6e10'
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
$env:OPENBLAS_NUM_THREADS='8'
$project=$PSScriptRoot
$python='E:/codex-download/environments/datascience-s6e10/Scripts/python.exe'
if (!(Test-Path -LiteralPath "$project/data/raw/train.csv")) {throw 'Missing official train.csv in data/raw'}
& $python "$project/src/check_resume.py"
if ($LASTEXITCODE -ne 0) {throw 'Source or data mismatch: prediction caches cannot be reused'}
& $python "$project/src/smoke.py"
if ($LASTEXITCODE -ne 0) {throw 'Environment smoke failed'}
& $python -u "$project/src/run_experiments.py" 2>&1 | Tee-Object -FilePath "$project/logs/rerun.log"
if ($LASTEXITCODE -ne 0) {throw 'Experiment failed; inspect logs/rerun.log'}
& $python "$project/src/verify_results.py"
if ($LASTEXITCODE -ne 0) {throw 'Verification failed'}
& $python "$project/src/postprocess.py"
if ($LASTEXITCODE -ne 0) {throw 'Model reload verification or report postprocessing failed'}
