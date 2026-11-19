param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
& $Python (Join-Path $PSScriptRoot 'src/course_pipeline.py')
if ($LASTEXITCODE -ne 0) { throw 'Course pipeline failed; see error above.' }
& $Python (Join-Path $PSScriptRoot 'src/course_tree.py')
if ($LASTEXITCODE -ne 0) { throw 'Decision tree experiment failed.' }
