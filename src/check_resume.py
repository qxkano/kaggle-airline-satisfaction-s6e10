"""Refuse prediction-cache reuse when source, data or protocol has changed."""
import hashlib,json
from pathlib import Path
ROOT=Path('E:/codex-download/项目/数据科学课程大作业-S6E10')
sources=[ROOT/'src/run_experiments.py',ROOT/'EXPERIMENT_PLAN.md',ROOT/'requirements-lock.txt']
current={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
manifest=ROOT/'results/run_source_hashes.json'
if manifest.exists():
    assert json.loads(manifest.read_text(encoding='utf-8'))==current,'Source/protocol changed: back up all outputs and start a fresh experiment.'
else:
    assert not list((ROOT/'results/predictions').glob('*.npz')),'Untracked existing prediction caches: start a fresh experiment.'
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps(current,indent=2),encoding='utf-8')
profile=ROOT/'results/data_profile.json'
if profile.exists():
    for name,expected in json.loads(profile.read_text(encoding='utf-8'))['sha256'].items():
        with (ROOT/'data/raw'/name).open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==expected,'Raw data changed: back up all outputs and start a fresh experiment.'
print('RESUME_GUARD_PASS')
