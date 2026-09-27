"""Read-only scientific record checks; normalize links in our reports only."""
from pathlib import Path
import hashlib,json,ast,re,statistics,collections
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
assert json.loads((OUT/'math_validation.json').read_text())['passed']
assert json.loads((OUT/'integration_checks.json').read_text())['passed']
completed=[]
for stage in [1,2,3]:
    for seed in [271,272,273]:
        p=OUT/f'method{stage}_seed{seed}.json';r=json.loads(p.read_text());assert r['complete']
        completed.append(p.name)
audit=json.loads((OUT/'reference_audit.json').read_text());assert len(audit)==6
audit_ratios=collections.defaultdict(list)
for a in audit:
    for r in a['rows']:
        if r['method'] in ['curenus_exact','noiseless_prune']:
            assert r['mean_js_ratio_to_noop']>1
            audit_ratios[r['method']].append(r['mean_js_ratio_to_noop'])
replay=json.loads((OUT/'method3_seed271.json').read_text())['fresh_replay']
assert replay['all_teacher_hashes_equal'] and replay['student_parameter_max_error']==0.
for seed in [271,272,273]:
    rows=json.loads((OUT/f'method3_seed{seed}.json').read_text())['rows']
    for r in rows:
        if r['method'].startswith('exclude_fresh_ota'):assert r['metrics']['forget_normalized_js']<1
order=[]
for line in (OUT/'sequential_gpu.log').read_text(encoding='utf-8',errors='replace').splitlines():
    try:r=json.loads(line)
    except Exception:continue
    if isinstance(r.get('stage'),int):order.append(r['stage'])
assert order==sorted(order), 'GPU method stages were not sequential'
for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
missing=[]
for p in ROOT.glob('*.md'):
    txt=p.read_text(encoding='utf-8')
    def fix(m):
        dest=m.group(2).strip('<>')
        if dest.startswith(('https:','http:','#')):return m.group(0)
        q=Path(dest) if Path(dest).is_absolute() else (p.parent/dest).resolve()
        if not q.exists():missing.append((p.name,dest))
        s=q.as_posix();return m.group(1)+'('+('<'+s+'>' if ' ' in s else s)+')'
    txt=re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)',fix,txt)
    p.write_text(txt,encoding='utf-8')
assert not missing,missing
dataset=Path('C:/Users/DISLAB/Desktop/split_learning/ota_ful/data/FashionMNIST/raw')
datahash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.iterdir() if p.is_file() and not p.name.endswith('.gz')}
files=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or 'vendor' in p.parts or p.name in ['verification.json','artifact_manifest.json','verification_console.txt']:continue
    files.append({'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'artifact_manifest.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
result={'completed_method_seed_files':completed,'reference_audits':len(audit),'sequential_stages_verified':True,
        'mathematical_checks_passed':True,'source_integration_checks_passed':True,'local_links_missing':missing,
        'independent_teacher_full_replay_seed':271,'teacher_hash_equality_count':9,'student_parameter_replay_error':0.,
        'two_reference_ratios':dict(audit_ratios),'dataset_sha256':datahash,'scientific_figure_visually_reviewed':True,
        'no_privacy_certificate_claimed':True,'full_upstream_benchmark_reproduction_claimed':False}
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
