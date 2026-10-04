"""Read-only, standard-library verification of the SISA research snapshot."""
from pathlib import Path
import ast,gzip,hashlib,json,zipfile
R=Path(__file__).resolve().parent
m=json.loads((R/'SISA_EXPORT_MANIFEST_20261005.json').read_text(encoding='utf-8'))
counts=dict(files=0,python=0,json=0,compressed=0)
for row in m['included']:
    p=(R/row['path']).resolve();assert p.is_relative_to(R)
    raw=p.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],row['path']
    if p.suffix=='.py':ast.parse(raw.decode('utf-8-sig'),filename=str(p));counts['python']+=1
    if p.suffix=='.json':json.loads(raw.decode('utf-8-sig'));counts['json']+=1
    if p.suffix=='.npz':
        with zipfile.ZipFile(p) as z:assert z.testzip() is None
        counts['compressed']+=1
    if p.name.endswith('.json.gz'):
        decoded=gzip.decompress(raw).decode('utf-8-sig')
        try:json.loads(decoded)
        except json.JSONDecodeError as exc:
            if exc.msg!='Extra data':raise
            for line in decoded.splitlines():
                if line.strip():json.loads(line)
        counts['compressed']+=1
    assert p.stat().st_size<100*1024**2
    counts['files']+=1
latest=R/'aircomp_mechanism_20261004'
rows=[json.loads(p.read_text(encoding='utf-8')) for p in latest.glob('*_summary.json')]
assert len(rows)==10 and sum(len(r['deletions']) for r in rows)==200
assert not (latest/'results.json').exists()
for row in rows:
    assert sorted(i for g in row['groups'] for i in g)==list(range(20))
    assert all(len(g)==5 for g in row['groups'])
    assert sorted(d['client'] for d in row['deletions'])==list(range(20))
print(json.dumps(dict(passed=True,**counts,complete_conditions=10,complete_deletions=200,planned_conditions=18),indent=2))
