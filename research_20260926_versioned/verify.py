from pathlib import Path
import json, math, hashlib, re

root=Path(__file__).resolve().parent;out=root/'results'
checks={}
env=json.loads((out/'environment.json').read_text())
assert env['protocol_sha256']==hashlib.sha256((root/'PROTOCOL_KO.md').read_bytes()).hexdigest()
assert env['code_sha256']==hashlib.sha256((root/'benchmark.py').read_bytes()).hexdigest()
assert json.loads((out/'math_checks.json').read_text())['passed']
checks['protocol_and_main_code_hashes_match']=True


def finite(obj):
    if isinstance(obj,float):assert math.isfinite(obj)
    elif isinstance(obj,dict):
        for v in obj.values():finite(v)
    elif isinstance(obj,list):
        for v in obj:finite(v)


counts={};max_power=0.;equality=[]
for family,nrows in [('DS',147),('OG',18),('RTD',28),('RTDfp64',29)]:
    records=[json.loads(p.read_text()) for p in sorted(out.glob(f'{family}_seed*.json'))]
    assert len(records)==3
    for rec in records:
        assert rec['complete'];finite(rec)
        assert len(rec['rows'])==nrows,(family,len(rec['rows']))
        if family=='RTDfp64':
            for row in rec['rows']:
                if row['method']=='RTD-Air-noiseless':
                    ex=row['extra'];assert ex['target_max_error']==0 and ex['parameter_max_error']==0
                    equality.append([rec['seed'],row['version']])
        for row in rec['rows']:
            if row.get('peak') is False:continue
            diag=row.get('diagnostic') or row.get('extra') or {}
            power=diag.get('power',[])
            if isinstance(power,dict):power=[power]
            for p in power:
                if not p:continue
                assert p['max_average_power']<=1+2e-6 and p['max_peak_power']<=4+2e-6
                max_power=max(max_power,p['max_peak_power'])
    counts[family]=sum(len(r['rows']) for r in records)
checks['rows']=counts;checks['six_noiseless_rtd_equalities']=equality;checks['max_peak_power']=max_power
summary=json.loads((out/'summary.json').read_text());finite(summary)
assert all(r['seeds']==3 for r in summary['records'])
checks['summary_groups']=len(summary['records'])
missing=[]
for p in root.glob('*.md'):
    for ref in re.findall(r'\]\((C:/[^)]+)\)',p.read_text(encoding='utf-8')):
        if not Path(ref).exists():missing.append(ref)
assert not missing,missing
for name in ['comparison.png','comparison.pdf','RESULT_TABLES_KO.md','RESULTS_KO.md']:
    assert (root/name).exists() and (root/name).stat().st_size>0
checks['local_document_links_valid']=True;checks['passed']=True
(out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
