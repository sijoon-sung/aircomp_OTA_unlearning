from pathlib import Path
import hashlib,json,time
import torch
R=Path(__file__).resolve().parent;OUT=R.parent/'aircomp_mechanism_summary_20261004'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
start=time.perf_counter();files=0;vectors=0
for p in R.glob('*_states.pt'):
    obj=read(p.with_name(p.name.replace('_states.pt','.json')))
    states=torch.load(p,map_location='cpu',weights_only=True)
    for name,v in states.items():
        assert v.numel()==38282 and bool(torch.isfinite(v).all())
        sha=hashlib.sha256(v.numpy().tobytes()).hexdigest();assert sha==obj['models'][name]
        vectors+=1
    files+=1
assert files==18*24
for s in [64001,64002,64003]:
    data=read(R/f'{s}_data.json');flat=[i for p in data['pools'] for i in p]
    assert len(flat)==len(set(flat))==12000
    assert len(set(data['public_dev_original']))==2000 and not set(flat)&set(data['public_dev_original'])
    assert data['original_indices']==flat+data['public_dev_original']
    route=read(R/f'{s}_routing.json')
    for gs in route['groups'].values():
        assert sorted(i for g in gs for i in g)==list(range(20)) and all(len(g)==5 for g in gs)
result=dict(passed=True,state_files=files,model_vectors=vectors,public_dev_disjoint_seeds=3,seconds=time.perf_counter()-start)
with (OUT/'state_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
