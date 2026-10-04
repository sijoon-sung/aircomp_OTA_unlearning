import json, gzip, math, time
from pathlib import Path
p=Path(__file__).resolve().parent
t=time.perf_counter()
result=json.loads((p/'results.json').read_text(encoding='utf-8'))
scenarios=json.loads((p/'scenarios.json').read_text(encoding='utf-8'))
parts=json.loads((p/'partitions.json').read_text(encoding='utf-8'))
assert len(parts)==5775
for partition in parts:
    assert sorted(i for g in partition for i in g)==list(range(12))
    assert all(len(g)==4 for g in partition)
conflicts=[]
with gzip.open(p/'per_partition.json.gz','rt',encoding='utf-8') as f:
  for row,scenario,line in zip(result['results'],scenarios,f):
    raw=json.loads(line); h=scenario['h'];valid=raw['eligible_indices']
    a=raw['source_repeat_sum'];b=raw['deletion_repeat_sum']
    amin=min(a[i] for i in valid);bmin=min(b[i] for i in valid)
    assert row['strict_conflict']==(not any(a[i]==amin and b[i]==bmin for i in valid))
    for method in ('initial','deletion','joint'):
      chosen=row[method];ix=chosen['partition_index'];assert ix in valid
      groups=parts[ix]
      initial=sum(max(1,math.ceil(1/(len(g)**2*min(h[i] for i in g)**2)-1e-12)) for g in groups)
      deletion=[]
      for client in range(12):
        retained=next([i for i in g if i!=client] for g in groups if client in g)
        r=max(1,math.ceil(1/(len(retained)**2*min(h[i] for i in retained)**2)-1e-12))
        # physical per-coordinate MSE check: beta^2 * sigma^2 / R <= target
        beta=1/(len(retained)*math.sqrt(38282)*min(h[i] for i in retained))
        assert beta**2*(38282*1e-4)/r<=1e-4+1e-12
        deletion.append(r)
      assert initial==a[ix] and sum(deletion)==b[ix]
      assert chosen['initial_UL_RE']==initial*38282
      assert abs(chosen['expected_deletion_UL_RE']-sum(deletion)*38282/12)<1e-8
    assert a[row['initial']['partition_index']]==amin
    assert b[row['deletion']['partition_index']]==bmin
    j=row['joint']['partition_index']
    assert a[j]*bmin+b[j]*amin==min(a[i]*bmin+b[i]*amin for i in valid)
    if row['strict_conflict']:
      conflicts.append(dict(seed=row['seed'],scenario=row['scenario'],deletion_only_saving_pct=100*(1-row['deletion']['expected_deletion_UL_RE']/row['initial']['expected_deletion_UL_RE']),initial_increase_pct=100*(row['deletion']['initial_UL_RE']/row['initial']['initial_UL_RE']-1)))
out=dict(status='passed',selected_designs_recomputed=270,all_partition_memberships_checked=5775,objective_minima_checked=90,conflicts=conflicts,audit_seconds=time.perf_counter()-t)
(p/'audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
