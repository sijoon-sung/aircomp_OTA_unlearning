"""CPU-only checks and static scientific plots of the completed control study."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'correlation_control_v1'
assert (OUT/'complete.json').exists(), 'Only analyze completed study'
assert not (OUT/'audit.json').exists(), 'Never overwrite audit'
rows=json.loads((OUT/'results.json').read_text())
pairs=json.loads((OUT/'comparisons.json').read_text())
assoc=json.loads((OUT/'association_summary.json').read_text())
assert len(rows)==18 and len(pairs)==9 and len(assoc)==9
checks=[]
for seed in [10701,10702,10703]:
    selected=[r for r in rows if r['seed']==seed]
    configs=[json.loads((OUT/r['case']/'config.json').read_text()) for r in selected]
    reference=OUT/selected[0]['case']
    with np.load(reference/'partition.npz') as z:
        pools={k:z[k].copy() for k in z.files}
    for r,c in zip(selected,configs):
        assert sorted(sum(c['groups'],[]))==list(range(20))
        assert all(len(g)==5 for g in c['groups'])
        assert np.array_equal(np.sort(c['base_h']),np.sort(configs[0]['base_h']))
        with np.load(OUT/r['case']/'partition.npz') as z:
            assert all(np.array_equal(z[k],v) for k,v in pools.items())
        with np.load(OUT/r['case']/'reference_probabilities.npz') as a, np.load(OUT/r['case']/'replay_probabilities.npz') as b:
            assert all(np.array_equal(a[k],b[k]) for k in a.files)
        assert r['source_cost']['local_calls']==4800
        assert r['reference_cost']['local_calls']==4560
        assert r['delete_cost']['local_calls']==960
        assert r['delete_cost']['max_expected_mse']<=1e-4*(1+1e-12)
        assert r['replay_exact'] and r['max_parameter_difference']==0
    random=[c['groups'] for r,c in zip(selected,configs) if r['method']=='random_norm']
    assert all(g==random[0] for g in random)
    for level in [0.,.5,1.]:
        cs=[c for r,c in zip(selected,configs) if r['extra']['level']==level]
        assert np.array_equal(cs[0]['base_h'],cs[1]['base_h'])
    checks.append(dict(seed=seed,partition_matched=True,channel_marginals_matched=True,
                       random_routing_fixed=True,paired_channels_matched=True,probabilities_exact=True))

fig,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
specs=[(assoc,'rho','Achieved data–channel association','Spearman rho'),
       (pairs,'shard_js_delta','Shard representation: channel − random','Mean JS divergence difference'),
       (pairs,'accuracy_delta_pp','Deletion accuracy: channel − random','Percentage points'),
       (pairs,'total_re_saving','Deletion communication saving','Total resource saving (%)')]
colors=['#0072B2','#D55E00','#009E73']
for ax,(data,field,title,ylabel) in zip(axes.flat,specs):
    yy=[]
    for seed,color in zip([10701,10702,10703],colors):
        values=[next(r[field] for r in data if r['seed']==seed and r['level']==l) for l in [0.,.5,1.]]
        values=np.array(values)*(100 if field=='total_re_saving' else 1)
        yy.append(values)
        ax.plot([0,.5,1],values,'o-',color=color,alpha=.65,lw=1.2,label=f'Seed {seed}')
    ax.plot([0,.5,1],np.mean(yy,axis=0),'s-',color='#222222',lw=2.4,label='Mean of 3 seeds')
    ax.axhline(0,color='gray',lw=.8,ls='--'); ax.set_xticks([0,.5,1])
    ax.set(title=title,xlabel='Constructed latent coupling coefficient',ylabel=ylabel)
    ax.grid(alpha=.16); ax.spines[['top','right']].set_visible(False)
axes[0,0].legend(fontsize=8)
fig.suptitle('Synthetic sensitivity study — no measured wireless data\nFashionMNIST · 20 clients · 4 shards · 240 rounds · fixed aggregation MSE',fontsize=13)
fig.savefig(OUT/'control_results.png',dpi=180)
fig.savefig(OUT/'control_results.pdf')
plt.close(fig)
with (OUT/'audit.json').open('x',encoding='utf-8') as f:
    json.dump(dict(passed=True,cases=18,checks=checks,source_calls=86400,reference_calls=82080,replay_calls=17280),f,indent=2)
print(json.dumps(dict(audit_passed=True,plot=str(OUT/'control_results.png'))))
