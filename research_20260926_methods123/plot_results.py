from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
def get(stage,name):return next(x for x in summary if x['stage']==stage and x['method']==name)
fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
cases=[(1,['curenus_exact','curenus_ota20_r1','curenus_ota20_r4'],['Noiseless','20dB\n1 repeat','20dB\n4 repeats'],'1. HVP / cubic correction','Reference: plain FL retraining'),
       (2,['ortho_noiseless_prune','ota20_boundary','ota20_matched'],['Noiseless','Boundary','Uniform\nmatched cost'],'2. Prepared orthogonal pruning','Reference: orthogonal FL retraining'),
       (3,['exclude_fresh_ota20_r1','exclude_fresh_ota20_r4'],['20dB\n1 repeat','20dB\n4 repeats'],'3. Independent teachers + fresh KD','Reference: target-free teacher-KD algorithm')]
for ax,(stage,names,labels,title,reference) in zip(axes,cases):
    metric=[get(stage,n)['metrics']['forget_normalized_js'] for n in names]
    means=[m['mean'] for m in metric];sd=[m['sd'] for m in metric]
    ax.bar(range(len(names)),means,yerr=sd,capsize=4,color=['#586F8C','#007F72','#55A59A'][:len(names)])
    ax.axhline(1,color='#AA5544',linestyle='--',label='No-op = 1')
    ax.set_yscale('log');ax.set_ylim(.008,90);ax.set_xticks(range(len(names)),labels)
    ax.set_title(title,fontsize=12);ax.set_xlabel(reference,fontsize=9)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.spines[['right','top']].set_visible(False)
    for i,m in enumerate(means):ax.text(i,(m+sd[i])*1.15,f'{m:.2f}',ha='center',fontsize=10)
axes[0].set_ylabel('Forget prediction JS / no-op JS (lower is closer)')
axes[0].legend(loc='lower left',frameon=False,fontsize=9)
fig.suptitle('Three limited GPU pilots: the reference algorithm matters',fontsize=16)
fig.supxlabel('3 model/data seeds; mean ± sample SD | FashionMNIST, 10 clients | No recovery tuning',fontsize=10)
fig.savefig(ROOT/'results_comparison.png',dpi=180)
fig.savefig(ROOT/'results_comparison.pdf')
plt.close(fig)
print('Saved result figure.')
