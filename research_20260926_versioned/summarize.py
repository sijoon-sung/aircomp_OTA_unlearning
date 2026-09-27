"""Aggregate noise within seeds first; never use results to change the experiment."""
from pathlib import Path
import json, statistics, csv
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'


def mean_sd(vals):
    return dict(mean=statistics.mean(vals),sd=statistics.stdev(vals) if len(vals)>1 else 0.,seed_values=vals)


def key_for(family,row):
    parts=[row['method'],row.get('version','baseline')]
    if family=='DS' and 'peak' in row:parts+=['peak' if row['peak'] else 'average-only']
    if family=='RTD' and row.get('extra') and 'repeats' in row['extra']:parts += ['r'+str(row['extra']['repeats'])]
    return ' / '.join(parts)


def run():
    groups={};full=[];meta={}
    for family in ['DS','OG','RTD']:
        prefix='RTDfp64' if family=='RTD' else family
        records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(OUT.glob(f'{prefix}_seed*.json'))]
        assert len(records)==3 and all(r['complete'] for r in records),(family,'incomplete')
        meta[family]=[dict(seed=r['seed'],seconds=r['seconds'],reference=r['reference']) for r in records]
        for rec in records:
            for row in rec['rows']:
                key=key_for(family,row)
                groups.setdefault((family,key),{}).setdefault(rec['seed'],[]).append(row)
                full.append(dict(family=family,seed=rec['seed'],label=key,noise=row.get('noise'),
                    **row['metrics'],total_real_uses=row['cost']['total_real_uses'] if row.get('cost') else None))
            for row in rec.get('external_fedavg',[]):
                key=row['method'];groups.setdefault(('External-FedAvg',key),{}).setdefault(rec['seed'],[]).append(row)
    summary=[]
    for (family,label),seedgroups in groups.items():
        metrics={}
        for metric in sorted(set.intersection(*[set(rows[0]['metrics']) for rows in seedgroups.values()])):
            vals=[statistics.mean(r['metrics'][metric] for r in rows) for _,rows in sorted(seedgroups.items())]
            if all(isinstance(v,(int,float)) for v in vals):metrics[metric]=mean_sd(vals)
        costs=[statistics.mean(r['cost']['total_real_uses'] for r in rows if r.get('cost')) for rows in seedgroups.values() if any(r.get('cost') for r in rows)]
        noise=[len(rows) for rows in seedgroups.values()]
        summary.append(dict(family=family,label=label,metrics=metrics,cost=statistics.mean(costs) if costs else None,
            seeds=len(seedgroups),noise_per_seed=noise))
    result=dict(records=summary,metadata=meta,scope='3 seeds; channel noise averaged within seed first')
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=sorted({k for row in full for k in row})
    with (OUT/'all_rows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(full)
    lines=['**Original / V1 및 비교군 — 전체 수치표**','',
           '각 값은 noise를 seed 내부에서 평균한 뒤 구한3 seed 평균±표본표준편차다. 모든 비용은 real channel-use equivalent다.','']
    for family,title in [('DS','① DS-Air: fixed-feature ridge reference'),('OG','② OG-Air: paired orthogonal FedAvg reference'),
                         ('RTD','③ RTD-Air: independent-teacher KD reference'),('External-FedAvg','외부 비교: plain FedAvg reference')]:
        lines += [f'**{title}**','', '| 방법 / 버전 / 조건 | 삭제 오차 ↓ | Test accuracy % | Forget accuracy % | 전체 real uses |',
                  '|---|---:|---:|---:|---:|']
        for row in summary:
            if row['family']!=family:continue
            m=row['metrics'];metric='parameter_error_ratio' if family=='DS' else 'forget_normalized_js'
            e=m[metric];acc=m['test_accuracy']
            f=m.get('forget_accuracy');ft='—' if f is None else f"{f['mean']*100:.2f} ± {f['sd']*100:.2f}"
            cost='oracle' if row['cost'] is None else f"{row['cost']:,.0f}"
            lines.append(f"| {row['label']} | {e['mean']:.6f} ± {e['sd']:.6f} | {acc['mean']*100:.2f} ± {acc['sd']*100:.2f} | {ft} | {cost} |")
        lines += ['']
    lines += ['DS와 CNN의 오차 정의 및 reference가 다르므로 표 사이의 숫자로 전체 우열을 정하지 않는다.',
              'Original은 우리 기존 구현이다. FedOSD/FedQUIT 표기는 본문에 명시한 adaptation이며 원 논문 전체 benchmark 재현이 아니다.']
    (ROOT/'RESULT_TABLES_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(summary)
    print(json.dumps([dict(family=r['family'],label=r['label'],error=r['metrics'].get('parameter_error_ratio',r['metrics'].get('forget_normalized_js')),cost=r['cost']) for r in summary],ensure_ascii=False,indent=2))


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[('DS',[('DS-Air / Original / peak','Original'),('Original-cost-matched / baseline / peak','Matched'),
                   ('DS-Air / V1 / peak','V1'),('DS-Air-subspace48 / V1 / peak','V1 rank48'),('Noisy stats retrain / baseline / peak','Stats retrain')]),
            ('OG',[('OG-Air / Original','Original'),('Original-mild / ablation','Mild'),('OG-Air / V1','V1'),
                   ('FedOSD-core10 / baseline','FedOSD core'),('Legacy-Sketch-V1 / baseline','Legacy sketch')]),
            ('RTD',[('RTD-Air / Original / r1','Original r1'),('RTD-Air / V1 / r1','V1 r1'),
                    ('RTD-Air / Original / r4','Original r4'),('RTD-Air / V1 / r4','V1 r4'),('Digital8 / baseline / r1','Digital8')])]
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
    for ax,(family,labels) in zip(axes,panels):
        vals=[];errs=[];names=[];colors=[]
        for key,name in labels:
            r=next(r for r in summary if r['family']==family and r['label']==key)
            m=r['metrics'].get('parameter_error_ratio',r['metrics'].get('forget_normalized_js'))
            vals.append(m['mean']);errs.append(m['sd']);names.append(name);colors.append('#16857b' if name.startswith('V1') else '#8996a8')
        bars=ax.bar(names,vals,color=colors,yerr=errs,capsize=3,error_kw={'lw':1,'ecolor':'#384454'})
        ax.set_yscale('log');ax.axhline(1,color='#a43b3b',ls='--',lw=1,label='No-op = 1')
        ax.set_ylim(min(v-e for v,e in zip(vals,errs))*.72,max(1.,max(v+e for v,e in zip(vals,errs)))*1.55)
        for bar,val,err in zip(bars,vals,errs):ax.text(bar.get_x()+bar.get_width()/2,(val+err)*1.08,f'{val:.3g}',ha='center',fontsize=9)
        ax.tick_params(axis='x',rotation=25,labelsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        titles={'DS':'DS-Air: retained ridge optimum','OG':'OG-Air: orthogonal FedAvg retrain','RTD':'RTD-Air: independent-teacher KD'}
        ax.set_title(titles[family]);ax.set_ylabel('Parameter error / no-op error (log)' if family=='DS' else 'Prediction JS / no-op JS (log)')
    fig.suptitle('Original vs V1 | 3 fresh seeds (mean +/- SD) | 20 dB | peak-limited real MAC',fontsize=13)
    fig.savefig(ROOT/'comparison.png',dpi=190);fig.savefig(ROOT/'comparison.pdf');plt.close(fig)


if __name__=='__main__':run()
