from pathlib import Path
import json, statistics, collections, math, hashlib
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
allgroups=collections.defaultdict(lambda:collections.defaultdict(list))
raw=[]
cost_audit={r['seed']:r for r in json.loads((OUT/'cost_audit.json').read_text())} if (OUT/'cost_audit.json').exists() else {}
for stage in [1,2,3]:
    for seed in [271,272,273]:
        path=OUT/f'method{stage}_seed{seed}.json'
        obj=json.loads(path.read_text(encoding='utf-8'));assert obj['complete'],path
        raw.append((stage,obj))
        for row in obj['rows']:
            family=row.get('family','default');key=(stage,family,row['method'])
            vals=dict(row['metrics'])
            if row['method'].startswith('no_op'):vals['forget_normalized_js']=1.
            if 'cost' in row:vals['total_real_uses']=row['cost']['total_real_uses']
            if row['method']=='no_op_ensemble_student' and seed in cost_audit:
                vals['total_real_uses']=cost_audit[seed]['independent_teacher_preparation_corrected']['total_real_uses']
            for k in ['wall_seconds','mask_wrong_count','selection_regret','score_mse','forget_js_to_noiseless_prune','query_target_mse']:
                if k in row:vals[k]=row[k]
            allgroups[key][seed].append(vals)
summary=[]
for (stage,family,name),seeds in allgroups.items():
    perseed={s:{k:statistics.mean(v[k] for v in vals if k in v) for k in set().union(*(v.keys() for v in vals))} for s,vals in seeds.items()}
    metrics={}
    for k in set().union(*(v.keys() for v in perseed.values())):
        vs=[v[k] for v in perseed.values() if k in v]
        metrics[k]={'mean':statistics.mean(vs),'sd':statistics.stdev(vs) if len(vs)>1 else None,'n_seeds':len(vs)}
    summary.append({'stage':stage,'family':family,'method':name,'metrics':metrics,'per_seed':perseed})
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

lines=['**GPU 순차 실험의 원자료 요약**','','Channel noise 반복은 먼저 seed 내부에서 평균하고, 아래 mean±sd는 model/data seed 간 표본표준편차다. 단일 reference-variation 행은 seed271만이다. Ratio=forget JS / no-op forget JS; 낮을수록 paired reference에 가깝다.','']
for stage in [1,2,3]:
    lines.extend([f'**방법 {stage}**','','| 방식 | Reference | Forget JS 비율 | Test accuracy % | Forget accuracy % | MIA AUC | 삭제 통신 M real uses |','|---|---|---:|---:|---:|---:|---:|'])
    for row in summary:
        if row['stage']!=stage:continue
        def fmt(k,scale=1.,digits=3):
            m=row['metrics'].get(k)
            if m is None:return '—'
            v=m['mean']*scale;s=m['sd']
            return f'{v:.{digits}f}'+(f' ± {s*scale:.{digits}f}' if s is not None else '')
        ref='독립 teacher KD' if row['family']=='independent_teachers' else ('ortho FL' if stage==2 and row['method']!='no_ortho_noiseless_prune' else 'plain FL')
        lines.append(f"| {row['method']} | {ref} | {fmt('forget_normalized_js')} | {fmt('test_accuracy',100,2)} | {fmt('forget_accuracy',100,2)} | {fmt('loss_mia_auc')} | {fmt('total_real_uses',1e-6,6)} |")
    lines.append('')
lines.extend(['**방법2: 통신오류만의 비교**','','| 방식 | 오선택 channel 수 | Score MSE | 무잡음 pruning 대비 forget JS |','|---|---:|---:|---:|'])
for row in summary:
    if row['stage']==2 and row['method'].startswith('ota'):
        m=row['metrics'];lines.append(f"| {row['method']} | {m['mask_wrong_count']['mean']:.3f} | {m['score_mse']['mean']:.6f} | {m['forget_js_to_noiseless_prune']['mean']:.6f} |")
lines.extend(['','독립 teacher의 no-op ensemble student 행은 삭제비용이 아니라 최초 준비비용이며, cost_audit.json에서 보완한 model DL·초기 seed 비용을 사용했다. Local teacher/student 계산 및 public image 사전배포 비용은 별도 장부에 있다. Plain/ortho source 준비비용도 각 training JSON에 기록되어 있다.',''])
(ROOT/'RESULT_TABLES_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

diagnostic=[]
for stage,obj in raw:
    if stage==1:
        diagnostic.append({'stage':1,'seed':obj['seed'],'source_val':obj['training']['validation_accuracy'],
            'source_test':obj['rows'][0]['metrics']['test_accuracy'],'noop_forget_js':obj['rows'][0]['metrics']['forget_js'],
            'retrain_test':obj['rows'][0]['metrics']['test_reference_accuracy']})
    if stage==2:diagnostic.append({'stage':2,'seed':obj['seed'],'ortho':obj['diagnostic'],
        'plain':next(r['diagnostic'] for r in obj['rows'] if r['method']=='no_ortho_noiseless_prune')})
    if stage==3:
        diagnostic.append({'stage':3,'seed':obj['seed'],'replay':obj['fresh_replay'],
            'teacher_gradient_samples':sum(t['gradient_samples'] for t in obj['teacher_preparation']),
            'teacher_wall_seconds':sum(t['wall_seconds'] for t in obj['teacher_preparation']),
            'structural_vs_fedavg':obj['structural_reference_vs_fedavg_retrain']})
(OUT/'diagnostics.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf-8')
print('\n'.join(lines))
print('DIAGNOSTICS',json.dumps(diagnostic,ensure_ascii=False))
