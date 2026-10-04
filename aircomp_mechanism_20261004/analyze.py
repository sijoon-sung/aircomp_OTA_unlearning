from pathlib import Path
import json,time,itertools,math,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from select_partition import select
R=Path(__file__).resolve().parent;OUT=R.parent/'aircomp_mechanism_summary_20261004';OUT.mkdir(exist_ok=True)
SEEDS=[64001,64002,64003];METHODS=['random','channel','data'];LEVELS=['no_noise','nominal20dB'];TARGETS=['q65','q60']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def textfile(name,s):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(s)
def mean(xs):return float(np.mean(xs)) if len(xs) else None
def rank(a):
    a=np.array(a);o=np.argsort(a,kind='stable');r=np.empty(len(a),float)
    for v in np.unique(a):r[a==v]=np.mean(np.flatnonzero(a[o]==v))+1
    return r
def spearman(a,b):
    if len(a)<3 or len(set(a))<2 or len(set(b))<2:return None
    return float(np.corrcoef(rank(a),rank(b))[0,1])
start=time.perf_counter();rows=read(R/'results.json');resource=read(R/'resource.json');physical=read(R/'physical_audit.json')
assert len(rows)==18 and resource['completed_conditions']==18 and not resource['failed']
attribution={'wall_seconds':resource['seconds'],'paused_seconds':0.,'active_wall_seconds':resource['seconds'],'exclusive_observed_gpu_board_Wh':resource['gpu_board_Wh'],'total_observed_gpu_board_Wh':resource['gpu_board_Wh'],'shared_gpu_interval':False}
if (R/'pause_event.json').exists():
    pause=read(R/'pause_event.json');resume=read(R/'resume_event.json')
    pt=datetime.datetime.fromisoformat(pause['time']).timestamp();rt=datetime.datetime.fromisoformat(resume['time']).timestamp()
    shared_begin=datetime.datetime(2026,10,4,3,51,6).timestamp()-5;shared_end=rt+5
    exclusive=sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(resource['samples'],resource['samples'][1:]) if b[0]<shared_begin or a[0]>shared_end)/3600
    attribution.update(paused_seconds=rt-pt,active_wall_seconds=resource['seconds']-(rt-pt),exclusive_observed_gpu_board_Wh=exclusive,shared_gpu_interval=True,excluded_begin=shared_begin,excluded_end=shared_end,note='shared interval and suspension excluded; full experiment-only energy unavailable')
dump('resource_attribution.json',attribution)
def get(s,m,l):return next(r for r in rows if (r['seed'],r['method'],r['noise'])==(s,m,l))
summary=[];perseed=[];profiles=[];audit_events=0;maxpower=0.;max_reference=0.;references=0;source_delete=[];ensemble_effects=[]
for row in rows:
    prefix=f"{row['seed']}_{row['method']}_{row['noise']}"
    source_objects=[read(R/(prefix+f'_source_{c}.json')) for c in range(4)]
    for q in ['fixed160',*TARGETS]:
        indiv=mean([o['test'][q]['accuracy'] for o in source_objects]);ens=row['source_test'][q]['accuracy']
        ensemble_effects.append(dict(seed=row['seed'],method=row['method'],noise=row['noise'],target=q,mean_individual_test=indiv,ensemble_test=ens,ensemble_gain_pp=100*(ens-indiv)))
    for tag in [*(f'source_{c}' for c in range(4)),*(f'delete_{u}' for u in range(20))]:
        obj=read(R/(prefix+'_'+tag+'.json'));assert len(obj['events'])==200 and len(obj['curve'])==10
        n=len(obj['ids']);assert n==(5 if tag.startswith('source') else 4)
        if tag.startswith('delete'):assert int(tag.split('_')[1]) not in obj['ids']
        for e in obj['events']:
            assert e['n']==n and e['ids']==obj['ids'] and e['repeats']==1 and e['max_power']<=1.00001
            assert abs(e['expected_mse']-row['sigma2']/(n*n*e['min_h']**2))<1e-10
            assert e['ul_reals']==38282 and e['dl_bits']==32*38282 and e['local_calls']==n
            audit_events+=1;maxpower=max(maxpower,e['max_power'])
        for q,target in [('q60',.6),('q65',.65)]:
            stop=next((b['round'] for a,b in zip(obj['curve'],obj['curve'][1:]) if a['accuracy']>=target and b['accuracy']>=target),None)
            assert stop==obj['stops'][q];assert obj['costs'][q]['ul_reals']==38282*(stop or 200)
            integer_stop=next((b['round'] for a,b in zip(obj['curve'],obj['curve'][1:]) if round(a['accuracy']*2000)>=round(target*2000) and round(b['accuracy']*2000)>=round(target*2000)),None)
            assert integer_stop==stop,('float_threshold_boundary',prefix,tag,q,stop,integer_stop)
        if tag.startswith('delete'):
            profiles.append(dict(seed=row['seed'],method=row['method'],noise=row['noise'],client=int(tag.split('_')[1]),**obj['features'],mse=obj['mean_mse'],stops=obj['stops'],shard_acc160=obj['curve'][7]['accuracy']))
            so=read(R/(prefix+f"_source_{obj['shard']}.json"))
            source_delete.append(dict(seed=row['seed'],method=row['method'],noise=row['noise'],client=int(tag.split('_')[1]),shard=obj['shard'],
                source_js=so['features']['js'],delete_js=obj['features']['js'],js_change=obj['features']['js']-so['features']['js'],
                source_dev160=so['curve'][7]['accuracy'],delete_dev160=obj['curve'][7]['accuracy'],dev_change_pp=100*(obj['curve'][7]['accuracy']-so['curve'][7]['accuracy']),
                source_stops=so['stops'],delete_stops=obj['stops'],source_recall160=so['curve'][7]['class_recall'],delete_recall160=obj['curve'][7]['class_recall']))
    for d in row['deletions']:
        assert d['unaffected_prediction_hashes_unchanged']
        if d['reference']:
            references+=1;max_reference=max(max_reference,max(d['reference']['errors'].values()))
    for q in TARGETS:
        ds=row['deletions'];ok=[d for d in ds if d['stops'][q] is not None]
        perseed.append(dict(seed=row['seed'],method=row['method'],noise=row['noise'],target=q,
            source_complete=row['source_success'][q],source_rounds=row['source_stops'][q],hits=len(ok),total=len(ds),
            mean_rounds_success_only=mean([d['stops'][q] for d in ok]),mean_ul_all_complete=mean([d['costs'][q]['ul_reals'] for d in ds]) if len(ok)==20 else None,
            worst_rounds_all_complete=max(d['stops'][q] for d in ds) if len(ok)==20 else None,
            consumed_ul_including_caps=mean([d['costs'][q]['ul_reals'] for d in ds]),
            source_test=row['source_test'][q]['accuracy'],delete_test=mean([d['test'][q]['accuracy'] for d in ds]),
            delete_fixed160=mean([d['test']['fixed160']['accuracy'] for d in ds]),worst_recall=mean([d['test'][q]['worst_recall'] for d in ds])))
for m,l,q in itertools.product(METHODS,LEVELS,TARGETS):
    rr=[r for r in perseed if (r['method'],r['noise'],r['target'])==(m,l,q)];ds=[d for s in SEEDS for d in get(s,m,l)['deletions']];ok=[d for d in ds if d['stops'][q] is not None]
    summary.append(dict(method=m,noise=l,target=q,hits=len(ok),total=60,source_complete_seeds=sum(r['source_complete'] for r in rr),
        mean_rounds_success_only=mean([d['stops'][q] for d in ok]),mean_ul_all_complete=mean([d['costs'][q]['ul_reals'] for d in ds]) if len(ok)==60 else None,
        mean_rounds_all_complete=mean([d['stops'][q] for d in ds]) if len(ok)==60 else None,
        delete_test=mean([r['delete_test'] for r in rr]),source_test=mean([r['source_test'] for r in rr]),delete_fixed160=mean([r['delete_fixed160'] for r in rr]),
        worst_recall=mean([r['worst_recall'] for r in rr]),mean_mse=mean([d['mean_mse'] for d in ds]),
        mean_delete_js=mean([d['features']['js'] for d in ds])))
data_gaps=[100*(mean([d['test']['fixed160']['accuracy'] for d in get(s,'data','no_noise')['deletions']])-mean([d['test']['fixed160']['accuracy'] for d in get(s,'channel','no_noise')['deletions']])) for s in SEEDS]
hdata=dict(seed_gaps_pp=data_gaps,mean_gap_pp=mean(data_gaps),supported=mean(data_gaps)>=1 and all(g>0 for g in data_gaps))
noise_effects=[]
for m,q in itertools.product(METHODS,TARGETS):
    paired=[];seedchanges=[];accuracy=[];transitions={'both_success':0,'noise_only_failure':0,'noise_only_success':0,'both_failure':0}
    for s in SEEDS:
        a=get(s,m,'no_noise');b=get(s,m,'nominal20dB');local=[]
        for da,db in zip(a['deletions'],b['deletions']):
            ta=da['stops'][q];tb=db['stops'][q]
            key='both_success' if ta and tb else 'noise_only_failure' if ta else 'noise_only_success' if tb else 'both_failure';transitions[key]+=1
            if ta and tb:paired.append((ta,tb));local.append(tb-ta)
        seedchanges.append(mean(local));accuracy.append(100*mean([db['test']['fixed160']['accuracy']-da['test']['fixed160']['accuracy'] for da,db in zip(a['deletions'],b['deletions'])]))
    ratio=sum(b for a,b in paired)/sum(a for a,b in paired)-1 if paired else None
    harm=transitions['noise_only_failure']>transitions['noise_only_success'] or (ratio is not None and ratio>=.05 and all(x is not None and x>0 for x in seedchanges))
    noise_effects.append(dict(method=m,target=q,transitions=transitions,paired_cost_change=ratio,seed_mean_round_change=seedchanges,seed_acc_effect_pp=accuracy,mean_acc_effect_pp=mean(accuracy),cost_harm_supported=bool(harm),accuracy_harm_supported=bool(mean(accuracy)<=-.5 and all(x<0 for x in accuracy))))
cost_comparisons=[]
for q in TARGETS:
    for a,b in itertools.permutations(METHODS,2):
        aa=[r for r in perseed if r['method']==a and r['noise']=='nominal20dB' and r['target']==q];bb=[r for r in perseed if r['method']==b and r['noise']=='nominal20dB' and r['target']==q]
        feasible=all(r['source_complete'] and r['hits']==20 for r in aa+bb)
        saving=1-mean([r['mean_ul_all_complete'] for r in aa])/mean([r['mean_ul_all_complete'] for r in bb]) if feasible else None
        gaps=[100*(ra['delete_test']-rb['delete_test']) for ra,rb in zip(aa,bb)]
        supported=feasible and saving>=.05 and mean(gaps)>=-1 and min(gaps)>=-2
        cost_comparisons.append(dict(target=q,method=a,baseline=b,both_feasible=feasible,saving=saving,seed_test_gap_pp=gaps,supported=bool(supported)))
correlations=[]
for q,l in itertools.product(TARGETS,LEVELS):
    pp=[p for p in profiles if p['noise']==l];ok=[p for p in pp if p['stops'][q] is not None]
    correlations.append(dict(target=q,noise=l,n=len(pp),completed=len(ok),
        js_vs_fixed160_acc=spearman([p['js'] for p in pp],[p['shard_acc160'] for p in pp]),
        q_vs_fixed160_acc=spearman([p['q'] for p in pp],[p['shard_acc160'] for p in pp]),
        js_vs_completed_rounds=spearman([p['js'] for p in ok],[p['stops'][q] for p in ok]),
        q_vs_completed_rounds=spearman([p['q'] for p in ok],[p['stops'][q] for p in ok]),
        caveat='pooled correlated observations; descriptive, no causal claim; rounds correlations condition on completion'))
noise_pairs=[]
for p in profiles:
    if p['noise']!='nominal20dB':continue
    base=next(a for a in profiles if (a['seed'],a['method'],a['client'],a['noise'])==(p['seed'],p['method'],p['client'],'no_noise'))
    noise_pairs.append(dict(seed=p['seed'],method=p['method'],client=p['client'],mse=p['mse'],dev_effect_pp=100*(p['shard_acc160']-base['shard_acc160'])))
selection=[select(rows,s,q) for q in TARGETS for s in SEEDS]
posthoc=[]
for l in LEVELS:
    pp=[p for p in source_delete if p['noise']==l]
    worst=min(pp,key=lambda p:p['dev_change_pp'])
    rt=read(R/f"{worst['seed']}_routing.json");gs=rt['groups'][worst['method']];g=gs[worst['shard']];h=np.array(rt['histograms']);left=[u for u in g if u!=worst['client']]
    posthoc.append(dict(noise=l,label='post-hoc descriptive analysis, not preregistered confirmatory hypothesis',
        js_change_vs_dev_change=spearman([p['js_change'] for p in pp],[p['dev_change_pp'] for p in pp]),
        source_success_delete_failure={q:sum(p['source_stops'][q] is not None and p['delete_stops'][q] is None for p in pp) for q in TARGETS},
        worst_dev_drop_example={**worst,'groups':gs,'source_hist':h[g].mean(0).tolist(),'remaining_hist':h[left].mean(0).tolist(),'deleted_hist':h[worst['client']].tolist()}))
audit=dict(passed=True,conditions=18,deletions=360,events=audit_events,reference_replays=references,max_reference_error=max_reference,max_power=maxpower,physical=physical['passed'],seconds=time.perf_counter()-start)
analysis=dict(summary=summary,per_seed=perseed,Hdata=hdata,noise_effects=noise_effects,cost_comparisons=cost_comparisons,correlations=correlations,selection=selection,posthoc_source_delete=posthoc,posthoc_mse_vs_noise_accuracy_effect=spearman([p['mse'] for p in noise_pairs],[p['dev_effect_pp'] for p in noise_pairs]),posthoc_ensemble_effects=ensemble_effects)
dump('analysis.json',analysis);dump('audit.json',audit)
for r in selection:dump(f"{r['seed']}_{r['target']}_selection.json",r)
colors={'random':'#6b7280','channel':'#2563eb','data':'#059669'}
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for j,m in enumerate(METHODS):
    r0=next(r for r in summary if r['method']==m and r['noise']=='no_noise' and r['target']=='q65');r1=next(r for r in summary if r['method']==m and r['noise']=='nominal20dB' and r['target']=='q65')
    axes[0].plot([0,1],[100*r0['delete_fixed160'],100*r1['delete_fixed160']],'-o',label=m,color=colors[m])
    for k,q in enumerate(TARGETS):
        rr=next(r for r in summary if r['method']==m and r['noise']=='nominal20dB' and r['target']==q)
        axes[1].bar(j+(k-.5)*.34,rr['hits']/60*100,width=.32,color=colors[m],alpha=1 if k==0 else .4)
        axes[1].text(j+(k-.5)*.34,rr['hits']/60*100+1,str(rr['hits'])+'/60',ha='center',fontsize=8,rotation=90)
        vals=[d['stops'][q] for s in SEEDS for d in get(s,m,'nominal20dB')['deletions'] if d['stops'][q] is not None]
        if vals:axes[2].scatter([j+(k-.5)*.25], [np.mean(vals)],s=90,marker='o' if k==0 else '^',color=colors[m])
axes[0].set_xticks([0,1],['No noise','Nominal 20 dB']);axes[0].set_ylabel('Post-delete ensemble accuracy (%)');axes[0].set_title('Equal size, fixed 160 rounds');axes[0].legend()
axes[1].set_xticks(range(3),METHODS);axes[1].set_ylim(0,135);axes[1].set_ylabel('Deletion completion (%)');axes[1].set_title('Dark: 65% target; light: 60% target')
axes[2].set_xticks(range(3),METHODS);axes[2].set_ylabel('Rounds, completed requests only');axes[2].set_title('Circle: 65%; triangle: 60% (selection bias)')
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.savefig(OUT/'mechanisms.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
for m in METHODS:
    pp=[p for p in profiles if p['method']==m and p['noise']=='no_noise']
    nn=[p for p in noise_pairs if p['method']==m]
    axes[0].scatter([p['js'] for p in pp],[100*p['shard_acc160'] for p in pp],s=24,alpha=.6,label=m,color=colors[m])
    axes[1].scatter([p['mse'] for p in nn],[p['dev_effect_pp'] for p in nn],s=24,alpha=.6,label=m,color=colors[m])
axes[0].set_xlabel('Remaining-group class JS (after deleting one client)');axes[0].set_ylabel('Shard dev accuracy at 160 rounds (%)');axes[0].set_title('Grouping path: no receiver noise')
axes[1].set_xlabel('Expected aggregate noise MSE');axes[1].set_ylabel('Same-group noise effect on dev accuracy (pp)');axes[1].set_title('Noise path: only variance changes');axes[1].axhline(0,color='gray',lw=1)
for ax in axes:ax.grid(alpha=.2);ax.legend()
fig.savefig(OUT/'causal_paths.png',dpi=170);plt.close(fig)
def table(head,rs):return '<table><thead><tr>'+''.join('<th>'+str(h)+'</th>' for h in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in r)+'</tr>' for r in rs)+'</tbody></table>'
def fmt(v,d=2):return '—' if v is None else f'{v:.{d}f}'
def link(name):return '../aircomp_mechanism_20261004/'+name
main=table(['배정','잡음','65% 완료','60% 완료','고정160 삭제정확도','삭제 MSE','삭제 JS'],[[m,l,str(next(r['hits'] for r in summary if (r['method'],r['noise'],r['target'])==(m,l,'q65')))+'/60',str(next(r['hits'] for r in summary if (r['method'],r['noise'],r['target'])==(m,l,'q60')))+'/60',fmt(100*next(r['delete_fixed160'] for r in summary if (r['method'],r['noise'],r['target'])==(m,l,'q65')))+'%',fmt(next(r['mean_mse'] for r in summary if (r['method'],r['noise'],r['target'])==(m,l,'q65')),5),fmt(next(r['mean_delete_js'] for r in summary if (r['method'],r['noise'],r['target'])==(m,l,'q65')),4)] for m,l in itertools.product(METHODS,LEVELS)])
costtable=table(['목표','배정','완료/60','source 전체완료 seed','전체 완료 평균UL','성공사례만 평균round','삭제 ensemble test (실패는cap모델 포함)'],[[r['target'],r['method'],str(r['hits'])+'/60',str(r['source_complete_seeds'])+'/3',fmt(r['mean_ul_all_complete'],0),fmt(r['mean_rounds_success_only']),fmt(100*r['delete_test'])+'%'] for r in summary if r['noise']=='nominal20dB'])
noisetable=table(['배정/목표','양쪽완료','잡음추가후 실패','잡음추가후 성공','짝지은 완료비용 변화','고정160 정확도 변화','비용악화 지지'],[[r['method']+'/'+r['target'],r['transitions']['both_success'],r['transitions']['noise_only_failure'],r['transitions']['noise_only_success'],fmt(None if r['paired_cost_change'] is None else 100*r['paired_cost_change'])+'%',fmt(r['mean_acc_effect_pp'])+'%p',r['cost_harm_supported']] for r in noise_effects])
selecttable=table(['목표','seed','실제 비용기준 선택','평균 삭제 UL','후보검사 UL합계'],[[r['target'],r['seed'],r['winner']['method'] if r['winner'] else '가능 후보 없음',fmt(r['winner']['mean_delete_ul'] if r['winner'] else None,0),fmt(r['total_profile_consumed_ul'],0)] for r in selection])
seedtable=table(['seed','배정','잡음','목표','source완료','삭제완료','완료시평균UL','source test','삭제 test'],[[r['seed'],r['method'],r['noise'],r['target'],r['source_complete'],str(r['hits'])+'/20',fmt(r['mean_ul_all_complete'],0),fmt(100*r['source_test']),fmt(100*r['delete_test'])] for r in perseed])
corrtable=table(['목표/잡음','완료','JS↔고정160정확도','q↔고정160정확도','JS↔완료round','q↔완료round'],[[r['target']+'/'+r['noise'],str(r['completed'])+'/'+str(r['n']),fmt(r['js_vs_fixed160_acc']),fmt(r['q_vs_fixed160_acc']),fmt(r['js_vs_completed_rounds']),fmt(r['q_vs_completed_rounds'])] for r in correlations])
posthoctable=table(['잡음','삭제후 JS 변화↔dev 변화','source 성공→삭제실패 (65%)','source 성공→삭제실패 (60%)'],[[p['noise'],fmt(p['js_change_vs_dev_change']),p['source_success_delete_failure']['q65'],p['source_success_delete_failure']['q60']] for p in posthoc])
example=posthoc[0]['worst_dev_drop_example'];example_rows=[]
for m in METHODS:
    d=read(R/f"{example['seed']}_{m}_no_noise_delete_{example['client']}.json");s=read(R/f"{example['seed']}_{m}_no_noise_source_{d['shard']}.json")
    example_rows.append([m,fmt(100*s['curve'][7]['accuracy'])+'%',fmt(100*d['curve'][7]['accuracy'])+'%',fmt(d['features']['js'],4),d['stops']['q60'] or '미완료',d['stops']['q65'] or '미완료'])
exampletable=table(['배정','삭제 전 shard dev160','삭제 후 shard dev160','삭제 후 JS','60% 완료round','65% 완료round'],example_rows)
ensrows=[]
for m in METHODS:
    ee=[r for r in ensemble_effects if (r['method'],r['noise'],r['target'])==(m,'no_noise','fixed160')]
    ensrows.append([m,fmt(100*mean([r['mean_individual_test'] for r in ee]))+'%',fmt(100*mean([r['ensemble_test'] for r in ee]))+'%',fmt(mean([r['ensemble_gain_pp'] for r in ee]))+'%p'])
enstable=table(['무잡음 배정','4개 shard 평균 test160','앙상블 test160','앙상블 이득'],ensrows)
dataword='사전 기준 충족' if hdata['supported'] else '사전 기준 미충족'
hcost=[r for r in cost_comparisons if r['supported']]
costword='; '.join(f"{r['target']} {r['method']} vs {r['baseline']}: {100*r['saving']:.1f}%" for r in hcost) if hcost else '모든 seed·모든 삭제 완료와 품질 조건을 함께 만족하는 5% 통신절감 비교는 없었습니다.'
resource_words=f"벽시계 {resource['seconds']/60:.2f}분, 일시중단 {attribution['paused_seconds']/60:.2f}분, 그 외 실행 {attribution['active_wall_seconds']/60:.2f}분. GPU 독점 관측 구간 {attribution['exclusive_observed_gpu_board_Wh']:.3f}Wh. 다른 작업과 공유한 구간은 이 실험의 전력으로 귀속하지 않으므로 총 단독 에너지 사용량은 미확정."
conclusion=f'동일크기·무잡음에서 data−channel 삭제 정확도 차이 {hdata["mean_gap_pp"]:+.2f}%p ({dataword}). {costword}'
algorithm='''가설: 같은 학습 품질을 충족하는 후보 중 실제 삭제비용 최소 배정을 고를 수 있다 | 독립변수: 균등 random/channel/data 후보 | 종속변수: 완료율·평균/최악삭제UL | 통제: 같은 N/K/크기/학습완료규칙 | 상태: 기록된 후보에 대한 사후 선택기 구현·실행, 새로운 온라인 알고리즘 성능 미검증

알고리즘 이름: 품질 제약 아래 실측 삭제비용으로 선택하는 고정 AirComp shard 배정 v0
목적: 데이터 대표성+MSE를 임의 가중합하여 비용이라고 부르지 않고, 같은 학습완료 규칙으로 직접 측정한 삭제통신비를 최소화한다.
입력: client IDs, 허용된 class histogram과 데이터 수, long-term channel amplitude, 전력/라운드 자원 예산, 공개 검증셋, K, 품질목표 a, cap L.
이번 구현 범위: N20,K4,각5명, source와삭제 둘다20round마다검증하고 연속2회 a이상에서완료, L200. a=.65(primary), .60(별도민감도).

1. 후보 생성: 동일크기 random, channel-q 최소, class-JS 최소의 3개 배정. channel/data는 같은 두 시작점과 pair swap 최적화, 최대15회. 전력/크기제약을 고정한다.
2. source 검증: 후보별 독립 shard를 동일 a/L로 학습한다. 모든 shard가 완료했는지 기록한다.
3. 삭제 프로파일링: 각 client를 한명씩 제외하고 해당 shard만 initialization에서 동일규칙으로 재학습한다. T_u 및 UL=D*T_u, DL, 연산량을 별도로 기록한다. source 다른 shard는 바꾸지 않는다.
4. 제약검사: source 모든 shard 성공 AND 모든 client 삭제 성공인 후보만 선택 가능. 미완료비용은 infinity/null이며 cap소모량을 완료비용으로 대체하지 않는다.
5. 선택: 평균_u UL삭제가 가장 작은 후보, 동률이면 최악삭제UL, 그다음 sourceUL, 그다음 이름순. 테스트 정확도는 선택 입력이 아니다.
6. 가능한 후보가 없으면 no_feasible_candidate 반환. 목표를 자동으로 낮추지 않고 낮은품질목표/더큰cap/더작은K 등을 별도설정으로 재검토한다. 이번실험에서 미실행인 대안을 결과로 표현하지 않는다.
7. 배포 학습전에 선택한 배정을 고정하고 shard별 모델/초기화를 서버에 저장한다. 요청은 해당 shard만 재학습. 본 프로파일링이 사용한 batch/noise조건 밖의 미래비용 예측은 검증되지 않았다.

얻는 것: 실제 삭제 후 집합과 실제 완료비용이 선택에 들어가며 삭제확률 예측이 필요없다. class대표성/집계MSE를 곧바로정확도/비용이라고 가정하지 않는다.
지불하는 것: 3후보의 source + 모든client 삭제를 먼저 실행하는 높은준비비용. select결과의 total_profile_consumed_ul에 이론적조기중단 기준 준비UL이 포함된다. 실제연구계측은 모든궤적200까지학습하므로 더비싸다. 통신절감순이익/미래다중요청의상각효과를 증명하지 않았다.
저장: D=38282, FP32. shard4개 initialization+현재모델 두세트 최대 1,225,024byte(약1.17MiB), 모두같은초기화는공유가능. SGD momentum없음. optimizer/RNG/metadata도일반적으로필요. client원본데이터 저장은별도. 임의후기checkpoint는삭제client의영향을포함하므로 전체client삭제시안전rollback은초기화.
노출: class10bins와count/channel 값 7680bit 초기보고. AirComp합관측만으로개별정보보호/DP/암호학적secureagg를증명하지않음.
보장: 고정routing/public-dev/학습규칙 조건부 affected shard 모델재현. 데이터의존routing과메타데이터자체삭제를보장하지않는다.
연구에서 남은 개선: 실측비용을 예측할 값싼 모델을 학습하고 새로운 data/channel/noise seed에서 검증한뒤, 후보평가를줄이는 move/swap 탐색으로확장. 새로운surrogate는실제cost예측력을확인하기전채택하지않는다. 현재selector 자체의문헌최초성은주장하지않는다.
'''
textfile('알고리즘_v0.txt',algorithm)
summarytext=f'''가설: 동일크기에서 데이터배정과잡음은삭제학습에다른경로로영향 | 독립변수: random/channel/data×수신분산0/.01 | 종속변수: 고정160정확도·완료율·삭제UL | 통제: N20/K4/각5명,seed3,동일SGD/전력/R1 | 판정/상태: 18조건360삭제 실행완료; {conclusion}

사전등록: ../aircomp_mechanism_20261004/PREREG.txt
주요분석: analysis.json
원자료: ../aircomp_mechanism_20261004/results.json 및 조건별source/delete JSON/PT
알고리즘: 알고리즘_v0.txt 및 ../aircomp_mechanism_20261004/select_partition.py
실행비용: {resource_words} local calls {resource['local_calls']}, aggregation rounds {resource['aggregation_rounds']}, 외부유료0. GPU보드전체샘플, 호스트전력/실제RF아님.
검증: 물리9조건통과, events {audit_events}, reference {references} 최대오차{max_reference}, 완료시점/전력/삭제client배제/비용검산통과.
한계: seed3, 1dataset/CNN, perfectCSI/합성독립flat채널, 고정균등크기. 실제학습의크기효과·privacy·MIMO·새알고리즘미래성능은미검증. 임계값에못도달한사례를숨기지않는다.
'''
textfile('한눈에_결과.txt',summarytext)
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AirComp shard: 원인 분리와 배정 알고리즘</title><style>body{{font-family:Segoe UI,Malgun Gothic,sans-serif;max-width:1220px;margin:32px auto;padding:0 24px;line-height:1.75;color:#203047}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}.box{{padding:18px;background:#edf5ff;border-left:5px solid #2563eb}}.note{{padding:16px;background:#fff4dd}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:18px 0}}td,th{{padding:8px;border:1px solid #d8e0e9;text-align:left}}th{{background:#f1f5f9}}img{{width:100%}}a{{color:#175cd3}}code{{background:#eef2f6;padding:3px}}.flow{{display:flex;gap:12px;flex-wrap:wrap}}.flow span{{padding:12px;background:#ecfdf5;border:1px solid #a7d4c3}}pre{{white-space:pre-wrap;font-family:inherit}}details{{margin:20px 0}}</style>
<p class="box"><b>가설</b> 동일크기에서 데이터배정과잡음은삭제학습에다른경로로영향 | <b>독립변수</b> 3배정×수신분산0/.01 | <b>종속변수</b> 정확도·완료율·삭제UL | <b>통제</b> N20/K4/각5명,같은SGD·전력 | <b>판정/상태</b> 18조건360삭제 완료. 사전기준 상세는 아래.</p>
<h1>무엇이 바뀌어 정확도와 삭제 비용이 달라지는가?</h1><p><b>{conclusion}</b></p>
<div class="flow"><span>배정 → 함께 학습하는 데이터 → 업데이트/모델</span><span>배정 → 최약 채널 → 평균집계 잡음</span><span>각 shard의 완료시점 → 삭제 통신비</span></div>
<p>두 경로를 분리하기 위해 이번에는 모든 shard를 5명으로 고정했습니다. 삭제 후에는 항상 4명입니다. 각 client의 실제 데이터는 이동·수정하지 않고 그룹만 바꿉니다. 배정 간 무잡음 비교는 grouping의 학습효과, 같은배정의잡음0/.01 비교는 receiver noise intervention입니다.</p>
<img src="mechanisms.png" alt="균등 shard 배정의 정확도, 삭제 완료율, 성공사례의 라운드 비교">
<h2>1. 크기와 채널이 집계잡음에 미치는 물리 효과</h2><p class="box">가설: MSE∝1/(n²hmin²) | 독립: n3/5/8,hmin.1/.3/1 | 종속: 복원오차·MSE·전력 | 통제: 같은평균update·clip1·P1·수신분산.01 | 판정: 9조건 모두통과</p>
<p>50,000회 noise draw/조건, 최대 MSE 상대오차 {100*max(r['relative_error'] for r in physical['rows']):.3f}%. 무선 복원식 검산이며 실제학습에서 shard크기의 인과효과를 검증한 실험은 아닙니다. 균등 n에서 한명을 균일하게 삭제하면, 그룹의 평균 삭제 MSE/source MSE 비는 n/(n−1)+n/(n−1)²·(h₁/h₂)²입니다. n5이면 1.25~1.5625. h₁/h₂는 가장약한/두번째약한 채널이며 이식은 순수잡음의기하만설명합니다.</p>
<h2>2. 데이터 구성과 잡음의 학습 경로</h2><p class="box">가설: 무잡음에서도 data배정이channel보다낫다 | 독립: 동일크기배정 | 종속: 고정160 삭제정확도 | 통제: 무잡음·크기5·모델·학습량 | 판정: {dataword}; 평균 {hdata['mean_gap_pp']:+.2f}%p</p>{main}
<p>무잡음 data−channel seed별 차이: {', '.join(f'{g:+.2f}%p' for g in data_gaps)}. class-JS 최소화가 좋은지와 무선MSE 최소화가좋은지는별개질문입니다. 낮은MSE가반드시높은정확도/낮은완료비용으로이어진다고가정하지않습니다.</p>
<h2>3. 같은 완료 규칙에서의 실제 삭제 통신비</h2><p class="box">가설: 배정별 동일품질 삭제완료비용이다르다 | 독립: 3배정 | 종속: 완료율·UL·정확도 | 통제: 각shard검증65%(primary)/60%(별도),2회연속,cap200 | 판정: {costword}</p>
<p>초기학습과삭제 모두 같은 shard별 중단규칙입니다. 20round마다 공개dev2000개를측정하며 연속2회목표이상인 최초시점에완료합니다. 미달은실패입니다. 표의 전체평균UL은60요청모두완료한경우만표시합니다. 실제연구는200까지실행해사전정한두품질기준/160고정비교를함께계측했습니다.</p>{costtable}
<p class="note">성공한 요청만의 평균 라운드는 선택 편향이 있으므로 이 값만으로 배정의 우열을 정하지 않습니다. 미완료는 <b>사전 정의한 shard별 정확도 기준을 200라운드 안에 달성하지 못했다</b>는 뜻입니다. 삭제 대상 데이터를 사용했다거나 SISA 독립성이 깨졌다는 뜻은 아닙니다. 각 shard에 65%를 요구하는 조건은 앙상블 전체 품질만 요구하는 조건보다 엄격할 수 있습니다. 60%와 65% 모두 사전등록했고 결과를 본 뒤 기준을 바꾸지 않았습니다.</p>
<h2>4. 같은 배정에 잡음만 추가하면?</h2>{noisetable}
<p>완료율악화 또는 양쪽완료쌍의비용5%증가와모든seed증가가사전비용악화기준입니다. 잡음효과가작거나반대면그대로보고합니다. 실제SNR은업데이트/채널에따라달라지므로 .01을정규화nominal20dB로표기합니다.</p>
<h2>5. 상관관계: 어떤 대리값을 비용 대신 쓸 수 있는가?</h2>{corrtable}
<img src="causal_paths.png" alt="무잡음에서의 데이터 분포와 shard 정확도, 같은 배정의 잡음 효과">
<p>Spearman rank상관이며 seed/배정/client 관측이서로독립이아닙니다. p-value나인과효과로해석하지않습니다. 완료round상관은성공사례만포함하므로특히선택편향이있습니다. 무잡음에서도 q↔정확도상관이생기면 그것은채널잡음의인과효과가아닙니다.</p>
<p><b>사후 탐색 분석:</b> 삭제 전에는 완료했지만 한 명이 빠진 뒤 완료하지 못하는 사례도 확인했습니다. 이 집계와 JS변화의 상관은 사전 가설 판정과 분리합니다.</p>{posthoctable}
<p><b>구체적 사례:</b> 무잡음에서 shard dev 정확도 감소가 가장 컸던 seed {example['seed']}, client {example['client']} 삭제를 고르고, 같은 client의 삭제를 세 배정에서 비교했습니다. 큰 감소 사례를 사후에 선택한 설명용 예시이며 전체 평균 효과를 대신하지 않습니다. 아래에서는 무선 잡음이 0이므로 배정 간 차이를 수신 잡음으로 설명할 수 없습니다.</p>{exampletable}
<p><b>사후 기술 통계:</b> 각 shard 모델의 품질과 최종 앙상블 품질을 구분합니다. 앙상블 이득은 두 정확도의 차이이며, 이것만으로 모델 다양성이 원인이라고 증명하지는 않습니다.</p>{enstable}
<h2>6. 지금 구현 가능한 알고리즘 v0</h2><p class="box">가설: 같은품질을만족하는후보중실측삭제비용최소선택 | 독립: 3후보 | 종속: 평균/최악삭제UL·완료율 | 통제: 동일학습규칙 | 상태: 사후 선택기 구현·실행, 온라인일반화미검증</p>
<p><b>후보생성 → source/각client삭제프로파일링 → 품질제약확인 → 평균삭제UL최소선택 → 배정고정.</b> test정확도는선택입력에사용하지않습니다. JS와MSE는후보를만드는정보이며실제cost를대체하지않습니다.</p>{selecttable}
<p class="note">이선택기는모든후보를직접학습·삭제하는준비비용을지불합니다. 따라서이를곧바로저비용신규알고리즘이라고주장하지않습니다. 준비비용까지상쇄하는순통신절감, 새로운무선실현에서의우월성, 문헌최초성은아직확인하지않았습니다.</p>
<p><a href="알고리즘_v0.txt">입력·수식·절차·비용·저장·보장 상세</a> · <a href="{link('select_partition.py')}">실행 가능한 선택 코드</a></p>
<h2>조건·비용·검증</h2><p>FashionMNIST12000train/2000publicdev/10000test, Dirichlet.5, seeds64001~64003, CNN38282params. 각client2SGDsteps×64,lr.05,clip1,average-symbol P1,R1,shard순차SISO,perfectCSI/timing,독립합성flat채널. 각shard 초기화같음,예측equalensemble. shard내개별update이력저장없음. 초기hist/count/channel노출7680bit/seed.</p>
<p>{resource_words} local calls {resource['local_calls']:,}, aggregation rounds {resource['aggregation_rounds']:,}, public-dev 평가 {resource['dev_evaluations']:,}, test 평가 {resource['test_evaluations']:,}, 외부 유료 0. power 5초 샘플, GPU 보드 전체이며 호스트 전력은 포함하지 않습니다. 실제 무선 장비의 시간/전력 측정이 아닙니다. <a href="resource_attribution.json">자원 구간별 기록</a> · <a href="{link('GPU_RESOURCE_NOTE.txt')}">다른 GPU 작업으로 인한 일시중단 기록</a></p>
<p>18조건/360삭제/{audit_events:,}events검산, representative삭제18재실행 parameter최대오차{max_reference}, 삭제대상미참여/타shard예측hash불변/중단시점재현/전력/R1/cost 통과. 고정routing조건부재현으로한정하며정보의존routing과metadata삭제, DP/secureaggregation, 실제RF는검증하지않았습니다.</p>
<details><summary>모든 seed별 결과</summary>{seedtable}</details>
<h2>사전등록과 원자료</h2><p><a href="{link('PREREG.txt')}">실행 전 가설·판정기준</a> · <a href="{link('results.json')}">원결과</a> · <a href="{link('resource.json')}">실행비용/전력샘플</a> · <a href="{link('provenance.json')}">코드hash</a> · <a href="analysis.json">전체분석</a> · <a href="audit.json">감사결과</a> · <a href="한눈에_결과.txt">한눈에 요약</a></p>
<p>이보고서는실행폴더와분리했습니다. 이전실험을덮어쓰지않았으며 과거의크기가변조건결과를이번균등크기실험에섞지않았습니다. 동일SISA전체재학습 대비이득과새배정의추가이득을구분하며 단일global모델FL을비교분모로사용하지않았습니다.</p></html>'''
textfile('report.html',page)
print(json.dumps(dict(summary=summary,Hdata=hdata,noise_effects=noise_effects,cost_comparisons=cost_comparisons,selection=[dict(seed=s['seed'],target=s['target'],winner=s['winner']['method'] if s['winner'] else None) for s in selection],resource={k:v for k,v in resource.items() if k!='samples'},audit=audit),ensure_ascii=False,indent=2))
