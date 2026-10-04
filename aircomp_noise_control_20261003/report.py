from pathlib import Path
import json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;OLD=R.parent/'aircomp_deletion_partition_20261003'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,v):
    with (R/name).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
start=time.perf_counter();new=read(R/'results.json');old=read(OLD/'results.json');fixed=read(OLD/'fixed160_results.json');resource=read(R/'resource.json')
first_seconds_lower_bound=(R/'62001_channel_nominal0dB_summary.json').stat().st_mtime-(R/'provenance.json').stat().st_mtime
METHODS=['channel','data','delete_joint'];LEVELS=['no_noise','nominal20dB','nominal0dB'];combined=[];eventcount=0
for r in new:
    source=read(R/f"{r['seed']}_{r['method']}_{r['noise']}_source.json")
    oldr=next(x for x in old if x['seed']==r['seed'] and x['method']==r['method']);assert r['groups']==oldr['groups']
    assert source['cost']['ul_reals']==oldr['source_cost']['ul_reals']
    for d in r['deletions']:
        previous=next(x for x in fixed if x['seed']==r['seed'] and x['method']==r['method'] and x['client']==d['client'])
        for k in ['total_re','local_calls','ul_reals','dl_bits','pilot_reals','control_bits']:assert d['cost'][k]==previous['metrics']['cost'][k]
        obj=read(R/f"{r['seed']}_{r['method']}_{r['noise']}_delete_{d['client']}.json")
        assert len(obj['events'])==160
        for e in obj['events']:
            assert e['repeats']==1 and e['max_power']<=1.00001
            assert abs(e['expected_mse']-r['sigma2']/(e['n']**2*e['min_h']**2))<1e-9
            eventcount+=1
    combined.append({'seed':r['seed'],'method':r['method'],'noise':r['noise'],'new_run':True,'source_test':r['source']['test'],
        'delete_test':r['mean_delete_test'],'delete_worst_recall':float(np.mean([d['metrics']['test_worst_recall'] for d in r['deletions']])),
        'mean_delete_mse':float(np.mean([d['mean_expected_mse'] for d in r['deletions']]))})
for r in old:
    if r['method'] not in METHODS:continue
    ds=[d for d in fixed if d['seed']==r['seed'] and d['method']==r['method']]
    combined.append({'seed':r['seed'],'method':r['method'],'noise':'nominal20dB','new_run':False,'source_test':r['source_metrics']['test'],
        'delete_test':float(np.mean([d['metrics']['test'] for d in ds])),
        'delete_worst_recall':float(np.mean([d['metrics']['test_worst_recall'] for d in ds])),
        'mean_delete_mse':next(x['mean_delete_mse']/100 for x in combined if x['seed']==r['seed'] and x['method']==r['method'] and x['noise']=='nominal0dB')})
def get(seed,method,noise):return next(r for r in combined if r['seed']==seed and r['method']==method and r['noise']==noise)
summary=[]
for method in METHODS:
    for noise in LEVELS:
        rr=[get(s,method,noise) for s in [62001,62002,62003]]
        penalties=[100*(r['delete_test']-get(r['seed'],method,'no_noise')['delete_test']) for r in rr]
        summary.append({'method':method,'noise':noise,'source_test':float(np.mean([r['source_test'] for r in rr])),
            'delete_test':float(np.mean([r['delete_test'] for r in rr])), 'seed_delete_test':[r['delete_test'] for r in rr],
            'mean_noise_effect_pp':float(np.mean(penalties)),'seed_noise_effect_pp':penalties,
            'worst_recall':float(np.mean([r['delete_worst_recall'] for r in rr])),
            'mean_delete_mse':float(np.mean([r['mean_delete_mse'] for r in rr]))})
comparisons=[]
for noise in LEVELS:
    gap=[100*(get(s,'channel',noise)['delete_test']-get(s,'data',noise)['delete_test']) for s in [62001,62002,62003]]
    base=[100*(get(s,'channel','no_noise')['delete_test']-get(s,'data','no_noise')['delete_test']) for s in [62001,62002,62003]]
    comparisons.append({'noise':noise,'channel_minus_data_pp':float(np.mean(gap)),'seed_gaps_pp':gap,
                        'difference_in_differences_pp':float(np.mean(np.array(gap)-base))})
g0=comparisons[0];nonradio=g0['channel_minus_data_pp']<=-1 and all(v<0 for v in g0['seed_gaps_pp'])
audit={'passed':True,'new_conditions':len(new),'new_deletions':sum(len(r['deletions']) for r in new),'reused_conditions':9,'reused_deletions':180,
       'paired_cost_checks':360,'events_checked':eventcount,'max_reference_error':max(d['reference_maxabs'] for r in new for d in r['deletions'] if d['reference_maxabs'] is not None),'seconds':time.perf_counter()-start}
write('analysis.json',{'summary':summary,'comparisons':comparisons,'nonradio_criterion_pass':nonradio,'per_seed':combined,'first_attempt_seconds_lower_bound':first_seconds_lower_bound});write('audit.json',audit)
colors={'channel':'#2563eb','data':'#059669','delete_joint':'#d97706'}
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for method in METHODS:
    rr=[next(r for r in summary if r['method']==method and r['noise']==l) for l in LEVELS]
    values=np.array([r['seed_delete_test'] for r in rr])*100
    axes[0].plot(range(3),values.mean(1),marker='o',label=method,color=colors[method])
    for j in range(3):axes[0].scatter(range(3),values[:,j],s=17,alpha=.4,color=colors[method])
    axes[1].plot(range(3),[r['mean_noise_effect_pp'] for r in rr],marker='o',label=method,color=colors[method])
for ax in axes:
    ax.set_xticks(range(3),['No noise','Nominal 20 dB','Nominal 0 dB']);ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0].set_ylabel('Post-deletion test accuracy (%)');axes[0].set_title('160 rounds; lines: 3-seed mean, dots: seeds')
axes[1].set_ylabel('Change from same partition without noise (pp)');axes[1].set_title('Only receiver noise is changed');axes[1].axhline(0,color='gray',lw=.7)
fig.savefig(R/'noise_control.png',dpi=170);plt.close(fig)
def table(head,rows):return '<table><tr>'+''.join('<th>'+str(h)+'</th>' for h in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
main_table=table(['배정','수신 잡음','최초 test','삭제160 test','최저 class recall 평균','무잡음 대비 변화','삭제 집계MSE'],[[r['method'],r['noise'],f"{100*r['source_test']:.2f}%",f"{100*r['delete_test']:.2f}%",f"{100*r['worst_recall']:.2f}%",f"{r['mean_noise_effect_pp']:+.2f}%p",f"{r['mean_delete_mse']:.4g}"] for r in summary])
compare_table=table(['잡음 조건','채널−데이터 정확도','무잡음 대비 격차 변화','seed별 채널−데이터'],[[r['noise'],f"{r['channel_minus_data_pp']:+.2f}%p",f"{r['difference_in_differences_pp']:+.2f}%p",', '.join(f'{v:+.2f}' for v in r['seed_gaps_pp'])] for r in comparisons])
details=table(['seed','배정','잡음','source test','삭제 test','신규실행'],[[r['seed'],r['method'],r['noise'],f"{100*r['source_test']:.2f}%",f"{100*r['delete_test']:.2f}%",r['new_run']] for r in sorted(combined,key=lambda r:(r['seed'],r['method'],LEVELS.index(r['noise'])))])
verdict='무선 잡음을 제거해도 채널 중심 배정의 열세가 남는다는 사전 기준을 충족했습니다.' if nonradio else '무잡음에서도 채널 중심 배정이 일관되게 열세라는 사전 기준은 충족하지 못했습니다.'
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>고정 배정·잡음만 변경한 대조 실험</title><style>body{{font-family:Segoe UI,Malgun Gothic,sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;color:#203047;line-height:1.75}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:30px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}td,th{{border:1px solid #d4dce8;padding:10px;text-align:left}}th{{background:#edf3fa}}.box{{padding:20px;background:#edf5ff;border-left:5px solid #2463ce}}.note{{padding:16px;background:#fff5df}}img{{width:100%;height:auto}}a{{color:#175cd3}}</style>
<h1>배정·데이터·학습량을 고정하고 수신 잡음만 바꿨습니다</h1>
<p class="box"><b>가설</b> 채널 중심 배정의 정확도 손실은 수신 잡음 때문에 발생했는가?<br><b>독립변수</b> noise variance 0 / .01 / 1 · <b>종속변수</b> 삭제160round 정확도 및 동일배정의잡음효과 · <b>통제</b> 배정·데이터·초기화·batch·채널·round·전력한도·전송량 · <b>상태</b> 신규18조건360삭제 완료, 기존9조건180삭제 재사용</p>
<h2>판정</h2><p><b>{verdict}</b> 무잡음에서 channel−data 차이는 {g0['channel_minus_data_pp']:+.2f}%p입니다. 이것은 수신 잡음으로 설명되지 않는 배정/크기/협업학습 효과이며, 데이터 편중 하나의 인과효과로 단정하지 않습니다.</p>
<p>기존결과가불리하다는이유로배정·seed·목표값을바꾸지않았습니다. 강한잡음조건도0dB라는스트레스시험으로명시하고무잡음·20dB와동시에보고합니다. 채널집계MSE개선과실제정확도개선을구분합니다.</p>
<img src="noise_control.png" alt="배정별 삭제 후 정확도와 같은 배정의 무잡음 대비 정확도 변화">
<h2>전체 조건 결과</h2>{main_table}
<h2>채널 중심 배정이 잡음에 더 강한가?</h2>{compare_table}
<p>격차 변화 = [채널(noise)−채널(무잡음)] − [데이터(noise)−데이터(무잡음)]. 양수면 채널 중심 배정이 상대적으로 잡음에 강한 방향입니다. 이는 그 조건에서 channel이 최종 정확도도 가장 높다는 뜻은 아닙니다. 3개 seed의 같은 그룹·같은client 삭제를 짝지어 비교했습니다.</p>
<h2>무엇을 같게 두었나</h2>
{table(['구분','통제'],[['배정','기존 channel/data/delete_joint 그룹 그대로. shard크기·client소속 재검색 없음'],['데이터','FashionMNIST12000/2000/10000, Dirichlet.5, 20clients, seeds62001~62003'],['모델','CNN38282 parameters, 모든조건동일초기값'],['학습','2SGDsteps×64, lr.05, momentum0, clip1, source/삭제모두160round'],['무선','SISO순차4blocks, R=1, P=1평균symbol전력한도, 동일h/CSI/동기화'],['유일한 개입','receiver sigma²=0/.01/1. 동일standard-normal noise에크기만변경'],['삭제','모든20명을각각삭제, 해당shard만초기화재학습, 다른shard source모델유지'],['예산','같은배정과삭제자에서통신량·localcalls가noise조건간완전히같음'],['선택','test/dev로조건선택·조기중단안함. 20dB기존자료재사용표시']])}
<p class="note">같은배정 안에서 수신잡음의 효과는 분리했습니다. 배정방법 사이에는 원래부터 shard크기와 데이터구성이 다르므로 무잡음에서의차이를 ‘데이터편중만의효과’라고해석하지않습니다. 이것은 원인을 숨기지 않고 결론의 범위를 정확히 나누기 위한 대조입니다.</p>
<h2>검증</h2><p>원자료 indices/pools/base채널 일치, 기존코드와20dB 1step 재현9건 parameter오차0. 삭제 {eventcount:,}events 물리식/전력/R1 검산. 360개삭제의통신·연산이이전같은배정160round와일치. representative client0 초기화재실행18건 최대parameter오차 {audit['max_reference_error']:.2g}. 공개고정routing조건부모델검증이며privacy/metadata삭제/실제RF검증아님.</p>
<details><summary>27조건 seed별 결과</summary>{details}</details>
<h2>실행 오류와 복구</h2><p>첫2조건을완료한뒤호환검사에서잡음설정초기화누락으로중단했다. 완료조건의noise설정은정상이었으며해당결과를보존하고,검사직전sigma²=.01초기화를추가한뒤남은16조건만재개했다. 실험조건/성공기준/seed는변경하지않았다. <a href="EXECUTION_NOTE.txt">오류·복구 기록</a>. 첫2조건의최종tensor묶음은소실됐지만원자료events/metrics는보존됐다.</p>
<h2>비용·한계·다음 판단</h2><p>첫실행완료구간 시간하한 {first_seconds_lower_bound/60:.2f}분 + 재개 {resource['seconds']/60:.2f}분 = 총실행시간하한 {(resource['seconds']+first_seconds_lower_bound)/60:.2f}분. 재개구간GPU보드 {resource['gpu_board_Wh']:.3f}Wh(호스트전체아님), 첫실행GPU샘플소실로총에너지는미측정. 재개 local training calls {resource['local_calls']:,} + 기존엔진호환검사 {resource['legacy_check_calls']['local_calls']}회. 외부결제0, 전기요금미산정. 재사용20dB실험의과거비용은이번비용에중복합산하지않음.</p>
<p>한데이터셋·3seed·작은CNN·합성채널·완전CSI에대한통제실험입니다. AirCluster MIMO비교/새배정최적화/새삭제확률모델은실행하지않았습니다. 조건통제가문제없음이나새알고리즘우위를보장하지않습니다. 수신잡음영향과배정전체영향을구분한결과를기반으로후속알고리즘을설계하며,이번결과를보고추가조건을골라실행하지않습니다.</p>
<p><a href="PREREG.txt">사전등록 조건</a> · <a href="analysis.json">분석 원자료</a> · <a href="results.json">새 실행 결과</a> · <a href="audit.json">검산</a> · <a href="control_checks.json">기존실험 일치 확인</a> · <a href="resource.json">GPU 비용</a> · <a href="run.py">실행코드</a> · <a href="provenance.json">입력/코드 SHA256</a> · <a href="../aircomp_deletion_partition_20261003/report.html">기존20dB 실험</a></p></html>'''
with (R/'report.html').open('x',encoding='utf-8') as f:f.write(page)
print(json.dumps({'summary':summary,'comparisons':comparisons,'nonradio_criterion':nonradio,'audit':audit,'resource':{k:v for k,v in resource.items() if k!='samples'}},ensure_ascii=False,indent=2))
