"""Independent CPU audit, finite-grid budget comparison, report and plots."""
from pathlib import Path
import json, math, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'snr_privacy_v1'
DEST=ROOT/'snr_privacy_summary_v1'
D=38282; C=.1; delta=1e-5

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def epsilon(rho): return float(rho+2*np.sqrt(rho*np.log(1/delta)))
def resources(K,R,t):
    # source+one affected shard, including histogram/count/shard metadata.
    source=t*(K*(R*D+16*D+32)+40*20)+3840
    deletion=t*(R*D+16*D+32+40*(20//K-1))
    return source+deletion

def audit(rows):
    assert len(rows)==63
    assert len({r['case'] for r in rows})==57
    for r in rows:
        folder=OUT/r['case']; cfg=read(folder/'config.json')
        assert r['replay_exact'] and r['checkpoints_exact']
        assert sorted(sum(cfg['groups'],[]))==list(range(20))
        assert all(len(g)==20//r['K'] for g in cfg['groups'])
        sl=read(folder/'source_ledger.json'); ul=read(folder/f"delete{r['deleted_client']}_replay_ledger.json")
        assert all(r['deleted_client'] not in e['ids'] for e in ul['events'])
        for ledger in [sl,ul]:
            rho=np.zeros(20)
            for event in ledger['events']:
                sensitivity=2*C/event['active']
                calculated=sensitivity**2/(2*event['variance'])
                assert np.isclose(calculated,event['per_client_rho'],rtol=1e-11)
                assert event['max_power']<=1.00001
                assert np.isclose(event['expected_mse'],D*event['variance'])
                assert event['ul_reals']==r['repeats']*D
                rho[event['ids']]+=calculated
            assert np.allclose(rho,ledger['curve'][-1]['rho_per_client'])
        for point in r['curve']:
            assert np.isclose(point['combined_total_re'],resources(r['K'],r['repeats'],point['round']))
            assert np.isclose(point['combined_epsilon'],epsilon(max(point['rho_per_client'])))
            if r['cap'] is not None: assert point['combined_epsilon']<=r['cap']+1e-7
    for seed in [10801,10802,10803]:
        cases=sorted({r['case'] for r in rows if r['seed']==seed})
        first=read(OUT/cases[0]/'config.json')
        with np.load(OUT/cases[0]/'partition.npz') as z: part={k:z[k].copy() for k in z.files}
        for case in cases:
            cfg=read(OUT/case/'config.json')
            assert np.array_equal(cfg['base_h'],first['base_h'])
            assert np.array_equal(cfg['histogram'],first['histogram'])
            with np.load(OUT/case/'partition.npz') as z: assert all(np.array_equal(z[k],v) for k,v in part.items())
    return dict(passed=True,source_configurations=57,deletion_evaluations=63,checkpoints=252,
                radio_and_privacy_recomputed=True,paired_channels_histograms_partitions_matched=True,
                exact_models='Worker independently trained full reference and replay; exact at all checkpoints')

def main():
    start=time.perf_counter()
    assert (OUT/'complete.json').exists()
    DEST.mkdir(exist_ok=False)
    rows=read(OUT/'results.json'); dump(DEST/'audit.json',audit(rows))
    mainrows=[r for r in rows if r['deleted_client']==0]
    table=[]
    for r in mainrows:
        p=r['curve'][-1]
        table.append(dict(seed=r['seed'],K=r['K'],method=r['method'],snr=r['snr'],R=r['repeats'],cap=r['cap'],
                  test=p['deleted']['test']['accuracy'],dev=p['deleted']['dev']['accuracy'],
                  macro_f1=p['deleted']['test']['macro_f1'],worst_recall=p['deleted']['test']['worst_recall'],
                  epsilon=p['combined_epsilon'],source_epsilon=p['source_epsilon'],
                  total_re=p['combined_total_re'],delete_re=r['delete_cost']['total_re'],
                  mse=r['mean_mse'],actual_snr_db=r['mean_actual_snr_db'],clip_fraction=r['clip_fraction'],
                  source_signal_energy=r['source_cost']['signal_energy'],mean_js=r['mean_js']))
    dump(DEST/'all_final_metrics.json',table)
    summaries=[]
    configs=sorted({(r['K'],r['snr'],r['method'],r['R'],str(r['cap'])) for r in table})
    for K,snr,method,R,cap in configs:
        selected=[r for r in table if (r['K'],r['snr'],r['method'],r['R'],str(r['cap']))==(K,snr,method,R,cap)]
        summaries.append(dict(K=K,snr=snr,method=method,R=R,cap=cap,n=len(selected),
            **{key:float(np.mean([r[key] for r in selected])) for key in ['test','dev','worst_recall','epsilon','total_re','mse','actual_snr_db','mean_js']},
            test_min=min(r['test'] for r in selected),test_max=max(r['test'] for r in selected)))
    dump(DEST/'summary.json',summaries)
    targets=[]
    for seed in [10801,10802,10803]:
        for snr in [0,10,20]:
            for method in ['random','channel','joint']:
                for cap in [8,100,100000,10000000]:
                    candidates=[]
                    for r in mainrows:
                        if (r['seed'],r['snr'],r['method'])!=(seed,snr,method): continue
                        for p in r['curve']:
                            if p['deleted']['dev']['accuracy']>=.70 and p['combined_epsilon']<=cap+1e-7:
                                candidates.append(dict(case=r['case'],K=r['K'],R=r['repeats'],power_cap_epsilon=r['cap'],
                                  round=p['round'],resources=p['combined_total_re'],dev=p['deleted']['dev']['accuracy'],
                                  test=p['deleted']['test']['accuracy'],epsilon=p['combined_epsilon']))
                    best=min(candidates,key=lambda x:(x['resources'],x['case'],x['round'])) if candidates else None
                    targets.append(dict(seed=seed,snr=snr,method=method,epsilon_limit=cap,feasible=best is not None,best=best))
    dump(DEST/'matched_accuracy_privacy.json',targets)
    # Preregistered practical grouping check: same K4, R1 natural, dev-only stopping.
    paired=[]
    for seed in [10801,10802,10803]:
        for snr in [0,10,20]:
            selected={r['method']:r for r in mainrows if r['seed']==seed and r['snr']==snr and r['K']==4 and r['repeats']==1 and r['cap'] is None}
            for baseline in ['random','channel']:
                a=selected[baseline]; b=selected['joint']
                ac=next((p for p in a['curve'] if p['deleted']['dev']['accuracy']>=.70),None)
                bc=next((p for p in b['curve'] if p['deleted']['dev']['accuracy']>=.70),None)
                paired.append(dict(seed=seed,snr=snr,baseline=baseline,
                    accuracy_delta_pp=100*(b['curve'][-1]['deleted']['test']['accuracy']-a['curve'][-1]['deleted']['test']['accuracy']),
                    baseline_target=ac is not None,joint_target=bc is not None,
                    target_saving=None if ac is None or bc is None else 1-bc['combined_total_re']/ac['combined_total_re'],
                    epsilon_ratio=b['curve'][-1]['combined_epsilon']/a['curve'][-1]['combined_epsilon']))
    dump(DEST/'paired_grouping.json',paired)
    decisions=[]
    for snr in [0,10,20]:
        for baseline in ['random','channel']:
            p=[r for r in paired if r['snr']==snr and r['baseline']==baseline]
            allfeasible=all(r['target_saving'] is not None for r in p)
            mean_saving=float(np.mean([r['target_saving'] for r in p])) if allfeasible else None
            accuracy=float(np.mean([r['accuracy_delta_pp'] for r in p]))
            improved=sum(r['target_saving'] is not None and r['target_saving']>0 for r in p)
            decisions.append(dict(snr=snr,baseline=baseline,mean_accuracy_delta_pp=accuracy,all_targets_feasible=allfeasible,
                 mean_target_saving=mean_saving,improved_seeds=improved,
                 practical_pass=bool(allfeasible and mean_saving>=.1 and accuracy>=-2 and improved>=2)))
    dump(DEST/'decisions.json',decisions)
    fixedbudget=[]
    for r in mainrows:
        if r['cap'] is not None or r['repeats']!=1: continue
        budget=resources(2,1,160)
        points=[p for p in r['curve'] if p['combined_total_re']<=budget]
        best=max(points,key=lambda p:p['round']) if points else None
        fixedbudget.append(dict(case=r['case'],seed=r['seed'],snr=r['snr'],K=r['K'],method=r['method'],budget=budget,
                               round=best['round'] if best else None,test=best['deleted']['test']['accuracy'] if best else None,
                               used_re=best['combined_total_re'] if best else None))
    dump(DEST/'equal_total_budget.json',fixedbudget)
    deletioncheck=[dict(case=r['case'],deleted=r['deleted_client'],accuracy=r['curve'][-1]['deleted']['test']['accuracy'],
                       total_re=r['curve'][-1]['combined_total_re'],epsilon=r['curve'][-1]['combined_epsilon'])
                   for r in rows if r['seed']==10801 and r['K']==4 and r['snr']==10 and r['repeats']==1 and r['cap'] is None]
    dump(DEST/'deletion_target_check.json',deletioncheck)
    repeat_pairs=[]
    for seed in [10802,10803]:
        for method in ['random','channel','joint']:
            a=next(r for r in table if r['seed']==seed and r['method']==method and r['K']==4 and r['snr']==10 and r['R']==1 and r['cap'] is None)
            b=next(r for r in table if r['seed']==seed and r['method']==method and r['K']==4 and r['snr']==10 and r['R']==4 and r['cap'] is None)
            repeat_pairs.append(dict(seed=seed,method=method,accuracy_delta_pp=100*(b['test']-a['test']),
                mse_ratio=b['mse']/a['mse'],epsilon_ratio=b['epsilon']/a['epsilon'],
                resource_ratio=b['total_re']/a['total_re']))
    dump(DEST/'repetition_pairs.json',repeat_pairs)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    colors=dict(random='#0072B2',channel='#D55E00',joint='#009E73')
    for method,color in colors.items():
        selected=[next(s for s in summaries if s['K']==4 and s['snr']==snr and s['method']==method and s['R']==1 and s['cap']=='None') for snr in [0,10,20]]
        axes[0,0].plot([0,10,20],[s['test']*100 for s in selected],'o-',color=color,label=method)
        axes[0,1].semilogy([0,10,20],[s['epsilon'] for s in selected],'o-',color=color,label=method)
        ks=[next(r for r in table if r['seed']==10801 and r['K']==K and r['snr']==10 and r['method']==method and r['R']==1 and r['cap'] is None) for K in [2,4,5]]
        axes[1,0].plot([2,4,5],[r['test']*100 for r in ks],'o-',color=color,label=method)
        capped=[next(r for r in table if r['seed']==10801 and r['K']==4 and r['snr']==10 and r['method']==method and r['R']==1 and r['cap']==cap) for cap in [8,100,None]]
        axes[1,1].plot([0,1,2],[r['test']*100 for r in capped],'o-',color=color,label=method)
    axes[0,0].set(title='Fixed repetitions: K=4, 3 seeds',xlabel='Nominal SNR (dB)',ylabel='Deletion test accuracy (%)')
    axes[0,1].set(title='Conditional privacy: source + one replay',xlabel='Nominal SNR (dB)',ylabel='Worst-client epsilon bound (log scale)')
    axes[1,0].set(title='Shard count: SNR=10, screen seed only',xlabel='Number of shards',ylabel='Deletion test accuracy (%)',xticks=[2,4,5])
    axes[1,1].set(title='Privacy power control: screen seed only',ylabel='Deletion test accuracy (%)',xticks=[0,1,2],xticklabels=['epsilon ≤ 8','epsilon ≤ 100','Natural noise'])
    for ax in axes.flat: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('SISA / SISO AirComp: accuracy, noise and conditional client privacy\nPublic fixed class histograms · 20 clients · 160 rounds · simulation',fontsize=13)
    fig.savefig(DEST/'snr_privacy_results.png',dpi=170); fig.savefig(DEST/'snr_privacy_results.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for method,color in colors.items():
        rs=[r for r in mainrows if r['K']==4 and r['snr']==10 and r['method']==method and r['repeats']==1 and r['cap'] is None]
        x=[p['combined_total_re']/1e6 for p in rs[0]['curve']]
        ys=np.array([[p['deleted']['test']['accuracy']*100 for p in r['curve']] for r in rs])
        axes[0].plot(x,ys.mean(0),'o-',color=color,label=method)
        axes[0].fill_between(x,ys.min(0),ys.max(0),color=color,alpha=.12)
        for K,marker in [(2,'o'),(4,'s'),(5,'^')]:
            row=next(r for r in table if r['seed']==10801 and r['K']==K and r['snr']==10 and r['method']==method and r['R']==1 and r['cap'] is None)
            axes[1].scatter(row['total_re']/1e6,row['test']*100,color=color,marker=marker,s=65,label=f'{method}, K={K}')
    axes[0].set(title='K=4: mean and seed range',xlabel='Source + one deletion (million RE)',ylabel='Deletion test accuracy (%)')
    axes[1].set(title='K sweep at 160 rounds: screen seed only',xlabel='Source + one deletion (million RE)',ylabel='Deletion test accuracy (%)')
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Communication–accuracy observations at nominal SNR=10 dB')
    fig.savefig(DEST/'communication_accuracy.png',dpi=170); fig.savefig(DEST/'communication_accuracy.pdf'); plt.close(fig)
    cost=read(OUT/'cost_total.json'); preflight=read(ROOT/'snr_privacy_preflight_v1/cost_total.json')
    costbrief={k:v for k,v in cost.items() if k!='samples'}
    costbrief['preflight_seconds']=preflight['seconds']; costbrief['preflight_gpu_board_Wh']=preflight['gpu_board_Wh']
    lines=['# SISA / AirComp SNR·shard·privacy 실험 결과','',
           '57개 source 조건, 63개 deletion reference/replay 비교 완료. 252개 체크포인트에서 고정 shard 배정의 동일 SISA reference와 exact 일치. 실측 무선 데이터가 아닌 시뮬레이션이다. 공개 histogram/routing 자체를 삭제하거나 삭제 후 배정 알고리즘을 다시 실행하는 절차의 동일성까지 검증한 것은 아니다.', '',
           'FashionMNIST train12000/dev2000/test10000, client20, Dirichlet .5, local2x64 SGD lr.05, clip.1, 160 rounds. SNR=Pmax/sigma2, Pmax1, R1 기본. K2/5는 screen1seed, K4는3seeds. 원자료와 사전기록을 함께 확인한다.', '',
           '**프라이버시 범위:** 공개·고정된 client별 클래스 분포/크기/CSI/routing 조건에서 client의 모든 feature를 교체하는 conditional DP 상한. 클래스 분포/참여 여부 보호, 악성 수신기, 공모, 여러 안테나로 개별 신호 분리, 반복 삭제 요청 전체의 보장은 아니다. delta1e-5. 매우 큰 epsilon은 유용한 프라이버시 보장으로 해석하지 않는다.', '',
           '## K4 고정 라운드/SNR 비교 (3 seeds 평균)',
           '|SNR|배정|정확도|최저 클래스 recall|조건부 epsilon 상한|집계 MSE|source+delete RE|',
           '|---|---|---|---|---|---|---|']
    for s in summaries:
        if s['K']==4 and s['R']==1 and s['cap']=='None':
            lines.append(f"|{s['snr']}|{s['method']}|{100*s['test']:.2f}%|{100*s['worst_recall']:.2f}%|{s['epsilon']:.3g}|{s['mse']:.3g}|{s['total_re']:.0f}|")
    lines+=['','같은 K/R/라운드에서는 배정별 통신 자원 수가 같다. 좋은 grouping만으로 송신 symbol 수가 줄었다고 주장하지 않는다.','',
            '## Joint의 목표 dev accuracy 70% 도달 비용 판정',
            '|SNR|비교 기준|평균 최종 정확도 차이 pp|모든 seed에서 두 방식 도달|평균 자원 절감|개선 seed|실용 기준|','|---|---|---|---|---|---|---|']
    for s in decisions:
        saving='N/A' if s['mean_target_saving'] is None else f"{100*s['mean_target_saving']:.2f}%"
        lines.append(f"|{s['snr']}|{s['baseline']}|{s['mean_accuracy_delta_pp']:+.2f}|{s['all_targets_feasible']}|{saving}|{s['improved_seeds']}/3|{s['practical_pass']}|")
    lines+=['','실용 기준: 정확도 평균 손실2pp 이내, 같은 dev 목표에 대한 총 RE 평균10% 이상 절감 및2/3seed 이상 개선. 사전등록한 전체3seed 판정이며 screen/confirm 차이는 원자료에 명시된다. N/A는 실패/미도달을 삭제한 평균이 아니라 비교 불가다.', '',
            '## 같은 정확도·조건부 privacy 목표에서 grid 탐색',
            '|epsilon 상한|실현 가능한 seed/SNR/method 셀|전체 셀|','|---|---|---|']
    for cap in [8,100,100000,10000000]:
        cells=[r for r in targets if r['epsilon_limit']==cap]
        lines.append(f"|{cap}|{sum(r['feasible'] for r in cells)}|{len(cells)}|")
    lines+=['','각셀의 최저 source+delete RE, 선택K/R/round, dev와test는 matched_accuracy_privacy.json. 이 유한grid에서 미실현이라는 뜻이며 불가능성 증명은 아니다. seed10801만 K2/5와privacy-cap 후보가 있어 seed간 탐색공간이 다르므로 최저비용을 단순 pooled 비교하지 않는다.', '',
            '## 프라이버시 제한 전력 제어 (screen seed, K4/SNR10)',
            '|배정|목표epsilon|실제누적epsilon|정확도|','|---|---|---|---|']
    for r in table:
        if r['cap'] is not None: lines.append(f"|{r['method']}|{r['cap']}|{r['epsilon']:.3f}|{r['test']*100:.2f}%|")
    lines+=['','## R1에서 R4 반복 전송으로 변경 (confirmation 2 seeds)',
            '|seed|배정|정확도 차이 pp|MSE 비율|epsilon 비율|총 RE 비율|','|---|---|---|---|---|---|']
    for r in repeat_pairs:
        lines.append(f"|{r['seed']}|{r['method']}|{r['accuracy_delta_pp']:+.2f}|{r['mse_ratio']:.3f}|{r['epsilon_ratio']:.3f}|{r['resource_ratio']:.3f}|")
    lines+=['','## 세부자료와 한계',
            '- all_final_metrics.json:57개 전체 조건. summary.json:조건별 평균/범위. paired_grouping.json:seed별 차이. equal_total_budget.json:K2/R1/160라운드 상당의 동일 총 예산에서 가능한 최신checkpoint 비교.',
            '- deletion_target_check.json:screen K4/SNR10에서client0/7/14 삭제. 각source+한삭제 요청의 별도 시나리오이며 요청3개를 순차 공개하는 privacy budget이 아니다.',
            '- natural R1과 R4 비교는 confirmation 2 seeds만 대응 비교해야 한다. 반복은 UL/energy와 privacy composition을 증가시킨다. R4의 MSE 감소와 정확도 변화 및 epsilon 증가를 함께 확인한다.',
            '- 고정 clip/고정 전력 한도, 이상적 SISO 정렬, 완전 CSI, orthogonal shard transmission. MIMO 동시 분리나 실제 무선 간섭을 평가하지 않았다. 고정 라운드이며 수렴 보장이 없다.',
            '- 잡음 공분산과 민감도가 모두 shard 크기에 의존하므로, 크기가 커진다고 자동으로 privacy가 강해지지 않는다. 높은 epsilon 상한은 강한 보호를 입증하지 못한다는 뜻이며, 상한 차이는 실제 공격 성공률의 측정이 아니다.',
            '- 전체 데이터에 대한 client DP를 원하면 공개 label histogram도 보호하는 별도 protocol이 필요하다. raw histogram 기반 routing의 조건부 범위를 숨기지 않는다.', '',
            '## 비용','```json',json.dumps(costbrief,ensure_ascii=False,indent=2),'```','',
            'GPU board Wh는 GPU 보드 샘플 적분이며 host 전체/통신 RF Joule이 아니다. 외부 유료 서비스0. 정규화 signal energy는 각 ledger에 별도로 보존한다. 실패 시 failure.json도 확인한다.', '',
            '재현: ../snr_privacy.py, ../snr_privacy_analyze.py, ../SNR_PRIVACY_PREREG.md. 원자료 ../snr_privacy_v1/{case}/config,partition,labels,source와reference/replay models·ledgers·curves. 코드 hash는 environment.json. 기존 실험 결과는 변경하지 않았다.', '']
    (DEST/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    dump(DEST/'analysis_cost.json',dict(cpu_analysis_wall_seconds=time.perf_counter()-start))
    print(json.dumps(dict(audit='passed',decisions=decisions,cost=costbrief),ensure_ascii=False))

if __name__=='__main__': main()
