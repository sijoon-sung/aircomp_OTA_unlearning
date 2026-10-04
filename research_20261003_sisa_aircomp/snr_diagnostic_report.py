"""Report a separately registered post-hoc clipping diagnostic; retain main conclusions."""
from pathlib import Path
import json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    started=time.perf_counter(); out=ROOT/'snr_clip_diagnostic_v1'
    assert (out/'complete.json').exists()
    mainrows=read(ROOT/'snr_privacy_v1/results.json'); rows=read(out/'results.json')
    assert len(rows)==8 and all(r['replay_exact'] and r['checkpoints_exact'] for r in rows)
    comparisons=[]
    for b in rows:
        if b['snr']==120: continue
        a=next(r for r in mainrows if r['seed']==10801 and r['K']==4 and r['method']==b['method'] and r['snr']==b['snr'] and r['repeats']==1 and r['cap'] is None and r['deleted_client']==0)
        ac=read(ROOT/'snr_privacy_v1'/a['case']/'config.json'); bc=read(out/'C1.0'/b['case']/'config.json')
        assert ac['groups']==bc['groups'] and ac['base_h']==bc['base_h'] and ac['histogram']==bc['histogram']
        assert np.isclose(a['curve'][-1]['combined_epsilon'],b['curve'][-1]['combined_epsilon'])
        assert np.isclose(b['mean_mse']/a['mean_mse'],100)
        first=next((p for p in b['curve'] if p['deleted']['dev']['accuracy']>=.70),None)
        comparisons.append(dict(snr=b['snr'],method=b['method'],C01_test=a['curve'][-1]['deleted']['test']['accuracy'],
             C1_test=b['curve'][-1]['deleted']['test']['accuracy'],C01_clip=a['clip_fraction'],C1_clip=b['clip_fraction'],
             test_delta_pp=100*(b['curve'][-1]['deleted']['test']['accuracy']-a['curve'][-1]['deleted']['test']['accuracy']),
             C1_dev=b['curve'][-1]['deleted']['dev']['accuracy'],epsilon=b['curve'][-1]['combined_epsilon'],
             C1_first_dev70_round=None if first is None else first['round'],
             C1_first_dev70_total_re=None if first is None else first['combined_total_re']))
    a=next(r for r in rows if r['snr']==120 and r['C']==.1)
    b=next(r for r in rows if r['snr']==120 and r['C']==1.)
    diagnostic=dict(test_delta_pp=100*(b['curve'][-1]['deleted']['test']['accuracy']-a['curve'][-1]['deleted']['test']['accuracy']),
        dev_delta_pp=100*(b['curve'][-1]['deleted']['dev']['accuracy']-a['curve'][-1]['deleted']['dev']['accuracy']),
        C01_test=a['curve'][-1]['deleted']['test']['accuracy'],C1_test=b['curve'][-1]['deleted']['test']['accuracy'],
        C01_clip=a['clip_fraction'],C1_clip=b['clip_fraction'])
    diagnostic['criterion_pass']=bool(diagnostic['test_delta_pp']>=2 and diagnostic['dev_delta_pp']>=2)
    dump(out/'comparisons.json',comparisons); dump(out/'near_noiseless_diagnostic.json',diagnostic)
    dump(out/'audit.json',dict(passed=True,exact_cases=8,paired_group_channel_histogram_match=True,natural_epsilon_C_invariant=True,mse_C_squared=True))
    costs={name:read(ROOT/folder/'cost_total.json') for name,folder in [('preflight','snr_privacy_preflight_v1'),('main','snr_privacy_v1'),('diagnostic','snr_clip_diagnostic_v1')]}
    totals=dict(experiment_seconds=sum(c['seconds'] for c in costs.values()),gpu_board_Wh=sum(c['gpu_board_Wh'] for c in costs.values()),
                local_calls=sum(c['local_calls'] for c in costs.values()),external_payment=0,
                primary_source_cases=57,diagnostic_source_cases=8,primary_deletions=63,diagnostic_deletions=8,preflight_cases=1,
                energy_note='Approximate sampled whole-board integral; preflight shorter than 5s has no integration interval, not zero actual energy')
    dump(out/'combined_cost.json',totals)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    methods=['random','channel','joint']; x=np.arange(3)
    for ax,snr in zip(axes[:2],[0,20]):
        ps=[next(r for r in comparisons if r['snr']==snr and r['method']==m) for m in methods]
        ax.bar(x-.18,[100*r['C01_test'] for r in ps],width=.36,label='C=0.1',color='#0072B2')
        ax.bar(x+.18,[100*r['C1_test'] for r in ps],width=.36,label='C=1.0',color='#D55E00')
        ax.set(title=f'Nominal SNR={snr} dB',xticks=x,xticklabels=methods,ylabel='Deletion accuracy (%)',ylim=(0,100))
        ax.legend(fontsize=8)
    axes[2].bar([0,1],[100*diagnostic['C01_test'],100*diagnostic['C1_test']],color=['#0072B2','#D55E00'])
    axes[2].set(title='Near-noiseless diagnostic (120 dB)',xticks=[0,1],xticklabels=['C=0.1','C=1.0'],ylim=(0,100),ylabel='Deletion accuracy (%)')
    for ax in axes: ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.15)
    fig.suptitle('Post-hoc clipping diagnostic — one screen seed, K=4, 160 rounds\nSeparate from the 3-seed preregistered main comparison',fontsize=12)
    fig.savefig(out/'clipping_diagnostic.png',dpi=170); plt.close(fig)
    lines=['# SNR 실험의 clipping 진단과 최종 판단','',
           '주57조건을 대체하지 않는 사후 진단8조건이다. 주 실험에서91~95.5% clipping과 dev70% 미도달을 확인한 뒤 별도 사전기록을 남기고 실행했다. seed10801 하나만 사용했으므로 확인실험3seeds 결과와 합쳐 평균하지 않는다.', '',
           '|SNR|배정|C0.1 정확도|C1 정확도|차이 pp|C0.1 clipping|C1 clipping|','|---|---|---|---|---|---|---|']
    for r in comparisons: lines.append(f"|{r['snr']}|{r['method']}|{100*r['C01_test']:.2f}%|{100*r['C1_test']:.2f}%|{r['test_delta_pp']:+.2f}|{100*r['C01_clip']:.2f}%|{100*r['C1_clip']:.2f}%|")
    lines+=['','## 잡음을 거의 없앤 수치 기준',
            f"120dB/random에서 C0.1 정확도{100*diagnostic['C01_test']:.2f}%, C1 정확도{100*diagnostic['C1_test']:.2f}%. Test 차이{diagnostic['test_delta_pp']:+.2f}pp, dev 차이{diagnostic['dev_delta_pp']:+.2f}pp. 진단 기준(test/dev 모두2pp 이상 개선) 통과={diagnostic['criterion_pass']}.", '',
            '120dB는 현실적인 무선 SNR 주장이 아니라 작은 잡음의 수치 기준이다. 0/20dB에서 C를 바꾸면 clipping뿐 아니라 송신 alignment와 잡음 규모도 함께 바뀐다. 따라서 이 두 조건만으로 clipping의 순수 인과효과를 주장하지 않는다.', '',
            '## 해석',
            '- 주 실험의 실패/미도달 판정은 유지한다. 현재 조건부 client DP 상한은 자연잡음만으로 실용적인 보호를 입증하지 못하고, 프라이버시 제한 조건의 정확도 결과도 함께 읽어야 한다.',
            '- 진단에서 정확도가 회복된다면, 기본 실험의 작은 SNR 정확도 변화만으로 AirComp grouping 방향을 기각하지 않는다. 후속 확인은 calibration한 clipping/충분한 학습량에서 grouping과 SNR을 다시 비교하는 것이다. 아직 실행하지 않았다.',
            '- 공동 배정은 고정된 .5/.5의 MSE/JS heuristic이며 SNR별 최적화나 최적성/신규성을 주장하지 않는다. 크기 자체의 privacy 효과와 bottleneck channel 효과도 구분해야 한다.',
            '- 삭제 후 정확성은 고정 routing의 동일 SISA reference와 비교했다. 공개된 histogram/routing 자체의 삭제를 검증한 것은 아니다.', '',
            '## 전체 실행 비용 (주 실험+진단+preflight)','```json',json.dumps(totals,indent=2),'```','',
            'GPU board Wh는 host 전체나 통신 RF Joule이 아니다. 각 조건의 통신 RE/정규화 signal energy는 ledger에 별도로 있다. 외부 유료 서비스0. 주 실험 보고서: ../snr_privacy_summary_v1/REPORT_KO.md. 원자료: ../snr_privacy_v1/ 및 이 폴더의 C0.1/C1.0 하위폴더. 비교 수치 comparisons.json, near_noiseless_diagnostic.json, 전체비용 combined_cost.json.', '']
    (out/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    dump(out/'analysis_cost.json',dict(cpu_analysis_wall_seconds=time.perf_counter()-started))
    print(json.dumps(dict(diagnostic=diagnostic,comparisons=comparisons,combined_cost=totals),ensure_ascii=False))

if __name__=='__main__': main()
