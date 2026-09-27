from pathlib import Path
import json, statistics as st, math, hashlib
from collections import defaultdict
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def mean(xs):return st.mean(xs)
def sd(xs):
    xs=list(xs)
    return st.stdev(xs) if len(xs)>1 else 0.


def main():
    runs=[json.loads((OUT/f'seed{s}_complete.json').read_text(encoding='utf-8')) for s in [381,382,383]]
    rows=[r for run in runs for r in run['rows']]
    assert len(rows)==198
    groups=defaultdict(list)
    for row in rows:groups[(row['mode'],row['normalization'],row['step'],row['beta'])].append(row)
    summary=[]
    for key,group in groups.items():
        mode,norm,step,beta=key;seed_means={}
        for seed in [381,382,383]:
            items=[r for r in group if r['seed']==seed]
            assert len(items)==(4 if mode.startswith('ota') else 1)
            seed_means[str(seed)]={m:mean(r['metrics'][m] for r in items) for m in items[0]['metrics']}
        means={m:mean(x[m] for x in seed_means.values()) for m in group[0]['metrics']}
        sds={m:sd(x[m] for x in seed_means.values()) for m in means}
        summary.append(dict(mode=mode,normalization=norm,step=step,beta=beta,metrics=means,std=sds,
            seed_metrics=seed_means,cost=mean(r['cost']['total_real_uses'] for r in group),
            cost_at20=mean(r['cost']['total_real_uses_at_20db'] for r in group),
            first_geometry={m:mean(r['first_geometry'][m] for r in group) for m in [
                'ideal_residual_norm','max_abs_retained_cosine','received_vs_ideal_cosine',
                'sketch_vs_exact_cosine','noise_to_signal_norm','step_norm','active_clients']},
            exploration_gate=means['forget_normalized_js_ref0']<.8 and
                all(x['forget_normalized_js_ref0']<1 for x in seed_means.values())))
    lookup={(x['mode'],x['normalization'],x['step'],x['beta']):x for x in summary}
    comparisons=[]
    for mode in ['exact','sketch','ota20','ota40']:
        for step in [1,10]:
            soft=lookup[(mode,'target_norm',step,.5)];full=lookup[(mode,'target_norm',step,1.)]
            diffs={str(seed):{f'ref{ref}':soft['seed_metrics'][str(seed)][f'forget_normalized_js_ref{ref}']-
                full['seed_metrics'][str(seed)][f'forget_normalized_js_ref{ref}'] for ref in [0,1]} for seed in [381,382,383]}
            comparisons.append(dict(mode=mode,step=step,soft_minus_full=diffs,
                relative_improvement=1-soft['metrics']['forget_normalized_js_ref0']/full['metrics']['forget_normalized_js_ref0'],
                all_seed_both_reference_improvement=all(v<0 for d in diffs.values() for v in d.values())))
    verify=dict(passed=True,rows=len(rows),groups=len(summary),models=99,
                max_peak_power=max(r['last_geometry']['power']['max_peak_power'] for r in rows),
                max_average_power=max(r['last_geometry']['power']['max_average_power'] for r in rows),
                max_exact_retained_cosine=max(r['last_geometry']['max_abs_retained_cosine'] for r in rows if r['mode']=='exact' and r['beta']==1),
                max_mac_error=max(r['last_geometry']['mac_max_error'] for r in rows))
    assert verify['max_peak_power']<4.00001 and verify['max_average_power']<1.00001
    assert verify['max_exact_retained_cosine']<1e-5 and verify['max_mac_error']<2e-6
    beta0_differences=[]
    for seed in [381,382,383]:
        for step in [1,10]:
            a=[r for r in rows if r['seed']==seed and r['mode']=='exact' and r['beta']==0 and r['step']==step]
            assert len(a)==2
            left=a[0]['metrics']['forget_normalized_js_ref0'];right=a[1]['metrics']['forget_normalized_js_ref0']
            beta0_differences.append(dict(seed=seed,step=step,absolute=abs(left-right),relative=abs(left-right)/max(abs(left),abs(right),1e-12)))
            assert math.isclose(left,right,rel_tol=1e-6,abs_tol=1e-7)
    verify['beta0_float32_normalization_differences']=beta0_differences
    env=json.loads((OUT/'environment.json').read_text())
    assert env['protocol_sha256']==hashlib.sha256((ROOT/'PROTOCOL_KO.md').read_bytes()).hexdigest()
    assert env['code_sha256']==hashlib.sha256((ROOT/'run_experiment.py').read_bytes()).hexdigest()
    write(OUT/'verification.json',verify)
    write(OUT/'summary.json',dict(groups=summary,comparisons=comparisons,verification=verify))
    lines=['**직교성 필요 여부 — GPU 실험 결과, 2026-09-27**','',
        '새 seed381/382/383, FashionMNIST, client0 전체 삭제. 동일 UCE loss·gradient minibatch·학습률·target-gradient norm으로 projection 강도만0/.5/1로 변경했다. Plain FedAvg source150rounds와 seed별 독립 retained reference2개를 새로 학습했다. 99개 삭제 trajectory의 step1/10,총198개 평가를 수행했다.','',
        '오차는 reference0 대비 forget JS / no-op JS다. 1은 미삭제 상태,0은 해당 reference의 예측과 같음이다. 표는 channel noise를 seed 안에서 먼저 평균한 뒤 3seed 평균±표준편차다. 데이터·학습 seed는3개이며 noise4개나 reference2개를 독립 표본으로 부풀리지 않는다.','']
    for step in [1,10]:
        lines += [f'**A 참여 {step}회: 동일 step norm 주 비교**','',
            '| 전달 조건 | 투영 없음 β=0 | 부분 β=.5 | 완전 β=1 | .5의 1 대비 개선 |',
            '|---|---:|---:|---:|---:|']
        for mode in ['exact','sketch','ota20','ota40']:
            gg=[lookup[(mode,'target_norm',step,beta)] for beta in [0.,.5,1.]]
            vals=[f"{g['metrics']['forget_normalized_js_ref0']:.4f} ± {g['std']['forget_normalized_js_ref0']:.4f}" for g in gg]
            imp=1-gg[1]['metrics']['forget_normalized_js_ref0']/gg[2]['metrics']['forget_normalized_js_ref0']
            lines.append('|'+mode+'|'+'|'.join(vals)+f'|{100*imp:.1f}%|')
        lines.append('')
    lines += ['**대체 reference1: 평균 정규화 forget JS**','','| 참여 | 전달 | β=0 | β=.5 | β=1 |','|---:|---|---:|---:|---:|']
    for step in [1,10]:
        for mode in ['exact','sketch','ota20','ota40']:
            vals=[lookup[(mode,'target_norm',step,beta)]['metrics']['forget_normalized_js_ref1'] for beta in [0.,.5,1.]]
            lines.append(f'|{step}|{mode}|'+ '|'.join(f'{v:.4f}' for v in vals)+'|')
    lines += ['','**재정규화 없는 raw-scale 진단 (exact oracle)**','','| 참여 | β=0 | β=.5 | β=1 |','|---:|---:|---:|---:|']
    for step in [1,10]:
        vals=[lookup[('exact','raw',step,beta)]['metrics']['forget_normalized_js_ref0'] for beta in [0.,.5,1.]]
        lines.append(f'|{step}|'+ '|'.join(f'{v:.4f}' for v in vals)+'|')
    lines += ['','**첫 step의 신호 기하 (3seed 및 noise 평균)**','',
              '| 조건 | β | 이상적 residual norm | 수신/이상적 cosine | noise/signal norm | 최대 잔존 cosine |',
              '|---|---:|---:|---:|---:|---:|']
    for mode in ['exact','sketch','ota20','ota40']:
        for beta in [0.,.5,1.]:
            g=lookup[(mode,'target_norm',1,beta)]['first_geometry']
            lines.append(f"|{mode}|{beta}|{g['ideal_residual_norm']:.4f}|{g['received_vs_ideal_cosine']:.4f}|{g['noise_to_signal_norm']:.4f}|{g['max_abs_retained_cosine']:.4f}|")
    lines += ['','**계산·정보·통신 해석**','',
        '- Exact는 full-gradient oracle로서 정보 제약을 위반하는 계산 대조군이다. Sketch는2048차원16bit 관계 정보와 계수 방송으로 가중 OTA 합산한다. 같은 모델공간 projection을 비교하며 CDMA/OFDMA를 도입하지 않았다.',
        '- β=0은 잔존 송신계수가0이므로 A-only 신호가 된다. 좋은 숫자가 나와도 프라이버시를 만족하는 AirComp 해법으로 채택하지 않는다. β=.5/1도 sketch/norm/aggregate 공개가 있으며 full-transcript 비복원이나 DP 보장은 없다.',
        '- Step1만 A의 마지막1회 참여 조건이다. Step10은 A가10번 gradient를 계산하고 참여한 확장이며 one-shot으로 표현하지 않는다.',
        '- 모든 noisy 조건은 coherent real MAC,정확한CSI,평균power1/peak4,AWGN이다. 40dB는20dB보다 좋은 채널이며 동일 비용의 개선 기법이 아니다. 반복은1이다.',
        '- Exact/raw-scale은 projection으로 step 자체가 작아지는 효과를 확인한다. 주 비교는 target norm을 맞춰 단순히 덜 움직여 좋은지와 방향 효과를 구분한다.',
        '- 사전 탐색 성공은 평균<.8 및3seed 모두<1이다. beta1보다 덜 나쁜 것과 삭제 성공을 구분한다. Loss-MIA가 source부터 약하면 privacy 증거로 쓰지 않는다.',
        '- 재학습 reference는 확률적이므로 second reference 결과도 공개했다. 둘과 가까워져도 전체 재학습 분포에 대한 certified unlearning 증명은 아니다.','',
        '**비용·성능 전체표 (target norm,평균)**','','| 참여 | 조건 | β | test acc % | forget acc % | JS 절대 | 총 real uses | MIA AUC | gate |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for step in [1,10]:
        for mode in ['exact','sketch','ota20','ota40']:
            for beta in [0.,.5,1.]:
                g=lookup[(mode,'target_norm',step,beta)];m=g['metrics']
                lines.append(f"|{step}|{mode}|{beta}|{100*m['test_accuracy']:.2f}|{100*m['forget_accuracy']:.2f}|{m['forget_js_ref0']:.6f}|{g['cost']:,.0f}|{m['loss_mia_auc']:.4f}|{g['exploration_gate']}|")
    lines += ['','**Source/reference와 전 데이터 gradient 진단**','','| seed | source test % | ref0 test % | ref1 test % | no-op forget JS | ref간 forget JS | CE 정상점 잔차비 | CE aggregate residual | UCE span residual |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        a=run['baseline'];g=a['full_data_geometry'];s=a['source']
        lines.append(f"|{run['seed']}|{100*s['test_accuracy']:.2f}|{100*a['reference0']['test_accuracy']:.2f}|{100*a['reference1']['test_accuracy']:.2f}|{s['forget_js_ref0']:.6f}|{a['reference_difference']['forget']:.6f}|{g['full_ce_stationarity_ratio']:.4f}|{g['ce_aggregate_residual_fraction']:.4f}|{g['uce_span_residual_fraction']:.4f}|")
    prep=sum(r['preparation']['seconds']+sum(x['seconds'] for x in r['reference_preparation']) for r in runs)
    case_seconds=sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('seed*_noise*.json'))
    lines += ['',f'새 모델9개 준비 GPU simulator 시간 합계 {prep:.1f}초, 삭제99trajectory 시간 합계 {case_seconds:.1f}초. 실제 RF latency/단말 latency는 아니다. 개별 비용 ledger와 source/reference 준비비용은 원자료에 보존했다.',
        '', '2차 손실 반례에서 β=0/.5/1의 삭제 잔여 제곱오차는0/.25/1이다. 실제 CNN의 UCE 결과를 이 반례로 미리 결론 내리지 않았다.',
        '', '선행 구성은 [FedOSD](https://arxiv.org/html/2412.20200v1)의 UCE/투영 및 [COTAF](https://arxiv.org/abs/2009.12787)의 송신전처리/공중합산에 연결된다. 이번은 기존 Legacy-Sketch-V1의 투영 강도 진단이며 새 알고리즘의 노벨티를 주장하지 않는다.',
        '', '조건은 PROTOCOL_KO.md, 전체 seed/noise 수치는 results/summary.json 및 results/seed*_complete.json, 검증은 results/verification.json에 있다.']
    (ROOT/'RESULT_TABLES_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    colors={'exact':'#555555','sketch':'#2376b7','ota20':'#d66b29','ota40':'#30966b'}
    for ax,step in zip(axes,[1,10]):
        for mode in ['exact','sketch','ota20','ota40']:
            gs=[lookup[(mode,'target_norm',step,bb)] for bb in BETAS]
            ax.plot(BETAS,[g['metrics']['forget_normalized_js_ref0'] for g in gs],marker='o',label=mode,color=colors[mode])
        ax.axhline(1,color='black',linestyle=':',label='No-op' if step==1 else None)
        ax.set(xlabel='Projection strength beta (0=none, 1=full)',ylabel='Normalized deletion JS (lower is better)',title=f'{step} deletion participation(s)',yscale='log',xticks=BETAS)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Same UCE objective and step norm | 3 fresh seeds | no recovery')
    fig.savefig(ROOT/'comparison.png',dpi=180);fig.savefig(ROOT/'comparison.pdf');plt.close(fig)
    print(json.dumps(dict(verification=verify,comparisons=comparisons),indent=2))


BETAS=[0.,.5,1.]
if __name__=='__main__':main()
