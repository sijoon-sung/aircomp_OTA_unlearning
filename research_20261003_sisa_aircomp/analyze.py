"""Independent CPU audit and Korean report. Does not start a CUDA worker."""
from pathlib import Path
import csv, hashlib, json, math, sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def audit_case(run,row):
    folder=run/row['case']
    cfg=read(folder/'config.json')
    D=cfg['D']
    vectors={s:torch.load(folder/f'{s}_models.pt',map_location='cpu',weights_only=True) for s in ['source','reference','replay']}
    maxdiff=max(float((a-b).abs().max()) for a,b in zip(vectors['reference'],vectors['replay']))
    assert all(torch.equal(a,b) for a,b in zip(vectors['reference'],vectors['replay']))
    assert all(torch.equal(vectors['source'][c],vectors['replay'][c]) for c in range(cfg['K']) if c!=cfg['affected'])
    errors=[]
    metrics={}
    labels=np.load(folder/'labels.npz')
    probs={s:np.load(folder/f'{s}_probabilities.npz') for s in ['source','reference','replay']}
    for phase in ['source','reference','replay']:
        ledger=read(folder/f'{phase}_ledger.json')
        ev=ledger['events']
        if phase=='replay':
            assert len(ev)==cfg['rounds']
        else:
            assert len(ev)==cfg['rounds']*cfg['K']
        sums={k:0. for k in ['total_re','ul_reals','dl_bits','pilot_reals','control_bits','local_calls','gradient_samples','ul_signal_energy','ul_control_pilot_energy','norm_scalar_count']}
        for e in ev:
            expected_ids=[i for i in cfg['groups'][e['c']] if phase=='source' or i!=0]
            assert e['ids']==expected_ids and len(e['ids'])>=3
            if phase=='replay':
                assert e['c']==cfg['affected'] and 0 not in e['ids']
            expected_repeat=max(1,math.ceil(D*e['beta']**2*cfg['sigma2']/cfg['epsilon']))
            assert e['repeats']==expected_repeat
            assert e['ul_reals']==expected_repeat*D
            assert e['dl_bits']==32*D and e['gradient_samples']==128*len(expected_ids)
            assert e['pilot_reals']==8*len(expected_ids)
            adaptive=cfg['method'].endswith('norm')
            assert e['norm_scalar_count']==len(expected_ids)*adaptive
            assert e['control_bits']==64*len(expected_ids)+64+32*len(expected_ids)*adaptive
            re=e['ul_reals']+e['pilot_reals']+(e['control_bits']+e['dl_bits'])/2
            assert re==e['total_re']
            assert e['max_client_symbol_power']<=1.00001
            mse=D*e['beta']**2*cfg['sigma2']/e['repeats']
            assert abs(mse-e['expected_mse'])<1e-12 and mse<=cfg['epsilon']*(1+1e-12)
            for s in sums:sums[s]+=e[s]
        for s,v in sums.items():
            assert math.isclose(v,ledger['cost'][s],rel_tol=1e-12,abs_tol=1e-6),(row['case'],phase,s)
        assert ledger['cost']['tdma_delay_units']==ledger['cost']['elastic_fdma_delay_units']==sums['total_re']
        fixed=sum(cfg['K']*max(e['total_re'] for e in ev if e['t']==t) for t in range(cfg['rounds']))
        assert math.isclose(fixed,ledger['cost']['fixed_fdma_delay_units'])
        for s in ['dev','test']:
            value=float(np.mean(probs[phase][s].argmax(1)==labels[s]))
            metrics[f'{phase}_{s}_accuracy']=value
            if phase=='replay':
                errors.append(abs(value-row[f'{s}_accuracy']))
    for s in ['dev','test','forgotten','retain']:
        assert np.array_equal(probs['reference'][s],probs['replay'][s])
    assert max(errors)==0
    return dict(case=row['case'],max_parameter_difference=maxdiff,max_accuracy_recompute_error=max(errors),
                all_checks_pass=True,**metrics)

def paired(rows,phase):
    rs=[r for r in rows if r['phase']==phase and r['K']>1]
    pairs=[]
    for seed in sorted({r['seed'] for r in rs}):
        b=next(r for r in rs if r['seed']==seed and r['method']=='random_norm')
        c=next(r for r in rs if r['seed']==seed and r['method']=='channel_norm')
        g=next(r for r in rows if r['phase']==phase and r['seed']==seed and r['K']==1)
        pairs.append(dict(seed=seed,K=c['K'],random_accuracy=b['test_accuracy'],channel_accuracy=c['test_accuracy'],
             global_accuracy=g['test_accuracy'],accuracy_delta_pp=100*(c['test_accuracy']-b['test_accuracy']),
             ul_ratio=c['delete_cost']['ul_reals']/b['delete_cost']['ul_reals'],
             total_ratio=c['delete_cost']['total_re']/b['delete_cost']['total_re'],
             energy_ratio=(c['delete_cost']['ul_signal_energy']+c['delete_cost']['ul_control_pilot_energy'])/(b['delete_cost']['ul_signal_energy']+b['delete_cost']['ul_control_pilot_energy']),
             init_ratio=c['source_cost']['total_re']/b['source_cost']['total_re'],
             global_delete_re_ratio=c['delete_cost']['total_re']/g['delete_cost']['total_re'],
             global_delete_local_ratio=c['delete_cost']['local_calls']/g['delete_cost']['local_calls'],
             same_sisa_full_re_ratio=c['delete_cost']['total_re']/c['reference_cost']['total_re'],
             global_init_re_ratio=c['source_cost']['total_re']/g['source_cost']['total_re'],
             min_active=c['delete_cost']['min_active_clients'],
             source_random=b['source_cost']['total_re'],source_channel=c['source_cost']['total_re'],source_global=g['source_cost']['total_re'],
             delete_random=b['delete_cost']['total_re'],delete_channel=c['delete_cost']['total_re'],delete_global=g['delete_cost']['total_re']))
    return pairs

def main():
    run=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'run_v1')
    assert (run/'complete.json').exists()
    out=ROOT/(sys.argv[2] if len(sys.argv)>2 else 'summary_v1')
    out.mkdir(exist_ok=False)
    rows=read(run/'results.json')
    selection=read(run/'selection.json')
    audits=[audit_case(run,r) for r in rows]
    write(out/'audit.json',dict(cases=len(audits),all_pass=all(a['all_checks_pass'] for a in audits),rows=audits))
    confirmation=paired(rows,'confirm')
    stress=paired(rows,'stress')
    candidate=[r for r in rows if r['phase']=='confirm' and r['method']=='channel_norm']
    baseline=[r for r in rows if r['phase']=='confirm' and r['K']>1 and r['method']=='random_norm']
    ratio_of_means=lambda p:sum(r['delete_cost'][p] for r in candidate)/sum(r['delete_cost'][p] for r in baseline)
    total_ratio=ratio_of_means('total_re')
    avg_delta=float(np.mean([p['accuracy_delta_pp'] for p in confirmation]))
    wins=sum(p['total_ratio']<1 for p in confirmation)
    success=total_ratio<=.9 and avg_delta>=-2 and wins>=2
    decisions=dict(chosen_K=selection['chosen_K'],eligible_selection=selection['eligible_selection'],
       additional_grouping_success=success,confirmation_ratio_of_mean_total_re=total_ratio,
       confirmation_ratio_of_mean_ul_re=ratio_of_means('ul_reals'),
       confirmation_ratio_of_mean_total_re_dl6=ratio_of_means('total_re_dl6'),
       confirmation_mean_accuracy_delta_pp=avg_delta,total_re_wins=wins,
       scope='3-seed baseband pilot, not a novelty proof or privacy certification')
    write(out/'decisions.json',decisions)
    write(out/'paired_results.json',dict(confirmation=confirmation,stress=stress))
    costs=[]
    for path in ROOT.glob('*/cost_total.json'):
        if path.parent.name.startswith('preflight') or path.parent==run:
            costs.append(dict(run=path.parent.name,**read(path)))
    for path in ROOT.glob('*/cost_interrupted.json'):
        costs.append(dict(run=path.parent.name,**read(path)))
    totalcost=dict(total_seconds=sum(c['seconds'] for c in costs),gpu_board_Wh=sum(c['gpu_board_Wh'] for c in costs),
                   completed_cases=sum(c['completed_cases'] for c in costs),external_payment=0,
                   disk_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
                   runs=[{k:v for k,v in c.items() if k!='samples'} for c in costs],
                   limits='Board energy via 5-second samples; short preflight may have <2 samples. CPU audit/report not included; host/electricity cost not measured.')
    write(out/'experiment_cost.json',totalcost)
    flatrows=[]
    for r in rows:
        a={k:r[k] for k in ['case','phase','seed','K','method','dev_accuracy','test_accuracy','replay_exact','checkpoint_bytes','case_seconds']}
        for prefix,field in [('init','source_cost'),('full_sisa','reference_cost'),('delete','delete_cost')]:
            for metric in ['local_calls','gradient_samples','ul_reals','pilot_reals','control_bits','dl_bits','total_re','total_re_dl6','ul_signal_energy','ul_control_pilot_energy','norm_scalar_count','min_active_clients','seconds','fixed_fdma_delay_units','elastic_fdma_delay_units']:
                a[f'{prefix}_{metric}']=r[field][metric]
        flatrows.append(a)
    with (out/'all_metrics.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(flatrows[0]));writer.writeheader();writer.writerows(flatrows)

    lines=['# SISA-AirComp 1차 실험 결과','',
        '2026-10-03. 실제 신규 학습·전체 재학습·부분 replay를 실행했다. 기존 결과는 수정하지 않았다.', '',
        f"판단: **{'사전 기준에서 추가 grouping 개선 확인' if success else '사전 기준에서 추가 grouping 개선 미확인'}**. 선택 K={selection['chosen_K']}, dev 선택 적격={selection['eligible_selection']}. 문헌 최초성은 검증하지 않았다.", '',
        '## 조건과 개선 기전','',
        'FashionMNIST train12000/dev2000/test10000, 20 clients, Dirichlet .5, raw CNN, 120 rounds, local SGD 2x64, client 균등 평균. 전원 t=0 참여; client0 삭제. 해당 shard 초기 checkpoint부터 같은 120 rounds를 다시 수행한다. 이번 client 전체 삭제에서는 중간 slice checkpoint 절감이 없다.', '',
        '무작위/채널 기반 고정 배정 × 공개 clipping bound/현재 norm scalar gain 조절을 교차 비교했다. long-term channel amplitude -20~0dB, bounded per-round fading, 완전 CSI, 이상적 직교 shard 전송, aggregate squared noise norm 목표1e-4. 반복 전송 횟수는 이 목표와 전력 상한으로 정했다. 실제 무선 장비가 아닌 baseband simulation이다.', '',
        'random_norm은 channel_norm과 동일한 현재 norm 정보 및 power control을 사용하는 강한 비교군이다. 새로운 grouping의 성능은 이 둘로 판단하며, fixed-bound 대비 큰 차이를 신규 grouping 기여로 바꾸어 쓰지 않는다.', '',
        '## 실행 전 기준과 검증','',
        f'- 전체 {len(rows)}조건에서 source, A-free 모든 shard 재학습, 해당 shard replay를 각각 실행. CPU audit {len(audits)}/{len(audits)} 통과.',
        '- 모든 shard parameter와 reference/replay 예측 bitwise 동일. 무영향 shard 불변. 삭제 client 및 타 shard client의 replay 호출0.',
        '- 파형 reconstruction, 직교 bin leakage0, Monte Carlo noise variance, 모든 송신 power cap, ledger 및 accuracy 독립 재계산 통과.',
        '- 신규 개선 채택 기준: confirmation 평균 total RE 10% 이상 절감, 평균 정확도 감소2pp 이내, 3 seeds 중2개 이상 RE 감소. 테스트로 K를 재선택하지 않았다.', '',
        '## Screening: seed10601, dev 선택에만 사용','',
        '| K | 방식 | Dev % | Test % (선택 미사용) | 삭제 local calls | UL RE (million) | 전체 RE (million) | 초기 전체 RE (million) |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['phase']=='screen':
            d=r['delete_cost'];s=r['source_cost']
            lines.append(f"| {r['K']} | {r['method']} | {100*r['dev_accuracy']:.2f} | {100*r['test_accuracy']:.2f} | {d['local_calls']} | {d['ul_reals']/1e6:.3f} | {d['total_re']/1e6:.3f} | {s['total_re']/1e6:.3f} |")
    lines+=['',f"선택 점수: J=.5×(삭제 local calls/K1)+.5×(삭제 전체 RE/K1). channel_norm이 random_norm보다 dev2pp, K1보다5pp를 넘게 낮아지지 않는 K에서 최소 점수를 선택. K={selection['chosen_K']}를 confirmation 전에 고정했다. 상세값: ../run_v1/selection.json.",'',
        '## 새 seed 3개: 추가 grouping 개선','',
        '| Seed | Random 정확도 % | Channel 정확도 % | 차이 pp | UL 절감 % | 전체 RE 절감 % | 정규화 UL 에너지 변화 % |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for p in confirmation:
        lines.append(f"| {p['seed']} | {100*p['random_accuracy']:.2f} | {100*p['channel_accuracy']:.2f} | {p['accuracy_delta_pp']:+.2f} | {100*(1-p['ul_ratio']):.2f} | {100*(1-p['total_ratio']):.2f} | {100*(p['energy_ratio']-1):+.2f} |")
    lines+=['',f"주 판단의 분모는 random_norm. 평균 자원량 비율로 UL 절감 {100*(1-decisions['confirmation_ratio_of_mean_ul_re']):.2f}%, 전체 RE 절감 {100*(1-total_ratio):.2f}%, 평균 정확도 차이 {avg_delta:+.2f}pp. RE 개선 seed {wins}/3. DL 효율을2에서6 bits/RE로 바꾼 ledger 민감도에서 전체 RE 절감 {100*(1-decisions['confirmation_ratio_of_mean_total_re_dl6']):.2f}%. 이는 채널 모델을 다시 실행한 결과가 아니다.",'',
        '## 단일 모델 재학습과 동일 SISA 재학습의 분모 구분','',
        '| Seed | K1 정확도 % | 제안 정확도 % | 삭제 계산 / K1 | 삭제 RE / K1 | 삭제 RE / 동일 SISA 전체 | 초기 RE / K1 | 최소 삭제 후 인원 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in confirmation:
        lines.append(f"| {p['seed']} | {100*p['global_accuracy']:.2f} | {100*p['channel_accuracy']:.2f} | {p['global_delete_local_ratio']:.3f} | {p['global_delete_re_ratio']:.3f} | {p['same_sisa_full_re_ratio']:.3f} | {p['global_init_re_ratio']:.3f} | {p['min_active']} |")
    lines+=['','K1은 다른 학습 알고리즘이므로 정확 언러닝 기준이 아니다. 동일 SISA 전체 재학습의 reference와 bitwise 같다는 결과를 단일 전역 모델 재학습과 같다고 해석하지 않는다.','',
        '## 초기 비용을 포함한 lifecycle projection','',
        '| 가정한 삭제 횟수 | Channel / Random 전체 RE | Channel / K1 전체 RE |',
        '|---:|---:|---:|']
    for n in [1,10,100]:
        a=sum(p['source_channel']+n*p['delete_channel'] for p in confirmation)
        b=sum(p['source_random']+n*p['delete_random'] for p in confirmation)
        c=sum(p['source_global']+n*p['delete_global'] for p in confirmation)
        lines.append(f'| {n} | {a/b:.3f} | {a/c:.3f} |')
    lines+=['','위 표는 각 삭제가 이번 측정과 같은 비용이라는 대입 계산이다. 실제 누적·순차 삭제나 shard 고갈을 실행하지 않았다.','',
        '## 채널-데이터 상관 스트레스 (별도 seed)','',
        '| Seed | Random 정확도 % | Channel 정확도 % | 차이 pp | 전체 RE 절감 % |',
        '|---:|---:|---:|---:|---:|']
    for p in stress:
        lines.append(f"| {p['seed']} | {100*p['random_accuracy']:.2f} | {100*p['channel_accuracy']:.2f} | {p['accuracy_delta_pp']:+.2f} | {100*(1-p['total_ratio']):.2f} |")
    lines+=['','dominant label과 채널 순위를 연결해 채널 grouping이 데이터 분포까지 몰리게 했다. 이는 알고리즘에 label 정보를 준 것이 아니라 evaluator의 환경 생성이다. 주 confirmation과 평균을 섞지 않는다.','',
        '## 정보·전송·비용 해석','',
        '- norm 방식은 client당 round마다 현재 clipped update norm32bits를 추가로 노출한다. CSI/ID도 사용한다. 개별 gradient/history를 서버에 저장하지 않는다. 최소집계 인원은 보장된 프라이버시 지표가 아니며 MIA/gradient inversion 실험은 이번 범위에 없다.',
        '- 고정 FDMA와 elastic FDMA 차이에는 비어 있는 대역폭을 회수하는 효과가 있다. 그러나 work-conserving TDMA와 elastic FDMA의 유효 RE와 이상적 통신 시간은 이 모델에서 동일하다. 동시 전송 자체의 spectral gain을 주장하지 않는다.',
        '- 모든 RE는 전체 공유 시간-주파수 자원을 센 값. 2 bits/RE의 control/DL 가정, pilots 포함. 본문 time은 GPU simulation wall time과 통신 ledger를 구분한다. 실제 ms 지연이나 RF Joule은 측정하지 않았다.',
        '- normalized UL energy에는 analog waveform와 단위 symbol energy의 control/pilot를 포함한다. 수신기/회로/BS DL 에너지 제외. 전송 횟수 절감이 같은 비율의 에너지 절감은 아니다.',
        '- exactness 검증은 같은 고정 routing, RNG coupling, 이상적 채널 simulator의 같은 알고리즘 비교다. 실제 RF에서의 분포 인증, metadata 자체 삭제, 공격 방어 보장이 아니다.', '',
        '## 총 실행 비용','',
        f"- preflight와 본실험 포함 측정 wall time {totalcost['total_seconds']:.2f}s ({totalcost['total_seconds']/60:.2f}min). GPU board energy {totalcost['gpu_board_Wh']:.4f}Wh. 유료 외부 서비스 사용0. host 전체 소비전력/전기요금 미측정.",
        f"- 결과 폴더 파일 크기(보고서 생성 전) {totalcost['disk_bytes']/2**20:.2f}MiB. 각 실행의 power samples는 cost_total.json에 보존. preflight는5초 미만이라 에너지 적분 표본이 부족할 수 있다.",
        '- 단일 신규 GPU worker로 직렬 실행. CPU audit/report 생성 시간은 위 GPU 실험비에 포함하지 않았다.', '',
        '## 다음 판단','',
        ('같은 정보 조건의 random_norm 대비 확인된 범위에서 channel_norm을 후속 후보로 유지한다. 채널-데이터 상관, 여러 삭제 요청, 이탈/CSI 오차와 초기 비용을 포함한 확장이 필요하다.' if success else
         '현재 channel_norm을 전체 통신 비용의 검증된 개선으로 채택하지 않는다. UL-only 이득, 정확도 손실, 에너지 변화를 분리해서 다음 설계를 정한다. 단순 대역폭 재배정을 독립적인 신규 개선으로 주장하지 않는다.'), '',
        '## 원자료','',
        '- ../PREREGISTRATION.md: 실행 전 조건/기준. ../experiment.py: 실행 코드.',
        '- ../run_v1/: 각 source/reference/replay 모델, 확률, partition, labels, event ledger, 코드 hash, selection, completion.',
        '- all_metrics.csv / paired_results.json / decisions.json: 수치와 결정.',
        '- audit.json: CPU 독립 검증. experiment_cost.json: 총비용.',
        '- frontier.png / confirmation.png: 그림. scientific pilot이며 일반적인 우월성을 확정하지 않는다.','']
    (out/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for method,color in [('random_norm','#64748b'),('channel_norm','#0f766e')]:
        rr=sorted([r for r in rows if r['phase']=='screen' and r['method']==method and r['K']>1],key=lambda r:r['K'])
        ks=[r['K'] for r in rr]
        axes[0].plot(ks,[r['dev_accuracy']*100 for r in rr],'o-',label=method,color=color)
        axes[1].plot(ks,[r['delete_cost']['total_re']/1e6 for r in rr],'o-',label=method,color=color)
        axes[2].plot(ks,[r['source_cost']['total_re']/1e6 for r in rr],'o-',label=method,color=color)
    for ax,title in zip(axes,['Dev accuracy (%)','Deletion resource elements (million)','Initial training RE (million)']):
        ax.set_xlabel('Number of shards K');ax.set_title(title);ax.set_xticks([2,4,5]);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Screening: fixed rounds, full-client deletion, seed 10601',y=1.01)
    fig.tight_layout();fig.savefig(out/'frontier.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    x=np.arange(len(confirmation));width=.25
    for offset,(field,label,color) in enumerate([('ul_ratio','UL RE','#0284c7'),('total_ratio','UL + DL + control','#0f766e'),('energy_ratio','Normalized UL energy','#d97706')]):
        axes[0].bar(x+(offset-1)*width,[p[field] for p in confirmation],width,label=label,color=color)
    axes[0].axhline(1,color='black',linewidth=1);axes[0].set_xticks(x,[str(p['seed']) for p in confirmation]);axes[0].set_ylabel('Channel / random grouping');axes[0].legend(fontsize=8)
    axes[1].bar(x-.18,[p['random_accuracy']*100 for p in confirmation],.36,label='Random',color='#64748b')
    axes[1].bar(x+.18,[p['channel_accuracy']*100 for p in confirmation],.36,label='Channel',color='#0f766e')
    axes[1].set_xticks(x,[str(p['seed']) for p in confirmation]);axes[1].set_ylabel('Test accuracy (%)');axes[1].set_ylim(0,100);axes[1].legend()
    fig.suptitle(f"Held-out confirmation, K={selection['chosen_K']}; both use norm adaptation")
    fig.tight_layout();fig.savefig(out/'confirmation.png',dpi=160);plt.close(fig)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'PREREGISTRATION.md',ROOT/'experiment.py',ROOT/'analyze.py',out/'REPORT_KO.md',out/'all_metrics.csv',out/'audit.json']}
    write(out/'manifest.json',manifest)
    print(json.dumps(dict(decisions=decisions,cost=totalcost,confirmation=confirmation,stress=stress),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
