"""Audit the communication/privacy sweep and assess claims without changing settings."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    done=json.loads((OUT/'completion.json').read_text());rows=[];refs={};groups=defaultdict(list)
    dep=ROOT.parent/'research_20260928_cr_realdata/run_experiment.py'
    assert done['source_sha256']==sha(ROOT/'run_experiment.py')
    assert done['dependency_sha256']==sha(dep)
    assert done['protocol_sha256']==sha(ROOT/'PROTOCOL_KO.md')
    wf=json.loads((OUT/'waveform_checks.json').read_text())
    assert wf['passed'] and wf['source_sha256']==done['source_sha256']
    frame=json.loads((OUT/'full_frame_accounting.json').read_text())
    assert frame['passed'] and frame['main_source_sha256']==done['source_sha256']
    assert frame['audit_sha256']==sha(ROOT/'audit_full_frame.py')
    for p in sorted(OUT.glob('*_seed*_K*.json')):
        o=json.loads(p.read_text());assert len(o['rows'])==1200
        assert all(o[k]==done[k] for k in ['source_sha256','dependency_sha256','protocol_sha256'])
        rows+=o['rows'];r=o['rows'][0];refs[(r['dataset'],r['K'],r['seed'])]=o['references']
    assert len(rows)==done['expected_rows']==21600 and len(refs)==18
    assert len({tuple(r[x] for x in ['dataset','seed','K','rho_cap','condition','method','channel_draw']) for r in rows})==len(rows)
    for r in rows:
        assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))
        assert r['cost_max_block_energy']<=1+1e-10
        assert r['cost_payload_energy']<=r['cost_payload_energy_bound_conservative']*(1+1e-10)+1e-12
        # Independent ledger reconstruction uses integer frames, not reported total.
        n=int(r['cost_active']);rep=int(r['cost_repeats'])
        digits=math.ceil((480+64*n+5120)/(.5*math.log2(101)))
        expected=333*rep+9*n+9+digits+math.ceil(digits/8)
        assert r['cost_total_real']==expected
        assert abs(r['cost_deletion_ms_at_1M']-expected/1000)<1e-10
        if r['condition']!='csi5':
            assert r['rho_all']<=r['rho_cap']+1e-9
            assert r['statistic_KL_matched_variance']<1e-18
            assert r['coupled_head_relative_error_max']<1e-9
            assert r['counterfactual_A_impact_mse']<1e-18
            if r['method']=='A-Only':
                assert abs(r['statistic_KL_original_reference']-296/2*(1-math.log(2)))<1e-10
            else:assert r['statistic_KL_original_reference']<1e-18
        groups[(r['dataset'],r['K'],r['condition'],r['rho_cap'],r['method'])].append(r)
    numbers=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k not in ['K','seed','channel_draw']]
    summary={}
    for key,rr in groups.items():
        o={k:st.mean(r[k] for r in rr) for k in numbers}
        for metric in ['accuracy','cost_total_real','cost_deletion_ms_at_1M','cost_accounted_tx_energy']:
            o[metric+'_seed_sd']=st.stdev(st.mean(r[metric] for r in rr if r['seed']==s) for s in done['seeds'])
        times=[r['cost_deletion_ms_at_1M'] for r in rr]
        o.update(delay_p50_ms=float(np.percentile(times,50)),delay_p95_ms=float(np.percentile(times,95)),
                 delay_over_50ms_fraction=st.mean(t>50 for t in times),
                 privacy_cap_violation_fraction=st.mean(r['rho_all']>r['rho_cap']+1e-9 for r in rr),
                 max_rho=max(r['rho_all'] for r in rr),max_matched_KL=max(r['statistic_KL_matched_variance'] for r in rr))
        o.update(frame['groups']['|'.join(map(str,key))])
        summary[key]=o
    reference={}
    for ds in ['FashionMNIST','MNIST']:
        for k in [10,100,1000]:
            sub=[v for (dd,kk,ss),v in refs.items() if dd==ds and kk==k]
            reference[(ds,k)]={x:st.mean(o[x] for o in sub) for x in sub[0]}
    paired={};decisions=[]
    for key,rr in groups.items():
        ds,k,ch,rho,method=key
        if method=='Independent':continue
        baseline={(r['seed'],r['channel_draw']):r for r in groups[(ds,k,ch,rho,'Independent')]}
        a=summary[key];b=summary[(ds,k,ch,rho,'Independent')]
        p={x+'_ratio':a[x]/b[x] for x in ['cost_total_real','cost_payload_energy','cost_accounted_tx_energy','lifecycle_real','lifecycle_accounted_tx_energy']}
        p.update(mean_accuracy_delta_pp=100*(a['accuracy']-b['accuracy']),
                 paired_communication_win_fraction=st.mean(r['cost_total_real']<baseline[(r['seed'],r['channel_draw'])]['cost_total_real'] for r in rr),
                 paired_symbol_ratio_median=st.median(r['cost_total_real']/baseline[(r['seed'],r['channel_draw'])]['cost_total_real'] for r in rr))
        for kapp in ['0.0','1e-06','0.0001','0.01']:
            metric='cost_tx_plus_payload_circuit_'+kapp
            p['tx_and_payload_circuit_ratio_'+kapp]=a[metric]/b[metric]
        saved_exposure=b['cost_payload_client_active_real']-a['cost_payload_client_active_real']
        extra_tx=a['cost_accounted_tx_energy']-b['cost_accounted_tx_energy']
        p['circuit_break_even_kappa_from_group_means']=max(0.,extra_tx)/saved_exposure if saved_exposure>0 else None
        paired[key]=p
        if k==1000 and ch=='rayleigh' and method.startswith('CR-'):
            clean=reference[(ds,k)]['clean_accuracy'];gap=100*(clean-a['accuracy'])
            seed_gaps=[100*(refs[(ds,k,s)]['clean_accuracy']-st.mean(r['accuracy'] for r in rr if r['seed']==s)) for s in done['seeds']]
            decisions.append(dict(dataset=ds,method=method,nominal_rho_cap=rho,
                                  full_frame_rho_cap=a['full_frame_rho_cap'],
                                  full_frame_epsilon_cap=a['full_frame_epsilon_cap'],
                                  clean_gap_pp=gap,seed_clean_gaps_pp=seed_gaps,
                                  communication_reduction_pct=100*(1-p['cost_total_real_ratio']),
                                  mean_criterion_pass=gap<=5 and p['cost_total_real_ratio']<=.9,
                                  all_seeds_within_5pp=all(g<=5 for g in seed_gaps)))
    accepted={}
    for method in ['CR-Fixed','CR-Power','CR-Uses']:
        accepted[method]=None
        for rho in [.5,1.,2.,4.,8.]:
            match=[o for o in decisions if o['method']==method and o['nominal_rho_cap']==rho]
            if all(o['mean_criterion_pass'] for o in match):accepted[method]=rho;break
    verification=dict(passed=True,rows=len(rows),group_count=len(groups),config_count=len(refs),
                      counted_waveform_cases=len(wf['checks']),max_waveform_variance_relative_error=max(r['relative_variance_error'] for r in wf['checks']),
                      source_sha256=done['source_sha256'],dependency_sha256=done['dependency_sha256'],
                      protocol_sha256=done['protocol_sha256'],summary_sha256=sha(__file__),
                      first_passing_nominal_rho_both_datasets=accepted,
                      first_passing_full_frame_rho_both_datasets={k:2*v if v is not None else None for k,v in accepted.items()},
                      full_frame_accounting_sha256=sha(OUT/'full_frame_accounting.json'),decisions=decisions)
    write(OUT/'verification.json',verification)
    write(OUT/'summary.json',dict(groups={'|'.join(map(str,k)):v for k,v in summary.items()},
                                references={'|'.join(map(str,k)):v for k,v in reference.items()},
                                paired={'|'.join(map(str,k)):v for k,v in paired.items()},first_passing=accepted))
    lines=['# 통신·프라이버시 완화 결과표','',
           '지연은 시뮬레이션의 frame real symbol 수를 1M real/s로 환산한 값이다. SDR 실측이 아니다. 3 seed, K1000, 정확 CSI Rayleigh 결과다. A-Only의 final variance는 다른 방법의 2배이고 원래 noisy-retrain reference와 일치하지 않는다.','',
           '**프라이버시 보정:** 복사한 cyclic prefix도 서버 관측이다. 아래 rho/epsilon은 prefix를 포함한 보수적 전체 관측 상한이다. 동결 raw JSON/PROTOCOL의 nominal rho는 이 값의 절반이며 epsilon도 그대로 쓰면 안 된다. 근거는 README와 audit_full_frame.py를 참조한다.','',
           '| Dataset | Full-frame rho cap | Full-frame epsilon cap | Independent acc / ms | CR-Power acc / ms | A-Only acc / ms | CR 통신 절감 |','|---|---:|---:|---|---|---|---:|']
    for ds in ['FashionMNIST','MNIST']:
        for rho in [.5,1.,2.,4.,8.]:
            cells=[]
            for m in ['Independent','CR-Power','A-Only']:
                o=summary[(ds,1000,'rayleigh',rho,m)]
                cells.append(f'{100*o["accuracy"]:.2f}% / {o["cost_deletion_ms_at_1M"]:.3f}')
            p=paired[(ds,1000,'rayleigh',rho,'CR-Power')]
            lines.append(f'| {ds} | {2*rho} | {o["full_frame_epsilon_cap"]:.2f} | '+ ' | '.join(cells)+f' | {100*(1-p["cost_total_real_ratio"]):.2f}% |')
    lines+=['','## Full-frame rho=4 (nominal rho=2) 세부 비교','',
            '| Dataset | 방법 | Accuracy ± seed SD | 지연 mean / p50 / p95(ms) | >50ms | 원래 reference KL | oracle AUC |','|---|---|---:|---:|---:|---:|---:|']
    for ds in ['FashionMNIST','MNIST']:
        for m in ['Independent','CR-Fixed','CR-Power','CR-Uses','A-Only']:
            o=summary[(ds,1000,'rayleigh',2.,m)]
            lines.append(f'| {ds} | {m} | {100*o["accuracy"]:.3f} ± {100*o["accuracy_seed_sd"]:.3f} | {o["cost_deletion_ms_at_1M"]:.3f} / {o["delay_p50_ms"]:.3f} / {o["delay_p95_ms"]:.3f} | {o["delay_over_50ms_fraction"]:.3f} | {o["statistic_KL_original_reference"]:.4g} | {o["full_frame_oracle_AUC"]:.4f} |')
    lines+=['','## 모든 privacy 단계의 채택 기준','',
            '| Dataset | 방법 | Full-frame rho | Clean gap(%p) | 통신 절감(%) | 평균 기준 | 모든 seed utility 기준 |','|---|---|---:|---:|---:|---|---|']
    for o in decisions:
        lines.append(f'| {o["dataset"]} | {o["method"]} | {o["full_frame_rho_cap"]} | {o["clean_gap_pp"]:.3f} | {o["communication_reduction_pct"]:.3f} | {o["mean_criterion_pass"]} | {o["all_seeds_within_5pp"]} |')
    lines+=['','## 전체 조건','',
            '| Dataset | K | Channel | Full-frame rho cap | 방법 | acc(%) | ms | TX energy | Full-frame rho bound max | cap초과 fraction |','|---|---:|---|---:|---|---:|---:|---:|---:|---:|']
    for key,o in sorted(summary.items()):
        ds,k,ch,rho,m=key
        lines.append(f'| {ds} | {k} | {ch} | {2*rho} | {m} | {100*o["accuracy"]:.3f} | {o["cost_deletion_ms_at_1M"]:.3f} | {o["cost_accounted_tx_energy"]:.4g} | {o["full_frame_rho_all_bound_max"]:.3f} | {o["privacy_cap_violation_fraction"]:.3f} |')
    (ROOT/'RESULT_TABLES_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),constrained_layout=True)
    colors={'Independent':'#7a8fa5','CR-Power':'#168b68','A-Only':'#cb7133'}
    for ax,ds in zip(axes[0],['FashionMNIST','MNIST']):
        for method in ['Independent','CR-Power','A-Only']:
            oo=[summary[(ds,1000,'rayleigh',r,method)] for r in [.5,1.,2.,4.,8.]]
            ax.plot([o['cost_deletion_ms_at_1M'] for o in oo],[o['accuracy']*100 for o in oo],'-o',color=colors[method],label=method)
            for ii in [0,2,4]:
                o=oo[ii];ax.annotate(f'e={o["full_frame_epsilon_cap"]:.1f}',(o['cost_deletion_ms_at_1M'],o['accuracy']*100),
                                     xytext=(4,5 if method!='Independent' else -10),textcoords='offset points',fontsize=7,color=colors[method])
        ax.axhline(reference[(ds,1000)]['clean_accuracy']*100,color='k',ls=':',label='Clean retained')
        ax.set(xscale='log',xlabel='Modeled deletion latency (ms, 1M real/s)',ylabel='Test accuracy (%)',title=ds+' / K1000 / exact CSI')
        ax.set_xlim(1.8,110)
        ax.xaxis.set_major_locator(FixedLocator([2,5,10,20,40,80]))
        ax.xaxis.set_major_formatter(FixedFormatter(['2','5','10','20','40','80']))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.2)
    for ax,metric,title,ylabel in [(axes[1,0],'saving','Matched-output communication gain','Real-symbol reduction (%)'),
                                   (axes[1,1],'auc','Privacy cost of weaker noise','Oracle two-hypothesis AUC')]:
        for ds,marker in [('FashionMNIST','o'),('MNIST','s')]:
            oo=[summary[(ds,1000,'rayleigh',r,'CR-Power')] for r in [.5,1.,2.,4.,8.]]
            yy=[100*(1-paired[(ds,1000,'rayleigh',r,'CR-Power')]['cost_total_real_ratio']) for r in [.5,1.,2.,4.,8.]] if metric=='saving' else [o['full_frame_oracle_AUC'] for o in oo]
            ax.plot([o['full_frame_epsilon_cap'] for o in oo],yy,'-'+marker,label=ds)
        ax.set(xlabel='Full-frame epsilon upper bound (delta = 1e-5)',ylabel=ylabel,title=title)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('CR-Air communication / utility / privacy trade-off: fixed 16-d head, 3 seeds',fontsize=13)
    fig.savefig(ROOT/'tradeoff.png',dpi=170);fig.savefig(ROOT/'tradeoff.pdf')
    print(json.dumps({'first_passing_nominal_rho':accepted,'row_count':len(rows),'waveforms':len(wf['checks'])},indent=2))
    for ds in ['FashionMNIST','MNIST']:
        print(ds,'rho2',json.dumps(paired[(ds,1000,'rayleigh',2.,'CR-Power')]))


if __name__=='__main__':main()
