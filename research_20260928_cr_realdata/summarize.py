"""Audit saved outputs and generate complete tables and scientific plots."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    completion=json.loads((OUT/'completion.json').read_text())
    assert completion['completed']
    assert completion['code_sha256']==sha(ROOT/'run_experiment.py')
    assert completion['protocol_sha256']==sha(ROOT/'PROTOCOL_KO.md')
    rows=[];refs={};bygroup=defaultdict(list)
    for p in sorted(OUT.glob('*_seed*_K*.json')):
        o=json.loads(p.read_text())
        assert o['code_sha256']==completion['code_sha256']
        assert o['protocol_sha256']==completion['protocol_sha256']
        assert len(o['rows'])==320
        rows.extend(o['rows'])
        first=o['rows'][0]
        refs[(first['dataset'],first['K'],first['seed'])]=o['references']
    assert len(refs)==18 and len(rows)==completion['expected_rows']==5760
    assert len({tuple(r[x] for x in ['dataset','seed','K','condition','method','channel_draw']) for r in rows})==len(rows)
    for r in rows:
        for val in r.values():
            if isinstance(val,float):assert math.isfinite(val)
        assert r['noise_draws']==8 and r['utility_noise_draws']==2
        assert r['cost_max_block_energy']<=1+1e-10
        assert r['cost_payload_energy']<=r['cost_payload_energy_bound']*(1+1e-10)+1e-12
        if r['condition']!='csi5' and r['method']!='No-op':
            assert r['statistic_KL_ideal']<=1e-18
            assert r['coupled_model_relative_error_max']<=1e-9
            assert r['rho_all']<=.5+1e-10
            assert r['counterfactual_A_model_mse']==0
        if r['method']=='Independent':
            assert r['statistic_KL_radio_reference']<=1e-18
            assert r['counterfactual_A_model_mse']==0
        if r['method']=='No-op':
            assert r['cost_total_real']==0 and abs(r['counterfactual_A_influence_ratio']-1)<1e-10
        bygroup[(r['dataset'],r['K'],r['condition'],r['method'])].append(r)
    numeric=[x for x in rows[0] if isinstance(rows[0][x],(int,float)) and x not in ['seed','K','channel_draw']]
    summaries={}
    for key,rr in bygroup.items():
        means={x:st.mean(r[x] for r in rr) for x in numeric}
        seed_acc=[st.mean(r['accuracy'] for r in rr if r['seed']==s) for s in completion['seeds']]
        means.update(accuracy_seed_sd=st.stdev(seed_acc),accuracy_seed_means=seed_acc,
                     conditional_privacy_cap_violation_fraction=st.mean(r['rho_all']>.5+1e-10 for r in rr),
                     max_rho=max(r['rho_all'] for r in rr),
                     max_KL_ideal=max(r['statistic_KL_ideal'] for r in rr),
                     max_coupled_relative_error=max(r['coupled_model_relative_error_max'] for r in rr),
                     max_counterfactual_A_influence_ratio=max(r['counterfactual_A_influence_ratio'] for r in rr))
        summaries[key]=means
    references={}
    for ds in ['FashionMNIST','MNIST']:
        for k in [10,100,1000]:
            items=[r for (dd,kk,ss),r in refs.items() if dd==ds and kk==k]
            references[(ds,k)]={x:st.mean(r[x] for r in items) for x in items[0]}
    paired={}
    for ds,k,cond,method in summaries:
        if method not in ['CR-Fixed','CR-Power']:continue
        rr=bygroup[(ds,k,cond,method)]
        base={(r['seed'],r['channel_draw']):r for r in bygroup[(ds,k,cond,'Independent')]}
        sm=summaries[(ds,k,cond,method)];sb=summaries[(ds,k,cond,'Independent')]
        rat={x+'_ratio_of_means':sm[x]/sb[x] for x in ['cost_payload_energy','cost_total_real','cost_accounted_tx_energy_pilot_1.0','lifecycle_real','lifecycle_accounted_tx_energy']}
        rat.update(payload_ratio_median=st.median(r['cost_payload_energy']/base[(r['seed'],r['channel_draw'])]['cost_payload_energy'] for r in rr),
                   total_energy_win_fraction=st.mean(r['cost_accounted_tx_energy_pilot_1.0']<base[(r['seed'],r['channel_draw'])]['cost_accounted_tx_energy_pilot_1.0'] for r in rr),
                   mean_payload_energy_saved=sb['cost_payload_energy']-sm['cost_payload_energy'],
                   extra_nonpayload_energy=sm['cost_accounted_tx_energy_pilot_1.0']-sm['cost_payload_energy']-sb['cost_accounted_tx_energy_pilot_1.0']+sb['cost_payload_energy'],
                   accuracy_delta_pp=100*(sm['accuracy']-sb['accuracy']))
        for pv in ['0.0','0.001','0.01','1.0']:
            metric='cost_accounted_tx_energy_pilot_'+pv
            rat['total_energy_ratio_pilot_'+pv]=sm[metric]/sb[metric]
        paired[(ds,k,cond,method)]=rat
    decisions=[]
    for ds in ['FashionMNIST','MNIST']:
        for method in ['CR-Fixed','CR-Power']:
            key=(ds,1000,'rayleigh',method);a=summaries[key];p=paired[key];ref=references[(ds,1000)]
            deletion=a['max_KL_ideal']<=1e-18 and a['max_coupled_relative_error']<=1e-9 and a['max_rho']<=.5+1e-10
            utility=a['accuracy']>=ref['clean_clipped_retained_accuracy']-.05 and a['accuracy']>=ref['public_only_accuracy']+.10
            radio=p['cost_accounted_tx_energy_pilot_1.0_ratio_of_means']<=.95 and p['cost_total_real_ratio_of_means']<=1.05
            decisions.append(dict(dataset=ds,method=method,deletion_and_privacy_pass=deletion,
                                  utility_pass=utility,radio_pass=radio,adopt=deletion and utility and radio,
                                  accuracy_loss_to_clean_pp=100*(ref['clean_clipped_retained_accuracy']-a['accuracy'])))
    exact=[r for r in rows if r['condition']!='csi5' and r['method']!='No-op']
    verification=dict(passed=True,rows=len(rows),configuration_files=len(refs),
                      evaluated_utility_models=sum(r['utility_noise_draws'] for r in rows),
                      parameter_noise_realizations=sum(r['noise_draws'] for r in rows),
                      exact_max_KL=max(r['statistic_KL_ideal'] for r in exact),
                      exact_max_model_relative_error=max(r['coupled_model_relative_error_max'] for r in exact),
                      exact_max_rho=max(r['rho_all'] for r in exact),
                      max_power=max(r['cost_max_block_energy'] for r in rows),
                      decisions=decisions,source_sha256=completion['code_sha256'],
                      protocol_sha256=completion['protocol_sha256'],summary_code_sha256=sha(__file__))
    write(OUT/'verification.json',verification)
    write(OUT/'summary.json',dict(groups={'|'.join(map(str,k)):v for k,v in summaries.items()},
                                references={'|'.join(map(str,k)):v for k,v in references.items()},
                                paired={'|'.join(map(str,k)):v for k,v in paired.items()},decisions=decisions))
    lines=['# CR-Air 실제 데이터 결과표','',
           '정확도는3seed 평균 ± seed SD(%). 각 seed/조건은16 channel draw × 첫2 noise의32개 모델을 평가했다. '
           '채널 계산과 FP64 head solve를 GPU로 실행했다. RF 실측값이 아니다. 정확 CSI에서 Independent와 두 CR 방법은 같은 최종 noisy law를 목표로 하므로 작은 정확도 차이를 알고리즘 개선으로 해석하지 않는다. No-op은 이 분포 일치의 대상이 아니다.','',
           '## Rayleigh, 정확 CSI: 정확도','',
           '| Dataset | K | Clean retained | No-op | Independent | CR-Fixed | CR-Power |','|---|---:|---:|---:|---:|---:|---:|']
    for ds in ['FashionMNIST','MNIST']:
        for k in [10,100,1000]:
            cells=[f'{ds}',str(k),f'{references[(ds,k)]["clean_clipped_retained_accuracy"]*100:.2f}']
            for method in ['No-op','Independent','CR-Fixed','CR-Power']:
                v=summaries[(ds,k,'rayleigh',method)]
                cells.append(f'{v["accuracy"]*100:.2f} ± {v["accuracy_seed_sd"]*100:.2f}')
            lines.append('| '+' | '.join(cells)+' |')
    lines+=['','## Rayleigh: 평균 비용 비율','',
            '비율은Independent=1. counted TX energy는 payload/UL pilot/DL pilot/control/CSI/head 방송의 정규화 모형이며 circuit/PA/rx energy가 없다.','',
            '| Dataset | K | 방법 | Payload energy | Counted TX energy | 삭제 real uses | Lifecycle real uses |','|---|---:|---|---:|---:|---:|---:|']
    for ds in ['FashionMNIST','MNIST']:
        for k in [10,100,1000]:
            for method in ['CR-Fixed','CR-Power']:
                v=paired[(ds,k,'rayleigh',method)]
                cols=[ds,str(k),method]+[f'{v[x]:.6f}' for x in ['cost_payload_energy_ratio_of_means','cost_accounted_tx_energy_pilot_1.0_ratio_of_means','cost_total_real_ratio_of_means','lifecycle_real_ratio_of_means']]
                lines.append('| '+' | '.join(cols)+' |')
    lines+=['','## CSI5: 실제 계수로 감사한 조건부 privacy 상한 위반','',
            'rho_cap=.5. 이것은 random CSI 모형 전체의 새 DP 증명이 아니다. nominal 증명을 실제 오차에 그대로 적용할 수 없음을 진단한다.','',
            '| Dataset | K | 방법 | 위반 fraction | 최대 rho | 평균 A 영향 잔여비 | ideal KL | radio-reference KL |','|---|---:|---|---:|---:|---:|---:|---:|']
    for ds in ['FashionMNIST','MNIST']:
        for k in [10,100,1000]:
            for method in ['Independent','CR-Fixed','CR-Power']:
                v=summaries[(ds,k,'csi5',method)]
                lines.append(f'| {ds} | {k} | {method} | {v["conditional_privacy_cap_violation_fraction"]:.3f} | {v["max_rho"]:.4f} | {v["counterfactual_A_influence_ratio"]:.6f} | {v["statistic_KL_ideal"]:.6g} | {v["statistic_KL_radio_reference"]:.6g} |')
    lines+=['','## 사전 채택 판정','',
            '| Dataset | 방법 | 삭제·privacy | utility | 통신 | 전체 채택 |','|---|---|---|---|---|---|']
    for row in decisions:
        lines.append('| '+' | '.join([row['dataset'],row['method']]+['통과' if row[x] else '실패' for x in ['deletion_and_privacy_pass','utility_pass','radio_pass','adopt']])+' |')
    lines+=['','## 전체 조건','',
            '| Dataset | K | Channel | 방법 | Accuracy(%) | ideal KL | rho max | Payload E | Counted E | Real uses |','|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for key,v in sorted(summaries.items()):
        ds,k,ch,method=key
        lines.append(f'| {ds} | {k} | {ch} | {method} | {v["accuracy"]*100:.3f} | {v["statistic_KL_ideal"]:.4g} | {v["max_rho"]:.4f} | {v["cost_payload_energy"]:.6g} | {v["cost_accounted_tx_energy_pilot_1.0"]:.3f} | {v["cost_total_real"]:.3f} |')
    (ROOT/'RESULT_TABLES_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    colors={'Independent':'#2878b5','CR-Fixed':'#ef843b','CR-Power':'#29996e','No-op':'#929292'}
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),constrained_layout=True)
    ks=[10,100,1000]
    for ax,ds in zip(axes[0],['FashionMNIST','MNIST']):
        ax.plot(ks,[references[(ds,k)]['clean_clipped_retained_accuracy']*100 for k in ks],'-k',marker='s',label='Clean retained')
        for method in ['Independent','CR-Fixed','CR-Power','No-op']:
            vv=[summaries[(ds,k,'rayleigh',method)] for k in ks]
            ax.errorbar(ks,[v['accuracy']*100 for v in vv],yerr=[v['accuracy_seed_sd']*100 for v in vv],
                        color=colors[method],marker='o',label=method,ls='--' if method=='No-op' else '-')
        ax.set(xscale='log',xticks=ks,xticklabels=ks,xlabel='Clients (10,000 private samples total)',
               ylabel='Test accuracy (%)',title=ds+' / exact CSI',ylim=(0,65))
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
    ax=axes[1,0]
    for method in ['CR-Fixed','CR-Power']:
        for metric,style,label in [('cost_payload_energy_ratio_of_means','--','payload'),('cost_accounted_tx_energy_pilot_1.0_ratio_of_means','-','counted TX')]:
            vv=[st.mean(paired[(ds,k,'rayleigh',method)][metric] for ds in ['FashionMNIST','MNIST']) for k in ks]
            ax.plot(ks,vv,style,color=colors[method],marker='o',label=method+' '+label)
    ax.axhline(1,color='k',lw=.8)
    ax.set(xscale='log',xticks=ks,xticklabels=ks,xlabel='Clients',ylabel='Energy ratio (Independent = 1)',
           title='Rayleigh: signal energy vs protocol overhead',ylim=(0,1.13))
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,1];xx=np.arange(3);width=.24
    for idx,method in enumerate(['Independent','CR-Fixed','CR-Power']):
        vv=[100*st.mean(summaries[(ds,k,'csi5',method)]['conditional_privacy_cap_violation_fraction'] for ds in ['FashionMNIST','MNIST']) for k in ks]
        ax.bar(xx+(idx-1)*width,vv,width,color=colors[method],label=method)
    ax.set(xticks=xx,xticklabels=ks,xlabel='Clients',ylabel='Conditional cap violations (%)',
           title='5% CSI error: nominal privacy is insufficient',ylim=(0,110))
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('CR-Air feasibility: fixed 16-d random features, client-level privacy cap rho = 0.5',fontsize=13)
    fig.savefig(ROOT/'comparison.png',dpi=170)
    fig.savefig(ROOT/'comparison.pdf')
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    for ds in ['FashionMNIST','MNIST']:
        for method in ['CR-Fixed','CR-Power']:
            print(ds,method,'K1000 rayleigh',json.dumps(paired[(ds,1000,'rayleigh',method)],ensure_ascii=False))


if __name__=='__main__':main()
