"""Stdlib audit/summary of cohort diversity; optional Matplotlib figures."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import math
import statistics as st

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
KEY=['dataset','condition','method']
METRICS=['MSE_ratio','theoretical_MSE_ratio','bias_ratio','variance_ratio','test_accuracy','forget_JS_ratio']
METHODS=['Original','V1','SubspaceV1','FD2-V1','FD8-Original','FD8-V1','FD8-SubspaceV1','CG-V1','CG-SubspaceV1']


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def finite(v):
    if isinstance(v,float):
        assert math.isfinite(v)
    elif isinstance(v,dict):
        for x in v.values():finite(x)
    elif isinstance(v,list):
        for x in v:finite(x)


def aggregate(rows):
    g=dict(zip(KEY,[rows[0][k] for k in KEY]))
    byseed=defaultdict(list)
    for r in rows:byseed[r['seed']].append(r)
    g['seed_metrics']={str(s):{m:st.mean(r['metrics'][m] for r in rs) for m in METRICS}
                       for s,rs in sorted(byseed.items())}
    for m in METRICS:
        values=[v[m] for v in g['seed_metrics'].values()]
        g[m]=st.mean(values)
        g[m+'_seed_SD']=st.stdev(values)
    values=sorted(r['metrics']['MSE_ratio'] for r in rows)
    complete=[r for r in rows if r['status']=='ok']
    g.update(sessions=len(rows),completion_rate=len(complete)/len(rows),
             completed_and_closer_than_noop=sum(r['status']=='ok' and r['metrics']['MSE_ratio']<1 for r in rows)/len(rows),
             median_session_MSE=st.median(values),p95_session_MSE=values[math.ceil(.95*len(values))-1],
             mean_active=st.mean(r['cost']['active_total'] for r in rows),
             completed_mean_active=st.mean(r['cost']['active_total'] for r in complete) if complete else None,
             completed_MSE=st.mean(r['metrics']['MSE_ratio'] for r in complete) if complete else None,
             budget=rows[0]['budget'],mean_attempts=st.mean(r['attempts'] for r in rows),
             completed_mean_repeats=st.mean(r['correction_repeats'] for r in complete) if complete else None,
             latency={key:st.mean(r['cost']['latency'][key] for r in rows) for key in ['1','4','16']},
             worst_latency={key:max(r['cost']['latency'][key] for r in rows) for key in ['1','4','16']})
    attack=[r['attack'] for r in complete if r['attack'] is not None]
    if attack:
        g['attack_full_MSE']=st.mean(r['full_gradient_relative_MSE'] for r in attack)
        g['attack_projected_MSE']=st.mean(r['projected_gradient_relative_MSE'] for r in attack)
    if complete:
        g['selected_actual_min_gain_mean']=st.mean(r['selected_actual_min_gain'] for r in complete)
        g['H_scale_squared_mean']=st.mean(r['curvature']['scale']**2 for r in complete)
    rate=.5*math.log2(1+10**(rows[0]['SNR']/10))
    final_bits=64*10*32
    prep=1.125*(64*65/2+64*10+160+(2*(10*32+64)+final_bits+256)/rate)
    g['source_preparation_ideal']=prep
    g['mean_lifecycle']=prep+g['mean_active']
    g['completed_lifecycle']=prep+g['completed_mean_active'] if complete else None
    g['mean_cold']=g['mean_active']+g['completion_rate']*1.125*final_bits/rate
    return g


def main(plots=False):
    completion=load(OUT/'completion.json')
    checks=load(OUT/'math_checks.json')
    files=sorted(OUT.glob('*_seed*.json'))
    assert len(files)==6 and completion['complete'] and checks['passed']
    for key,path in [('code_sha256',ROOT/'run_experiment.py'),('protocol_sha256',ROOT/'PROTOCOL_KO.md'),
                     ('reused_code_sha256',ROOT.parent/'research_20260928_ds_revision/run_experiment.py')]:
        assert sha(path)==completion[key]==checks[key]
    rows=[]
    refs={}
    for p in files:
        raw=load(p)
        assert raw['complete']
        assert all(raw[k]==completion[k] for k in ['code_sha256','protocol_sha256','reused_code_sha256'])
        rows.extend(raw['rows'])
        refs[p.stem]=raw['references']
    assert len(rows)==completion['rows']==10368
    feedback=load(OUT/'feedback_verification.json')
    assert feedback['passed'] and feedback['decisions_compared']==len(rows)
    assert feedback['decisions_changed_by_FP32_feedback']==0
    assert feedback['audit_code_sha256']==sha(ROOT/'audit_feedback.py')
    assert feedback['code_sha256']==completion['code_sha256']
    finite(rows)
    groups=defaultdict(list)
    session=defaultdict(dict)
    max_cost_error=max_mc_z=0.
    for r in rows:
        cost=r['cost']
        m=r['metrics']
        snr=r['SNR']
        rate=.5*math.log2(1+10**(snr/10))
        assert cost['active_total'] <= r['budget']+1e-7
        assert r['retained_target_clients']==9
        if r['status']=='ok':
            assert r['participating_clients']==9 and cost['complete']
            b,e=r['base_ledger'],r['search_cost']
            reconstructed=1.125*(b['analog_H']+b['analog_correction']+b['pilot_real']+b['digital_bits']/rate)
            reconstructed+=1.125*(e['extra_pilot_real']+e['extra_digital_bits']/rate)
            max_cost_error=max(max_cost_error,abs(cost['active_total']-reconstructed))
            assert r['power']['mean']<=1+1e-8 and r['power']['peak']<=4+1e-8
            assert r['curvature']['power']['mean']<=1+1e-8 and r['curvature']['power']['peak']<=4+1e-8
            assert abs(m['theoretical_MSE_ratio']-m['bias_ratio']-m['variance_ratio'])<1e-7
            z=abs(m['MSE_ratio']-m['theoretical_MSE_ratio'])/max(m['MonteCarlo_SE_ratio'],1e-15)
            max_mc_z=max(max_mc_z,z)
            if r['CSI_error']==0:assert m['CSI_mean_shift_norm']<1e-9
            assert b['analog_correction']==80*sum(b['repeats'])
            assert r['correction_repeats']==sum(b['repeats'])
            if r['method'].startswith('CG'):
                assert r['selected_estimated_min_gain']>=.4
                assert r['selected_band']==r['attempts']-1
        else:
            assert r['status']=='deletion_pending' and r['method'].startswith('CG')
            assert not r['unlearning_success'] and not cost['complete']
            assert r['participating_clients']==0 and m['MSE_ratio']==1
            t=r['timeout_cost']
            expected=1.125*(t['pilot_real']+t['digital_bits']/rate)
            max_cost_error=max(max_cost_error,abs(expected-cost['active_total']))
        for factor in [1,4,16]:
            if r['method'].startswith('CG'):
                expected=((r['attempts']-1)*factor*r['budget']+cost['active_total']-cost['previous_probe_airtime']) if cost['complete'] else 8*factor*r['budget']
            else:expected=cost['active_total']
            assert abs(expected-cost['latency'][str(factor)])<1e-7
        groups[tuple(r[k] for k in KEY)].append(r)
        session[(r['dataset'],r['seed'],r['condition'],r['draw'])][r['method']]=r
    assert max_cost_error<1e-7 and max_mc_z<8
    bias_error=0.
    static_selected=0
    for key, rs in session.items():
        assert sorted(rs)==sorted(METHODS)
        for a,b in [('Original','V1'),('FD8-Original','FD8-V1')]:
            bias_error=max(bias_error,abs(rs[a]['metrics']['bias_ratio']-rs[b]['metrics']['bias_ratio']))
        if key[2]=='static20':
            assert all(rs[m]['selected_band']==0 for m in ['Original','FD2-V1','FD8-V1'])
            static_selected+=1
    assert bias_error<1e-8
    results=[aggregate(rs) for _,rs in sorted(groups.items())]
    assert len(results)==108
    index={tuple(g[k] for k in KEY):g for g in results}
    def get(ds='FashionMNIST',condition='iid20',method='V1'):
        return index[(ds,condition,method)]
    gates={}
    for ds in ['FashionMNIST','MNIST']:
        reference=st.mean(v['retained_accuracy'] for k,v in refs.items() if k.startswith(ds+'_'))
        for method,baseline in [('FD8-V1','V1'),('FD8-SubspaceV1','SubspaceV1')]:
            g,b=get(ds,method=method),get(ds,method=baseline)
            improve=1-g['MSE_ratio']/b['MSE_ratio']
            drop=reference-g['test_accuracy']
            passgate=(g['MSE_ratio']<1 and all(v['MSE_ratio']<1 for v in g['seed_metrics'].values()) and improve>=.3 and drop<=.01 and g['completion_rate']==1)
            gates[ds+'/'+method]=dict(passed=passgate,improvement=improve,all_seeds_below_noop=all(v['MSE_ratio']<1 for v in g['seed_metrics'].values()),accuracy_drop=drop)
    write(OUT/'summary.json',dict(groups=results,primary_gates=gates,references=refs))
    audit=dict(passed=True,rows=len(rows),groups=len(results),raw_files=len(files),
               maximum_cost_error=max_cost_error,maximum_MC_standard_error_multiple=max_mc_z,
               equal_mean_bias_error=bias_error,all_completed_sessions_include_nine_clients=True,
               no_timeout_counted_as_success=True,all_power_and_active_budget_checks=True,
               paired_CSI_math_check=True,static_bank_selection_sessions=static_selected,
               FP32_feedback_decisions_verified=len(rows),
               source_and_protocol_hashes_match=True,privacy_certificate=False,
               scope='Implementation/cost audit; not a privacy or deletion certificate')
    write(OUT/'verification.json',audit)
    lines=['# Cohort diversity: 전체 조건 결과','',
           '오차 미삭제=1, retained ridge=0. 3seed 평균±seed SD. seed당32channel/H draw, draw당16correction noise. Utility는 draw당첫1noise. Pending도 E=1로 포함하며 완료율을 별도 표시한다.', '',
           '|Dataset|Condition|Method|E ± SD|완료%|완료 및 E<1 %|Test %|Active cost|Latency(1 window)|',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for g in results:
        lines.append(f"|{g['dataset']}|{g['condition']}|{g['method']}|{g['MSE_ratio']:.5f} ± {g['MSE_ratio_seed_SD']:.5f}|{g['completion_rate']*100:.2f}|{g['completed_and_closer_than_noop']*100:.2f}|{g['test_accuracy']*100:.3f}|{g['mean_active']:.2f}|{g['latency']['1']:.2f}|")
    (ROOT/'RESULT_TABLES_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit=audit,primary_gates=gates),ensure_ascii=False,indent=2))
    if plots:draw_plots(get)


def draw_plots(get):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    labels=['V1','FD2-V1','FD8-V1','SubspaceV1','FD8-SubspaceV1']
    display=['V1','FD2\nV1','FD8\nV1','Subspace','FD8\nSubspace']
    for ax,ds in zip(axes[0],['FashionMNIST','MNIST']):
        values=[get(ds,method=m)['MSE_ratio'] for m in labels]
        for j,(m,value,col) in enumerate(zip(labels,values,['#888888','#d59b32','#1965b0','#9e94b5','#12826b'])):
            ax.errorbar(j,value,yerr=get(ds,method=m)['MSE_ratio_seed_SD'],fmt='o',color=col,capsize=5,markersize=7)
        ax.axhline(1,color='black',linestyle=':',label='No-op')
        ax.set_xticks(range(len(labels)),display)
        ax.set(yscale='log',ylabel='Normalized MSE (log)',title=f'{ds}: 20dB independent Rayleigh')
        ax.legend(fontsize=8)
    ax=axes[1,0]
    conditions=['iid20','correlated20','static20','csi05_20','weak_client20']
    for m,col in [('V1','#777777'),('FD8-V1','#1965b0'),('FD8-SubspaceV1','#12826b')]:
        ax.plot(range(len(conditions)),[get('MNIST',c,m)['MSE_ratio'] for c in conditions],marker='o',label=m,color=col)
    ax.axhline(1,color='black',linestyle=':')
    ax.set_xticks(range(len(conditions)),['Independent','Corr .9','Static','CSI 5%','Weak client'])
    ax.set(yscale='log',ylabel='Normalized MSE (log)',title='MNIST: limitations at 20dB')
    ax.legend(fontsize=8)
    ax=axes[1,1]
    for m,col in [('V1','#777777'),('FD8-V1','#1965b0'),('CG-V1','#cc7722')]:
        g=get('MNIST',method=m)
        ax.plot([1,4,16],[g['latency'][str(x)]/g['budget'] for x in [1,4,16]],marker='o',label=f"{m}: complete {100*g['completion_rate']:.1f}%",color=col)
    ax.set(xlabel='Coherence window / active budget',ylabel='Mean latency / active budget',title='Waiting is charged, including timeouts')
    ax.set_xticks([1,4,16]);ax.legend(fontsize=8)
    fig.suptitle('Cohort-diversity DS-Air | 9 retained clients in every completed aggregate',fontsize=14)
    fig.savefig(ROOT/'comparison.png',dpi=170)
    fig.savefig(ROOT/'comparison.pdf')
    plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--plots',action='store_true')
    main(ap.parse_args().plots)
