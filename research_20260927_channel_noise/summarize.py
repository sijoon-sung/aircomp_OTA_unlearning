"""Recompute clustered summaries and check the recorded ledgers (stdlib only)."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics as st
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def finite(x):
    if isinstance(x, float):
        assert math.isfinite(x)
    elif isinstance(x, dict):
        for v in x.values(): finite(v)
    elif isinstance(x, list):
        for v in x: finite(v)


def stats(rows, field):
    seeds = defaultdict(list)
    for row in rows:
        value = row
        for key in field.split('.'):
            value = value[key]
        seeds[row['seed']].append(value)
    vals = [st.mean(seeds[s]) for s in sorted(seeds)]
    return dict(mean=st.mean(vals), seed_sd=st.stdev(vals), seed_values=vals,
                seed_count=len(vals), H_replicates_per_seed=[len(seeds[s]) for s in sorted(seeds)])


def group(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [dict(zip(keys, key), metrics={f:stats(rr, f) for f in fields})
            for key, rr in sorted(groups.items())]


def run():
    records = [load(OUT/f'seed{s}.json') for s in [202609271, 202609272, 202609273]]
    assert all(r['complete'] for r in records)
    for r in records:
        assert r['code_sha256']==sha(ROOT/'run_experiment.py')
        assert r['protocol_sha256']==sha(ROOT/'PROTOCOL_KO.md')
    finite(records)
    rows = [r for s in records for r in s['rows']]
    pts = [r for s in records for r in s['point_rows']]
    assert len(rows)==1458 and len(pts)==648
    max_cost_error = 0.
    for r in rows+pts:
        c = r['cost']
        if c is not None:
            total = 1.125*(c['analog_real']+c['pilot_real']+c['digital_bits']/c['rate_bits_per_real'])
            max_cost_error = max(max_cost_error, abs(total-c['deletion_cached_real_uses']))
            assert abs(total-c['deletion_cached_real_uses'])<1e-8
            assert sum(c['repeats'])*130+2145*32==c['analog_real']
            assert all(n>=1 for n in c['repeats'])
        if 'power' in r:
            assert r['power']['max_mean_power']<=1+1e-9
            assert r['power']['max_peak_power']<=4+1e-9
    paired = defaultdict(dict)
    for r in rows:
        paired[(r['seed'], r['snr_db'], r['tau'], r['curvature'], r['H_noise'])][r['method']] = r
        if r['cost'] is not None:
            var = r['output_variance']
            kl = r['mean_bias_squared']/(2*r['tau']**2)+65*sum(v/r['tau']**2-1-math.log(v/r['tau']**2) for v in var)
            assert abs(kl-r['conditional_KL'])<1e-7
    for methods in paired.values():
        nm, uf, aa = [methods[k] for k in ['NM-Air', 'Uniform-match', 'Aware-fixed40']]
        assert nm['cost']['deletion_cached_real_uses']<=aa['cost']['deletion_cached_real_uses']+1e-8
        assert nm['cost']['deletion_cached_real_uses']<=uf['cost']['deletion_cached_real_uses']+1e-8
        assert nm['conditional_KL']==aa['conditional_KL']==uf['conditional_KL']
        assert nm['empirical_point_MSE_ratio']==uf['empirical_point_MSE_ratio']
        assert all(abs(v-nm['tau']**2)<1e-14 for v in nm['physical_variance'])
    fields = ['conditional_KL','conditional_covariance_KL','conditional_TV_upper',
              'theoretical_point_MSE_ratio','empirical_point_MSE_ratio','mean_bias_ratio',
              'test_accuracy','forget_normalized_js']
    gg = group(rows, ['curvature','snr_db','tau','method'], fields)
    cg = group([r for r in rows if r['cost']], ['curvature','snr_db','tau','method'],
               ['cost.deletion_cached_real_uses','cost.lifecycle_real_uses','cost.correction_real',
                'power.tx_symbol_energy_normalized'])
    pp = group(pts, ['curvature','snr_db','budget','method'],
               ['empirical_point_MSE_ratio','forget_normalized_js','test_accuracy','cost.deletion_cached_real_uses'])
    refs = [dict(seed=s['seed'], **r) for s in records for r in s['references']]
    rsummary = group(refs, ['tau'], ['randomized_retrain.test_accuracy','clean_retrain_accuracy','source_accuracy'])

    def find(arr, **where):
        return next(r['metrics'] for r in arr if all(r[k]==v for k,v in where.items()))
    main = dict(curvature='OTA32', snr_db=20, tau=.003)
    nm = find(gg, **main, method='NM-Air')
    nm_cost = find(cg, **main, method='NM-Air')['cost.deletion_cached_real_uses']['mean']
    aware_cost = find(cg, **main, method='Aware-fixed40')['cost.deletion_cached_real_uses']['mean']
    refacc = find(rsummary, tau=.003)['randomized_retrain.test_accuracy']['mean']
    verdict = dict(main_setting=main, cost_saving_fraction=1-nm_cost/aware_cost,
        mean_TV_upper=nm['conditional_TV_upper']['mean'],
        test_accuracy_drop_from_randomized_reference=refacc-nm['test_accuracy']['mean'])
    verdict['prespecified_pass'] = verdict['cost_saving_fraction']>=.05 and verdict['mean_TV_upper']<=.1 and verdict['test_accuracy_drop_from_randomized_reference']<=.01
    summary = dict(aggregation='128 payload draws within each H draw, then H draws within each seed, then 3 seeds; seed SD is not 128 independent data seeds',
        gaussian=gg, costs=cg, point=pp, references=rsummary, verdict=verdict)
    save(OUT/'summary.json', summary)
    check = dict(passed=True, gaussian_rows=len(rows), point_rows=len(pts),
        complete_seed_count=3, code_and_protocol_hashes_match=True,
        ledger_max_error=max_cost_error, power_constraints_pass=True,
        same_law_comparators_checked=len(paired), NM_covariance_and_integer_costs_checked=True,
        gaussian_KL_recomputed=True, main_success=verdict['prespecified_pass'])
    save(OUT/'verification.json', check)
    lines = ['**채널 잡음 GPU 결과표 — 2026-09-27**','',
        '통계는 noise를 seed 안에서 평균한 뒤 3개 데이터 seed로 계산했다. 모든 수치와 seed별 평균/표준편차는 results/summary.json에 있다. 주 표는20dB,tau=.003,실제 OTA Hessian32 조건이다.','',
        '| 방법 | 조건부 KL 평균 | 공분산 KL | Point MSE 비율(이론 기대값) | Forget JS 비율 | Test accuracy | 총 삭제 real uses |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in ['NM-Air','Uniform-match','Aware-fixed40','DS-fixed40','Naive-add40','Noise-only']:
        a = find(gg, **main, method=method)
        c = '별도 software 진단' if method=='Noise-only' else f"{find(cg, **main, method=method)['cost.deletion_cached_real_uses']['mean']:,.1f}"
        lines.append(f"|{method}|{a['conditional_KL']['mean']:.4f}|{a['conditional_covariance_KL']['mean']:.4f}|{a['theoretical_point_MSE_ratio']['mean']:.6f}|{a['forget_normalized_js']['mean']:.6f}|{a['test_accuracy']['mean']*100:.3f}%|{c}|")
    lines += ['',f"NM-Air 비용 절감={verdict['cost_saving_fraction']*100:.3f}%. 같은 tau의 randomized retrain accuracy={refacc*100:.3f}%. 사전 판정={verdict['prespecified_pass']}.",
        '조건부 KL/TV의 평균은 주변 분포에 대한 상계이며 실제 주변 KL/TV의 측정값이 아니다. Noise-only는 source에 software noise만 넣은 반례로 RF 비용 경쟁 방식이 아니다. Point MSE와 JS의 reference는 깨끗한 retained ridge이고,KL의 reference는 사전에 고정한 randomized ridge다.',
        '', '**모든 NM-Air 설정: OTA32만**','',
        '| SNR | tau | 총 삭제비 | Aware-fixed40 대비 절감 | 조건부 KL 평균 | TV 상계 평균 |',
        '|---|---:|---:|---:|---:|---:|']
    for snr in [10,20,30]:
        for tau in [.001,.003,.01]:
            where = dict(curvature='OTA32',snr_db=snr,tau=tau)
            a = find(gg, **where, method='NM-Air')
            c = find(cg, **where, method='NM-Air')['cost.deletion_cached_real_uses']['mean']
            b = find(cg, **where, method='Aware-fixed40')['cost.deletion_cached_real_uses']['mean']
            lines.append(f"|{snr}|{tau}|{c:,.1f}|{100*(1-c/b):.3f}%|{a['conditional_KL']['mean']:.4f}|{a['conditional_TV_upper']['mean']:.6f}|")
    lines += ['','**Deterministic reference: JS-Air와 동일 비용의 raw 비교**','',
        '| SNR | 총 반복 | Raw MSE 비율 | JS-Air MSE 비율 | 감소율 | 동일 총 삭제비 |',
        '|---|---:|---:|---:|---:|---:|']
    for snr in [10,20,30]:
        for budget in [5,10,20,40]:
            where = dict(curvature='OTA32',snr_db=snr,budget=budget)
            a = find(pp, **where, method='DS-raw')
            b = find(pp, **where, method='JS-Air')
            av,bv = a['empirical_point_MSE_ratio']['mean'], b['empirical_point_MSE_ratio']['mean']
            lines.append(f"|{snr}|{budget}|{av:.6f}|{bv:.6f}|{100*(1-bv/av):.3f}%|{a['cost.deletion_cached_real_uses']['mean']:,.1f}|")
    lines += ['','10dB는 JS-Air로 개선되어도 MSE 비율이1보다 크다. No-op보다 나쁜 조건을 언러닝 성공으로 표시하지 않는다. JS-Air는 결정론적 보정의 통계적 추정 개선이며 randomized reference의 Gaussian law를 유지하지 않는다.',
        '', '**Hessian oracle 진단 — 무선 실험 성과와 구분**','',
        '| 방식 | KL 평균 | TV 평균 | Point MSE 비율 |', '|---|---:|---:|---:|']
    for method in ['NM-Air','Aware-fixed40','DS-fixed40','Naive-add40']:
        a = find(gg, curvature='oracle',snr_db=20,tau=.003,method=method)
        lines.append(f"|{method}|{a['conditional_KL']['mean']:.9g}|{a['conditional_TV_upper']['mean']:.9g}|{a['theoretical_point_MSE_ratio']['mean']:.6f}|")
    lines += ['', 'Oracle는 Hessian 수집 잡음을 없앤 원인 진단이다. 이 결과에 기록된 통신 원장은 가상 비교용이며 실제 장비에서 같은 비용으로 정확한 곡률을 얻었다는 주장이 아니다.',
        '', '![비용과 삭제 오차](comparison.png)', '',
        '추가 sampler 구분 검사와 잡음 분산 오보정/추정비용은 results/diagnosis.json, 해석과 수식은 METHOD_KO.md와 README_KO.md를 참고한다.']
    (ROOT/'RESULT_TABLES_KO.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(verification=check, verdict=verdict), indent=2))
    return summary


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.1), constrained_layout=True)
    colors = {'DS-raw':'#718096','JS-Air':'#137c75'}
    for method in ['DS-raw','JS-Air']:
        rr = sorted([r for r in summary['point'] if r['curvature']=='OTA32' and r['snr_db']==20 and r['method']==method], key=lambda r:r['budget'])
        axs[0].errorbar([r['metrics']['cost.deletion_cached_real_uses']['mean']/1000 for r in rr],
            [r['metrics']['empirical_point_MSE_ratio']['mean'] for r in rr],
            yerr=[r['metrics']['empirical_point_MSE_ratio']['seed_sd'] for r in rr],
            marker='o', label=method, color=colors[method], capsize=3)
    axs[0].set(title='Point target: shrinkage helps', xlabel='Total deletion cost (1,000 real uses)', ylabel='Normalized parameter MSE')
    axs[0].legend()
    names = ['NM-Air','Aware-fixed40','Uniform-match']
    rr = [next(r for r in summary['costs'] if r['curvature']=='OTA32' and r['snr_db']==20 and r['tau']==.003 and r['method']==name) for name in names]
    costs = [r['metrics']['cost.deletion_cached_real_uses']['mean']/1000 for r in rr]
    axs[1].bar(names, costs, color=['#137c75','#718096','#d39b39'])
    for i,v in enumerate(costs): axs[1].text(i,v+1,f'{v:.1f}',ha='center')
    axs[1].set(title='Same output covariance, modest savings', ylabel='Total deletion cost (1,000 real uses)', ylim=(0,155))
    axs[1].tick_params(axis='x', labelrotation=12)
    dg = load(OUT/'diagnosis.json')
    names = ['OTA32','oracle','noise_only']
    vals = [st.mean(r['radius_discriminator_AUC'] for r in dg['rows'] if r['sampler']==k) for k in names]
    axs[2].bar(['Noisy H','Exact-H\ndiagnostic','Noise only'], vals, color=['#ba5049','#137c75','#718096'])
    axs[2].axhline(.5, color='black',linestyle='--',linewidth=1)
    axs[2].set(title='Randomized retraining remains distinguishable',ylabel='Sampler discriminator AUC (not MIA)',ylim=(0,1.1))
    for ax in axs: ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'comparison.png', dpi=180)
    fig.savefig(ROOT/'comparison.pdf')
    plt.close(fig)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plots', action='store_true')
    args = ap.parse_args()
    summary = run()
    if args.plots: plot(summary)
