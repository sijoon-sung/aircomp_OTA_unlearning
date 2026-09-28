"""Independent, stdlib-only result/ledger audit. Optional scientific plots."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
KEY = ['dataset', 'D', 'SNR', 'channel', 'H_repeats', 'method']


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def aggregate(rows):
    seeds = sorted({r['seed'] for r in rows})
    names = ['MSE_ratio', 'theoretical_MSE_ratio', 'bias_ratio', 'variance_ratio',
             'test_accuracy', 'forget_JS_ratio', 'CSI_mean_shift_norm', 'deployed_MSE_ratio']
    ans = dict(zip(KEY, [rows[0][k] for k in KEY]))
    ans['seed_metrics'] = {}
    for s in seeds:
        rs = [r for r in rows if r['seed'] == s]
        ans['seed_metrics'][str(s)] = {m: st.mean(r['metrics'][m] for r in rs) for m in names}
    for m in names:
        vals = [ans['seed_metrics'][str(s)][m] for s in seeds]
        ans[m] = st.mean(vals)
        ans[m+'_seed_SD'] = st.stdev(vals)
    ans['cost'] = st.mean(r['cost']['total'] for r in rows)
    ans['lifecycle_cost'] = st.mean(r['cost']['lifecycle'] for r in rows)
    ans['budget'] = rows[0]['budget']
    ans['curvature_analog_cost'] = rows[0]['cost']['analog_H']*1.125
    ans['basis_digital_cost'] = rows[0]['cost']['basis_bits']/rows[0]['cost']['digital_rate']*1.125
    ans['pre_fraction'] = None if rows[0]['chosen_pre_blocks'] is None else st.mean(st.mean(r['chosen_pre_blocks']) for r in rows)
    ans['mean_repeats'] = st.mean(sum(r['cost']['repeats']) for r in rows)
    if rows[0]['attack'] is not None:
        ans['full_gradient_attack_MSE_ratio'] = st.mean(r['attack']['full_gradient_relative_MSE'] for r in rows)
        ans['projected_gradient_attack_MSE_ratio'] = st.mean(r['attack']['projected_gradient_relative_MSE'] for r in rows)
    return ans


def main(plots=False):
    completion = json.loads((OUT/'completion.json').read_text())
    checks = json.loads((OUT/'math_checks.json').read_text())
    for key, path in [('code_sha256', ROOT/'run_experiment.py'), ('protocol_sha256', ROOT/'PROTOCOL_KO.md')]:
        assert sha(path) == checks[key] == completion[key]
    rows, references = [], {}
    files = sorted(OUT.glob('*_D*_seed*.json'))
    assert len(files) == 18
    for path in files:
        value = json.loads(path.read_text())
        assert value['complete']
        rows.extend(value['rows'])
        references[path.stem] = value['references']
    assert len(rows) == completion['rows'] == 5568
    assert all(r['status'] == 'ok' for r in rows)
    max_ledger_error, max_mc_z, same_bias_error = 0., 0., 0.
    rowgroups = defaultdict(dict)
    for row in rows:
        cc, mm = row['cost'], row['metrics']
        recalculated = cc['cp']*(cc['analog_H']+cc['analog_correction']+cc['pilot_real']+cc['digital_bits']/cc['digital_rate'])
        max_ledger_error = max(max_ledger_error, abs(cc['total']-recalculated))
        assert cc['total'] <= row['budget']+1e-7
        assert abs(cc['analog_correction']-80*sum(cc['repeats'])) < 1e-8
        assert min(cc['repeats']) >= 1
        assert row['power']['mean'] <= 1+1e-8 and row['power']['peak'] <= 4+1e-8
        if row['curvature'] is not None:
            assert row['curvature']['power']['mean'] <= 1+1e-8
            assert row['curvature']['power']['peak'] <= 4+1e-8
        assert abs(mm['theoretical_MSE_ratio']-mm['bias_ratio']-mm['variance_ratio']) < 1e-8
        z = abs(mm['MSE_ratio']-mm['theoretical_MSE_ratio'])/max(mm['MonteCarlo_SE_ratio'], 1e-15)
        max_mc_z = max(max_mc_z, z)
        if row['channel'] != 'rayleigh_csi5':
            assert mm['CSI_mean_shift_norm'] < 1e-10
        key = tuple(row[k] for k in ['dataset', 'D', 'seed', 'SNR', 'channel', 'H_repeats', 'H_draw'])
        rowgroups[key][row['method']] = row
    for methods in rowgroups.values():
        bs = [methods[m]['metrics']['bias_ratio'] for m in ['Original', 'V1', 'Switch']]
        same_bias_error = max(same_bias_error, max(bs)-min(bs))
    assert max_ledger_error < 1e-7 and same_bias_error < 1e-8
    assert max_mc_z < 8, 'Investigate Monte Carlo/model inconsistency before reporting'
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in KEY)].append(row)
    groups = [aggregate(rs) for _, rs in sorted(buckets.items())]
    lookup = {tuple(g[k] for k in KEY): g for g in groups}
    def get(ds='FashionMNIST', d=64, snr=20, channel='rayleigh', hr=32, method='Original'):
        return lookup[(ds, d, snr, channel, hr, method)]
    primary = {m: get(method=m) for m in ['Original', 'V1', 'Switch', 'LocalNewton', 'Diagonal', 'StatsRetrain', 'SubspaceV1', 'SubspaceV1-nominal']}
    orig, v1, sw, sub = [primary[m] for m in ['Original', 'V1', 'Switch', 'SubspaceV1-nominal']]
    retained_acc = st.mean(v['retained_accuracy'] for k, v in references.items() if k.startswith('FashionMNIST_D64_'))
    switch_improvement = 1-sw['MSE_ratio']/min(orig['MSE_ratio'], v1['MSE_ratio'])
    allseed = all(sw['seed_metrics'][s]['MSE_ratio'] < min(orig['seed_metrics'][s]['MSE_ratio'], v1['seed_metrics'][s]['MSE_ratio']) for s in sw['seed_metrics'])
    decision = dict(switch_improvement_vs_stronger=switch_improvement, switch_all_seeds_improve=allseed,
                    switch_accuracy_drop=retained_acc-sw['test_accuracy'],
                    switch_primary_pass=switch_improvement >= .05 and allseed and retained_acc-sw['test_accuracy'] <= .01,
                    subspace_nominal_cost_saving=1-sub['cost']/orig['cost'],
                    subspace_nominal_relative_MSE=sub['MSE_ratio']/orig['MSE_ratio'],
                    subspace_nominal_cost_relative_error_gate=(1-sub['cost']/orig['cost'] >= .2 and sub['MSE_ratio'] <= 1.05*orig['MSE_ratio']),
                    subspace_nominal_better_than_noop=sub['MSE_ratio'] < 1,
                    retained_reference_accuracy=retained_acc,
                    privacy_certificate=False, full_transcript_requirement_satisfied=False)
    counts = dict(conditions=54, V1_beats_Original=0, Switch_beats_stronger_endpoint=0,
                  Switch_5percent_better=0, Subspace_matched_beats_Original=0,
                  Subspace_nominal_beats_noop=0, V1_beats_noop=0)
    for ds, d, snr, ch in itertools_product(['FashionMNIST', 'MNIST'], [32, 64, 128], [10, 20, 30], ['unit', 'rayleigh', 'rayleigh_csi5']):
        oo, vv, ss, rr, nn = [get(ds, d, snr, ch, method=m) for m in ['Original', 'V1', 'Switch', 'SubspaceV1', 'SubspaceV1-nominal']]
        counts['V1_beats_Original'] += vv['MSE_ratio'] < oo['MSE_ratio']
        counts['Switch_beats_stronger_endpoint'] += ss['MSE_ratio'] < min(oo['MSE_ratio'], vv['MSE_ratio'])
        counts['Switch_5percent_better'] += ss['MSE_ratio'] < .95*min(oo['MSE_ratio'], vv['MSE_ratio'])
        counts['Subspace_matched_beats_Original'] += rr['MSE_ratio'] < oo['MSE_ratio']
        counts['Subspace_nominal_beats_noop'] += nn['MSE_ratio'] < 1
        counts['V1_beats_noop'] += vv['MSE_ratio'] < 1
    result = dict(primary=primary, decision=decision, counts=counts, groups=groups, references=references)
    write(OUT/'summary.json', result)
    verification = dict(passed=True, rows=len(rows), groups=len(groups), raw_files=len(files),
                        ledger_max_error=max_ledger_error, equal_mean_bias_max_error=same_bias_error,
                        maximum_MC_standard_error_multiple=max_mc_z, all_power_and_budget_checks=True,
                        source_and_protocol_hashes_match=True, interpretation='implementation audit, not successful unlearning certificate')
    write(OUT/'verification.json', verification)
    lines = ['# DS-Air 보완 실험: 전체 결과', '', 'MSE는 미삭제=1, 정확한 retained ridge=0. 각 H draw 내부32회 평균 -> H4회 평균 -> 3seed 평균±seed SD. 전 데이터셋/조건을 포함한다. 비용은 real-use equivalent이며 측정 RF 시간/에너지가 아니다.', '',
             '| 데이터 | D | SNR | 채널 | H반복 | 방법 | MSE±SD | mean bias | noise variance | 정확도% | 총비용 |',
             '|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"|{g['dataset']}|{g['D']}|{g['SNR']}|{g['channel']}|{g['H_repeats']}|{g['method']}|{g['MSE_ratio']:.5f}±{g['MSE_ratio_seed_SD']:.5f}|{g['bias_ratio']:.5f}|{g['variance_ratio']:.5f}|{100*g['test_accuracy']:.3f}|{g['cost']:.2f}|")
    (ROOT/'RESULT_TABLES_KO.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(verification=verification, decision=decision, counts=counts), ensure_ascii=False, indent=2))
    if plots:
        draw_plots(get)


def itertools_product(*args):
    import itertools
    return itertools.product(*args)


def draw_plots(get):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {'Original':'#777777', 'V1':'#1965b0', 'Switch':'#e07a21', 'SubspaceV1':'#12826b', 'SubspaceV1-nominal':'#9c4c99'}
    for ax, ds in zip(axes[0], ['FashionMNIST', 'MNIST']):
        for method, color in colors.items():
            yy = [get(ds, 64, s, 'rayleigh', method=method)['MSE_ratio'] for s in [10, 20, 30]]
            ax.plot([10, 20, 30], yy, marker='o', label=method, color=color)
        ax.axhline(1, color='black', linestyle=':', label='No-op')
        ax.set(xlabel='Nominal SNR (dB)', ylabel='Normalized parameter MSE (log)', yscale='log', title=f'{ds}: D=64, Rayleigh, H32')
        ax.set_xticks([10, 20, 30])
        ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    ax = axes[1, 0]
    for method in ['Original', 'V1', 'Switch', 'SubspaceV1']:
        xx = [get(hr=r, method=method)['cost']/1000 for r in [8, 32, 128]]
        yy = [get(hr=r, method=method)['MSE_ratio'] for r in [8, 32, 128]]
        ax.plot(xx, yy, marker='o', color=colors[method], label=method)
    ax.axhline(1, color='black', linestyle=':')
    ax.set(xlabel='Total deletion cost (thousand real uses)', ylabel='Normalized parameter MSE (log)', yscale='log', title='Curvature cost: FashionMNIST, D64, 20dB')
    ax.grid(alpha=.18)
    ax = axes[1, 1]
    labels = ['Original', 'V1', 'Switch', 'SubspaceV1', 'SubspaceV1-nominal']
    bias = [get(method=m)['bias_ratio'] for m in labels]
    variance = [get(method=m)['variance_ratio'] for m in labels]
    ax.bar(range(len(labels)), bias, label='Squared mean bias', color='#6b83ad')
    ax.bar(range(len(labels)), variance, bottom=bias, label='Channel variance', color='#d9a441')
    ax.axhline(1, color='black', linestyle=':')
    ax.set_xticks(range(len(labels)), ['Original','V1','Switch','Subspace\nmatched','Subspace\nnominal'])
    ax.set(ylabel='Analytic normalized MSE', title='Primary condition: why full-H methods fail')
    ax.legend(fontsize=8)
    fig.suptitle('DS-Air revision | 2 datasets, 3 new seeds | lower error is better', fontsize=14)
    fig.savefig(ROOT/'comparison.png', dpi=170)
    fig.savefig(ROOT/'comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plots', action='store_true')
    main(ap.parse_args().plots)
