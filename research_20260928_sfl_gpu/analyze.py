"""Aggregate independent training seeds, not channel draws as extra seeds."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SCENARIOS = ["fashion20", "fashion10", "width20"]


def stat(xs):
    x = np.asarray(xs, dtype=float)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x)>1 else 0., values=x.tolist())


def main():
    summary, manifest = {}, {}
    for scenario in SCENARIOS:
        grouped = {}
        for path in sorted((ROOT/scenario).glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            grouped.setdefault(row["method"], []).append(row)
            manifest[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary[scenario] = {}
        for method, rows in grouped.items():
            assert sorted(r["seed"] for r in rows) == [311, 312, 313], (scenario, method)
            assert len({r["code_sha"] for r in rows}) == 1
            tests = {}
            for condition in rows[0]["tests"]:
                tests[condition] = {k: stat([np.mean([v[k] for v in r["tests"][condition]]) for r in rows])
                                    for k in ["accuracy", "loss"]}
            ss = dict(seeds=[r["seed"] for r in rows], config=rows[0]["config"], tests=tests,
                      ledger={k: stat([r["ledger"][k] for r in rows]) for k in rows[0]["ledger"]},
                      compute=rows[0]["compute"], gpu_wall_seconds=stat([r["gpu_wall_seconds"] for r in rows]),
                      correction_gpu_ms=stat([r["correction_gpu_milliseconds"] for r in rows]),
                      peak_cuda_bytes=stat([r["process_peak_cuda_bytes"] for r in rows]),
                      traces=[r["trace_mean"] for r in rows])
            # A secondary validation-budget diagnostic; not a test accuracy guarantee.
            hits = [next((v for v in r["history"] if v["validation"]["accuracy"] >= .6), None) for r in rows]
            ss["validation_60pct"] = dict(reached=sum(v is not None for v in hits),
                                         radio_seconds=[v["ledger"]["modeled_radio_seconds"] if v else None for v in hits])
            summary[scenario][method] = ss
    # Check paired data allocation, and shared cost for identical architecture.
    for scenario in SCENARIOS:
        for seed in [311, 312, 313]:
            rows = [json.loads(p.read_text(encoding="utf-8")) for p in (ROOT/scenario).glob(f"*_seed{seed}.json")]
            assert len({r["dataset"]["train_index_sha"] for r in rows}) == 1
            assert len({json.dumps(r["dataset"]["cohort_histograms"]) for r in rows}) == 1
            for r in rows:
                assert np.isfinite(r["ledger"]["total_uses"])
                if r["config"]["kind"] == "ota":
                    assert r["ledger"]["max_block_power"] <= 1.00001
                assert abs(r["ledger"]["total_uses"]-sum(v for k,v in r["ledger"].items()
                           if k.endswith("_uses") and k!="total_uses")) < 1e-5
    comparisons = {}
    for scenario, methods in summary.items():
        cond = "rician_10dB_csi0" if scenario == "fashion10" else "rician_20dB_csi0"
        comparisons[scenario] = {}
        for left, right in [("OTA-Stein-Q16-W8", "OTA-Q16-W8"),
                            ("OTA-Stein-Q16-W8", "OTA-Reg-Q16-W8"),
                            ("OTA-Q16-W8", "Latent4-Q16-W8"),
                            ("OTA-Reg-Q16-W8", "Latent4-Reg-Q16-W8"),
                            ("Latent4-Reg-Q16-W8", "Latent4-Q16-W8"),
                            ("OTA-Stein-Q16-HET", "OTA-Reg-Q16-HET")]:
            if left not in methods or right not in methods:
                continue
            a,b = methods[left], methods[right]
            d = np.array(a["tests"][cond]["accuracy"]["values"])-np.array(b["tests"][cond]["accuracy"]["values"])
            aa,bb = a["ledger"]["modeled_radio_seconds"]["mean"],b["ledger"]["modeled_radio_seconds"]["mean"]
            comparisons[scenario][left+" vs "+right] = dict(accuracy_difference_pp=stat(100*d),
                                                          radio_reduction_percent=100*(1-aa/bb))
    result = dict(scenarios=summary, paired_comparisons=comparisons, audit_passed=True,
                  note="SD across 3 independent training seeds; channel draws averaged within each seed")
    (ROOT/"summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    for p in Path(__file__).resolve().parent.glob("*.py"):
        manifest['code/'+p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['math_validation.json','noise_gradient_validation.json','channel_sum_validation.json']:
        p=Path(__file__).resolve().parent/name
        manifest['source/'+name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    lines = ["# GPU 결과 전체 표", "", "평균 ± 학습 시드 3개의 sample SD. 같은 시드의 test 채널 3회는 먼저 평균했다. 시간은 1MHz 무선 모델의 총 점유시간이며 실측 RF latency가 아니다.", ""]
    for scenario, methods in summary.items():
        cond = "rician_10dB_csi0" if scenario == "fashion10" else "rician_20dB_csi0"
        lines += [f"## {scenario}: {cond}", "", "|방법|정확도 %|Radio s|UL energy J|DL energy J|Client max MAC/sample|GPU 학습+val s|보정 CUDA ms|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, row in methods.items():
            acc = row["tests"][cond]["accuracy"]
            l = row["ledger"]
            lines.append(f"|{name}|{100*acc['mean']:.2f} ± {100*acc['sd']:.2f}|{l['modeled_radio_seconds']['mean']:.3f}|{l['modeled_ul_joules_at_1W_1MHz']['mean']:.3f}|{l['modeled_dl_joules_at_1W_1MHz']['mean']:.3f}|{row['compute']['client_forward_MAC_per_sample']:,}|{row['gpu_wall_seconds']['mean']:.3f}|{row['correction_gpu_ms']['mean']:.1f}|")
        lines += ["", "|방법|Forward %|Backward %|Model sync %|Control+pilot %|Raw input %|", "|---|---:|---:|---:|---:|---:|"]
        for name,row in methods.items():
            l={k:v['mean'] for k,v in row['ledger'].items()}
            if not l['total_uses']:
                continue
            values=[l['forward_uses'],l['backward_uses'],l['sync_ul_uses']+l['sync_dl_uses'],l['control_uses']+l['pilot_uses'],l.get('raw_input_dl_uses',0.)]
            lines.append('|'+name+'|'+ '|'.join(f'{100*v/l["total_uses"]:.2f}' for v in values)+'|')
        lines += ["", "|방법|Rician20 정확도 %|Rician10 정확도 %|Rayleigh20 정확도 %|CSI RMS5% 정확도 %|", "|---|---:|---:|---:|---:|"]
        for name,row in methods.items():
            values=[row['tests'][key]['accuracy']['mean'] for key in ['rician_20dB_csi0','rician_10dB_csi0','rayleigh_20dB_csi0','rician_20dB_csi0.05']]
            lines.append('|'+name+'|'+'|'.join(f'{v*100:.2f}' for v in values)+'|')
        lines.append('')
    (ROOT/"RESULT_TABLES_KO.md").write_text('\n'.join(lines).rstrip()+'\n', encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    names=['Latent4-Q16-W8','OTA-Q16-W8','OTA-Stein-Q16-W8','OTA-Reg-Q16-W8','Latent4-Reg-Q16-W8']
    colors=['#6c7785','#3875b6','#c47828','#42916c','#845998']
    for ax,scenario in zip(axes,SCENARIOS):
        methods=summary[scenario]
        cond='rician_10dB_csi0' if scenario=='fashion10' else 'rician_20dB_csi0'
        for name,col in zip(names,colors):
            if name not in methods:
                continue
            row=methods[name]; ac=row['tests'][cond]['accuracy']
            ax.errorbar(row['ledger']['modeled_radio_seconds']['mean'],ac['mean']*100,
                        yerr=ac['sd']*100,fmt='o',color=col,capsize=3,label=name.replace('-Q16-W8',''))
        ax.set_title(scenario+' / mean ± seed SD'); ax.set_xlabel('Modeled radio time (s, 1 MHz)')
        ax.set_ylabel('Final test accuracy (%)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='best')
    fig.tight_layout(); fig.savefig(ROOT/'tradeoffs.png',dpi=160); plt.close(fig)
    print(json.dumps(comparisons,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    ROOT = ap.parse_args().root.resolve()
    main()
