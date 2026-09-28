"""Correct full-transcript privacy for the informative cyclic prefix.

Frozen main results are preserved. No accuracy or communication result changes.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import importlib.util
import json
import math
import statistics
import time
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_sweep', ROOT/'run_experiment.py')
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
base = exp.base


@torch.no_grad()
def main(paths):
    start = time.perf_counter()
    done = json.loads((exp.OUT/'completion.json').read_text())
    assert done['source_sha256'] == base.sha(ROOT/'run_experiment.py')
    assert done['dependency_sha256'] == base.sha(exp.DEP)
    # One transmitted frame is T s, with a repeated suffix. T^T T has
    # diagonal 1 (259 coordinates) and 2 (37 coordinates).
    eye = torch.eye(exp.Q, device=exp.DEV, dtype=exp.DT)
    transform = torch.cat((eye[-exp.CP:], eye))
    gram = transform.T @ transform
    diag = torch.ones(exp.Q, device=exp.DEV, dtype=exp.DT)
    diag[-exp.CP:] = 2
    assert torch.equal(gram, torch.diag(diag))
    assert float(torch.linalg.eigvalsh(gram).max()) == 2
    # Check precision against explicit 333-coordinate Gaussian observations.
    d = base.normal((exp.Q,), 91928)
    explicit = float((transform @ d).square().sum())
    formula = float(d.square().sum()+d[-exp.CP:].square().sum())
    assert abs(explicit-formula) < 1e-10
    grouped = defaultdict(list)
    for name, path in paths.items():
        assert {p.name: base.sha(p) for p in sorted(path.glob('*ubyte'))} == done['data_hashes'][name]
        x, y, xt, yt = base.dataset(path)
        proj = base.normal((784, exp.D-1), 2026092890).float()/math.sqrt(784)
        z = torch.cat((F.relu((x-.5)@proj), torch.ones(len(x), 1, device=exp.DEV)), 1).to(exp.DT)
        del x, xt, yt
        for seed in done['seeds']:
            for k in exp.KS:
                obj = json.loads((exp.OUT/f'{name}_seed{seed}_K{k}.json').read_text())
                ids = base.partition(y, seed, k)
                stats, _, _ = base.stats(z, y, ids)
                norm = float(stats[0].square().sum())
                prefix_norm = float(stats[0, -exp.CP:].square().sum())
                for row in obj['rows']:
                    # rho_A in the frozen file uses sensitivity 2B in a
                    # prefix-discarding receiver. An actual s_A vs empty test
                    # has Mahalanobis d^2 = rho_A*(||s_A||^2+||s_A,CP||^2)/2.
                    d2 = row['rho_A']*(norm+prefix_norm)/2
                    auc = .5*(1+math.erf(math.sqrt(d2/2)/math.sqrt(2)))
                    key = (name, k, row['condition'], row['rho_cap'], row['method'])
                    grouped[key].append(dict(
                        full_frame_rho_cap=2*row['rho_cap'],
                        full_frame_epsilon_cap=exp.eps(2*row['rho_cap']),
                        full_frame_rho_A_bound=2*row['rho_A'],
                        full_frame_rho_all_bound=2*row['rho_all'],
                        full_frame_oracle_AUC=auc,
                        cap_violation=row['rho_all'] > row['rho_cap']+1e-9,
                        prefix_norm_fraction=prefix_norm/norm))
        del z, y
    groups = {}
    for key, rows in grouped.items():
        value = {field: statistics.mean(row[field] for row in rows) for field in rows[0]}
        value['full_frame_rho_all_bound_max'] = max(row['full_frame_rho_all_bound'] for row in rows)
        groups['|'.join(map(str, key))] = value
    result = dict(passed=True, correction='Copied cyclic prefix supplies extra observations; use T^T T, not receiver-discarded payload alone.',
                  private_statistic_dimension=exp.Q, copied_coordinates=exp.CP,
                  sensitivity_squared_multiplier=2, group_count=len(groups),
                  main_source_sha256=done['source_sha256'], audit_sha256=base.sha(__file__),
                  original_results_unchanged=True, elapsed_seconds=time.perf_counter()-start,
                  gpu=torch.cuda.get_device_name(), groups=groups)
    assert len(groups) == 450
    base.write(exp.OUT/'full_frame_accounting.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fashion', required=True, type=Path)
    parser.add_argument('--mnist', required=True, type=Path)
    args = parser.parse_args()
    main({'FashionMNIST':args.fashion, 'MNIST':args.mnist})
