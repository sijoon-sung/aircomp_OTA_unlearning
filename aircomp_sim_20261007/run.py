#!/usr/bin/env python
"""라운드 단위 AirComp-SISA 시뮬레이션 실행기.

    python run.py                 # P0(문제 상황 확인) -> P1(배정 트레이드오프)
    python run.py --quick         # 수 분짜리 동작 확인
    python run.py --only p1
결과: runs/<시각>_<호스트>/{p0_problem,p1_assignment}/REPORT_KO.md, results.json, SUMMARY_KO.md
"""
import argparse, importlib, socket, sys, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'sisa_arch_20261006'))

ORDER = ['p0', 'p1']
MODULES = {'p0': 'exp_p0_problem', 'p1': 'exp_p1_assignment'}
TITLES = {'p0': 'P0 문제 상황 확인', 'p1': 'P1 배정 트레이드오프'}

def main():
    for st in (sys.stdout, sys.stderr):
        try: st.reconfigure(encoding='utf-8', errors='replace')
        except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(HERE / 'data'))
    ap.add_argument('--dataset', default='fashionmnist', choices=['fashionmnist', 'mnist'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=160)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--eps', type=float, default=10.0, help='허용 집계 오차 (G0 측정값 10)')
    ap.add_argument('--only', default=None); ap.add_argument('--skip', default=None)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    from sisa_arch import common
    common.setup_torch(); device = common.pick_device(a.device)
    if a.quick:
        a.seeds, a.rounds = 1, 20
    out = Path(a.out) if a.out else HERE / 'runs' / f'{time.strftime("%Y%m%d_%H%M%S")}_{socket.gethostname()}{"_quick" if a.quick else ""}'
    out.mkdir(parents=True, exist_ok=True)
    log = common.Logger(out / 'run_log.txt'); env = common.environment(device); env['args'] = vars(a)
    common.write_json(out / 'environment.json', env)
    log(f'결과 폴더: {out}'); log(f'장치: {device} {env.get("gpu", "")}')
    raw = common.ensure_dataset(a.data, a.dataset); log(f'데이터: {raw}')
    cfg = dict(out=out, raw=raw, device=device, seeds=[81001 + k for k in range(a.seeds)], T=a.rounds, chunk=a.chunk, quick=a.quick, eps=a.eps)
    todo = [e for e in ORDER if (not a.only or e in a.only.split(',')) and (not a.skip or e not in a.skip.split(','))]
    status = {}
    for e in todo:
        log(f'===== {TITLES[e]} 시작 ====='); t0 = time.time()
        mod = importlib.import_module(f'aircomp_sim.{MODULES[e]}')
        try:
            with common.PowerMeter() as pm:
                res = mod.run(cfg, log)
            res.update(status='완료', **pm.report())
        except Exception as ex:
            res = dict(name=e, status='실패', error=repr(ex), trace=traceback.format_exc(), seconds=time.time() - t0)
            log(f'!!! {e} 실패: {ex!r}'); log(res['trace'])
        status[e] = res; common.write_json(out / 'status.json', status)
        log(f'===== {TITLES[e]} 끝: {res["status"]}, {res.get("seconds", 0) / 60:.1f}분 =====')
    L = ['# 라운드 단위 AirComp-SISA 시뮬레이션 요약', '', '| 실험 | 상태 | 시간(분) |', '|---|---|---|']
    L += [f'| {TITLES[e]} | {status[e]["status"]} | {status[e].get("seconds", 0) / 60:.1f} |' for e in todo]
    for e in todo:
        p = out / {'p0': 'p0_problem', 'p1': 'p1_assignment'}[e] / 'REPORT_KO.md'
        if p.exists(): L += ['', '---', '', p.read_text(encoding='utf-8')]
    (out / 'SUMMARY_KO.md').write_text('\n'.join(L), encoding='utf-8'); log(f'요약: {out / "SUMMARY_KO.md"}')

if __name__ == '__main__':
    main()
