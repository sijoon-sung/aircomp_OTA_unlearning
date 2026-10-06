#!/usr/bin/env python
"""SISA–AirComp 아키텍처 실험 원클릭 실행기.

    python run_all.py            # 전체 실행 (G0 -> E1 -> E2 -> E3 -> E4 -> E5)
    python run_all.py --quick    # 2~5분짜리 동작 확인
    python run_all.py --only e2  # 일부만 (G0 결과 eps_star.json 이 같은 --out 에 있어야 함)

결과: runs/<시각>_<호스트>/ 아래 실험별 폴더(REPORT_KO.md, results.json)와 SUMMARY_KO.md.
"""
import argparse, importlib, json, socket, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ORDER = ['g0', 'e1', 'e2', 'e3', 'e4', 'e5', 'x1', 'x2']
MODULES = {'g0': 'g0_noise', 'e1': 'e1_stable', 'e2': 'e2_entry', 'e3': 'e3_lifetime', 'e4': 'e4_bandwidth', 'e5': 'e5_guard', 'x1': 'x1_lifecycle', 'x2': 'x2_radio'}
TITLES = {'g0': 'G0 잡음 허용 한계', 'e1': 'E1 삭제해도 바뀌지 않는 배정', 'e2': 'E2 채널 순서 참여 시점',
          'e3': 'E3 연속 삭제와 shard 크기', 'e4': 'E4 삭제 중 대역 몰아주기', 'e5': 'E5 guard 대역과 누설', 'x1': 'X1 수명 비용과 손익분기', 'x2': 'X2 무선 제어 격리와 노출'}

def parse():
    ap = argparse.ArgumentParser(description='SISA-AirComp 아키텍처 실험')
    ap.add_argument('--data', default=str(HERE / 'data'), help='데이터 폴더 (없으면 FashionMNIST 자동 다운로드)')
    ap.add_argument('--out', default=None, help='결과 폴더 (기본: runs/<시각>_<호스트>)')
    ap.add_argument('--device', default='auto', help='auto / cuda / cuda:1 / cpu')
    ap.add_argument('--dataset', default='fashionmnist', choices=['fashionmnist', 'mnist'])
    ap.add_argument('--seeds', type=int, default=5, help='GPU 학습 seed 수')
    ap.add_argument('--draws', type=int, default=1000, help='CPU 해석 채널 draw 수 (E1, E2, E4)')
    ap.add_argument('--draws-e3', type=int, default=300, help='E3 채널 draw 수')
    ap.add_argument('--seqs', type=int, default=20, help='E3 draw 당 삭제 순서 수')
    ap.add_argument('--cost-draws', type=int, default=200, help='E1 비용 계산 draw 수')
    ap.add_argument('--draws-x1', type=int, default=300, help='X1 채널 draw 수')
    ap.add_argument('--search-draws', type=int, default=100, help='X1 배정 탐색 draw 수')
    ap.add_argument('--rounds', type=int, default=160)
    ap.add_argument('--chunk', type=int, default=64, help='한 번에 계산할 client 수. GPU 메모리 부족이면 64 나 32')
    ap.add_argument('--eps', type=float, default=None, help='G0 대신 쓸 집계 오차 한계(직접 지정)')
    ap.add_argument('--only', default=None, help='예: g0,e2')
    ap.add_argument('--skip', default=None, help='예: e5')
    ap.add_argument('--quick', action='store_true', help='작은 규모로 동작만 확인')
    return ap.parse_args()

def main():
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    a = parse()
    from sisa_arch import common
    common.setup_torch()
    device = common.pick_device(a.device)
    if a.quick:
        a.seeds, a.draws, a.draws_e3, a.seqs, a.cost_draws, a.rounds, a.draws_x1, a.search_draws = 1, 12, 3, 3, 6, 20, 4, 2
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = Path(a.out) if a.out else HERE / 'runs' / f'{stamp}_{socket.gethostname()}{"_quick" if a.quick else ""}'
    out.mkdir(parents=True, exist_ok=True)
    log = common.Logger(out / 'run_log.txt')
    env = common.environment(device); env['args'] = vars(a)
    common.write_json(out / 'environment.json', env)
    log(f'결과 폴더: {out}')
    log(f'장치: {device} {env.get("gpu", "")} / torch {env["torch"]}')
    if device == 'cpu' and a.device == 'auto':
        log('주의: GPU 를 찾지 못했다. CPU 로도 돌지만 매우 느리다(--quick 로 먼저 확인 권장).')
    raw = common.ensure_dataset(a.data, a.dataset)
    log(f'데이터: {raw}')

    todo = [e for e in ORDER if (not a.only or e in a.only.split(',')) and (not a.skip or e not in a.skip.split(','))]
    cfg = dict(out=out, raw=raw, device=device, seeds=[71001 + k for k in range(a.seeds)], draws=a.draws, draws_e3=a.draws_e3,
               seqs=a.seqs, cost_draws=a.cost_draws, T=a.rounds, chunk=a.chunk, quick=a.quick, eps=a.eps, repetition_matters=None,
               draws_x1=a.draws_x1, search_draws=a.search_draws)
    status = {}
    for e in todo:
        if e != 'g0' and cfg['eps'] is None:
            p = out / 'eps_star.json'
            if p.exists():
                g0 = common.read_json(p); cfg['eps'] = g0['eps_used']; cfg['repetition_matters'] = g0.get('repetition_matters')
                log(f'G0 결과 사용: eps = {cfg["eps"]:g}')
            else:
                cfg['eps'] = 1e-4
                log('경고: G0 결과가 없어 이전 실험의 eps=1e-4 를 쓴다(--eps 로 지정 가능).')
        log(f'===== {TITLES[e]} 시작 =====')
        mod = importlib.import_module(f'sisa_arch.{MODULES[e]}')
        t0 = time.time()
        try:
            with common.PowerMeter() as pm:
                res = mod.run(cfg, log)
            res.update(status='완료', **pm.report())
            if e == 'g0':
                cfg['repetition_matters'] = res['repetition_matters']
                if a.eps is None:
                    cfg['eps'] = res['eps_used']
        except Exception as ex:
            res = dict(name=e, status='실패', error=repr(ex), trace=traceback.format_exc(), seconds=time.time() - t0)
            log(f'!!! {e} 실패: {ex!r}')
            log(res['trace'])
            if 'out of memory' in repr(ex).lower():
                log('GPU 메모리 부족: --chunk 64 (또는 32) 로 다시 실행하고 --only 로 남은 실험만 돌리면 된다.')
        status[e] = res
        common.write_json(out / 'status.json', status)
        log(f'===== {TITLES[e]} 끝: {res["status"]}, {res.get("seconds", 0) / 60:.1f}분 =====')
    write_summary(out, status, cfg)
    log(f'전체 완료. 요약: {out / "SUMMARY_KO.md"}')

def write_summary(out, status, cfg):
    from sisa_arch.report import table, fmt
    rows = []
    for e in ORDER:
        if e not in status:
            continue
        r = status[e]
        v = '; '.join(f'{n}: {"지지" if ok is True else "미지지" if ok is False else "보류"}' for n, ok in r.get('verdicts', []))
        rows.append([TITLES[e], r['status'], v or r.get('error', ''), f'{r.get("seconds", 0) / 60:.1f}',
                     fmt(r.get('gpu_board_Wh')) if r.get('gpu_board_Wh') is not None else '-'])
    L = ['# SISA–AirComp 아키텍처 실험 요약', '', f'eps(집계 오차 한계) = {cfg["eps"]}', '',
         table(['실험', '상태', '판정', '시간(분)', 'GPU 보드 Wh'], rows), '',
         '각 실험의 표와 판정 근거는 아래에 이어 붙였다. 원자료는 실험 폴더의 results.json 이다.', '']
    for e in ORDER:
        p = out / MODULES[e] / 'REPORT_KO.md'
        if p.exists():
            L += ['---', '', p.read_text(encoding='utf-8'), '']
    (out / 'SUMMARY_KO.md').write_text('\n'.join(L), encoding='utf-8')

if __name__ == '__main__':
    main()
