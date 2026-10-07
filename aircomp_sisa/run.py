#!/usr/bin/env python
"""AirComp-SISA 실험 실행기.

    python run.py                          # 전부 (noise ... placement, control, stability, differencing, dropout)
    python run.py --only control,stability,differencing,dropout   # E1~E4
    python run.py --only lifecycle,subcarrier                     # E2b, R1~R3
    python run.py --only retrain,earlystop,sharding,fairness      # 추가 실험 B, D, A·E, C
    python run.py --only interference      # 일부만 (쉼표로 여러 개)
    python run.py --quick                  # seed 1개, 20라운드, 축소 격자 (동작 확인용, 수치는 의미 없음)

결과: runs/<시각>_<호스트>/<실험>/REPORT_KO.md, results.json, 그리고 SUMMARY_KO.md (보고서 모음)
"""
import argparse, importlib, socket, sys, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from asisa import config
from asisa.data import ensure_dataset
from asisa.util import setup_torch, pick_device, Logger, environment, write_json, PowerMeter

ORDER = ['noise', 'resources', 'assignment', 'interference', 'codes', 'placement', 'control', 'stability', 'differencing', 'dropout', 'lifecycle', 'subcarrier',
         'retrain', 'earlystop', 'sharding', 'fairness']
TITLE = {'noise': '허용 집계 잡음', 'resources': '자원·삭제 비용·노출', 'assignment': '배정의 학습·삭제 비용', 'interference': 'shard 간 간섭과 도착 크기 맞추기', 'codes': '시간 오차에 강한 코드(ZCZ)', 'placement': '같은 자원을 쓰는 shard 의 배정', 'control': 'E1 shard 밖 제어 (전체 정렬·스케줄링)', 'stability': 'E2 배정 안정성과 재학습 범위', 'differencing': 'E3 삭제 전후 합의 차분', 'dropout': 'E4 학습 중 이탈', 'lifecycle': 'E2b 배정·삭제 처리의 누적 비용', 'subcarrier': 'R1~R3 부반송파·시간 배치 (FDMA, OFDMA, 혼합, 보호 대역)',
         'retrain': 'B 재학습 전송이 다른 shard 에 남기는 흔적', 'earlystop': 'D 전체 검증으로 조기 종료', 'sharding': 'A·E shard 수에 따른 정확도와 추론 비용',
         'fairness': 'C 남의 삭제 비용을 누가 내는가'}

def main():
    for st in (sys.stdout, sys.stderr):
        try: st.reconfigure(encoding='utf-8', errors='replace')
        except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='쉼표로 구분: ' + ','.join(ORDER))
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=config.ROUNDS)
    ap.add_argument('--eps', type=float, default=config.EPS, help='허용 집계 오차 (noise 실험의 측정값)')
    ap.add_argument('--dataset', default='fashionmnist', choices=['fashionmnist', 'mnist'])
    ap.add_argument('--data', default=str(HERE / 'data'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--chunk', type=int, default=64, help='vmap 묶음 크기. GPU 메모리가 부족하면 32')
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.rounds = 1, 20
    todo = [e for e in ORDER if not a.only or e in a.only.split(',')]
    setup_torch(); device = pick_device(a.device)
    out = Path(a.out) if a.out else HERE / 'runs' / f'{time.strftime("%Y%m%d_%H%M%S")}_{socket.gethostname()}{"_quick" if a.quick else ""}'
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / 'run_log.txt')
    write_json(out / 'environment.json', dict(environment(device), args=vars(a)))
    log(f'결과 폴더: {out}'); log(f'장치: {device}')
    cfg = dict(out=out, raw=ensure_dataset(a.data, a.dataset), device=device, seeds=[config.SEED0 + k for k in range(a.seeds)],
               T=a.rounds, chunk=a.chunk, quick=a.quick, eps=a.eps)
    status = {}
    for e in todo:
        log(f'===== {e}: {TITLE[e]} 시작 =====')
        try:
            with PowerMeter() as pm:
                res = importlib.import_module(f'asisa.exp.{e}').run(cfg, log)
            status[e] = dict(res or {}, status='완료', **pm.report())
        except Exception as ex:
            status[e] = dict(status='실패', error=repr(ex), trace=traceback.format_exc())
            log(f'!!! {e} 실패: {ex!r}'); log(status[e]['trace'])
        write_json(out / 'status.json', status)
        log(f'===== {e} 끝: {status[e]["status"]}, {status[e].get("seconds", 0) / 60:.1f}분 =====')
    L = ['# AirComp-SISA 실험 요약', '', '| 실험 | 상태 | 시간(분) |', '|---|---|---|']
    L += [f'| {e} ({TITLE[e]}) | {status[e]["status"]} | {status[e].get("seconds", 0) / 60:.1f} |' for e in todo]
    for e in todo:
        p = out / e / 'REPORT_KO.md'
        if p.exists():
            L += ['', '---', '', p.read_text(encoding='utf-8')]
    (out / 'SUMMARY_KO.md').write_text('\n'.join(L), encoding='utf-8')
    log(f'요약: {out / "SUMMARY_KO.md"}')

if __name__ == '__main__':
    main()
