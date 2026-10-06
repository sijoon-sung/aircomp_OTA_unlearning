#!/usr/bin/env python
"""코드 분리 다중 shard 전송(aircomp_sim/cdma.py)의 식 검증. CPU 에서 수십 초.

    python check_cdma.py

확인 항목
  1. 무잡음 + 완전 직교 walsh 에서 두 전력 정렬 규칙 모두 shard 평균을 그대로 복원한다.
  2. near-far: 강한 shard -> 약한 shard 간섭 에너지의 (최대 전력 / 공통 수신 크기) 비 = (beta_약 / beta_강)^2.
  3. walsh L4 에 순환 칩 오차를 주면 간섭은 코드 2 <-> 3 사이에서만 생긴다.
  4. 삭제 대상처럼 추적하는 client 한 명의 간섭이 따로 기록된다.
  5. 반복: R = ceil(MSE1/eps), 측정 잡음 분산 = D alpha^2 sigma2 / (L R), 슬롯 수 = max R, 간섭은 min(R_j, R_k)/R_k 배로 준다.
  6. 공통 수신 크기에서 전력 상한 때문에 beta0 를 못 맞추는 shard 만 beta_min + 반복, 나머지는 beta0. 전력 상한 준수.
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'sisa_arch_20261006'))
from aircomp_sim import cdma

def main():
    for st in (sys.stdout, sys.stderr):
        try: st.reconfigure(encoding='utf-8', errors='replace')
        except Exception: pass
    torch.manual_seed(0); rng = np.random.default_rng(0)
    D, K, dev = 3000, 4, 'cpu'
    groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
    def mkx(n):
        X = torch.randn(n, D, dtype=torch.float64); return X / X.norm(dim=1, keepdim=True) * 0.9
    X_list = [mkx(5) for _ in range(K)]
    delays = rng.uniform(0, 1, 20)
    hs = [np.array([0.9, 0.95, 1.0, 0.92, 0.97]), np.array([0.1, 0.6, 0.7, 0.8, 0.9]), np.array([0.5, 0.6, 0.7, 0.8, 0.9]), np.array([0.3, 0.6, 0.7, 0.8, 0.9])]
    hl = [(g, hh) for g, hh in zip(groups, hs)]
    ok = True
    def check(name, cond, info=''):
        nonlocal ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info)
    for align in ['maxpow', 'common']:
        c = cdma.CdmaConfig(sigma2=1e-30, code='walsh', L=4, align=align, target=1e-20, n_nom=5)
        r, l, d = cdma.CdmaSystem(c, K, 1, np.zeros(20)).transmit(list(range(K)), X_list, hl, D, 7, dev)
        err = max(float((r[k] - X_list[k].mean(0)).abs().max()) for k in range(K))
        check(f'1. 무잡음 직교 복원 ({align})', err < 1e-6, f'max 오차 {err:.1e}')
    res = {}
    for align in ['maxpow', 'common']:
        c = cdma.CdmaConfig(sigma2=1e-30, code='walsh', L=4, delay_max=0.3, align=align, target=1.0, n_nom=5)
        res[align] = cdma.CdmaSystem(c, K, 1, delays * 0.3).transmit(list(range(K)), X_list, hl, D, 7, dev, track=(10,))[2]
    bm = [res['maxpow'][k]['beta'] for k in range(K)]
    ratio = res['maxpow'][3]['leak_by_src'][2] / res['common'][3]['leak_by_src'][2]
    pred = (bm[3] / bm[2]) ** 2
    check('2. near-far 증폭 = (beta 비)^2', abs(ratio / pred - 1) < 1e-5, f'측정 {ratio:.4f}, 식 {pred:.4f} (h_min 비 제곱 {(0.5 / 0.3) ** 2:.4f})')
    check('2. 공통 수신 크기에서 beta 동일', len({round(res['common'][k]['beta'], 12) for k in range(K)}) == 1)
    check('3. walsh 순환 칩 오차의 간섭은 코드 2<->3 사이만', all(res['maxpow'][k]['leak_energy'] < 1e-20 for k in [0, 1]) and res['maxpow'][3]['leak_energy'] > 0)
    check('4. 추적 client(코드 2 shard 의 client 10) 간섭 기록', res['maxpow'][3]['u_leak'].get(10, 0) > 0 and 10 not in res['maxpow'][2]['u_leak'])
    c = cdma.CdmaConfig(sigma2=100.0, code='walsh', L=4, delay_max=0.3, align='maxpow', eps=10.0)
    s = cdma.CdmaSystem(c, K, 1, delays * 0.3)
    Rs = [s.plan(5, float(hs[k].min()), D)[1] for k in range(K)]
    nz = {k: [] for k in range(K)}
    for t in range(40):
        r, l, d = s.transmit(list(range(K)), X_list, hl, D, 1000 + t, dev)
        for k in range(K):
            nz[k].append(d[k]['noise_energy'])
    for k in range(K):
        a = d[k]['alpha']; pv = D * a * a * 100.0 / (4 * Rs[k]); mv = float(np.mean(nz[k]))
        check(f'5. 반복 잡음 분산 shard{k} (R={Rs[k]})', abs(mv / pv - 1) < 0.05, f'측정 {mv:.3f}, 식 {pv:.3f}')
    check('5. 슬롯 수 = max R, UL 심볼 = L D max R', all(d[k]['slots'] == max(Rs) for k in range(K)) and l[0].ul_symbols == 4 * D * max(Rs))
    d1 = cdma.CdmaSystem(cdma.CdmaConfig(sigma2=100.0, code='walsh', L=4, delay_max=0.3, align='maxpow', eps=0.0), K, 1, delays * 0.3).transmit(list(range(K)), X_list, hl, D, 5, dev)[2]
    wp = (min(Rs[2], Rs[3]) / Rs[3]) ** 2; wm = d[3]['leak_by_src'][2] / d1[3]['leak_by_src'][2]
    check('5. 반복 시 간섭 희석 = (min(R_j,R_k)/R_k)^2', abs(wm / wp - 1) < 1e-6, f'측정 {wm:.4f}, 식 {wp:.4f}')
    c = cdma.CdmaConfig(sigma2=100.0, code='walsh', L=4, align='common', target=10.0, n_nom=5, eps=10.0)
    s = cdma.CdmaSystem(c, K, 1, np.zeros(20))
    beta0 = 5 * math.sqrt(10.0 * 4 / (D * 100.0))
    pl = [s.plan(5, float(hs[k].min()), D) for k in range(K)]
    check('6. 공통 수신 크기: 맞출 수 있는 shard 는 beta0·R=1, 못 맞추는 shard 만 beta_min + 반복',
          all((abs(b - beta0) < 1e-12 and R == 1) or (b > beta0 and R >= 1) for b, R in pl), str([(round(b / beta0, 3), R) for b, R in pl]))
    l = s.transmit(list(range(K)), X_list, hl, D, 9, dev)[1]
    check('6. 전력 상한 준수', max(l[k].max_power_ratio for k in range(K)) <= 1 + 1e-6, f'최대 {max(l[k].max_power_ratio for k in range(K)):.4f}')
    print('모두 통과' if ok else '실패 항목 있음')
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
