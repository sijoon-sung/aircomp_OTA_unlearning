#!/usr/bin/env python
"""물리 계층(asisa/radio.py) 식 검증. CPU, 수십 초. 실험 전에 한 번 돌린다.

    python check.py
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from asisa.radio import RadioConfig, CodeSystem, transmit_orth, plan

OK = True
def check(name, cond, info=''):
    global OK
    OK &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info)

def main():
    for st in (sys.stdout, sys.stderr):
        try: st.reconfigure(encoding='utf-8', errors='replace')
        except Exception: pass
    torch.manual_seed(0); rng = np.random.default_rng(0)
    D, K = 3000, 4
    groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
    def mkx(n):
        X = torch.randn(n, D, dtype=torch.float64); return X / X.norm(dim=1, keepdim=True) * 0.9
    X = [mkx(5) for _ in range(K)]
    hs = [np.array([0.9, 0.95, 1.0, 0.92, 0.97]), np.array([0.1, 0.6, 0.7, 0.8, 0.9]), np.array([0.5, 0.6, 0.7, 0.8, 0.9]), np.array([0.3, 0.6, 0.7, 0.8, 0.9])]
    delays = rng.uniform(0, 1, 20)
    shards = lambda: [(k, X[k], groups[k], hs[k]) for k in range(K)]

    # 직교 블록
    for al in ['maxpow', 'common']:
        r, l, d = transmit_orth(RadioConfig(sigma2=1e-30, align=al, target=1e-20), X[1], hs[1], 1)
        check(f'직교 블록 무잡음 복원 ({al})', float((r - X[1].mean(0)).abs().max()) < 1e-9, f'전력 상한 비 {l.max_power_ratio:.3f}')
    cfg = RadioConfig(sigma2=1.0, eps=10.0)
    e = [float(transmit_orth(cfg, X[1], hs[1], 100 + t)[2]['noise_energy']) for t in range(200)]
    beta, R = plan(cfg, 5, 0.1, D); a = beta / 5
    check('직교 블록 잡음 분산 = D alpha^2 sigma2 / R', abs(np.mean(e) / (D * a * a / R) - 1) < 0.05, f'측정 {np.mean(e):.4f}, 식 {D * a * a / R:.4f}')
    c100 = RadioConfig(sigma2=100.0, eps=10.0); b, R = plan(c100, 5, 0.1, D)
    check('반복 수 R = ceil(MSE1/eps)', R == math.ceil(D * (b / 5) ** 2 * 100 / 10 - 1e-9), f'R={R}')
    lm = transmit_orth(RadioConfig(sigma2=.01), X[1], hs[1], 1)[1]; lc = transmit_orth(RadioConfig(sigma2=.01, align='common'), X[1], hs[1], 1)[1]
    check('공통 수신 크기는 전력을 낮춰 에너지가 줄고 집계 오차는 eps', lc.energy < lm.energy and abs(lc.mse_sum - 10.0) < 1e-9,
          f'에너지 {lc.energy:.3e} < {lm.energy:.3e}')
    lp = transmit_orth(RadioConfig(sigma2=.01, blocks=4), X[1], hs[1], 1)[1]
    check('블록 4개 동시 사용: 시간 1/4, 블록당 전력 상한 P/4 준수', lp.ul_time == lp.ul_symbols / 4 and lp.max_power_ratio <= 1 + 1e-6)

    # 코드 분할
    for al in ['maxpow', 'common']:
        out = CodeSystem(RadioConfig(sigma2=1e-30, mux='code', L=4, align=al, target=1e-20, n_nom=5), K, 1, np.zeros(20)).transmit(shards(), 7)[0]
        check(f'코드 분할 무잡음 직교 복원 ({al})', max(float((out[k] - X[k].mean(0)).abs().max()) for k in range(K)) < 1e-6)
    res = {al: CodeSystem(RadioConfig(sigma2=1e-30, mux='code', L=4, delay_max=0.3, align=al, target=1.0, n_nom=5), K, 1, delays * 0.3)
           .transmit(shards(), 7, track=(10,))[2] for al in ['maxpow', 'common']}
    bm = [res['maxpow'][k]['beta'] for k in range(K)]
    ratio = res['maxpow'][3]['leak_by_src'][2] / res['common'][3]['leak_by_src'][2]
    check('near-far: 간섭 에너지 비 = (beta 비)^2', abs(ratio / (bm[3] / bm[2]) ** 2 - 1) < 1e-5,
          f'측정 {ratio:.4f}, 식 {(bm[3] / bm[2]) ** 2:.4f}, h_min 비 제곱 {(0.5 / 0.3) ** 2:.4f}')
    check('공통 수신 크기에서 shard 마다 beta 동일', len({round(res['common'][k]['beta'], 12) for k in range(K)}) == 1)
    check('walsh L4 순환 칩 오차의 간섭은 코드 2<->3 사이만', all(res['maxpow'][k]['leak_energy'] < 1e-20 for k in [0, 1]) and res['maxpow'][3]['leak_energy'] > 0)
    check('추적 client 의 간섭이 따로 기록됨', res['maxpow'][3]['u_leak'].get(10, 0) > 0 and 10 not in res['maxpow'][2]['u_leak'])
    sysc = CodeSystem(RadioConfig(sigma2=100.0, eps=10.0, mux='code', L=4, delay_max=0.3), K, 1, delays * 0.3)
    Rs = [plan(sysc.cfg, 5, float(hs[k].min()), D, 4)[1] for k in range(K)]
    nz = {k: [] for k in range(K)}
    for t in range(40):
        out, l, d = sysc.transmit(shards(), 1000 + t)
        for k in range(K):
            nz[k].append(d[k]['noise_energy'])
    for k in range(K):
        a = d[k]['alpha']; pv = D * a * a * 100.0 / (4 * Rs[k])
        check(f'코드 분할 반복 잡음 분산 shard{k} (R={Rs[k]})', abs(np.mean(nz[k]) / pv - 1) < 0.05, f'측정 {np.mean(nz[k]):.3f}, 식 {pv:.3f}')
    check('슬롯 수 = max R, UL 칩 수 = L D max R', all(d[k]['slots'] == max(Rs) for k in range(K)) and l[0].ul_symbols == 4 * D * max(Rs))
    d1 = CodeSystem(RadioConfig(sigma2=100.0, eps=0.0, mux='code', L=4, delay_max=0.3), K, 1, delays * 0.3).transmit(shards(), 5)[2]
    wp = (min(Rs[2], Rs[3]) / Rs[3]) ** 2
    check('반복 시 간섭 희석 = (min(R_j,R_k)/R_k)^2', abs(d[3]['leak_by_src'][2] / d1[3]['leak_by_src'][2] / wp - 1) < 1e-6)
    cc = RadioConfig(sigma2=100.0, eps=10.0, mux='code', L=4, align='common', target=10.0, n_nom=5)
    beta0 = 5 * math.sqrt(10.0 * 4 / (D * 100.0)); pl = [plan(cc, 5, float(hs[k].min()), D, 4) for k in range(K)]
    check('공통 수신 크기: 맞출 수 있는 shard 는 beta0·R=1, 못 맞추는 shard 만 최대 전력 + 반복',
          all((abs(b - beta0) < 1e-12 and R == 1) or (b > beta0 and R >= 1) for b, R in pl), str([(round(b / beta0, 3), R) for b, R in pl]))
    l = CodeSystem(cc, K, 1, np.zeros(20)).transmit(shards(), 9)[1]
    check('전력 상한 준수', max(l[k].max_power_ratio for k in range(K)) <= 1 + 1e-6)

    # ZCZ 코드: 시간 오차가 gap 칩 미만이면 간섭 0, 자기 신호 그대로
    d1s = rng.uniform(0, 1, 20) * 0.999; d2s = rng.uniform(0, 2, 20) * 0.999
    for code, gap, dl, zero in [('zcz', 1, d1s, True), ('zcz', 2, d2s, True), ('zcz', 1, 1 + d1s, False), ('walsh', 1, d1s, False)]:
        cs = CodeSystem(RadioConfig(sigma2=1e-30, mux='code', code=code, L=4, gap=gap, delay_max=2, target=1e-20), K, 1, dl)
        out, l, d = cs.transmit(shards(), 3)
        leak = max(d[k]['leak_energy'] for k in range(K))
        err = max(float((out[k] - X[k].mean(0)).abs().max()) for k in range(K))
        tag = f'{code} gap{gap}, 시간 오차 {dl.min():.2f}~{dl.max():.2f} 칩, 칩 수 {cs.chips}'
        if zero:
            check(f'ZCZ 간섭 0·자기 신호 복원 ({tag})', leak < 1e-20 and err < 1e-6, f'간섭 {leak:.1e}, 복원 오차 {err:.1e}')
        else:
            check(f'영 상관 구간 밖이면 간섭 생김 ({tag})', leak > 1e-8, f'간섭 {leak:.1e}')
    cz = RadioConfig(sigma2=1.0, eps=0, mux='code', code='zcz', L=4, gap=2)
    cs = CodeSystem(cz, K, 1, np.zeros(20)); e = []
    for t in range(60):
        out, l, d = cs.transmit(shards(), 500 + t); e.append(d[1]['noise_energy'])
    a = d[1]['alpha']; pv = D * a * a * 1.0 * 3 / 4
    check('ZCZ 잡음 분산 = D alpha^2 sigma2 (gap+1)/L, 칩 수 = L(gap+1)', abs(np.mean(e) / pv - 1) < 0.05 and cs.chips == 12,
          f'측정 {np.mean(e):.3f}, 식 {pv:.3f}')
    # 약한 shard 에 맞춤
    dw = CodeSystem(RadioConfig(sigma2=1e-30, mux='code', L=4, delay_max=0.3, align='weakest'), K, 1, delays * 0.3).transmit(shards(), 7)[2]
    bw = [dw[k]['beta'] for k in range(K)]
    check('약한 shard 에 맞춤: 모든 shard 의 beta = 가장 약한 shard 의 최대 크기', max(bw) / min(bw) - 1 < 1e-12 and abs(bw[0] - max(bm)) < 1e-12)
    # 직교 블록 전체 정렬: system 이 정한 beta (전체 최약 노드 기준) 로 보낸다
    bf = 1.0 / (0.05 * math.sqrt(D))
    r, l, d = transmit_orth(RadioConfig(sigma2=.01, align='global'), X[1], hs[1], 1, bf)
    check('직교 블록 전체 정렬: beta = 전체 최약 노드 기준, 전력 상한 준수', abs(d['beta'] - bf) < 1e-12 and l.max_power_ratio <= 1 + 1e-6,
          f'beta 비 {d["beta"] / plan(RadioConfig(sigma2=.01), 5, 0.1, D)[0]:.2f} (shard 자기 기준 대비)')
    print('모두 통과' if OK else '실패 항목 있음')
    sys.exit(0 if OK else 1)

if __name__ == '__main__':
    main()
