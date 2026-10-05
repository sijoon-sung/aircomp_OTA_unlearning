"""AirComp 한 shard 의 라운드 비용과 잡음 — 해석식(numpy). 학습과 시뮬레이터가 같은 식을 쓴다.

채널 역보상 AirComp (완전 CSI, 위상 보상), client update 는 L2<=C 로 clip.
  beta_phys = C / (n sqrt(D P_eff) h_min)   (가장 약한 client 가 전력 상한에 닿는 scale)
  1회 전송 집계 오차(벡터 전체 제곱합)  MSE1 = D beta^2 sigma2 = C^2 sigma2 / (n^2 P_eff h_min^2)
같은 update 를 R 번 보내 평균하면 MSE = MSE1 / R.
송신 에너지(공개 한계 C 기준 상한, 정규화 단위):
  E = C^2 D sigma2 sum_i h_i^-2 / (n^2 MSE_achieved)
  -> 같은 집계 오차를 얻는 데 드는 에너지는 반복으로 얻든 전력으로 얻든 같다.

모드
  none       잡음 없음
  mse        집계 오차를 eps 로 직접 지정(G0 의 통제 변수)
  R1         전력 상한으로 1회 전송(잡음은 채널에 따라 달라짐)
  fixed_mse  전력 상한으로 보내고 MSE<=eps 가 될 때까지 반복(R 증가)
  min_energy MSE1<=eps 이면 전력을 낮춰 MSE=eps 로 맞춤(R=1), 아니면 fixed_mse 와 같음
전력 제약
  symbol     심볼당 평균 전력 <= P (대역을 넓혀도 심볼당 전력 유지)
  total      client 총 전력 <= P. 대역 b 배면 심볼당 전력 P/b
시간 단위: 기본 블록(대역 1) 에서 심볼 1개를 보내는 시간. 대역 b 이면 같은 심볼 수가 1/b 시간.
"""
import math
import numpy as np
from .common import CLIP, P_MAX, D_PARAMS

DL_BITS_PER_USE = 2.0   # 디지털 downlink·control 환산(이전 ledger 와 같음)

def mse1(n, hmin, sigma2, bw=1.0, power='symbol', C=CLIP, P=P_MAX):
    p_eff = P / bw if power == 'total' else P
    n = np.asarray(n, dtype=float); hmin = np.asarray(hmin, dtype=float)
    return C * C * sigma2 / (n * n * p_eff * hmin * hmin)

def round_terms(n, hmin, suminv, mode, sigma2=0.0, eps=None, bw=1.0, power='symbol', D=D_PARAMS):
    """벡터화: n, hmin, suminv(=sum h_i^-2) 배열(라운드별). n==0 인 라운드는 비용 0.
    반환 dict of arrays: R, mse(달성 집계 오차), var(좌표당 잡음 분산), energy, ul_uses, ul_time, dl_time, total_re."""
    n = np.asarray(n, dtype=float); hmin = np.asarray(hmin, dtype=float); suminv = np.asarray(suminv, dtype=float)
    active = n > 0
    nn = np.where(active, n, 1.0); hh = np.where(active, hmin, 1.0); ss = np.where(active, suminv, 0.0)
    R = np.ones_like(nn)
    if mode == 'none' or (sigma2 == 0 and mode != 'mse'):
        mse = np.zeros_like(nn)
    elif mode == 'mse':
        mse = np.full_like(nn, float(eps))
    else:
        m1 = mse1(nn, hh, sigma2, bw, power)
        if mode == 'R1':
            mse = m1
        elif mode in ('fixed_mse', 'min_energy'):
            R = np.maximum(1.0, np.ceil(m1 / eps - 1e-9))
            mse = m1 / R
            if mode == 'min_energy':
                mse = np.where(m1 <= eps, float(eps), mse)
        else:
            raise ValueError(mode)
    if sigma2 > 0 and mode != 'mse':
        energy = np.where(mse > 0, CLIP ** 2 * D * sigma2 * ss / (nn * nn * np.maximum(mse, 1e-300)), 0.0)
    else:
        energy = np.full_like(nn, np.nan)
    ul_uses = D * R + 8 * nn                       # analog UL + pilot
    dl_uses = (32 * D + 64 * nn + 64) / DL_BITS_PER_USE  # 모델 broadcast + control
    out = dict(R=R, mse=mse, var=mse / D, energy=energy, ul_uses=ul_uses,
               ul_time=ul_uses / bw, dl_time=dl_uses / bw, total_re=ul_uses + dl_uses)
    for k in out:
        out[k] = np.where(active, out[k], 0.0)
    return out

def noise_var(n, h_active, mode, sigma2=0.0, eps=None, bw=1.0, power='symbol', D=D_PARAMS):
    """학습 시뮬레이터용 스칼라 버전. 반환 (좌표당 분산, 라운드 정보 dict)."""
    h_active = np.asarray(h_active, dtype=float)
    t = round_terms([len(h_active)], [h_active.min()], [(1 / h_active ** 2).sum()], mode, sigma2, eps, bw, power, D)
    info = {k: float(v[0]) for k, v in t.items()}
    return info['var'], info

# ---------------------------------------------------------------- 일정 -> 라운드 통계
def schedule_stats(h, members, entry, t0, T, exclude=()):
    """h: [T, N] 채널, members: shard client 목록, entry: {client: 최초 라운드}.
    라운드 t0..T-1 의 (n_t, hmin_t, suminv_t) 배열."""
    mem = [i for i in members if i not in exclude]
    if not mem:
        z = np.zeros(T - t0); return z, np.ones(T - t0), z
    hs = h[t0:T][:, mem]                                     # [T', m]
    e = np.array([entry[i] for i in mem])
    act = (np.arange(t0, T)[:, None] >= e[None, :])          # [T', m]
    n = act.sum(1).astype(float)
    hmin = np.where(act, hs, np.inf).min(1); hmin = np.where(n > 0, hmin, 1.0)
    suminv = np.where(act, 1 / hs ** 2, 0.0).sum(1)
    return n, hmin, suminv

METRICS = ['ul_uses', 'total_re', 'energy', 'ul_time', 'dl_time']

def schedule_cost(h, members, entry, t0, T, regime, exclude=()):
    """regime: dict(mode, sigma2, eps, bw, power). 반환: 지표별 합계 + 최대 R, 평균 MSE."""
    n, hmin, suminv = schedule_stats(h, members, entry, t0, T, exclude)
    r = round_terms(n, hmin, suminv, regime['mode'], regime.get('sigma2', 0.0), regime.get('eps'),
                    regime.get('bw', 1.0), regime.get('power', 'symbol'))
    act = n > 0
    out = {k: float(np.nansum(r[k])) for k in METRICS}
    out['rounds_active'] = int(act.sum())
    out['max_R'] = float(r['R'].max()) if act.any() else 0.0
    out['mean_mse'] = float(r['mse'][act].mean()) if act.any() else 0.0
    out['compute_calls'] = float(n.sum())
    return out

def multi_cost(stats_list, regime, metrics=('ul_uses', 'total_re', 'energy', 'ul_time', 'dl_time', 'R')):
    """여러 일정의 (n, hmin, suminv) 를 이어 붙여 한 번에 계산한다. 반환: {지표: 일정별 합계 배열}."""
    lens = [len(s[0]) for s in stats_list]
    n = np.concatenate([s[0] for s in stats_list]); hmin = np.concatenate([s[1] for s in stats_list])
    suminv = np.concatenate([s[2] for s in stats_list])
    r = round_terms(n, hmin, suminv, regime['mode'], regime.get('sigma2', 0.0), regime.get('eps'),
                    regime.get('bw', 1.0), regime.get('power', 'symbol'))
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    out = {}
    for k in metrics:
        v = np.nan_to_num(r[k] if k != 'R' else np.where(n > 0, r['R'], 0.0))
        out[k] = np.add.reduceat(v, starts) if len(v) else np.zeros(len(lens))
    out['max_R'] = np.maximum.reduceat(np.where(n > 0, r['R'], 0.0), starts)
    out['compute_calls'] = np.add.reduceat(n, starts)
    return out

def regime_name(rg):
    s = f"{rg['mode']}"
    if rg.get('sigma2') is not None and rg['mode'] != 'mse':
        s += f"_snr{round(10 * math.log10(P_MAX / rg['sigma2'])) if rg['sigma2'] > 0 else 'inf'}dB"
    if rg.get('eps') is not None:
        s += f"_eps{rg['eps']:g}"
    if rg.get('bw', 1) != 1:
        s += f"_b{rg['bw']:g}"
    if rg.get('power', 'symbol') != 'symbol':
        s += f"_{rg['power']}"
    return s
