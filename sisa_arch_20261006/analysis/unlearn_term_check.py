"""언러닝 항을 배정 목적함수에 넣으면 최적 배정이 실제로 바뀌는가 — CPU 전수 열거.

N=12 client, K=3 shard x 4명 (5,775 배정) 과 K=4 x 3명 (15,400 배정) 을 모두 열거한다.
학습 비용 T(G) 는 기존 AirComp 배정 연구가 쓰는 형태 3가지, 언러닝 비용 U(G) 는 삭제 대상 u 에 대한 기대값.
  학습 목적                          한 shard g 의 라운드 비용
  mse    (반복 전송/집계 오차형)       1 / (n^2 h_min^2)
  maxpow (최대 전력 에너지)             sum_i (h_min / h_i)^2
  minen  (잡음 한계까지 전력을 낮춘 에너지) (1/n^2) sum_i h_i^-2
  언러닝 항
  same      같은 형태의 비용을 g \\ u 에 적용 (재학습 통신)
  straggler 재학습 지연 = g \\ u 안의 가장 느린 기기 계산 시간 (학습 지연은 전체 최댓값이라 배정과 무관)
  mixed     same + straggler (각각 정규화)
측정: argmin T 와 argmin U 가 다른 비율(엄격 충돌), T 를 5%/10% 까지 허용할 때 U 를 몇 % 줄일 수 있는가.
"""
import itertools, json, sys, time
from pathlib import Path
import numpy as np

def partitions(n, k):
    size = n // k
    out = []
    def rec(rest, groups):
        if not rest:
            out.append(tuple(groups)); return
        first = rest[0]
        for comb in itertools.combinations(rest[1:], size - 1):
            g = (first,) + comb
            rec([x for x in rest if x not in g], groups + [g])
    rec(list(range(n)), [])
    return out

def group_cost(kind, h, idx):
    hs = h[list(idx)]; n = len(idx)
    if n == 0:
        return 0.0
    if kind == 'mse':
        return 1.0 / (n * n * hs.min() ** 2)
    if kind == 'maxpow':
        return float(((hs.min() / hs) ** 2).sum())
    if kind == 'minen':
        return float((1 / hs ** 2).sum() / (n * n))
    raise ValueError(kind)

def run(N, K, draws, seed=0):
    parts = partitions(N, K); size = N // K
    groups = sorted({g for p in parts for g in p}); gid = {g: i for i, g in enumerate(groups)}
    P = np.array([[gid[g] for g in p] for p in parts])          # [num_parts, K]
    rng = np.random.default_rng(seed)
    kinds = ['mse', 'maxpow', 'minen']; uterms = ['same', 'straggler', 'mixed']
    stat = {(t, u): dict(conflict=0, red_opt=[], inc_opt=[], red5=[], red10=[]) for t in kinds for u in uterms}
    for d in range(draws):
        h = 10 ** (rng.uniform(-20, 0, N) / 20)
        c = rng.lognormal(0, 0.5, N)                              # 기기별 로컬 계산 시간
        gT = {k: np.array([group_cost(k, h, g) for g in groups]) for k in kinds}
        gU = {k: np.array([sum(group_cost(k, h, [x for x in g if x != u]) for u in g) for g in groups]) for k in kinds}
        gS = np.array([sum(max(c[x] for x in g if x != u) for u in g) for g in groups])
        for t in kinds:
            T = gT[t][P].sum(1)
            Us = {'same': gU[t][P].sum(1) / N, 'straggler': gS[P].sum(1) / N}
            Us['mixed'] = Us['same'] / Us['same'].min() + Us['straggler'] / Us['straggler'].min()
            iT = int(np.argmin(T)); Tmin = T[iT]
            for u in uterms:
                U = Us[u]; iU = int(np.argmin(U)); s = stat[(t, u)]
                s['conflict'] += int(U[iT] > U.min() * (1 + 1e-9) and T[iU] > Tmin * (1 + 1e-9))
                s['red_opt'].append(1 - U[iU] / U[iT]); s['inc_opt'].append(T[iU] / Tmin - 1)
                for tol, key in [(0.05, 'red5'), (0.10, 'red10')]:
                    ok = T <= Tmin * (1 + tol)
                    s[key].append(1 - U[ok].min() / U[iT])
    out = {}
    for (t, u), s in stat.items():
        out[f'{t}|{u}'] = dict(conflict_rate=s['conflict'] / draws,
                               U_reduction_at_Uopt_median=float(np.median(s['red_opt'])),
                               T_increase_at_Uopt_median=float(np.median(s['inc_opt'])),
                               U_reduction_T5_median=float(np.median(s['red5'])), U_reduction_T5_mean=float(np.mean(s['red5'])),
                               U_reduction_T10_median=float(np.median(s['red10'])))
    return dict(N=N, K=K, size=size, partitions=len(parts), draws=draws, results=out)

if __name__ == '__main__':
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    t0 = time.time(); res = [run(12, 3, draws), run(12, 4, draws, seed=1)]
    out = Path(__file__).with_name('unlearn_term_check.json')
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in res:
        print(f"N={r['N']} K={r['K']} ({r['partitions']} 배정, draw {r['draws']})")
        print(f"{'학습목적|언러닝항':22s} 충돌률  U최적시U감소 그때T증가  T+5%허용U감소 T+10%허용U감소")
        for k, v in r['results'].items():
            print(f"{k:22s} {v['conflict_rate']:6.2f}  {v['U_reduction_at_Uopt_median']:9.1%} {v['T_increase_at_Uopt_median']:9.1%}"
                  f"  {v['U_reduction_T5_median']:11.1%} {v['U_reduction_T10_median']:12.1%}")
    print(f'{time.time() - t0:.1f}s')
