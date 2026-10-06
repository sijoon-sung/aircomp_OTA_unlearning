"""assignment — shard 배정 규칙에 따른 학습 비용 T 와 삭제 비용 U.

배정 규칙 (K=4)
  고정 규칙: 무작위 / 채널 정렬 자르기 / 채널 돌려 담기 / 채널 고정 구간 / ID 해시
  설계 규칙: 장기 채널만 보고 T/T0 + lam * U/U0 를 국소 탐색(두 client 맞바꾸기)으로 최소화 (lam=0 은 학습만 고려),
             그리고 T 가 lam=0 배정의 1% 이내인 배정 중 U 최소.
  탐색에 쓰는 비용(라운드당 송신 에너지, 장기 채널 기준):
      MSE1 = C^2 sigma2 / (n^2 P h_min^2),  달성 오차 mse (전력 정렬 규칙에 따라),  E = D sigma2 sum_i h_i^-2 / (n^2 mse)
      U = 각 member 를 지웠을 때 남은 shard 의 E 의 평균
전력 정렬: 최대 전력 / 집계 오차를 eps 에 맞춤. 직교 블록, 20dB.
측정: 실제 송수신 장부로 T (학습 4 shard 합) 와 U (채널 순위 0/25/50/75/100% 위치 5명을 각각 지우고 해당 shard 재학습, 평균).
"""
import math
import numpy as np
from . import seed_context, db
from ..channel import partition, shard_of, RULE_KO
from ..config import N_CLIENTS, CLIP, P_MAX
from ..fl import Shard, train
from ..radio import RadioConfig, Ledger
from ..util import write_json, fmt, pct, table

NAME = 'assignment'
K = 4
SIGMA2 = .01
RULES = ['random_split', 'rank_chunk', 'rank_rr', 'bins', 'hash', 'lam0', 'lam0.5', 'lam2', 'tbound']
NAME_KO = dict(RULE_KO, lam0='학습만 고려(λ=0)', **{'lam0.5': 'λ=0.5', 'lam2': 'λ=2'}, tbound='T+1% 안에서 U 최소')
ALIGN_KO = {'maxpow': '최대 전력', 'common': '집계 오차를 eps 에 맞춤'}

# ---------------------------------------------------------------- 탐색용 비용
def round_energy(align, h, g, eps, D):
    g = list(g); n = len(g)
    if n == 0:
        return 0.0
    hs = h[g]; m1 = CLIP ** 2 * SIGMA2 / (n * n * P_MAX * hs.min() ** 2)
    R = max(1.0, math.ceil(m1 / eps - 1e-9))
    mse = eps if (align == 'common' and m1 <= eps) else m1 / R
    return D * SIGMA2 * float((1 / hs ** 2).sum()) / (n * n * mse)

class Scorer:
    def __init__(self, align, h, eps, D):
        self.align, self.h, self.eps, self.D, self.memo = align, h, eps, D, {}
    def group(self, g):
        k = tuple(sorted(g))
        if k not in self.memo:
            self.memo[k] = (round_energy(self.align, self.h, k, self.eps, self.D),
                            sum(round_energy(self.align, self.h, [x for x in k if x != u], self.eps, self.D) for u in k))
        return self.memo[k]
    def cost(self, groups):
        T = U = 0.0
        for g in groups:
            a, b = self.group(g); T += a; U += b
        return T, U / N_CLIENTS

def local_search(sc, start, obj, passes=25):
    """두 shard 사이에서 client 한 쌍을 맞바꿔 obj(T, U) 가 줄면 받아들인다. 더 줄지 않을 때까지."""
    groups = [list(g) for g in start]; best = obj(*sc.cost(groups))
    for _ in range(passes):
        improved = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                for i in list(groups[a]):
                    for j in list(groups[b]):
                        cand = [list(g) for g in groups]
                        cand[a] = [x for x in groups[a] if x != i] + [j]; cand[b] = [x for x in groups[b] if x != j] + [i]
                        v = obj(*sc.cost(cand))
                        if v < best - 1e-12:
                            best, groups, improved = v, cand, True
                            break
                    if improved: break
                if improved: break
            if improved: break
        if not improved:
            break
    return [sorted(g) for g in groups]

def designed(base, align, eps, seed, D):
    sc = Scorer(align, base, eps, D); allc = list(range(N_CLIENTS))
    out = {r: partition(r, allc, base, K, seed) for r in ['random_split', 'rank_chunk', 'rank_rr', 'bins', 'hash']}
    T0, U0 = sc.cost(out['random_split'])
    starts = [out['rank_rr'], out['random_split'], out['rank_chunk']]
    for lam in [0.0, 0.5, 2.0]:
        obj = lambda t, u, lam=lam: t / T0 + lam * u / U0
        out[f'lam{lam:g}'] = min((local_search(sc, s, obj) for s in starts), key=lambda p: obj(*sc.cost(p)))
    Tstar = sc.cost(out['lam0'])[0]
    pen = lambda t, u: u / U0 + 1e3 * max(0.0, t / Tstar - 1.01)
    out['tbound'] = min((local_search(sc, s, pen) for s in [out['lam0']] + starts), key=lambda p: pen(*sc.cost(p)))
    return out

# ---------------------------------------------------------------- 실행
def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    rules = ['random_split', 'rank_rr', 'lam0', 'lam2'] if cfg['quick'] else RULES
    rows = []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed); T = cfg['T']
        order = np.argsort(c.base); dels = [int(order[int(round(q * (N_CLIENTS - 1)))]) for q in [0, .25, .5, .75, 1.0]]
        shards, parts = [], {}
        for al in ALIGN_KO:
            des = designed(c.base, al, cfg['eps'], seed, c.tr.D)
            for r in rules:
                groups = [list(g) for g in des[r]]; parts[(al, r)] = groups
                for k, g in enumerate(groups):
                    if g:
                        shards.append(Shard(f'{al}|{r}|src|{k}', g, f'{al}|{r}', slot=k, T=T))
                for u in dels:
                    cu = shard_of(groups, u); rest = [i for i in groups[cu] if i != u]
                    if rest:
                        shards.append(Shard(f'{al}|{r}|del|{u}', rest, f'{al}|{r}|del', slot=cu, T=T))
        log(f'[assignment] seed {seed}: shard 모델 {len(shards)}개')
        W, leds, _, _ = train(c.tr, shards, c.h, c.w0, seed, lambda s: RadioConfig(sigma2=SIGMA2, eps=cfg['eps'], align=s.split('|')[0]),
                              log=log, log_every=T // 4)
        P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
        for al in ALIGN_KO:
            for r in rules:
                groups = parts[(al, r)]
                Tl = Ledger()
                src = {}
                for k, g in enumerate(groups):
                    if g:
                        Tl.add(leds[ix[f'{al}|{r}|src|{k}']]); src[k] = c.ev.pick(P, ix[f'{al}|{r}|src|{k}'])
                Ul = Ledger(); dacc = []
                for u in dels:
                    nm = f'{al}|{r}|del|{u}'
                    if nm in ix:
                        Ul.add(leds[ix[nm]]); cu = shard_of(groups, u)
                        dacc.append(c.ev.ensemble([c.ev.pick(P, ix[nm]) if k == cu else p for k, p in src.items()])['test'])
                nd = max(1, len(dacc))
                rows.append(dict(seed=seed, align=al, rule=r, sizes=[len(g) for g in groups], acc=c.ev.ensemble(list(src.values()))['test'],
                                 del_acc=float(np.mean(dacc)), T=Tl.asdict(), U={k: v / nd for k, v in Ul.asdict().items()}))
        log(f'[assignment] seed {seed} 끝')
    write_json(out / 'results.json', dict(rows=rows, deletions='채널 순위 0/25/50/75/100% 위치 5명', eps=cfg['eps']))
    L = ['# assignment — shard 배정의 학습 비용과 삭제 비용 (실제 장부)', '',
         f'N={N_CLIENTS}, K={K}, {cfg["T"]}라운드, {db(SIGMA2)}, seed {len(cfg["seeds"])}개, eps={fmt(cfg["eps"])}. T = 학습 4 shard 합, U = 삭제 1회 평균.', '']
    for al in ALIGN_KO:
        L += [f'## {ALIGN_KO[al]}', '']
        tb, pts = [], []
        for r in rules:
            s = [x for x in rows if x['align'] == al and x['rule'] == r]
            if not s:
                continue
            dT, dU, q = [], [], []
            for x in s:
                b = next((y for y in rows if y['seed'] == x['seed'] and y['align'] == al and y['rule'] == 'lam0'), None)
                if b is None:
                    continue
                dT.append(x['T']['energy'] / b['T']['energy'] - 1); dU.append(x['U']['energy'] / b['U']['energy'] - 1)
                gain = b['U']['energy'] - x['U']['energy']; cost = x['T']['energy'] - b['T']['energy']
                q.append(cost / gain if gain > 0 else (0.0 if cost <= 0 else float('inf')))
            Te = float(np.mean([x['T']['energy'] for x in s])); Ue = float(np.mean([x['U']['energy'] for x in s])); pts.append((r, Te, Ue))
            tb.append([NAME_KO[r], str(s[0]['sizes']), fmt(Te), fmt(Ue), fmt(float(np.mean([x['U']['compute_calls'] for x in s])), 0),
                       pct(float(np.mean(dT))) if dT else '-', pct(float(np.mean(dU))) if dU else '-', fmt(float(np.median(q)), 2) if q else '-',
                       pct(float(np.mean([x['acc'] for x in s]))), pct(float(np.mean([x['del_acc'] for x in s])))])
        L.append(table(['배정', 'shard 크기(첫 seed)', '학습 에너지 T', '삭제 에너지 U', '삭제 로컬 학습 횟수', 'T 변화(λ=0 대비)', 'U 변화(λ=0 대비)',
                        '손익분기 삭제 수', '정확도', '삭제 후 정확도'], tb))
        front, best = [], float('inf')
        for name, te, ue in sorted(pts, key=lambda p: p[1]):
            if ue < best - 1e-9:
                front.append(name); best = ue
        L += ['', f'파레토 전선 (T 가 작은 순으로 볼 때 U 가 더 작아지는 배정): {", ".join(NAME_KO[n] for n in front)}. '
                  f'2개 이상이면 학습 비용을 더 써서 삭제 비용을 줄이는 교환이 있다.', '']
    L += ['손익분기 삭제 수 = (학습에서 더 쓴 에너지) / (삭제 1회에서 아낀 에너지), λ=0 배정 대비. 0 은 학습도 싸진 것, inf 는 회수 불가.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
