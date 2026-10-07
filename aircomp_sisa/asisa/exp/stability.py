"""stability (E2) — 불안정한 배정에서는 정확한 언러닝을 위해 몇 개의 shard 를 재학습해야 하는가, 그리고 '해시 + 국소 병합' 은 얼마나 싸게 막는가.

정확한 언러닝 = 'u 가 처음부터 없었다면' 의 학습 결과와 같은 분포. 배정 규칙도 학습의 일부이므로,
u 를 뺀 집단에 규칙을 다시 적용한 배정에서 구성이 바뀐 shard 는 모두 재학습해야 한다.
  재학습 집합 R(u) = {u 의 shard} ∪ {원래 shard k : S_k ∖ {u} 가 다시 적용한 배정의 어느 shard 와도 같지 않음}
배정 규칙
  random_split  무작위 순열 후 자르기 / rank_chunk 채널 정렬 자르기 / rank_rr 채널 돌려 담기 (집단 전체를 보고 정함)
  bins          채널 고정 구간 / hash ID 해시 (각자 자기 정보로만 정함)
  hash_merge    ID 해시 후, 인원이 n_min 미만인 shard 는 정해진 순서의 다음 비어 있지 않은 shard 에 합친다 (제안).
                합치는 조건이 shard 크기로만 정해지므로, 크기가 작아질 때만 짝 shard 를 함께 재학습한다.
(1) 계산 (학습 없음, 채널 draw 많이): u 하나 삭제의 재학습 shard 수·인원, 연속 삭제 8번 동안의 누적 재학습 인원, 최소 인원과 노출.
    노출은 resources 실험의 n 별 member 주 label 적중 (무잡음, 초기 모델) 실측값으로 읽는다.
(2) 학습 (GPU): 불안정한 배정에서 u 의 shard 만 재학습한 SISA 앙상블이 전체 재학습 (u 를 뺀 집단에 규칙을 다시 적용) 과 얼마나 다른가.
    직교 블록, shard 안 정렬, 20dB. 잡음 번호는 shard 구성 (최소 client 번호) 으로 정해서 구성이 같은 shard 는 같은 계산이 되게 한다.
"""
import numpy as np
from . import seed_context, db
from ..channel import partition, RULE_KO
from ..config import N_CLIENTS
from ..fl import Shard, train
from ..radio import RadioConfig
from ..trainer import disagreement
from .deletion import tv
from ..util import key, write_json, fmt, pct, table, select, mean

NAME = 'stability'
K = 4
N_MIN = 3
RULES = ['random_split', 'rank_chunk', 'rank_rr', 'bins', 'hash', 'hash_merge']
NAME_KO = dict(RULE_KO, hash_merge='해시 + 국소 병합 (제안)')
EXPOSURE = {1: .828, 2: .455, 4: .298, 5: .265, 10: .177, 20: .100}   # resources 실험 실측 (무잡음, 초기 모델)
SEQ = 8
DRAWS = 300

def exposure(n):
    if n <= 0:
        return float('nan')
    xs = sorted(EXPOSURE); n = min(max(n, xs[0]), xs[-1])
    return float(np.interp(np.log(n), np.log(xs), [EXPOSURE[x] for x in xs]))

def assign(rule, ids, base, seed):
    if rule != 'hash_merge':
        return [g for g in partition(rule, ids, base, K, seed)]
    g = [list(x) for x in partition('hash', ids, base, K, seed)]
    for k in range(K):
        if 0 < len(g[k]) < N_MIN:
            for d in range(1, K):
                p = (k + d) % K
                if g[p]:
                    g[p] = sorted(g[p] + g[k]); g[k] = []
                    break
    return g

def retrain_set(rule, ids, base, seed, u):
    before = [frozenset(x) for x in assign(rule, ids, base, seed) if x]
    after = {frozenset(x) for x in assign(rule, [i for i in ids if i != u], base, seed) if x}
    R = [g for g in before if u in g or (g - {u}) and (g - {u}) not in after]
    return R

def analytic(cfg, rules):
    rng = np.random.default_rng(key('stability'))
    single, life = [], []
    seeds = list(cfg['seeds']) + [10 ** 6 + d for d in range(DRAWS)]
    for sd in seeds:
        base = 10 ** (np.random.default_rng(key(sd, 'slow_channel')).uniform(-20, 0, N_CLIENTS) / 20)
        ids = list(range(N_CLIENTS))
        order = list(np.random.default_rng(key(sd, 'deletion_order')).permutation(ids))[:SEQ]
        for rule in rules:
            gs = [g for g in assign(rule, ids, base, sd) if g]
            Rs = [retrain_set(rule, ids, base, sd, u) for u in ids]
            single.append(dict(seed=sd, rule=rule, n_shards=len(gs), min_size=min(map(len, gs)), max_exposure=exposure(min(map(len, gs))),
                               retrain_shards=float(np.mean([len(R) for R in Rs])), retrain_clients=float(np.mean([sum(len(g) - (u in g) for g in R) for R, u in zip(Rs, ids)])),
                               all_shards=float(np.mean([len(R) == len(gs) for R in Rs]))))
            cur = list(ids); tot = 0; worst = 1.0; mins = []
            for u in order:
                R = retrain_set(rule, cur, base, sd, u)
                tot += sum(len(g) - (u in g) for g in R)
                cur = [i for i in cur if i != u]
                g2 = [g for g in assign(rule, cur, base, sd) if g]
                mins.append(min(map(len, g2)))
            life.append(dict(seed=sd, rule=rule, total_retrain=tot, final_min=mins[-1], worst_min=min(mins), worst_exposure=exposure(min(mins))))
    return single, life

def gpu_part(cfg, log, rules):
    T = cfg['T']; rows = []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed)
        order = np.argsort(c.base); targets = {'weakest': int(order[0]), 'median': int(order[N_CLIENTS // 2]), 'strongest': int(order[-1])}
        shards = []; plan = {}
        for rule in rules:
            gs = [g for g in assign(rule, list(range(N_CLIENTS)), c.base, seed) if g]
            plan[rule] = gs
            for kind, salt in [('src', ''), ('alt', 'alt')]:
                shards += [Shard(f'{rule}|{kind}|{min(g)}', g, f'{rule}|{kind}', slot=min(g), salt=salt, T=T) for g in gs]
            for tn, u in targets.items():
                g2 = [g for g in assign(rule, [i for i in range(N_CLIENTS) if i != u], c.base, seed) if g]
                plan[(rule, tn)] = g2
                shards += [Shard(f'{rule}|ref|{tn}|{min(g)}', g, f'{rule}|ref|{tn}', slot=min(g), T=T) for g in g2]
                gu = next(g for g in gs if u in g); rest = [i for i in gu if i != u]
                if rest:
                    shards.append(Shard(f'{rule}|rep|{tn}', rest, f'{rule}|rep|{tn}', slot=min(rest), T=T))
        log(f'[stability] seed {seed}: shard 모델 {len(shards)}개')
        W, leds, diags, _ = train(c.tr, shards, c.h, c.w0, seed, lambda s: RadioConfig(sigma2=.01, eps=cfg['eps']), log=log, log_every=T // 4)
        P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
        pick = lambda n: c.ev.pick(P, ix[n])
        for rule in rules:
            gs = plan[rule]
            src = {min(g): pick(f'{rule}|src|{min(g)}') for g in gs}; alt = {min(g): pick(f'{rule}|alt|{min(g)}') for g in gs}
            se, ae = c.ev.ensemble(list(src.values())), c.ev.ensemble(list(alt.values()))
            for tn, u in targets.items():
                gu = next(g for g in gs if u in g); rest = [i for i in gu if i != u]
                sisa = c.ev.ensemble([pick(f'{rule}|rep|{tn}') if (k == min(gu)) else p for k, p in src.items() if rest or k != min(gu)])
                full = c.ev.ensemble([pick(f'{rule}|ref|{tn}|{min(g)}') for g in plan[(rule, tn)]])
                R = retrain_set(rule, list(range(N_CLIENTS)), c.base, seed, u)
                rows.append(dict(seed=seed, rule=rule, target=tn, retrain_shards=len(R), ens_tv=tv(sisa['ptest'], full['ptest']),
                                 ens_dis=disagreement(sisa['ptest'], full['ptest']), yard_tv=tv(se['ptest'], ae['ptest']), yard_dis=disagreement(se['ptest'], ae['ptest']),
                                 acc_sisa=sisa['test'], acc_full=full['test']))
        del W, P
        log(f'[stability] seed {seed} 끝')
    return rows

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    rules = ['rank_rr', 'hash', 'hash_merge'] if cfg['quick'] else RULES
    single, life = analytic(cfg, rules)
    log(f'[stability] 계산 끝 (채널 draw {len(cfg["seeds"]) + DRAWS}개)')
    G = gpu_part(cfg, log, [r for r in rules if r != 'bins'])
    write_json(out / 'results.json', dict(single=single, lifecycle=life, gpu=G, n_min=N_MIN, seq=SEQ, exposure_table=EXPOSURE))
    L = ['# stability (E2) — 배정의 안정성과 정확한 언러닝에 필요한 재학습', '',
         f'N={N_CLIENTS}, K={K}, 병합 문턱 n_min={N_MIN}. 계산은 채널 draw {len(cfg["seeds"]) + DRAWS}개 평균. '
         '재학습 집합 = u 의 shard + u 를 뺀 집단에 규칙을 다시 적용했을 때 구성이 바뀌는 shard. 노출 = 가장 작은 shard 인원에서의 member 주 label 적중 (resources 실측).', '',
         '## 1. 삭제 1회', '']
    tb = []
    for rule in rules:
        s = select(single, rule=rule)
        tb.append([NAME_KO[rule], fmt(mean(s, 'n_shards'), 2), fmt(mean(s, 'retrain_shards'), 2), pct(mean(s, 'all_shards')), fmt(mean(s, 'retrain_clients'), 1),
                   fmt(mean(s, 'min_size'), 2), pct(float(np.mean([x['min_size'] < N_MIN for x in s]))), pct(mean(s, 'max_exposure'))])
    L += [table(['배정', 'shard 수', '재학습 shard 수', '전부 재학습 확률', '재학습 인원', '최소 인원', f'최소 인원 < {N_MIN} 확률', '최대 노출'], tb), '',
          f'## 2. 연속 삭제 {SEQ}번 (무작위 순서)', '']
    tb = []
    for rule in rules:
        s = select(life, rule=rule)
        tb.append([NAME_KO[rule], fmt(mean(s, 'total_retrain'), 1), fmt(mean(s, 'worst_min'), 2), pct(float(np.mean([x['worst_min'] < N_MIN for x in s]))), pct(mean(s, 'worst_exposure'))])
    L += [table(['배정', '누적 재학습 인원', '가장 작아진 shard 인원', f'< {N_MIN} 이 된 적 있는 비율', '최악 노출'], tb), '',
          '## 3. 학습 결과: u 의 shard 만 재학습한 SISA vs 전체 재학습', '',
          '확률 거리 = test 표본별 확률 분포의 total variation 평균. 난수 기준 = 원래 학습을 다른 배치·잡음으로 했을 때 두 앙상블 사이의 같은 거리.', '']
    tb = []
    for rule in [r for r in rules if r != 'bins']:
        for tn in ['weakest', 'median', 'strongest']:
            s = select(G, rule=rule, target=tn)
            if s:
                tb.append([NAME_KO[rule], {'weakest': '최약 노드', 'median': '중간 노드', 'strongest': '최강 노드'}[tn], fmt(mean(s, 'retrain_shards'), 2),
                           fmt(mean(s, 'ens_tv'), 4), pct(mean(s, 'ens_dis')), fmt(mean(s, 'yard_tv'), 4), pct(mean(s, 'yard_dis'))])
    L += [table(['배정', '삭제 대상', '정확성에 필요한 재학습 shard 수', '확률 거리 (SISA vs 전체 재학습)', '예측 불일치', '확률 거리 난수 기준', '예측 불일치 난수 기준'], tb), '',
          '## 읽는 법', '',
          '- 순위 기반 규칙 (채널 정렬·돌려 담기) 은 u 보다 순위가 낮은 사람이 한 칸씩 밀려, 정확성을 지키려면 대부분의 shard 를 재학습해야 한다. SISA 의 이득 (shard 하나만 재학습) 이 사라진다.',
          '- 그런데 SISA 대로 u 의 shard 만 재학습하면 3절의 확률 거리가 난수 기준 수준으로 커진다 (전체 재학습과 다른 결과 = 정확하지 않음).',
          '- 해시는 재학습이 항상 shard 하나지만 크기가 고르지 않아 최소 인원이 작고 노출이 크다. 해시 + 국소 병합은 크기 하한을 지키면서, 병합이 일어날 때만 재학습이 늘어난다.',
          '- 안정적인 규칙 (해시, 해시 + 병합) 에서 3절의 확률 거리는 0 에 가까워야 한다 (구성이 같은 shard 는 같은 계산).', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
