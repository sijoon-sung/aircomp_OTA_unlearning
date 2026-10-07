"""lifecycle (E2b) — 배정·삭제 처리 방법별로, 삭제와 신규 참여가 이어질 때 정확한 언러닝의 누적 비용과 노출을 비교한다.

정확한 언러닝 = 'u 가 처음부터 없었다면' 의 학습 결과와 같은 분포. 규칙을 u 없이 다시 적용했을 때 구성이 바뀐 shard 는
바뀐 member 중 가장 먼저 들어온 시점부터 다시 학습해야 한다 (체크포인트가 없으면 처음부터).
사건 흐름: 초기 client 20명이 160 라운드 학습한 뒤, 20 라운드마다 사건 하나 (삭제 70%: 현재 client 중 무작위 / 참여 30%: 새 client, 채널 무작위).
방법
  full            shard 없이 하나의 모델 (삭제마다 전원 처음부터)
  random_split / rank_rr          집단 전체로 정하는 배정 (정확성을 위해 바뀐 shard 전부 재학습)
  bounded         부하 상한 해시 (consistent hashing with bounded loads): 참여 순서대로 자기 해시 칸에 넣되 꽉 차면 다음 칸. 상한 = ceil(1.25 N/K)
  hash            ID 해시 (각자 자기 정보로만)
  hash_merge      해시 + 국소 병합: 인원 n_min 미만 shard 는 다음 비어 있지 않은 shard 에 합침
  hash_suspend    해시 + 정지: 인원 n_min 미만 shard 는 재학습하지 않고 앙상블에서 뺌 (그 데이터는 모델에서 빠짐). 다시 n_min 이 되면 처음부터 학습
  +batch4         삭제 요청을 4개씩 모아 한 번에 처리 (같은 shard 는 한 번만 재학습, 대신 기다림)
  +slice          초기 client 를 4 묶음으로 나눠 0/40/80/120 라운드에 참여시키고 참여 시점마다 체크포인트 저장 -> 삭제 대상이 들어온 시점부터 재학습
  hash_merge_K3   shard 수를 3으로 (작은 shard 가 덜 생기게)
비용: 계산 = 재학습한 client x 라운드, 통신 = 재학습한 shard x 라운드 (AirComp 는 shard 하나가 라운드당 D 심볼), 지연 = 요청부터 반영까지 라운드.
품질: 최소 인원 shard 의 노출 (resources 실측의 n 별 member 주 label 적중), 앙상블에 실제로 쓰인 데이터 비율.
GPU 확인: (1) slicing 재학습이 처음부터 재학습과 같은 결과인지 (정확성) (2) 단계 참여가 정확도를 떨어뜨리는지 (3) shard 하나를 정지하면 정확도가 얼마나 떨어지는지.
"""
import math
import numpy as np
from . import seed_context
from .stability import exposure
from ..channel import partition
from ..config import N_CLIENTS
from ..fl import Shard, train
from ..radio import RadioConfig
from ..util import key, write_json, fmt, pct, table, select, mean

NAME = 'lifecycle'
N_MIN = 3
T0, GAP, EVENTS, P_DEL, SLICES = 160, 20, 30, 0.7, 4
METHODS = ['full', 'random_split', 'rank_rr', 'bounded', 'hash', 'hash_merge', 'hash_suspend', 'hash_merge_batch4', 'hash_merge_slice',
           'hash_merge_slice_batch4', 'hash_merge_K3']
METHOD_KO = {'full': 'shard 없음 (전체 재학습)', 'random_split': '무작위 순열', 'rank_rr': '채널 돌려 담기', 'bounded': '부하 상한 해시',
             'hash': 'ID 해시', 'hash_merge': '해시 + 병합', 'hash_suspend': '해시 + 정지', 'hash_merge_batch4': '해시 + 병합 + 묶음 4',
             'hash_merge_slice': '해시 + 병합 + slicing', 'hash_merge_slice_batch4': '해시 + 병합 + slicing + 묶음 4', 'hash_merge_K3': '해시 + 병합, K=3'}
DRAWS = 200

def base_rule(m):
    for r in ['random_split', 'rank_rr', 'bounded', 'full']:
        if m == r:
            return r
    return 'hash'

def assign(method, ids, base, seed, K):
    """ids: 참여 순서. 반환: shard 별 client 목록 (빈 shard 포함)."""
    r = base_rule(method)
    if r == 'full':
        return [list(ids)]
    if r == 'bounded':
        cap = math.ceil(1.25 * len(ids) / K); g = [[] for _ in range(K)]
        for i in ids:
            h0 = key(seed, i, 'hash') % K
            for d in range(K):
                if len(g[(h0 + d) % K]) < cap:
                    g[(h0 + d) % K].append(i); break
        return g
    if r in ('random_split', 'rank_rr'):
        return partition(r, ids, base, K, seed)
    g = [list(x) for x in partition('hash', ids, base, K, seed)]
    if 'merge' in method:
        for k in range(K):
            if 0 < len(g[k]) < N_MIN:
                for d in range(1, K):
                    p = (k + d) % K
                    if g[p]:
                        g[p] = sorted(g[p] + g[k]); g[k] = []
                        break
    return g

def active(method, groups):
    gs = [frozenset(g) for g in groups if g]
    return [g for g in gs if len(g) >= N_MIN] if 'suspend' in method else gs

def retrain_cost(before, after, entry, t_now, slicing):
    """구성이 바뀐 새 shard 마다 (재학습 인원, 다시 도는 라운드).
    새로 들어온 사람만 더해졌으면 이어서 학습 (0 라운드). 그 밖에는 slicing 이면 바뀐 member 중 가장 먼저 들어온 시점부터, 아니면 처음부터."""
    out, bset = [], set(before)
    for g in after:
        if g in bset:
            continue
        old = max(before, key=lambda b: len(b & g), default=frozenset())
        added, removed = (g - old, old - g) if old & g else (g, frozenset())
        if not removed and all(entry[i] == t_now for i in added):
            start = t_now
        elif slicing:
            start = min(entry[i] for i in (added | removed))
        else:
            start = 0
        out.append((len(g), t_now - start))
    return out

def simulate(method, seed, base0):
    K = 3 if method.endswith('K3') else 4
    slicing = 'slice' in method; batch = 4 if 'batch4' in method else 1
    rng = np.random.default_rng(key(seed, 'lifecycle_events'))
    base = dict(enumerate(base0)); ids = list(range(N_CLIENTS))
    entry = {i: ((key(seed, i, 'slice') % SLICES) * (T0 // SLICES) if slicing else 0) for i in ids}
    order = sorted(ids, key=lambda i: (entry[i], i))
    groups = assign(method, order, base, seed, K); state = active(method, groups)
    comp = comm = 0.0; lat = []; shards_per_del = []; worst = 0.0; expo = []; frac = []; queue = []; nid = N_CLIENTS
    def quality(st, n_now):
        m = min((len(g) for g in st), default=0)
        return (exposure(m) if m else float('nan')), (sum(len(g) for g in st) / max(1, n_now))
    for e in range(EVENTS):
        t_now = T0 + (e + 1) * GAP
        if rng.random() < P_DEL and len(order) > 4:
            u = int(rng.choice(order)); order = [i for i in order if i != u]; queue.append(t_now)
        else:
            v = nid; nid += 1; base[v] = float(10 ** (rng.uniform(-20, 0) / 20)); entry[v] = t_now; order = order + [v]
        if queue and (len(queue) >= batch or e == EVENTS - 1) or not queue:
            new = active(method, assign(method, order, base, seed, K))
            cost = retrain_cost(state, new, entry, t_now, slicing)
            rounds = max((r for _, r in cost), default=0)
            comp += sum(n * r for n, r in cost); comm += sum(r for _, r in cost)
            for tq in queue:
                lat.append(t_now - tq + rounds); shards_per_del.append(sum(1 for _, r in cost if r > 0))
            queue = []; state = new
        x, f = quality(state, len(order))
        if not math.isnan(x):
            expo.append(x); worst = max(worst, x)
        frac.append(f)
    return dict(method=method, seed=seed, compute=comp, comm=comm, latency=float(np.mean(lat)) if lat else 0.0,
                shards_per_del=float(np.mean(shards_per_del)) if shards_per_del else 0.0, worst_exposure=worst,
                mean_exposure=float(np.mean(expo)) if expo else float('nan'), data_frac=float(np.mean(frac)))

def gpu_part(cfg, log):
    rows = []; T = cfg['T']
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed); K = 4
        groups = [g for g in partition('hash', range(N_CLIENTS), c.base, K, seed) if g]
        entry = {i: (key(seed, i, 'slice') % SLICES) * (T // SLICES) for i in range(N_CLIENTS)}
        late = [i for i in range(N_CLIENTS) if entry[i] >= T // 2 and len(next(g for g in groups if i in g)) >= 2]
        u = late[0] if late else max(range(N_CLIENTS), key=lambda i: entry[i]); gu = next(g for g in groups if u in g); tu = entry[u]
        rc = lambda s: RadioConfig(sigma2=.01, eps=cfg['eps'])
        sh = [Shard(f'all0|{min(g)}', g, 'all0', slot=min(g), T=T) for g in groups]
        sh += [Shard(f'slice|{min(g)}', g, 'slice', slot=min(g), entry={i: entry[i] for i in g}, T=T, save_at=(tu,) if g is gu else ()) for g in groups]
        rest = [i for i in gu if i != u]
        sh.append(Shard('ref', rest, 'ref', slot=min(gu), entry={i: entry[i] for i in rest}, T=T))
        log(f'[lifecycle] seed {seed}: GPU 확인 (삭제 대상 참여 라운드 {tu})')
        W, leds, diags, ck = train(c.tr, sh, c.h, c.w0, seed, rc)
        ix = {s.name: j for j, s in enumerate(sh)}
        w_ck = ck[ix[f'slice|{min(gu)}']][tu]
        W2, leds2, _, _ = train(c.tr, [Shard('slice_rep', rest, 'ref', slot=min(gu), entry={i: entry[i] for i in rest}, T=T, t0=tu, w0=w_ck)], c.h, c.w0, seed, rc)
        P = c.ev.probs(W)
        ens = lambda names: c.ev.ensemble([c.ev.pick(P, ix[n]) for n in names])['test']
        a0 = ens([f'all0|{min(g)}' for g in groups]); a1 = ens([f'slice|{min(g)}' for g in groups])
        small = min(groups, key=len)
        a_sus = ens([f'slice|{min(g)}' for g in groups if g is not small])
        rel = float((W2[0] - W[ix['ref']]).norm() / W[ix['ref']].norm())
        rows.append(dict(seed=seed, target_entry=tu, slice_vs_scratch_rel=rel, rounds_saved=tu, acc_all0=a0, acc_slice=a1, acc_suspend_smallest=a_sus,
                         smallest=len(small), compute_scratch=leds[ix['ref']].compute_calls, compute_slice=leds2[0].compute_calls))
        log(f'[lifecycle] seed {seed} GPU 끝')
    return rows

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    methods = ['full', 'rank_rr', 'hash', 'hash_merge', 'hash_merge_slice'] if cfg['quick'] else METHODS
    sims = []
    for sd in list(cfg['seeds']) + [2 * 10 ** 6 + d for d in range(DRAWS)]:
        base0 = 10 ** (np.random.default_rng(key(sd, 'slow_channel')).uniform(-20, 0, N_CLIENTS) / 20)
        sims += [simulate(m, sd, base0) for m in methods]
    log(f'[lifecycle] 계산 끝 (사건 흐름 {len(cfg["seeds"]) + DRAWS}개)')
    G = gpu_part(cfg, log)
    write_json(out / 'results.json', dict(sims=sims, gpu=G, setting=dict(T0=T0, gap=GAP, events=EVENTS, p_del=P_DEL, slices=SLICES, n_min=N_MIN, draws=DRAWS)))
    base = {m: mean(select(sims, method='full'), m2) for m, m2 in [('compute', 'compute'), ('comm', 'comm')]}
    L = ['# lifecycle (E2b) — 배정·삭제 처리 방법별 누적 비용과 노출', '',
         f'초기 client {N_CLIENTS}명, {T0} 라운드 학습 후 {GAP} 라운드마다 사건 {EVENTS}개 (삭제 {int(P_DEL * 100)}%, 참여 {int(100 - P_DEL * 100)}%). '
         f'사건 흐름 {len(cfg["seeds"]) + DRAWS}개 평균. 병합·정지 문턱 n_min = {N_MIN}.', '',
         '계산 = 재학습한 client × 라운드, 통신 = 재학습한 shard × 라운드 (AirComp 에서 shard 하나는 라운드당 D 심볼). 비는 shard 없는 전체 재학습 대비.', '']
    tb = []
    for m in methods:
        s = select(sims, method=m)
        tb.append([METHOD_KO[m], fmt(mean(s, 'compute'), 0), pct(mean(s, 'compute') / base['compute']), fmt(mean(s, 'comm'), 0), pct(mean(s, 'comm') / base['comm']),
                   fmt(mean(s, 'shards_per_del'), 2), fmt(mean(s, 'latency'), 0), pct(mean(s, 'worst_exposure')), pct(mean(s, 'mean_exposure')), pct(mean(s, 'data_frac'))])
    L += [table(['방법', '누적 계산', '계산 비', '누적 통신', '통신 비', '삭제당 재학습 shard', '삭제 지연 (라운드)', '최악 노출', '평균 노출', '앙상블에 쓰인 데이터'], tb), '',
          '## GPU 확인', '']
    if G:
        L += [f'- slicing 재학습 (삭제 대상이 들어온 라운드의 체크포인트부터) vs 처음부터 재학습 (같은 참여 일정): 파라미터 상대 차이 평균 {fmt(mean(G, "slice_vs_scratch_rel"), 6)} '
              f'(0 이면 정확히 같음), 아낀 라운드 평균 {fmt(mean(G, "rounds_saved"), 0)} / {cfg["T"]}, 로컬 학습 횟수 {fmt(mean(G, "compute_scratch"), 0)} → {fmt(mean(G, "compute_slice"), 0)}.',
              f'- 정확도: 전원 0 라운드 참여 {pct(mean(G, "acc_all0"))}, 4 묶음 단계 참여 {pct(mean(G, "acc_slice"))}, 가장 작은 shard (평균 {fmt(mean(G, "smallest"), 1)}명) 정지 {pct(mean(G, "acc_suspend_smallest"))}.', '']
    L += ['## 읽는 법', '',
          '- 집단 전체로 정하는 배정 (무작위 순열, 돌려 담기) 은 삭제뿐 아니라 참여만 있어도 다른 shard 의 구성이 바뀌어 재학습이 쌓인다.',
          '- 해시는 재학습이 가장 적지만 작은 shard 가 생겨 노출이 크다. 병합은 노출을 막는 대신 병합 때 재학습이 늘고, 정지는 재학습 없이 막는 대신 데이터를 잃는다.',
          '- 묶음은 같은 shard 중복 재학습을 줄이는 대신 지연이 늘어난다. slicing 은 AirComp 에서 통신 (다시 도는 라운드 수) 까지 줄이는 유일한 방법이다.',
          '- 가장 싼 조합은 표에서 계산·통신 비가 가장 작으면서 최악 노출이 목표 이하인 행이다.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
