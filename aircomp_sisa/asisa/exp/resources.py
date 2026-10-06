"""resources — shard 수 K 에 따른 자원·삭제 비용, 그리고 shard 크기에 따른 개인 노출.

(1) 자원과 삭제 비용: K in {1,2,4,5,10}, 무작위 배정, 직교 블록, 전력 정렬 2가지(최대 전력 / 집계 오차를 eps 에 맞춤).
    학습 장부(K 개 shard 합)와 삭제 1회 장부(해당 shard 만 처음부터 재학습, 삭제 대상 최대 4명 평균)를 실제 송수신으로 잰다.
    삭제 중 빈 블록 재사용: 재학습 shard 가 블록 K 개를 동시에 쓴다(총 전력 P 를 나눔).
    비교용 디지털 장부: client 마다 직교 자원으로 32 bit x D 를 Shannon 용량으로 보낼 때.
(2) 개인 노출: 서버가 받은 shard 평균(실제 송수신, 잡음 포함)으로 member 의 주 label 을 추정할 때의 적중률과,
    받은 평균과 member update 의 cosine. shard 인원 n in {1,2,4,5,10,20}, 무잡음/0dB/20dB, 초기 모델과 학습 중간 모델.
    주 label 추정: 받은 평균의 마지막 층 bias 10개 중 가장 큰 class.
"""
import numpy as np
import torch
from . import seed_context, db
from ..channel import partition, shard_of
from ..config import N_CLIENTS
from ..fl import Shard, train
from ..radio import RadioConfig, Ledger, transmit_orth, digital_ledger
from ..util import key, write_json, fmt, pct, table

NAME = 'resources'
KS = [1, 2, 4, 5, 10]
ALIGN_KO = {'maxpow': '최대 전력', 'common': '집계 오차를 eps 에 맞춤'}
SIGMA2 = .01

def part_cost(cfg, log):
    Ks = [1, 4] if cfg['quick'] else KS; rows = []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed); T = cfg['T']
        shards, plan = [], {}
        for K in Ks:
            groups = partition('random_split', range(N_CLIENTS), c.base, K, seed) if K > 1 else [list(range(N_CLIENTS))]
            dels = sorted({g[0] for g in groups} | {groups[0][-1]})[:4]
            plan[K] = (groups, dels)
            for al in ALIGN_KO:
                for k, g in enumerate(groups):
                    shards.append(Shard(f'{K}|{al}|src|{k}', g, f'{K}|{al}', slot=k, T=T))
                for u in dels:
                    cu = shard_of(groups, u); rest = [i for i in groups[cu] if i != u]
                    shards.append(Shard(f'{K}|{al}|del|{u}', rest, f'{K}|{al}|del', slot=cu, T=T))
                    shards.append(Shard(f'{K}|{al}|pool|{u}', rest, f'{K}|{al}|pool', slot=cu, T=T))
        def radio(s):
            K, al, kind = (s.split('|') + [''])[:3]
            return RadioConfig(sigma2=SIGMA2, eps=cfg['eps'], align=al, blocks=float(K) if kind == 'pool' else 1.0)
        log(f'[resources] seed {seed}: shard 모델 {len(shards)}개')
        W, leds, _, _ = train(c.tr, shards, c.h, c.w0, seed, radio, log=log, log_every=T // 4)
        P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
        dcfg = RadioConfig(sigma2=SIGMA2)
        for K in Ks:
            groups, dels = plan[K]
            dT, dU = Ledger(), Ledger()
            for t in range(T):
                for g in groups:
                    dT.add(digital_ledger(dcfg, c.h[t, g], c.tr.D))
                for u in dels:
                    g = groups[shard_of(groups, u)]
                    dU.add(digital_ledger(dcfg, c.h[t, [i for i in g if i != u]], c.tr.D))
            for al in ALIGN_KO:
                Tl, Ul, Pl = Ledger(), Ledger(), Ledger()
                for k in range(K):
                    Tl.add(leds[ix[f'{K}|{al}|src|{k}']])
                for u in dels:
                    Ul.add(leds[ix[f'{K}|{al}|del|{u}']]); Pl.add(leds[ix[f'{K}|{al}|pool|{u}']])
                src = [c.ev.pick(P, ix[f'{K}|{al}|src|{k}']) for k in range(K)]
                dacc = [c.ev.ensemble([c.ev.pick(P, ix[f'{K}|{al}|del|{u}']) if k == shard_of(groups, u) else src[k] for k in range(K)])['test'] for u in dels]
                nd = len(dels)
                rows.append(dict(seed=seed, K=K, align=al, acc=c.ev.ensemble(src)['test'], del_acc=float(np.mean(dacc)), train=Tl.asdict(),
                                 delete={k: v / nd for k, v in Ul.asdict().items()}, delete_pool={k: v / nd for k, v in Pl.asdict().items()},
                                 digital_train=dT.asdict(), digital_delete={k: v / nd for k, v in dU.asdict().items()}))
        log(f'[resources] seed {seed} 비용 끝')
    return rows

def part_exposure(cfg, log):
    ns = [1, 2, 4, 5, 10, 20]; snrs = [None, 1.0, .01]; reps = 40 if cfg['quick'] else 200; rows = []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed); T = cfg['T']; mid = T // 2
        _, _, _, ck = train(c.tr, [Shard('k1', list(range(N_CLIENTS)), 'k1', T=mid, save_at=(mid,))], c.h, c.w0, seed,
                            lambda s: RadioConfig(sigma2=SIGMA2, eps=cfg['eps']))
        dom = c.data['hist'].argmax(1)
        for stage, w in [('init', c.w0), ('mid', ck[0][mid])]:
            U = c.tr.updates(w[None].repeat(N_CLIENTS, 1), list(range(N_CLIENTS)), mid, [''] * N_CLIENTS)
            Un = U / U.norm(dim=1, keepdim=True)
            rng = np.random.default_rng(key(seed, 'exposure', stage))
            for s2 in snrs:
                rc = RadioConfig(sigma2=s2 if s2 else 1e-12, eps=cfg['eps'])
                for n in ns:
                    cm, cn, lm, ln = [], [], [], []
                    for rep in range(reps):
                        g = list(rng.choice(N_CLIENTS, n, replace=False)); rest = np.setdiff1d(np.arange(N_CLIENTS), g)
                        r, _, _ = transmit_orth(rc, U[g], c.h[mid, g], key(seed, stage, s2 or 0, n, rep))
                        rn = r / r.norm(); lab = int(r[-10:].argmax())
                        cm += [float(rn @ Un[i]) for i in g]; lm += [lab == dom[i] for i in g]
                        cn += [float(rn @ Un[j]) for j in rest]; ln += [lab == dom[j] for j in rest]
                    rows.append(dict(seed=seed, stage=stage, sigma2=s2, n=n, cos_member=float(np.mean(cm)),
                                     cos_nonmember=float(np.mean(cn)) if cn else float('nan'),
                                     label_member=float(np.mean(lm)), label_nonmember=float(np.mean(ln)) if ln else float('nan')))
        log(f'[resources] seed {seed} 노출 끝')
    return rows

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    B = part_cost(cfg, log); E = part_exposure(cfg, log)
    write_json(out / 'results.json', dict(cost=B, exposure=E, eps=cfg['eps']))
    L = ['# resources — shard 수에 따른 자원·삭제 비용과 개인 노출', '',
         f'N={N_CLIENTS}, {cfg["T"]}라운드, seed {len(cfg["seeds"])}개, eps={fmt(cfg["eps"])}, {db(SIGMA2)}. 매 라운드 실제 송수신과 장부 기록.', '',
         '## 1. shard 수 K 에 따른 학습·삭제 장부', '',
         '삭제 = 대상의 shard 만 처음부터 재학습 (대상 최대 4명 평균). "빈 블록 재사용" = 재학습 shard 가 블록 K 개를 동시에 쓸 때의 시간.', '']
    tb = []
    for al in ALIGN_KO:
        for K in sorted({r['K'] for r in B}):
            s = [r for r in B if r['K'] == K and r['align'] == al]
            g = lambda a, b: float(np.mean([r[a][b] for r in s]))
            tb.append([ALIGN_KO[al], K, fmt(g('train', 'ul_symbols')), fmt(g('train', 'energy')), fmt(g('delete', 'ul_symbols')),
                       fmt(g('delete', 'energy')), fmt(g('delete', 'compute_calls'), 0), fmt(g('delete', 'ul_time')), fmt(g('delete_pool', 'ul_time')),
                       fmt(g('digital_train', 'ul_symbols')), fmt(g('digital_delete', 'ul_symbols')),
                       pct(float(np.mean([r['acc'] for r in s]))), pct(float(np.mean([r['del_acc'] for r in s])))])
    L += [table(['전력 정렬', 'K', '학습 UL 심볼', '학습 에너지', '삭제 UL 심볼', '삭제 에너지', '삭제 로컬 학습 횟수', '삭제 UL 시간',
                 '삭제 UL 시간(빈 블록 재사용)', '디지털 학습 UL', '디지털 삭제 UL', '정확도', '삭제 후 정확도'], tb), '',
          '## 2. 서버가 받은 shard 평균에서의 개인 노출', '']
    tb = []
    for stage in ['init', 'mid']:
        for s2 in [None, 1.0, .01]:
            for n in [1, 2, 4, 5, 10, 20]:
                s = [r for r in E if r['stage'] == stage and r['sigma2'] == s2 and r['n'] == n]
                if s:
                    m = lambda f: (lambda v: float(np.mean(v)) if v else float('nan'))([r[f] for r in s if r[f] == r[f]])
                    tb.append(['초기' if stage == 'init' else '학습 중간', '무잡음' if s2 is None else db(s2), n, fmt(m('cos_member')), fmt(m('cos_nonmember')),
                               pct(m('label_member')), pct(m('label_nonmember'))])
    L += [table(['모델', 'SNR', 'shard 인원 n', 'cos(받은 평균, member)', 'cos(받은 평균, 비member)', 'member 주 label 적중', '비member 주 label 적중'], tb), '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
