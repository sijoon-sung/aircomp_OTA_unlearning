"""noise — 정확도를 해치지 않는 집계 잡음의 한계 eps*.

바꾸는 것: 라운드마다 shard 평균 update 에 더하는 잡음의 제곱합 기대값 eps (집계 오차). 잡음 방향은 모든 eps 에서 같고 크기만 다르다.
고정: 무작위 배정 4 shard x 5명, 데이터, 초기값, minibatch, 라운드 수.
측정: 학습 앙상블 정확도, shard 마다 1명 삭제(해당 shard 재학습) 후 앙상블 정확도, 2명짜리 shard 단독 정확도.
판정: 잡음 0 대비 정확도 하락이 모든 seed 에서 1pp 이하인 단계가 처음부터 연속되는 최대 eps 를 eps* 로 둔다.
"""
import numpy as np
from . import seed_context
from ..channel import partition
from ..config import N_CLIENTS
from ..fl import Shard, train
from ..radio import RadioConfig
from ..util import write_json, fmt, pp, table

NAME = 'noise'
LEVELS = [0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
K = 4

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    levels = [0, 10, 1000] if cfg['quick'] else LEVELS
    rows = []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed)
        groups = partition('random_split', range(N_CLIENTS), c.base, K, seed)
        shards = []
        for lv in levels:
            for k, g in enumerate(groups):
                shards.append(Shard(f'{lv}|src|{k}', g, str(lv), slot=k, T=cfg['T']))
                shards.append(Shard(f'{lv}|del|{k}', g[1:], str(lv), slot=k, T=cfg['T']))
            shards.append(Shard(f'{lv}|small', groups[0][:2], str(lv), slot=100, T=cfg['T']))
        log(f'[noise] seed {seed}: shard 모델 {len(shards)}개')
        W, _, _, _ = train(c.tr, shards, c.h, c.w0, seed, lambda s: RadioConfig(mux='ideal', noise_mse=float(s)), log=log, log_every=cfg['T'] // 4)
        P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
        for lv in levels:
            src = [c.ev.pick(P, ix[f'{lv}|src|{k}']) for k in range(K)]
            dels = [c.ev.ensemble([c.ev.pick(P, ix[f'{lv}|del|{k}']) if q == k else src[q] for q in range(K)])['test'] for k in range(K)]
            rows.append(dict(seed=seed, eps=lv, train_acc=c.ev.ensemble(src)['test'], delete_acc=float(np.mean(dels)),
                             small_acc=c.ev.acc(c.ev.pick(P, ix[f'{lv}|small'])), dev_acc=c.ev.ensemble(src)['dev']))
        log(f'[noise] seed {seed} 끝')
    drop = {}
    for f in ('train_acc', 'delete_acc', 'small_acc'):
        for r in rows:
            if r['eps'] > 0:
                b = next(x[f] for x in rows if x['seed'] == r['seed'] and x['eps'] == 0)
                drop.setdefault((f, r['eps']), []).append(b - r[f])
    pos = [lv for lv in levels if lv > 0]
    def contiguous(ok):
        best = 0
        for lv in pos:
            if not ok(lv):
                break
            best = lv
        return best
    eps_star = contiguous(lambda lv: max(drop[('train_acc', lv)]) <= .01 and max(drop[('delete_acc', lv)]) <= .01)
    eps_small = contiguous(lambda lv: max(drop[('small_acc', lv)]) <= .01)
    write_json(out / 'results.json', dict(rows=rows, eps_star=eps_star, eps_star_small_shard=eps_small))
    tb = [[fmt(lv, 0)] + [pp(f(drop[(k, lv)])) for k in ('train_acc', 'delete_acc', 'small_acc') for f in (np.mean, max)] for lv in pos]
    L = ['# noise — 정확도를 해치지 않는 집계 잡음의 한계', '',
         f'seed {len(cfg["seeds"])}개, {cfg["T"]}라운드, 무작위 배정 {K} shard. eps = 라운드마다 shard 평균 update 에 더한 잡음 벡터의 제곱합 기대값.', '',
         '## 잡음 0 대비 정확도 하락 (양수 = 나빠짐)', '',
         table(['eps', '학습 평균', '학습 최대', '삭제 후 평균', '삭제 후 최대', '2명 shard 평균', '2명 shard 최대'], tb), '',
         f'## 판정', '',
         f'- eps* (학습·삭제 후 모두 모든 seed 에서 하락 1pp 이하) = {fmt(eps_star, 0)}. 2명 shard 기준으로는 {fmt(eps_small, 0)}.',
         f'- 다른 실험은 eps = {fmt(cfg["eps"])} 를 쓴다 (`--eps` 로 바꿀 수 있음). eps* 가 이와 다르면 `--eps` 를 맞춰 다시 돌린다.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(eps_star=eps_star)
