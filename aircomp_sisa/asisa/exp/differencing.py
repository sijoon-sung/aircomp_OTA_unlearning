"""differencing (E3) — 재학습 때 서버가 삭제 전후의 shard 합을 모두 보면, 그 차이로 개인 update 가 드러나는가. 새 초기값 재학습이 막는가.

라운드 t 에 서버가 받는 값 (shard S, 인원 n)
  원래 학습:   y_t  = (1/n) sum_{S} x_i(theta_t) + 잡음
  재학습:      y'_t = (1/(n-1)) sum_{S - u} x_i(theta'_t) + 잡음'      (잡음과 잡음' 은 서로 다른 전송이라 독립)
  서버의 차분: d_t = n y_t - (n-1) y'_t
  같은 초기값·같은 배치로 재학습하면 theta'_0 = theta_0 이고 다른 member 의 update 가 같으므로 d_0 = x_u + 잡음 차이 -> u 의 개별 update 가 드러난다.
  재학습 첫 라운드에 다른 member v 가 이탈하면 d_0 = n y_0 - (n-2) y''_0 = x_u + x_v + 잡음 -> 삭제를 요청하지 않은 v 까지 섞여 드러난다.
재학습 방식
  same      같은 초기값, 같은 배치 순서 (보통의 '처음부터 다시')
  newbatch  같은 초기값, 새 배치 순서
  newinit   새 초기값, 새 배치 순서 (제안: 정확한 언러닝은 분포가 같으면 되므로 허용된다. 같은 모델 위에서 받은 두 합이 없어진다)
  same_drop same 에서 재학습 첫 라운드에 member v 가 이탈
측정: 라운드 0, 1, 2, 5, 10 에서 cos(d_t, x_u) (u 의 실제 update 와의 방향 일치), d_t 의 마지막 층 bias 로 추정한 u 의 주 label 적중,
      이탈 경우 cos(d_t, x_v) 와 v 의 주 label 적중. shard 인원 n in {2,3,5,10}, SNR 20 / 0 / -10dB.
      기준: 같은 방법으로 아무 관계 없는 client 의 주 label 을 맞히는 비율 (10 class 라 우연히 약 10%).
"""
import numpy as np
import torch
from . import seed_context, db
from ..config import N_CLIENTS
from ..fl import Shard, train
from ..model import init_vector
from ..radio import RadioConfig
from ..util import key, write_json, fmt, pct, table, select, mean

NAME = 'differencing'
NS = [2, 3, 5, 10]
REPS = 4
SNRS = [0.01, 1.0, 10.0]
MODES = ['same', 'newbatch', 'newinit', 'same_drop']
MODE_KO = {'same': '같은 초기값·같은 배치', 'newbatch': '같은 초기값·새 배치', 'newinit': '새 초기값 (제안)', 'same_drop': '같은 초기값·같은 배치 + 첫 라운드 이탈'}
ROUNDS = [0, 1, 2, 5, 10]

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    T = min(cfg['T'], max(ROUNDS) + 1)
    ns = [2, 5] if cfg['quick'] else NS
    snrs = [1.0] if cfg['quick'] else SNRS
    reps = 2 if cfg['quick'] else REPS
    rows = []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed); dom = c.data['hist'].argmax(1)
        w_new = init_vector(seed, cfg['device'], 'retrain')
        rng = np.random.default_rng(key(seed, 'differencing'))
        groups = []
        for n in ns:
            for r in range(reps):
                S = sorted(int(i) for i in rng.choice(N_CLIENTS, n, replace=False)); u = int(rng.choice(S))
                others = [i for i in S if i != u]; v = int(rng.choice(others)) if len(others) > 1 else None
                outsider = int(rng.choice([i for i in range(N_CLIENTS) if i not in S]))
                groups.append(dict(n=n, rep=r, S=S, u=u, v=v, outsider=outsider))
        shards, meta = [], {}
        for s2 in snrs:
            for gi, g in enumerate(groups):
                tag = f'{s2}|{gi}'; rest = [i for i in g['S'] if i != g['u']]
                shards.append(Shard(f'{tag}|orig', g['S'], f'{tag}|orig', slot=gi, T=T))
                shards.append(Shard(f'{tag}|same', rest, f'{tag}|same', slot=gi, noise_salt='re', T=T))
                shards.append(Shard(f'{tag}|newbatch', rest, f'{tag}|newbatch', slot=gi, salt='re', T=T))
                shards.append(Shard(f'{tag}|newinit', rest, f'{tag}|newinit', slot=gi, salt='re', w0=w_new, T=T))
                if g['v'] is not None:
                    shards.append(Shard(f'{tag}|same_drop', rest, f'{tag}|same_drop', slot=gi, noise_salt='re', absent={0: {g['v']}}, T=T))
        name_of = [s.name for s in shards]
        rec = {}
        def hook(j, t, active, r, U):
            if t in ROUNDS:
                nm = name_of[j]
                rec[(nm, t)] = (r.detach().float().cpu(), len(active))
                if nm.endswith('|orig'):
                    for q, i in enumerate(active):
                        rec[(nm, t, i)] = U[q].detach().float().cpu()
        log(f'[differencing] seed {seed}: shard 모델 {len(shards)}개, {T}라운드')
        train(c.tr, shards, c.h, c.w0, seed, lambda s: RadioConfig(sigma2=float(s.split('|')[0]), eps=cfg['eps']), hook=hook)
        for s2 in snrs:
            for gi, g in enumerate(groups):
                tag = f'{s2}|{gi}'; n = g['n']
                for mode in MODES:
                    if mode == 'same_drop' and g['v'] is None:
                        continue
                    for t in ROUNDS:
                        if (f'{tag}|orig', t) not in rec or (f'{tag}|{mode}', t) not in rec:
                            continue
                        y, n1 = rec[(f'{tag}|orig', t)]; y2, n2 = rec[(f'{tag}|{mode}', t)]
                        d = n1 * y - n2 * y2
                        xu = rec[(f'{tag}|orig', t, g['u'])]
                        cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-30))
                        lab = int(d[-10:].argmax())
                        row = dict(seed=seed, sigma2=s2, n=n, rep=g['rep'], mode=mode, t=t, cos_u=cos(d, xu), hit_u=float(lab == dom[g['u']]),
                                   hit_outsider=float(lab == dom[g['outsider']]))
                        if mode == 'same_drop':
                            xv = rec[(f'{tag}|orig', t, g['v'])]
                            row.update(cos_v=cos(d, xv), hit_v=float(lab == dom[g['v']]))
                        rows.append(row)
        rec.clear()
        log(f'[differencing] seed {seed} 끝')
    write_json(out / 'results.json', dict(rows=rows, ns=ns, snrs=snrs, rounds=ROUNDS, modes=MODES))
    L = ['# differencing (E3) — 삭제 전후 shard 합의 차이로 개인 update 가 드러나는가', '',
         f'서버의 차분 d_t = n·y_t − (n−1)·y\'_t (y: 원래 학습의 shard 평균, y\': 재학습의 shard 평균, 같은 라운드 t). seed {len(cfg["seeds"])}개 × shard {reps}개씩. '
         'cos = d_t 와 삭제 대상 u 의 실제 update 의 방향 일치 (1 이면 그대로 드러남). 적중 = d_t 의 마지막 층 bias 로 u 의 주 label 을 맞힌 비율. '
         '기준 = 같은 d_t 로 shard 밖 무관한 client 의 주 label 을 맞힌 비율.', '']
    for s2 in snrs:
        L += [f'## {db(s2)}: 재학습 첫 라운드 (t = 0)', '']
        tb = []
        for n in ns:
            for mode in MODES:
                s = select(rows, sigma2=s2, n=n, mode=mode, t=0)
                if not s:
                    continue
                tb.append([n, MODE_KO[mode], fmt(mean(s, 'cos_u'), 3), pct(mean(s, 'hit_u')), fmt(mean(s, 'cos_v'), 3) if mode == 'same_drop' else '-',
                           pct(mean(s, 'hit_v')) if mode == 'same_drop' else '-', pct(mean(s, 'hit_outsider'))])
        L += [table(['shard 인원 n', '재학습 방식', 'cos(d, x_u)', 'u 주 label 적중', 'cos(d, x_v) (이탈자)', 'v 주 label 적중', '기준 (무관한 client)'], tb), '']
    L += ['## 라운드가 지나면서 (n = 5, 같은 초기값·같은 배치)', '']
    tb = []
    for s2 in snrs:
        for t in ROUNDS:
            s = select(rows, sigma2=s2, n=5 if 5 in ns else ns[-1], mode='same', t=t)
            if s:
                tb.append([db(s2), t, fmt(mean(s, 'cos_u'), 3), pct(mean(s, 'hit_u'))])
    L += [table(['SNR', '라운드 t', 'cos(d, x_u)', 'u 주 label 적중'], tb), '',
          '## 읽는 법', '',
          '- 같은 초기값·같은 배치로 재학습하면 t = 0 에서 다른 member 의 update 가 정확히 상쇄되어 d 가 u 의 update 에 채널 잡음만 더한 값이 된다. 잡음이 작은 SNR 에서 cos 가 1 에 가까워야 한다.',
          '- 첫 라운드에 v 가 이탈하면 d 에 x_v 가 섞인다. 삭제를 요청하지 않은 v 의 정보가 드러나는 경로다.',
          '- 새 배치만 바꾸면 같은 모델에서의 update 가 조금씩 달라 상쇄가 불완전해진다. 새 초기값이면 두 합이 서로 다른 모델 위의 값이라 차분이 의미를 잃어야 한다 (cos ≈ 0, 적중 ≈ 기준).',
          '- 라운드가 지나면 원래 학습과 재학습의 모델이 갈라져 같은 초기값이어도 차분이 약해진다. 위험은 재학습 초반에 집중된다.',
          '- 새 초기값 재학습은 정확한 언러닝의 정의 (u 없이 처음부터 학습한 것과 같은 분포) 를 그대로 지키면서 추가 비용이 없다.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
