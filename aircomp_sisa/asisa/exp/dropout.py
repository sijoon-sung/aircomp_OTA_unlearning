"""dropout (E4) — 학습 중 client 가 라운드마다 빠지면 shard 합의 인원이 줄어 노출과 잡음이 얼마나 나빠지는가. 최소 인원 규칙이 얼마나 싸게 막는가.

라운드마다 client 는 확률 p 로 참여하지 못한다 (client·라운드로 정해짐). 그 라운드 shard 합의 실제 인원 n_t 와 최약 채널이 바뀐다.
AirComp 는 빠진 사람이 있어도 합이 그대로 계산된다 (SecAgg 처럼 비밀 분산으로 마스크를 복구하는 절차가 없다).
대신 n_t 가 작은 라운드의 합은 개인 정보를 많이 드러낸다.
설정: 무작위 배정, shard 수 K in {2, 4, 10}, 이탈률 p in {0, 0.2, 0.4}, 규칙 2가지 (그대로 보냄 / 그 라운드 인원이 3명 미만이면 보내지 않음),
      직교 블록, shard 안 최대 전력 정렬, 20dB.
측정: 라운드마다 실제로 받은 shard 평균의 마지막 층 bias 로 주 label 을 추정해, 그 라운드에 보낸 member 의 주 label 과 맞는 비율 (노출).
      n_t 별로 나눠서 본다. 기준 = shard 밖 client 의 주 label 과 맞는 비율.
      n_t <= 2 인 라운드 비율, 보내지 않은 라운드 비율, 평균 집계 오차, 앙상블 정확도.
"""
import numpy as np
from . import seed_context, db
from ..channel import partition
from ..config import N_CLIENTS
from ..fl import Shard, train
from ..radio import RadioConfig
from ..util import write_json, fmt, pct, table, select, mean

NAME = 'dropout'
KS = [2, 4, 10]
PS = [0.0, 0.2, 0.4]
POLICIES = {'send': 0, 'min3': 3}
POLICY_KO = {'send': '그대로 보냄', 'min3': '3명 미만이면 보내지 않음 (제안)'}

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    T = cfg['T']
    Ks = [4] if cfg['quick'] else KS
    rows, rounds = [], []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed); dom = c.data['hist'].argmax(1)
        shards, info = [], {}
        for K in Ks:
            groups = partition('random_split', range(N_CLIENTS), c.base, K, seed)
            for p in PS:
                for pol, mp in POLICIES.items():
                    tag = f'{K}|{p}|{pol}'; info[tag] = groups
                    shards += [Shard(f'{tag}|{k}', g, tag, slot=k, drop=p, min_present=mp, T=T) for k, g in enumerate(groups)]
        name_of = [s.name for s in shards]; memb = [s.members for s in shards]
        stat = {}
        def hook(j, t, active, r, U):
            tag = name_of[j].rsplit('|', 1)[0]; lab = int(r[-10:].argmax()); n = len(active)
            outs = [i for i in range(N_CLIENTS) if i not in memb[j]]
            s = stat.setdefault((tag, n), [0, 0.0, 0.0])
            s[0] += 1; s[1] += float(np.mean([lab == dom[i] for i in active])); s[2] += float(np.mean([lab == dom[i] for i in outs])) if outs else float('nan')
        log(f'[dropout] seed {seed}: shard 모델 {len(shards)}개')
        W, leds, diags, _ = train(c.tr, shards, c.h, c.w0, seed, lambda s: RadioConfig(sigma2=.01, eps=cfg['eps']), log=log, log_every=T // 4, hook=hook)
        P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
        for K in Ks:
            for p in PS:
                for pol in POLICIES:
                    tag = f'{K}|{p}|{pol}'; groups = info[tag]
                    js = [ix[f'{tag}|{k}'] for k in range(len(groups))]
                    sent = sum(diags[j]['rounds'] for j in js); total = T * len(js)
                    hist = {n: v for (tg, n), v in stat.items() if tg == tag}
                    nr = sum(v[0] for v in hist.values())
                    rows.append(dict(seed=seed, K=K, p=p, policy=pol, size=N_CLIENTS / K,
                                     sent_frac=sent / total, small_frac=sum(v[0] for n, v in hist.items() if n <= 2) / max(1, nr),
                                     one_frac=sum(v[0] for n, v in hist.items() if n == 1) / max(1, nr),
                                     mean_n=sum(n * v[0] for n, v in hist.items()) / max(1, nr),
                                     hit=sum(v[1] for v in hist.values()) / max(1, nr), hit_out=float(np.nanmean([v[2] / v[0] for v in hist.values()])) if hist else float('nan'),
                                     hit_small=(sum(v[1] for n, v in hist.items() if n <= 2) / max(1, sum(v[0] for n, v in hist.items() if n <= 2))) if any(n <= 2 for n in hist) else float('nan'),
                                     mse=float(np.mean([leds[j].mse_sum / max(1, leds[j].rounds) for j in js])),
                                     acc=c.ev.ensemble([c.ev.pick(P, j) for j in js])['test']))
                    for n, v in hist.items():
                        rounds.append(dict(seed=seed, K=K, p=p, policy=pol, n=n, count=v[0], hit=v[1] / v[0]))
        stat.clear(); del W, P
        log(f'[dropout] seed {seed} 끝')
    write_json(out / 'results.json', dict(rows=rows, by_n=rounds, Ks=Ks, ps=PS, policies=POLICIES))
    L = ['# dropout (E4) — 학습 중 이탈이 shard 합의 노출과 잡음을 얼마나 나쁘게 하는가', '',
         f'{db(.01)}, {cfg["T"]}라운드, seed {len(cfg["seeds"])}개. 노출 = 라운드마다 받은 shard 평균으로 추정한 주 label 이 그 라운드에 보낸 member 의 주 label 과 맞는 비율 (학습 전체 평균). '
         '기준 = shard 밖 client 의 주 label 과 맞는 비율.', '']
    tb = []
    for K in Ks:
        for p in PS:
            for pol in POLICIES:
                s = select(rows, K=K, p=p, policy=pol)
                if s:
                    tb.append([K, fmt(N_CLIENTS / K, 0), p, POLICY_KO[pol], fmt(mean(s, 'mean_n'), 2), pct(mean(s, 'small_frac')), pct(mean(s, 'one_frac')),
                               pct(mean(s, 'hit')), pct(mean(s, 'hit_small')), pct(mean(s, 'hit_out')), pct(1 - mean(s, 'sent_frac')), fmt(mean(s, 'mse'), 4), pct(mean(s, 'acc'))])
    L += [table(['K', 'shard 인원', '이탈률 p', '규칙', '라운드당 실제 인원', '인원 ≤ 2 라운드', '인원 = 1 라운드', '노출 (전체)', '노출 (인원 ≤ 2 라운드)',
                 '노출 기준', '보내지 않은 라운드', '평균 집계 오차', '앙상블 정확도'], tb), '',
          '## 실제 인원 n_t 별 노출 (모든 조건 합침, 그대로 보냄)', '']
    tb = []
    for n in sorted({r['n'] for r in rounds}):
        s = [r for r in rounds if r['n'] == n and r['policy'] == 'send']
        if s:
            cnt = sum(r['count'] for r in s)
            tb.append([n, cnt, pct(sum(r['hit'] * r['count'] for r in s) / cnt)])
    L += [table(['그 라운드 실제 인원 n_t', '라운드 수', '노출'], tb), '',
          '## 읽는 법', '',
          '- 이탈률이 오르면 shard 가 작을수록 (K 가 클수록) 인원 1~2명 라운드가 급격히 늘고, 그 라운드의 노출이 높다.',
          '- 3명 미만이면 보내지 않는 규칙은 그런 라운드를 없앤다. 대가는 보내지 않은 라운드 비율 (학습이 느려짐) 과 정확도 변화로 본다.',
          '- 이 규칙은 shard 자기 인원만 보므로 다른 shard 와 독립이고, u 를 지운 재학습에서도 같은 규칙이 적용되어 정확성과 충돌하지 않는다.',
          '- 이탈 처리 자체는 AirComp 에서 추가 절차가 없다 (SecAgg 는 비밀 분산 복구가 필요).', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
