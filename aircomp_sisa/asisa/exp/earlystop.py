"""earlystop (D) — 전력·스케줄링이 아니라도, '언제 멈출지' 를 전체 검증 정확도로 정하면 정확성이 깨지는가.

control (E1) 은 전력·송신 순서를 shard 밖 정보로 정하면 직교 블록에서도 흔적이 남는다는 것을 보였다. 같은 원리는 학습의 모든
전체 결정에 해당한다. 가장 흔한 것이 조기 종료다: 앙상블 (모든 shard) 의 검증 정확도를 보고 멈추는 라운드 t* 를 정하면,
u 를 지운 세계에서는 앙상블이 달라져 t* 가 달라질 수 있고, 그러면 u 와 무관한 shard 의 최종 모델 (W_k[t*]) 도 달라진다.
shard 마다 자기 검증 정확도로 멈추면 (shard 안 결정) u 와 무관한 shard 는 같은 모델이다.

정지 규칙 (체크포인트 간격 = T/16)
  best      앙상블 dev 정확도가 가장 높은 체크포인트 (전체 결정)
  target    앙상블 dev 정확도가 자기 최고의 97% 에 처음 닿는 체크포인트 (전체 결정)
  patience  앙상블 dev 정확도가 두 체크포인트 연속 안 오르면 멈춤 (전체 결정)
  local     shard 마다 자기 dev 정확도가 가장 높은 체크포인트 (shard 안 결정)
측정: 무작위 배정 4 shard, 직교 블록, shard 안 전력 정렬, 20dB. 삭제 대상 = 전체 최약 노드 / 채널 중간 노드.
  u 의 shard 가 아닌 shard 마다 rel = ||W_src[t*_src] − W_ref[t*_ref]|| / ||W_ref[t*_ref]||, t* 가 달라진 비율, 앙상블 정확도.
"""
import numpy as np
import torch
from . import seed_context, db
from ..channel import partition
from ..config import N_CLIENTS
from ..fl import Shard, train
from ..radio import RadioConfig
from ..util import write_json, fmt, pct, table, select, mean

NAME = 'earlystop'
K = 4
SIGMA2 = .01
RULES = ['best', 'target', 'patience', 'local']
RULE_KO = {'best': '앙상블 최고 (전체)', 'target': '앙상블 최고의 97% (전체)', 'patience': '앙상블 2번 정체 (전체)', 'local': 'shard 자기 최고 (shard 안)'}
TARGET_KO = {'weakest': '전체 최약 노드', 'median': '채널 중간 노드'}

def stop_round(rule, acc_by_t):
    ts = sorted(acc_by_t)
    if rule == 'best':
        return max(ts, key=lambda t: (acc_by_t[t], -t))
    if rule == 'target':
        goal = 0.97 * max(acc_by_t.values())
        return next(t for t in ts if acc_by_t[t] >= goal)
    if rule == 'patience':
        best, bad = -1.0, 0
        for t in ts:
            if acc_by_t[t] > best:
                best, bad = acc_by_t[t], 0
            else:
                bad += 1
                if bad >= 2:
                    return t
        return ts[-1]
    raise ValueError(rule)

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    T, eps = cfg['T'], cfg['eps']
    step = max(1, T // 16); CK = tuple(range(step, T + 1, step))
    targets_all = ['weakest'] if cfg['quick'] else ['weakest', 'median']
    rows, sysrows = [], []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed)
        groups = partition('random_split', range(N_CLIENTS), c.base, K, seed)
        order = np.argsort(c.base)
        targets = {tn: int(order[0] if tn == 'weakest' else order[N_CLIENTS // 2]) for tn in targets_all}
        shards = [Shard(f'src|{k}', groups[k], 'src', slot=k, T=T, save_at=CK) for k in range(K)]
        for tn, u in targets.items():
            shards += [Shard(f'ref|{tn}|{k}', [i for i in groups[k] if i != u], f'ref|{tn}', slot=k, T=T, save_at=CK) for k in range(K)]
        log(f'[earlystop] seed {seed}: shard 모델 {len(shards)}개, 체크포인트 {len(CK)}개')
        W, _, _, ck = train(c.tr, shards, c.h, c.w0, seed, lambda s: RadioConfig(sigma2=SIGMA2, eps=eps), log=log, log_every=T // 4)
        ix = {s.name: j for j, s in enumerate(shards)}
        def curve(prefix):
            """체크포인트마다 (앙상블 dev 정확도, shard 별 dev 정확도)."""
            ens, loc = {}, {k: {} for k in range(K)}
            for t in CK:
                Wt = torch.stack([ck[ix[f'{prefix}|{k}']][t] for k in range(K)])
                P = c.ev.probs(Wt)
                ens[t] = c.ev.ensemble([c.ev.pick(P, k) for k in range(K)])['dev']
                for k in range(K):
                    p = c.ev.pick(P, k); loc[k][t] = float((p['dev'].argmax(1) == c.ev.ydev).float().mean())
            return ens, loc
        src_ens, src_loc = curve('src')
        for tn, u in targets.items():
            ku = next(k for k in range(K) if u in groups[k])
            ref_ens, ref_loc = curve(f'ref|{tn}')
            for rule in RULES:
                if rule == 'local':
                    ts = {k: stop_round('best', src_loc[k]) for k in range(K)}; tr_ = {k: stop_round('best', ref_loc[k]) for k in range(K)}
                else:
                    t1, t2 = stop_round(rule, src_ens), stop_round(rule, ref_ens)
                    ts = {k: t1 for k in range(K)}; tr_ = {k: t2 for k in range(K)}
                for k in range(K):
                    if k == ku:
                        continue
                    a = ck[ix[f'src|{k}']][ts[k]]; b = ck[ix[f'ref|{tn}|{k}']][tr_[k]]
                    rows.append(dict(seed=seed, target=tn, rule=rule, shard=k, t_src=ts[k], t_ref=tr_[k], differs=float(ts[k] != tr_[k]),
                                     rel=float((a - b).norm() / b.norm())))
                Wf = torch.stack([ck[ix[f'src|{k}']][ts[k]] for k in range(K)]); P = c.ev.probs(Wf)
                sysrows.append(dict(seed=seed, target=tn, rule=rule, acc=c.ev.ensemble([c.ev.pick(P, k) for k in range(K)])['test'],
                                    t_mean=float(np.mean(list(ts.values())))))
        del W
        log(f'[earlystop] seed {seed} 끝')
    write_json(out / 'results.json', dict(rows=rows, systems=sysrows, setting=dict(K=K, T=T, checkpoints=list(CK), eps=eps, seeds=list(cfg['seeds']))))
    L = ['# earlystop (D) — 멈출 라운드를 전체 검증으로 정하면 정확성이 깨지는가', '',
         f'무작위 배정 {K} shard, 직교 블록, shard 안 정렬, {db(SIGMA2)}, {T}라운드, 체크포인트 {len(CK)}개 (간격 {step}), seed {len(cfg["seeds"])}개.', '',
         '열: 흔적 = u 의 shard 가 아닌 shard 의 ||W_src[t*_src] − W_ref[t*_ref]|| / ||W_ref[t*_ref]|| (평균·최대). t* 다름 = src 와 ref 의 정지 라운드가 다른 비율.', '']
    tb = []
    for tn in targets_all:
        for rule in RULES:
            rs = select(rows, target=tn, rule=rule); ss = select(sysrows, target=tn, rule=rule)
            if rs:
                tb.append([TARGET_KO[tn], RULE_KO[rule], fmt(mean(ss, 't_mean'), 0), pct(mean(rs, 'differs')), fmt(mean(rs, 'rel'), 4),
                           fmt(max(r['rel'] for r in rs), 4), pct(mean(ss, 'acc'))])
    L += [table(['삭제 대상', '정지 규칙', '정지 라운드 (src 평균)', 't* 다름', '흔적 평균', '흔적 최대', '앙상블 정확도'], tb), '',
          '## 읽는 법', '',
          '- shard 안 규칙 (local) 은 u 와 무관한 shard 의 정지 라운드가 src 와 ref 에서 같으므로 흔적이 0 이어야 한다.',
          '- 전체 규칙은 t* 가 다른 경우에만 흔적이 생긴다. t* 다름 비율이 0 이면 "이 설정에서는 안 걸렸다" 는 뜻이지 안전하다는 뜻이 아니다. '
          '학습이 단조롭게 오르는 짧은 학습에서는 best 가 거의 마지막 체크포인트라 잘 안 걸리고, target·patience 가 더 잘 걸린다.',
          '- 흔적이 생기면 그 크기는 체크포인트 간격 동안 모델이 움직인 양이다 (잡음이 클수록, 간격이 클수록 큼).', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
