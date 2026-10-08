"""retrain (B) — 재학습 전송이 학습 중인 다른 shard 에 남기는 흔적.

지금까지의 실험은 삭제 재학습을 따로 떼어 (다른 자원, 다른 시간에) 보냈다. 실제로는 삭제 요청이 학습 도중 (라운드 T_del) 에 오고,
u 의 shard 는 그 자리에서 처음부터 다시 시작하며, 다른 shard 들은 학습을 계속한다. 같은 자원을 나눠 쓰면 (코드 분할) 재학습 전송이
다른 shard 에 간섭한다. 'u 가 처음부터 없었다면' 의 세계에서는 그 자리에 (u 없이 처음부터 배운) 성숙한 shard 의 전송이 있었을 것이므로,
재학습 전송과 그 전송의 차이가 다른 shard 모델에 새 흔적으로 남는다. 직교 블록 (TDMA) 이면 이 흔적은 0 이어야 한다.

학습 경로 (system 하나 = 같은 자원을 쓰는 shard 묶음)
  src   실제 순서: 모든 shard 가 u 포함으로 시작, 라운드 T_del 에 u 의 shard 가 멈추고 그 자리 (같은 코드·슬롯) 에서 u 를 뺀 재학습이
        처음부터 시작 (라운드 T_del ~ T_del+T). 다른 shard 는 라운드 T 까지 학습.
  ref   u 가 처음부터 없음: u 의 shard 는 u 를 뺀 구성으로 라운드 0 부터. 재학습 없음.
  sep   지금까지의 실험 방식: u 포함으로 라운드 T 까지 학습, 재학습은 그 뒤 따로 (다른 shard 에 영향 없음).
  alt   src 와 같은 구성을 다른 배치·잡음으로 (학습 난수 기준선).
측정 (u 의 shard 가 아닌 shard k 마다)
  전체 흔적     rel(src_k, ref_k)   실제 순서에서 정확한 언러닝과의 거리
  u 자체 흔적   rel(sep_k, ref_k)   u 가 라운드 T 까지 같이 보낸 것만의 흔적 (interference 실험이 재던 양)
  재학습 흔적   rel(src_k, sep_k)   라운드 T_del 이후 전송이 '계속 학습' 에서 '재학습' 으로 바뀐 것만의 흔적
  난수 변동     rel(src_k, alt_k)
설정: 채널 정렬 자르기 배정 (near-far 가 큰 경우), u = 가장 강한 shard 의 최고 채널 member, 20dB, 최대 전력, T_del = T/2.
다중화: 직교 블록 (TDMA) / walsh L4 칩 오차 0.3 / PN L16 칩 오차 0.
"""
import numpy as np
from . import seed_context, db
from ..channel import partition, channels
from ..config import N_CLIENTS
from ..fl import Shard, train, per_round
from ..radio import RadioConfig
from ..util import write_json, fmt, pct, table, select, mean, paired, pm

NAME = 'retrain'
K = 4
SIGMA2 = .01
MUXES = [('tdma', dict(mux='orth')), ('walsh4_d03', dict(mux='code', code='walsh', L=4, delay_max=0.3)), ('pn16', dict(mux='code', code='pn', L=16))]
MUX_KO = {'tdma': '직교 블록 (TDMA)', 'walsh4_d03': 'walsh L4, 칩 오차 0.3', 'pn16': 'PN L16'}

def rel(a, b):
    return float((a - b).norm() / b.norm())

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    T, eps = cfg['T'], cfg['eps']; T_del = T // 2
    muxes = [MUXES[0], MUXES[1]] if cfg['quick'] else MUXES
    rows, sysrows = [], []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed)
        _, h = channels(seed, T_del + T)          # 재학습이 라운드 T 를 넘겨 돌므로 채널을 그만큼 더 뽑는다 (앞 T 라운드는 c.h 와 같음)
        groups = partition('rank_chunk', range(N_CLIENTS), c.base, K, seed)
        hmin = [float(min(c.base[i] for i in g)) for g in groups]
        strong, weak = int(np.argmax(hmin)), int(np.argmin(hmin))
        rest_k = [k for k in range(K) if k not in (strong, weak)]
        slot_of = {rest_k[0]: 0, rest_k[1]: 1, strong: 2, weak: 3}
        ku = strong; u = int(max(groups[ku], key=lambda i: c.base[i])); rest = [i for i in groups[ku] if i != u]
        shards, radio = [], {}
        for name, kw in muxes:
            rc = RadioConfig(sigma2=SIGMA2, eps=eps, **kw)
            for kind, salt in [('src', ''), ('alt', 'alt')]:
                sysn = f'{name}|{kind}'; radio[sysn] = rc
                for k in range(K):
                    shards.append(Shard(f'{sysn}|{k}', groups[k], sysn, slot=slot_of[k], salt=salt, T=T_del if k == ku else T))
                shards.append(Shard(f'{sysn}|rep', rest, sysn, slot=slot_of[ku], salt=salt + 'rep', t0=T_del, T=T_del + T))
            sysn = f'{name}|ref'; radio[sysn] = rc
            shards += [Shard(f'{sysn}|{k}', rest if k == ku else groups[k], sysn, slot=slot_of[k], T=T) for k in range(K)]
            sysn = f'{name}|sep'; radio[sysn] = rc
            shards += [Shard(f'{sysn}|{k}', groups[k], sysn, slot=slot_of[k], T=T) for k in range(K)]
        log(f'[retrain] seed {seed}: shard 모델 {len(shards)}개, 삭제 라운드 {T_del}')
        W, leds, diags, _ = train(c.tr, shards, h, c.w0, seed, lambda s: radio[s], track_of=lambda s: (u,) if s.endswith('|src') else (),
                                  log=log, log_every=T // 4)
        P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
        for name, _ in muxes:
            g = lambda kind, k: W[ix[f'{name}|{kind}|{k}']]
            for k in range(K):
                if k == ku:
                    continue
                d = diags[ix[f'{name}|src|{k}']]; own = max(d['own_energy'], 1e-30)
                rows.append(dict(seed=seed, mux=name, shard=k, code_index=slot_of[k], hmin_ratio=hmin[ku] / hmin[k],
                                 rel_total=rel(g('src', k), g('ref', k)), rel_u=rel(g('sep', k), g('ref', k)), rel_retrain=rel(g('src', k), g('sep', k)),
                                 rel_yard=rel(g('src', k), g('alt', k)), leak_to_own=d['leak_energy'] / own,
                                 leak_from_slot_to_own=d['leak_by_src'].get(slot_of[ku], 0.0) / own, uleak_to_own=d['u_leak'].get(u, 0.0) / own,
                                 acc_src=c.ev.acc(c.ev.pick(P, ix[f'{name}|src|{k}'])), acc_ref=c.ev.acc(c.ev.pick(P, ix[f'{name}|ref|{k}']))))
            ens = lambda kind, repname: c.ev.ensemble([c.ev.pick(P, ix[repname if k == ku else f'{name}|{kind}|{k}']) for k in range(K)])
            sysrows.append(dict(seed=seed, mux=name, acc_sisa=ens('src', f'{name}|src|rep')['test'], acc_ref=ens('ref', f'{name}|ref|{ku}')['test'],
                                rep_rounds=leds[ix[f'{name}|src|rep']].rounds, rep_energy=leds[ix[f'{name}|src|rep']].energy,
                                slots=per_round(diags[ix[f'{name}|src|rep']], 'slots')))
        del W, P
        log(f'[retrain] seed {seed} 끝')
    write_json(out / 'results.json', dict(rows=rows, systems=sysrows, setting=dict(K=K, T=T, T_del=T_del, eps=eps, seeds=list(cfg['seeds']), muxes=[m for m, _ in muxes])))
    L = ['# retrain (B) — 재학습 전송이 학습 중인 다른 shard 에 남기는 흔적', '',
         f'채널 정렬 자르기 배정, {db(SIGMA2)}, 최대 전력, {T}라운드, seed {len(cfg["seeds"])}개. 삭제 요청은 라운드 {T_del} 에 오고, '
         f'u (가장 강한 shard 의 최고 채널 member) 의 shard 는 그 자리에서 처음부터 다시 배운다 (라운드 {T_del}~{T_del + T}). 다른 shard 는 라운드 {T} 까지 계속 배운다.', '',
         '열: 파라미터 차이 = ||W_a − W_b|| / ||W_b||. 전체 흔적 = 실제 순서 vs u 가 처음부터 없음. u 자체 흔적 = u 가 끝까지 같이 보낸 경우 vs 없음 (interference 실험이 재던 양). '
         '재학습 흔적 = 실제 순서 vs u 가 끝까지 같이 보낸 경우 (라운드 T_del 이후 전송이 바뀐 것만). 난수 변동 = 같은 조건을 다른 배치·잡음으로.', '',
         '## 1. u 의 shard 가 아닌 shard 에 남는 흔적 (shard 평균 / 최대)', '']
    tb = []
    for name, _ in muxes:
        rs = select(rows, mux=name); ss = select(sysrows, mux=name)
        if not rs:
            continue
        tb.append([MUX_KO[name], fmt(mean(rs, 'rel_total'), 4), fmt(max(r['rel_total'] for r in rs), 4), fmt(mean(rs, 'rel_u'), 4), fmt(mean(rs, 'rel_retrain'), 4),
                   fmt(max(r['rel_retrain'] for r in rs), 4), fmt(mean(rs, 'rel_yard'), 4), fmt(mean(rs, 'leak_from_slot_to_own'), 4), fmt(mean(rs, 'uleak_to_own'), 4),
                   pct(mean(ss, 'acc_sisa')), pct(mean(ss, 'acc_ref'))])
    L += [table(['다중화', '전체 흔적', '전체 최대', 'u 자체 흔적', '재학습 흔적', '재학습 최대', '난수 변동', 'u 의 자원에서 온 간섭/자기', 'u 간섭/자기',
                 'SISA 앙상블 정확도', 'u 없이 처음부터 정확도'], tb), '',
          '## 2. 판정 (± 는 seed·shard 짝 차이의 2×표준오차)', '']
    for name, _ in muxes:
        rs = select(rows, mux=name)
        if not rs:
            continue
        m = mean(rs, 'rel_retrain'); mx = max(r['rel_retrain'] for r in rs)
        floor = max(r['rel_retrain'] for r in select(rows, mux=muxes[0][0])) if muxes[0][1].get('mux') == 'orth' else 1e-3   # 직교 블록의 값 = vmap 부동소수점 바닥
        L.append(f'- {MUX_KO[name]}: 재학습 흔적 평균 {fmt(m, 4)}, 최대 {fmt(mx, 4)} → '
                 + ('부동소수점 바닥 (재학습 전송이 남에게 닿지 않음)' if mx <= floor * 1.01 else f'재학습 전송이 다른 shard 를 바꿈 (바닥 {fmt(floor, 4)} 의 {mx / max(floor, 1e-12):.0f}배)'))
    if len(muxes) > 1:
        for name, _ in muxes[1:]:
            m, se, n = paired(rows, 'rel_total', {}, dict(mux=name), dict(mux=muxes[0][0]), keys=('seed', 'shard'))
            L.append(f'- 전체 흔적 ({MUX_KO[name]} − {MUX_KO[muxes[0][0]]}) = {pm(m, se, fmt)} (n={n})')
            m, se, n = paired(rows, 'rel_retrain', {}, dict(mux=name), dict(mux=muxes[0][0]), keys=('seed', 'shard'))
            L.append(f'- 재학습 흔적 ({MUX_KO[name]} − {MUX_KO[muxes[0][0]]}) = {pm(m, se, fmt)} (n={n})')
    L += ['', '## 읽는 법', '',
          '- 직교 블록이면 세 흔적이 모두 부동소수점 바닥 (약 1e-4) 이어야 한다. 재학습이 언제 어디서 돌든 다른 shard 는 모른다.',
          '- 같은 자원을 나눠 쓰면 재학습 흔적이 0 보다 크다. 이것은 u 가 직접 새는 것이 아니라, u 를 지운 "결과" (재학습 전송) 가 새는 것이다. '
          'u 를 안 보내게 해도 (예: 삭제 전 u 의 전송을 완벽히 분리해도) 남는 흔적이므로, 재학습 자원을 따로 떼어야만 없어진다.',
          '- 재학습 흔적이 u 자체 흔적보다 크면, 지금까지 재학습을 따로 돌린 실험은 흔적을 과소평가한 것이다.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
