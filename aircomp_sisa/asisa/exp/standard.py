"""standard — 기존 AirComp 논문의 조건으로 다시 돌리면 SISA 언러닝의 그림이 바뀌는가.

지금까지의 실험은 네 가지 점에서 기존 AirComp (Zhu·Wang·Huang 2020 BAA, Cao et al. 2022) 보다 순했다.
  (1) 페이딩이 U[0.85, 1.15] 였다 → 여기서는 Rayleigh (깊은 골).
  (2) 가장 약한 member 도 항상 보냈다 → 여기서는 절단 채널 역전: |h|^2 < g_th 인 client 는 그 라운드에 보내지 않고, 보내는 client 는
      공개 상수 beta = C / (sqrt(g_th) sqrt(PD)) 로 도착한다. 그래서 shard 합의 인원 n_t 가 라운드마다 채널 따라 바뀐다.
  (3) 집계 오차가 eps 를 넘으면 반복 전송으로 잡음을 시간으로 바꿨다 → 여기서는 반복 없음 (eps = 0). 잡음을 정확도로 받는다.
  (4) 동기·주파수 오차를 일부 실험에만 넣었고 RF 는 이상적이었다 → 여기서는 FDMA 에 주파수 오차 0.05 + CP 를 넘는 타이밍 오차 + 전력증폭기 비선형 (Rapp, IBO 6dB)
      + 수신 ADC 양자화 (8 bit, 풀스케일은 전체 대역) 를 두고, 코드 분할에 칩 오차 0.3 을 둔다. 이것들이 FDMA 로 갈라 놓은 shard 를 다시 묶는 현실의 경로다.
두 조건 (mild = 지금까지, std = 기존 AirComp) × SNR {20, 10, 0, -10dB} × 다중화 {TDMA, FDMA, 코드} 에서 같은 것을 잰다.
배정은 ID 해시 (shard 안 정보만). u = 가장 강한 shard 의 최고 채널 member.
측정 (deletion.py + hook)
  앙상블 정확도 (src), u 없이 처음부터의 정확도, 다른 shard 에 남는 흔적 rel 과 난수 변동 rel_yard, SISA vs 전체 재학습 앙상블 거리,
  라운드당 실제 송신 인원 n_t (평균, <= 2 인 라운드 비율, 0 인 라운드 비율), 노출 (받은 합의 bias 로 그 라운드 송신 member 의 주 label 적중, 기준 = 비member),
  평균 집계 오차, 라운드당 에너지, 슬롯 (반복 포함 시간).
"""
import numpy as np
from . import seed_context, db
from .deletion import add_system, measure
from ..channel import partition, channels
from ..config import N_CLIENTS
from ..fl import train
from ..radio import RadioConfig
from ..util import write_json, fmt, pct, pp, table, select, mean, paired, pm

NAME = 'standard'
K = 4
G_TH = 0.02                      # 절단 문턱 |h|^2 (-17dB). 장기 채널 10^(U[-20,0]/10) 에서 평균 참여율 약 70%
CFO, DELAY = 0.05, 0.3
RF = dict(timing=12.0, cp=8, pa_ibo=6.0, adc_bits=8)     # std 조건의 FDMA RF 손상: 타이밍 오차 U[0,12] 샘플 (CP 8 초과분이 ICI), PA IBO 6dB, ADC 8 bit
SNRS = [0.01, 0.1, 1.0, 10.0]
CONDS = ['mild', 'std']
COND_KO = {'mild': '지금까지 (순한 페이딩, 전원 송신, 반복 전송)', 'std': '기존 AirComp (Rayleigh, 절단 채널 역전, 반복 없음)'}
MUX_KO = {'tdma': 'TDMA', 'fdma': 'FDMA (주파수 오차 0.05; std 는 +타이밍·PA·ADC)', 'code': 'walsh L4 (칩 오차 0.3)'}

def radio_for(cond, s2, mux, eps):
    base = dict(sigma2=s2)
    if cond == 'mild':
        base.update(eps=eps, align='maxpow')
    else:
        base.update(eps=0.0, align='trunc', trunc=G_TH)
    if mux == 'tdma':
        return RadioConfig(mux='orth', **base)
    if mux == 'fdma':
        return RadioConfig(mux='ofdm', groups=1, alloc='block', cfo=CFO, **(RF if cond == 'std' else {}), **base)
    return RadioConfig(mux='code', code='walsh', L=4, delay_max=DELAY, **base)

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    T, eps = cfg['T'], cfg['eps']
    snrs = [0.01, 1.0] if cfg['quick'] else SNRS
    muxes = ['tdma', 'code'] if cfg['quick'] else ['tdma', 'fdma', 'code']
    rows, sysrows = [], []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed); dom = c.data['hist'].argmax(1)
        groups = [g for g in partition('hash', range(N_CLIENTS), c.base, K, seed)]
        live = [k for k in range(K) if groups[k]]
        hmin = [float(min(c.base[i] for i in g)) if g else float('nan') for g in groups]
        strong = max(live, key=lambda k: hmin[k]); weak = min(live, key=lambda k: hmin[k])
        rest_k = [k for k in live if k not in (strong, weak)]
        slot_of = {k: i for i, k in enumerate(rest_k)}; slot_of[strong] = 2; slot_of[weak] = 3
        u = int(max(groups[strong], key=lambda i: c.base[i])); targets = {'u': u}
        for cond in CONDS:
            _, h = channels(seed, T, 'rayleigh' if cond == 'std' else 'mild')
            shards, radio, track, tags = [], {}, {}, []
            for s2 in snrs:
                for mux in muxes:
                    tag = f'{cond}|{s2}|{mux}'; tags.append((tag, s2, mux))
                    add_system(shards, radio, track, tag, groups, slot_of, targets, radio_for(cond, s2, mux, eps), T)
            name_of = [s.name for s in shards]; members_of = [set(s.members) for s in shards]
            stat = {}
            def hook(j, t, active, r, U):
                nm = name_of[j]
                if '|src|' not in nm:
                    return
                tag = nm.rsplit('|', 2)[0]; s = stat.setdefault(tag, dict(rounds=0, n_sum=0, n_le2=0, hit=0.0, hit_n=0, base=0.0, base_n=0))
                lab = int(r[-10:].argmax()); n = len(active)
                s['rounds'] += 1; s['n_sum'] += n; s['n_le2'] += int(n <= 2)
                s['hit'] += sum(float(lab == dom[i]) for i in active); s['hit_n'] += n
                outs = [i for i in range(N_CLIENTS) if i not in members_of[j]]
                s['base'] += sum(float(lab == dom[i]) for i in outs); s['base_n'] += len(outs)
            log(f'[standard] seed {seed} {cond}: shard 모델 {len(shards)}개')
            W, leds, diags, _ = train(c.tr, shards, h, c.w0, seed, lambda s: radio[s], track_of=lambda s: track.get(s, ()), log=log, log_every=T // 4, hook=hook)
            P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
            for tag, s2, mux in tags:
                r, sr = measure(c, W, P, leds, diags, ix, tag, groups, slot_of, targets, hmin, None, dict(seed=seed, cond=cond, sigma2=s2, mux=mux))
                st = stat.get(tag, {}); total_rounds = T * len(live)
                sr.update(n_mean=st.get('n_sum', 0) / max(1, st.get('rounds', 1)), frac_le2=st.get('n_le2', 0) / max(1, st.get('rounds', 1)),
                          frac_skipped=1 - st.get('rounds', 0) / total_rounds,
                          exposure=st.get('hit', 0) / max(1, st.get('hit_n', 1)), exposure_base=st.get('base', 0) / max(1, st.get('base_n', 1)))
                rows += r; sysrows.append(sr)
            del W, P
        log(f'[standard] seed {seed} 끝')
    write_json(out / 'results.json', dict(rows=rows, systems=sysrows, setting=dict(K=K, T=T, g_th=G_TH, cfo=CFO, delay=DELAY, snrs=snrs, muxes=muxes, seeds=list(cfg['seeds']))))
    L = ['# standard — 기존 AirComp 조건 (Rayleigh, 절단 채널 역전, 반복 없음) 에서 다시 본 SISA 언러닝', '',
         f'ID 해시 배정 {K} shard, {T}라운드, seed {len(cfg["seeds"])}개. 절단 문턱 g_th = {G_TH} (|h|², 약 {round(10 * np.log10(G_TH))}dB). '
         f'FDMA 주파수 오차 {CFO}, 코드 분할 칩 오차 {DELAY}. mild = 지금까지의 조건 (U[0.85,1.15] 페이딩, 전원 송신, eps={fmt(eps)} 반복).', '',
         '열: 흔적 = u 의 shard 가 아닌 shard 의 ||W(u 있음) − W(u 없이 처음부터)|| / ||W(u 없이)|| (평균). 난수 변동 = 같은 조건을 다른 배치·잡음으로. '
         '앙상블 거리 = SISA (u 의 shard 만 재학습) vs 전체 재학습의 test 확률 거리 (TV). n_t = 그 라운드에 실제 보낸 인원. '
         '노출 = 받은 합의 bias 로 그 라운드 송신 member 의 주 label 을 맞힌 비율 (기준 = 비member). 슬롯 = 라운드당 시간 (반복 포함).', '']
    for cond in CONDS:
        L += [f'## {COND_KO[cond]}', '']
        tb = []
        for s2 in snrs:
            for mux in muxes:
                rs = select(rows, cond=cond, sigma2=s2, mux=mux); ss = select(sysrows, cond=cond, sigma2=s2, mux=mux)
                if not ss:
                    continue
                tb.append([db(s2), MUX_KO[mux], pct(mean(ss, 'acc_ens')), pct(mean(ss, 'del_acc_u')),
                           fmt(mean(rs, 'rel'), 4) if rs else '-', fmt(mean(rs, 'rel_yard'), 4) if rs else '-', fmt(mean(ss, 'ens_tv_u'), 4), fmt(mean(ss, 'ens_yard_tv'), 4),
                           fmt(mean(ss, 'n_mean'), 2), pct(mean(ss, 'frac_le2')), pct(mean(ss, 'frac_skipped')), pct(mean(ss, 'exposure')), pct(mean(ss, 'exposure_base')),
                           fmt(mean(ss, 'mse'), 3), fmt(mean(ss, 'energy')), fmt(mean(ss, 'slots'), 2)])
        L += [table(['SNR', '다중화', '앙상블 정확도', 'SISA 삭제 후 정확도', '흔적', '난수 변동', '앙상블 거리 (SISA vs 전체)', '앙상블 거리 난수 기준',
                     'n_t 평균', 'n_t ≤ 2 라운드', '안 보낸 라운드', '노출', '노출 기준', '평균 집계 오차', '에너지/라운드', '슬롯'], tb), '']
    L += ['## 판정 (std − mild, 같은 SNR·다중화, ± 는 2×표준오차)', '']
    for s2 in snrs:
        for mux in muxes:
            m, se, n = paired(sysrows, 'acc_ens', dict(sigma2=s2, mux=mux), dict(cond='std'), dict(cond='mild'))
            m2, se2, n2 = paired(rows, 'rel', dict(sigma2=s2, mux=mux), dict(cond='std'), dict(cond='mild'), keys=('seed', 'shard'))
            m3, se3, _ = paired(sysrows, 'exposure', dict(sigma2=s2, mux=mux), dict(cond='std'), dict(cond='mild'))
            if n:
                L.append(f'- {db(s2)}, {MUX_KO[mux]}: 정확도 {pm(m, se, pp)}, 흔적 {pm(m2, se2, fmt)} (n={n2}), 노출 {pm(m3, se3, pp)}')
    L += ['', '## 읽는 법', '',
          '- 기존 조건에서는 잡음을 반복으로 치우지 않으므로 SNR 이 내려가면 정확도가 떨어져야 한다 (mild 는 슬롯이 늘어나는 대신 정확도를 지킴).',
          '- 절단 때문에 n_t 가 라운드마다 바뀌고, 약한 client 가 자주 빠진다. n_t ≤ 2 인 라운드가 생기면 노출이 오르고, 0 이면 그 라운드는 안 보낸다.',
          '- 흔적은 난수 변동과 함께 읽는다. 기존 조건에서 난수 변동이 커지면 (잡음이 커서) 흔적이 같은 크기여도 상대적으로 묻힌다.',
          '- TDMA 는 어느 조건에서도 흔적이 부동소수점 바닥이어야 한다 (직교). FDMA·코드는 오차 때문에 흔적이 있고, 기존 조건에서 Rayleigh 가 near-far 를 키우면 더 커질 수 있다.',
          '- 이 표가 "SISA 가 기존 AirComp 조건에서 버티는가" 의 답이다. 정확도·흔적·노출 세 열을 SNR 축으로 본다.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
