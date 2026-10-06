"""P0 — 문제 상황이 실제로 있는가 (라운드 단위 AirComp 시뮬레이션).

(a) 무선 제어를 shard 끼리 공유하면 삭제 대상의 흔적이 다른 shard 에 남는가
    무선 제어: shard 별 전력 정렬 / 공통 전력 정렬 / 공통 스케줄링 문턱 / shard 별 스케줄링 문턱.
    측정: 삭제하지 않은 shard 의 모델이 u 있을 때 vs u 없이 처음부터(무선 제어 재계산) 학습했을 때의 예측 불일치.
    기준: 같은 reference 를 다른 잡음·배치 난수로 학습했을 때의 불일치(난수 변동), 그리고 shard 별 정렬(음성 대조).
(b) SISA 를 AirComp 에 얹으면 평상시 자원이 K 배가 되고 삭제 통신량은 줄지 않는가
    K in {1,2,4,5,10}, 실제 장부(심볼·에너지·계산·시간). 디지털 전송 기준선 장부도 같은 채널로 계산.
(c) 서버가 받은 (잡음 섞인) shard 집계에서 개인 정보가 shard 크기에 따라 얼마나 드러나는가
    실제 송수신을 거친 집계와 member update 의 cosine, 집계로 member 의 주 label 추정 적중률. SNR 별.
"""
import math
import numpy as np
import torch
from .fl import (ShardJob, run_rounds, load_split, channels, partition, shard_of, init_vector, key, N_CLIENTS,
                 LocalTrainer, Evaluator, disagreement, write_json)
from .radio import RadioConfig, Radio, Ledger

K = 4
CTRL = {'shard_align': dict(align='shard'), 'shared_align': dict(align='shared'),
        'shard_sched': dict(align='shard', sched=('shard', .2)), 'shared_sched': dict(align='shard', sched=('shared', .2))}
CTRL_KO = {'shard_align': 'shard별 전력 정렬', 'shared_align': '공통 전력 정렬', 'shard_sched': 'shard별 스케줄링 문턱', 'shared_sched': '공통 스케줄링 문턱'}

def snr_db(s2):
    return round(10 * math.log10(1 / s2))

# ---------------------------------------------------------------- (a)
def part_a(cfg, log):
    T = cfg['T']; s2s = [1.0] if cfg['quick'] else [1.0, .01]; rows = []
    ctrls = ['shard_align', 'shared_align', 'shared_sched'] if cfg['quick'] else list(CTRL)
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device']); base, h = channels(seed, T)
        groups = partition('random_split', range(N_CLIENTS), base, K, seed)
        order = np.argsort(base); targets = {'weakest': int(order[0]), 'median': int(order[N_CLIENTS // 2])}
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device']); ev = Evaluator(tr, data)
        jobs = []
        for cn in ctrls:
            for s2 in s2s:
                rc = RadioConfig(sigma2=s2, eps=cfg['eps'], mode='maxpow', **CTRL[cn])
                for c, g in enumerate(groups):
                    jobs.append(ShardJob(f'{cn}|{s2}|src|{c}', list(g), w0, T, system=f'{cn}|{s2}|src', tape=c, radio=rc))
                for tn, u in targets.items():
                    cu = shard_of(groups, u)
                    for c, g in enumerate(groups):
                        rest = [i for i in g if i != u]
                        jobs.append(ShardJob(f'{cn}|{s2}|ref|{tn}|{c}', rest, w0, T, system=f'{cn}|{s2}|ref|{tn}', tape=c, radio=rc))
                        jobs.append(ShardJob(f'{cn}|{s2}|alt|{tn}|{c}', rest, w0, T, system=f'{cn}|{s2}|alt|{tn}', tape=c, radio=rc, salt='alt'))
                    jobs.append(ShardJob(f'{cn}|{s2}|rep|{tn}', [i for i in groups[cu] if i != u], w0, T, system=f'{cn}|{s2}|rep|{tn}', tape=cu, radio=rc))
        log(f'[P0-a] seed {seed}: shard jobs={len(jobs)}')
        W, _, leds, _ = run_rounds(tr, jobs, h, seed, RadioConfig(), log, max(1, T // 4))
        P = ev.probs(W); ix = {j.name: k for k, j in enumerate(jobs)}
        def pr(n): return {k: v[ix[n]] for k, v in P.items()}
        for cn in ctrls:
            for s2 in s2s:
                src = {c: pr(f'{cn}|{s2}|src|{c}') for c in range(K)}
                for tn, u in targets.items():
                    cu = shard_of(groups, u); un = [c for c in range(K) if c != cu]
                    ref = {c: pr(f'{cn}|{s2}|ref|{tn}|{c}') for c in range(K)}; alt = {c: pr(f'{cn}|{s2}|alt|{tn}|{c}') for c in range(K)}
                    rep = pr(f'{cn}|{s2}|rep|{tn}')
                    dis = float(np.mean([disagreement(src[c]['test'], ref[c]['test']) for c in un]))
                    yard = float(np.mean([disagreement(ref[c]['test'], alt[c]['test']) for c in un]))
                    rel = float(np.mean([float((W[ix[f'{cn}|{s2}|src|{c}']] - W[ix[f'{cn}|{s2}|ref|{tn}|{c}']]).norm() / W[ix[f'{cn}|{s2}|ref|{tn}|{c}']].norm()) for c in un]))
                    mse_src = float(np.mean([leds[ix[f'{cn}|{s2}|src|{c}']].asdict()['mean_mse'] for c in un]))
                    mse_ref = float(np.mean([leds[ix[f'{cn}|{s2}|ref|{tn}|{c}']].asdict()['mean_mse'] for c in un]))
                    tx_src = float(np.mean([leds[ix[f'{cn}|{s2}|src|{c}']].tx_count for c in un])); tx_ref = float(np.mean([leds[ix[f'{cn}|{s2}|ref|{tn}|{c}']].tx_count for c in un]))
                    sisa = ev.ensemble([rep if c == cu else src[c] for c in range(K)]); refe = ev.ensemble(list(ref.values()))
                    rows.append(dict(seed=seed, ctrl=cn, sigma2=s2, target=tn, client=u, unaff_dis=dis, unaff_rel=rel, yard_dis=yard,
                                     unaff_mse_ratio=mse_src / mse_ref if mse_ref > 0 else float('nan'), unaff_tx_ratio=tx_src / tx_ref,
                                     ens_dis_sisa_ref=disagreement(sisa['_ptest'], refe['_ptest']), acc_sisa=sisa['test'], acc_ref=refe['test']))
        log(f'[P0-a] seed {seed} done')
    return rows

# ---------------------------------------------------------------- (b)
def part_b(cfg, log):
    T = cfg['T']; Ks = [1, 4] if cfg['quick'] else [1, 2, 4, 5, 10]; modes = ['maxpow', 'minen']; s2 = .01; rows = []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device']); base, h = channels(seed, T)
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device']); ev = Evaluator(tr, data)
        jobs = []; plan = {}
        for Kv in Ks:
            groups = partition('random_split', range(N_CLIENTS), base, Kv, seed) if Kv > 1 else [list(range(N_CLIENTS))]
            dels = sorted({g[0] for g in groups} | {groups[0][-1]})[:4]
            for m in modes:
                rc = RadioConfig(sigma2=s2, eps=cfg['eps'], mode=m)
                for c, g in enumerate(groups):
                    jobs.append(ShardJob(f'{Kv}|{m}|src|{c}', list(g), w0, T, system=f'{Kv}|{m}', tape=c, radio=rc))
                for u in dels:
                    cu = shard_of(groups, u); rest = [i for i in groups[cu] if i != u]
                    jobs.append(ShardJob(f'{Kv}|{m}|del|{u}', rest, w0, T, system=f'{Kv}|{m}|del{u}', tape=cu, radio=rc))
                    rc_tot = RadioConfig(sigma2=s2, eps=cfg['eps'], mode=m, power='total')
                    jobs.append(ShardJob(f'{Kv}|{m}|delidle|{u}', rest, w0, T, system=f'{Kv}|{m}|delidle{u}', tape=cu, radio=rc_tot, bw=float(Kv)))
            plan[Kv] = (groups, dels)
        log(f'[P0-b] seed {seed}: shard jobs={len(jobs)}')
        W, _, leds, _ = run_rounds(tr, jobs, h, seed, RadioConfig(), log, max(1, T // 4))
        P = ev.probs(W); ix = {j.name: k for k, j in enumerate(jobs)}
        rd = Radio(RadioConfig(sigma2=s2), tr.D, seed, cfg['device'])
        for Kv in Ks:
            groups, dels = plan[Kv]
            # 디지털 기준선 장부 (같은 채널·같은 라운드 수)
            dig_T = Ledger(); dig_U = Ledger()
            for t in range(T):
                for g in groups:
                    dig_T.add(rd.digital_ledger(h[t, g]))
                for u in dels:
                    cu = shard_of(groups, u); dig_U.add(rd.digital_ledger(h[t, [i for i in groups[cu] if i != u]]))
            for m in modes:
                Tl = Ledger()
                for c in range(Kv):
                    Tl.add(leds[ix[f'{Kv}|{m}|src|{c}']])
                Ul = Ledger(); Ui = Ledger()
                for u in dels:
                    Ul.add(leds[ix[f'{Kv}|{m}|del|{u}']]); Ui.add(leds[ix[f'{Kv}|{m}|delidle|{u}']])
                nd = len(dels)
                src = [{k: v[ix[f'{Kv}|{m}|src|{c}']] for k, v in P.items()} for c in range(Kv)]
                acc = ev.ensemble(src)['test']
                dacc = []
                for u in dels:
                    cu = shard_of(groups, u)
                    pl = [{k: v[ix[f'{Kv}|{m}|del|{u}']] for k, v in P.items()} if c == cu else src[c] for c in range(Kv)]
                    dacc.append(ev.ensemble(pl)['test'])
                rows.append(dict(seed=seed, K=Kv, mode=m, acc=acc, del_acc=float(np.mean(dacc)),
                                 T=Tl.asdict(), U={k: v / nd for k, v in Ul.asdict().items()}, U_idle={k: v / nd for k, v in Ui.asdict().items()},
                                 digital_T=dig_T.asdict(), digital_U={k: v / nd for k, v in dig_U.asdict().items()}))
        log(f'[P0-b] seed {seed} done')
    return rows

# ---------------------------------------------------------------- (c)
def part_c(cfg, log):
    T = cfg['T']; ns = [1, 2, 4, 5, 10, 20]; s2s = [None, 1.0, .01]; out = []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device']); base, h = channels(seed, T)
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device'])
        job = ShardJob('k1', list(range(N_CLIENTS)), w0, T // 2, save_at=(T // 2,))
        _, ck, _, _ = run_rounds(tr, [job], h, seed, RadioConfig(sigma2=.01, eps=cfg['eps']))
        dom = data['hist'].argmax(1)
        for stage, w in [('init', w0), ('mid', ck[0][T // 2])]:
            U, _ = tr.updates(w[None].repeat(N_CLIENTS, 1), list(range(N_CLIENTS)), T // 2)
            Un = U / U.norm(dim=1, keepdim=True)
            rng = np.random.default_rng(key(seed, 'expo', stage))
            for s2 in s2s:
                rd = Radio(RadioConfig(sigma2=s2 if s2 else 1e-12, eps=cfg['eps']), tr.D, seed, cfg['device'])
                for n in ns:
                    cm, cn, lm, ln = [], [], [], []
                    for rep in range(200 if not cfg['quick'] else 40):
                        gset = list(rng.choice(N_CLIENTS, n, replace=False)); rest = np.setdiff1d(np.arange(N_CLIENTS), gset)
                        hs = h[T // 2, gset]
                        r, _ = rd.transmit(U[gset], hs, float(hs.min()), noise_key=key(seed, stage, s2 or 0, n, rep))
                        rn = r / r.norm()
                        cm += [float(rn @ Un[i]) for i in gset]
                        if len(rest): cn += [float(rn @ Un[j]) for j in rest]
                        lab = int(r[-10:].argmax())
                        lm += [lab == dom[i] for i in gset]
                        if len(rest): ln += [lab == dom[j] for j in rest]
                    out.append(dict(seed=seed, stage=stage, sigma2=s2, n=n, cos_member=float(np.mean(cm)), cos_nonmember=float(np.mean(cn)) if cn else float('nan'),
                                    label_member=float(np.mean(lm)), label_nonmember=float(np.mean(ln)) if ln else float('nan')))
        log(f'[P0-c] seed {seed} done')
    return out

# ---------------------------------------------------------------- 보고서
def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)): return '-'
    if isinstance(v, (int, np.integer)): return f'{int(v):,}'
    if abs(v) >= 1e6 or (v != 0 and abs(v) < 1e-3): return f'{v:.3e}'
    return f'{v:,.{nd}f}'
def pct(v): return '-' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{100 * v:.2f}%'
def pp(v): return '-' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{100 * v:+.2f}pp'
def table(hd, rows):
    return '\n'.join(['| ' + ' | '.join(hd) + ' |', '|' + '|'.join(['---'] * len(hd)) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])

def run(cfg, log):
    out = cfg['out'] / 'p0_problem'; out.mkdir(parents=True, exist_ok=True)
    A = []   # (a) 는 코드 분리 실험 P0-A(exp_p0a_cdma.py)로 대체
    B = part_b(cfg, log); Cc = part_c(cfg, log)
    write_json(out / 'results.json', dict(a=A, b=B, c=Cc, eps=cfg['eps']))
    L = ['# P0 — 문제 상황 확인 (라운드 단위 AirComp 시뮬레이션)', '',
         f'N=20, K={K}, {cfg["T"]}라운드, seed {len(cfg["seeds"])}개, eps={cfg["eps"]}. 매 라운드 송신 전력 계산·심볼 송신·채널 가산·수신·장부가 실제로 실행된다.', '',
         '(a) 간섭과 흔적은 P0-A 보고서(p0a_cdma/REPORT_KO.md)를 본다.', '']
    L += ['', '## (b) shard 수 K 에 따른 실제 장부 (20dB, 삭제 대상 평균)', '']
    tb = []
    for m in ['maxpow', 'minen']:
        for Kv in sorted(set(r['K'] for r in B)):
            sel = [r for r in B if r['K'] == Kv and r['mode'] == m]
            g = lambda path: float(np.mean([eval('r' + path) for r in sel]))
            tb.append(['최대 전력' if m == 'maxpow' else '전력 낮춤', Kv, fmt(g("['T']['ul_symbols']")), fmt(g("['T']['energy']")), fmt(g("['U']['ul_symbols']")),
                       fmt(g("['U']['energy']")), fmt(g("['U']['compute_calls']"), 0), fmt(g("['U']['ul_time']")), fmt(g("['U_idle']['ul_time']")),
                       fmt(g("['digital_T']['ul_symbols']")), fmt(g("['digital_U']['ul_symbols']")), pct(g("['acc']")), pct(g("['del_acc']"))])
    L.append(table(['전송 규칙', 'K', '학습 UL 심볼', '학습 에너지', '삭제 UL 심볼', '삭제 에너지', '삭제 계산', '삭제 UL 시간', '삭제 UL 시간(빈 블록 재사용)',
                    '디지털 학습 UL', '디지털 삭제 UL', '정확도', '삭제 후 정확도'], tb))
    L += ['', '디지털 기준선: client 마다 직교 자원으로 32bit x D 를 Shannon 용량 log2(1+P|h|²/σ²) 로 보낼 때의 real 사용량.', '',
          '## (c) 서버가 받은 shard 집계에서의 개인 노출', '']
    tb = []
    for stage in ['init', 'mid']:
        for s2 in [None, 1.0, .01]:
            for n in [1, 2, 4, 5, 10, 20]:
                sel = [r for r in Cc if r['stage'] == stage and r['sigma2'] == s2 and r['n'] == n]
                if not sel: continue
                m = lambda f: float(np.nanmean([r[f] for r in sel]))
                tb.append(['초기' if stage == 'init' else '중간', '무잡음' if s2 is None else f'{snr_db(s2)}dB', n, fmt(m('cos_member')), fmt(m('cos_nonmember')), pct(m('label_member')), pct(m('label_nonmember'))])
    L.append(table(['모델', 'SNR', 'n', 'cos(수신 집계, member)', 'cos(수신 집계, 비member)', 'member 주 label 적중', '비member 주 label 적중'], tb))
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(name='P0')
