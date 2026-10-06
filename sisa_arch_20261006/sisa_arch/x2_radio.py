"""X2 — AirComp-SISA 에서 데이터 격리만으로 충분한가, 그리고 shard 크기가 노출에 주는 영향.

(A) 무선 제어 수준 격리
    AirComp 에서는 client 가 데이터뿐 아니라 채널로도 다른 client 의 학습에 영향을 준다(전력 정렬 scale, 수신기).
    무선 제어 세 가지에서, 삭제 대상 u 의 shard 만 재학습한 결과(SISA)와 u 없이 무선 제어까지 다시 계산해
    처음부터 학습한 reference 를 비교한다.
      local  : shard 마다 자기 최약 채널로 전력 정렬 (기준, 음성 대조)
      common : 모든 shard 가 전체 최약 채널 기준 공통 scale 을 씀
      zf     : 모든 shard 가 같은 자원에 동시에 보내고 MIMO ZF 로 분리 (수신 안테나 20개, 정적 채널)
    삭제 대상: 전체에서 채널이 가장 약한 client, 중간 client.
    핵심 지표: 삭제하지 않은 shard 가 u 의 유무에 따라 달라지는 정도(예측 불일치) vs 학습 난수 변동(다른 tape).
(B) shard 크기와 노출
    같은 모델에서 20명의 update 를 구하고, 크기 n 의 무작위 집합 평균(서버가 보는 shard 집계)에서
    개별 member 의 정보가 얼마나 드러나는지 잰다: update 의 cosine 유사도, 마지막 층 bias 로 member 의 주 label 추정.
"""
import math
import numpy as np
import torch
from .common import load_split, channels, partition, shard_of, init_vector, write_json, key, N_CLIENTS, CLIP, P_MAX, D_PARAMS
from .trainer import LocalTrainer, Job, run_jobs, Evaluator, split_probs, disagreement, kl
from .report import table, fmt, pct, pp, verdict_line

K = 4; M_ANT = 20
CONF_KO = {'local': 'shard별 전력 정렬', 'common': '공통 scale', 'zf': '같은 자원 MIMO ZF'}

def inv_var(sigma2, hs):
    n = len(hs)
    return CLIP ** 2 * sigma2 / (n * n * D_PARAMS * P_MAX * float(np.min(hs)) ** 2)

def zf_vars(Hc, systems_groups, sigma2):
    """Hc: [M, N] 복소 채널. 동시에 보내는 shard 들의 member 목록 -> shard 별 좌표당 잡음 분산."""
    out = []
    for g, mem in enumerate(systems_groups):
        others = [i for k, m2 in enumerate(systems_groups) if k != g for i in m2]
        if others:
            Q, _ = np.linalg.qr(Hc[:, others]); P = np.eye(Hc.shape[0]) - Q @ Q.conj().T
        else:
            P = np.eye(Hc.shape[0])
        Hg = P @ Hc[:, mem]
        cands = [Hg[:, k] / np.linalg.norm(Hg[:, k]) for k in range(len(mem))]
        ev = np.linalg.eigh(Hg @ Hg.conj().T)[1][:, -1]; cands.append(ev)
        best = max(cands, key=lambda w: np.min(np.abs(w.conj() @ Hc[:, mem])))
        gmin = float(np.min(np.abs(best.conj() @ Hc[:, mem])))
        eta = 1.0 / (len(mem) * gmin)
        out.append(CLIP ** 2 * sigma2 * eta ** 2 / (D_PARAMS * P_MAX))
    return out

def make_system(conf, groups_active, h, Hc, sigma2):
    """반환: shard 별 var_fn(t, act). groups_active: 이 시스템에서 동시에 학습하는 shard 의 member 목록(dict c->list)."""
    keys = sorted(groups_active)
    if conf == 'local':
        return {c: (lambda t, act: inv_var(sigma2, h[t, act])) for c in keys}
    if conf == 'common':
        def f(t, act, keys=keys):
            return max(inv_var(sigma2, h[t, groups_active[c]]) for c in keys)
        return {c: f for c in keys}
    if conf == 'zf':
        v = zf_vars(Hc, [groups_active[c] for c in keys], sigma2)
        return {c: (lambda t, act, vv=vv: vv) for c, vv in zip(keys, v)}
    raise ValueError(conf)

def part_a(cfg, log):
    T = cfg['T']; s2s = [1.0] if cfg['quick'] else [1.0, .01]; confs = ['local', 'common', 'zf']
    rows, noise = [], []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device']); base, h = channels(seed, T)
        rng = np.random.default_rng(key(seed, 'mimo'))
        Hc = (rng.normal(size=(M_ANT, N_CLIENTS)) + 1j * rng.normal(size=(M_ANT, N_CLIENTS))) / np.sqrt(2) * base[None, :]
        groups = partition('random_split', range(N_CLIENTS), base, K, seed)
        order = np.argsort(base); targets = {'weakest': int(order[0]), 'median': int(order[N_CLIENTS // 2])}
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device']); ev = Evaluator(tr, data)
        jobs = []; vinfo = {}
        for conf in confs:
            for s2 in s2s:
                full = {c: list(g) for c, g in enumerate(groups)}
                vf = make_system(conf, full, h, Hc, s2)
                for c, g in full.items():
                    jobs.append(Job(f'{conf}|{s2}|src|{c}', c, w0, {i: 0 for i in g}, 0, T, {'mode': 'none', 'var_fn': vf[c]}))
                for tn, u in targets.items():
                    cu = shard_of(groups, u)
                    minus = {c: [i for i in g if i != u] for c, g in full.items()}
                    vr = make_system(conf, minus, h, Hc, s2)
                    for c, g in minus.items():
                        jobs.append(Job(f'{conf}|{s2}|ref|{tn}|{c}', c, w0, {i: 0 for i in g}, 0, T, {'mode': 'none', 'var_fn': vr[c]}))
                        jobs.append(Job(f'{conf}|{s2}|alt|{tn}|{c}', c, w0, {i: 0 for i in g}, 0, T, {'mode': 'none', 'var_fn': vr[c]}, salt='alt'))
                    vp = make_system(conf, {cu: minus[cu]}, h, Hc, s2)
                    jobs.append(Job(f'{conf}|{s2}|rep|{tn}', cu, w0, {i: 0 for i in minus[cu]}, 0, T, {'mode': 'none', 'var_fn': vp[cu]}))
                    # 삭제하지 않은 shard 의 잡음 분산이 u 유무로 얼마나 바뀌었는가 (t=0 기준)
                    ratios = [vf[c](0, full[c]) / vr[c](0, minus[c]) for c in full if c != cu]
                    vinfo[(conf, s2, tn)] = float(np.mean(ratios))
        log(f'[X2-A] seed {seed}: jobs={len(jobs)}')
        W, _, _ = run_jobs(tr, jobs, h, seed, log_every=max(1, T // 4), logger=log)
        P = ev.probs(W); ix = {j.name: k for k, j in enumerate(jobs)}
        for conf in confs:
            for s2 in s2s:
                src = {c: split_probs(P, ix[f'{conf}|{s2}|src|{c}']) for c in range(K)}
                for tn, u in targets.items():
                    cu = shard_of(groups, u)
                    ref = {c: split_probs(P, ix[f'{conf}|{s2}|ref|{tn}|{c}']) for c in range(K)}
                    alt = {c: split_probs(P, ix[f'{conf}|{s2}|alt|{tn}|{c}']) for c in range(K)}
                    rep = split_probs(P, ix[f'{conf}|{s2}|rep|{tn}'])
                    un = [c for c in range(K) if c != cu]
                    dis_u = float(np.mean([disagreement(src[c]['test'], ref[c]['test']) for c in un]))
                    rel_u = float(np.mean([float((W[ix[f'{conf}|{s2}|src|{c}']] - W[ix[f'{conf}|{s2}|ref|{tn}|{c}']]).norm() /
                                                 W[ix[f'{conf}|{s2}|ref|{tn}|{c}']].norm()) for c in un]))
                    yard = float(np.mean([disagreement(ref[c]['test'], alt[c]['test']) for c in un]))
                    acc_u_src = float(np.mean([float((src[c]['test'].argmax(1) == data['ty']).float().mean()) for c in un]))
                    acc_u_ref = float(np.mean([float((ref[c]['test'].argmax(1) == data['ty']).float().mean()) for c in un]))
                    sisa = ev.ensemble([rep if c == cu else src[c] for c in range(K)]); refe = ev.ensemble(list(ref.values()))
                    alte = ev.ensemble(list(alt.values()))
                    rows.append(dict(seed=seed, conf=conf, sigma2=s2, target=tn, client=u, unaff_dis=dis_u, unaff_rel=rel_u, yard_dis=yard,
                                     unaff_acc_src=acc_u_src, unaff_acc_ref=acc_u_ref, noise_ratio=vinfo[(conf, s2, tn)],
                                     ens_dis_sisa_ref=disagreement(sisa['_ptest'], refe['_ptest']),
                                     ens_dis_ref_alt=disagreement(refe['_ptest'], alte['_ptest']),
                                     acc_sisa=sisa['test'], acc_ref=refe['test']))
        log(f'[X2-A] seed {seed} done')
    return rows

def part_b(cfg, log):
    """shard 크기 n 과 개별 노출."""
    T = cfg['T']; ns = [1, 2, 4, 5, 10, 20]; out = []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device']); base, h = channels(seed, T)
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device'])
        job = Job('k1', 0, w0, {i: 0 for i in range(N_CLIENTS)}, 0, T // 2, {'mode': 'none'}, save_at=(T // 2,))
        W, ck, _ = run_jobs(tr, [job], h, seed)
        dom = data['hist'].argmax(1)
        for stage, w in [('init', w0), ('mid', ck[0][T // 2])]:
            U, _ = tr.updates(w[None].repeat(N_CLIENTS, 1), list(range(N_CLIENTS)), T // 2)
            U = U.double(); Un = U / U.norm(dim=1, keepdim=True)
            bias = U[:, -10:]
            rng = np.random.default_rng(key(seed, 'expo', stage))
            for n in ns:
                cm, cn, lm, ln = [], [], [], []
                for _ in range(300):
                    gset = rng.choice(N_CLIENTS, n, replace=False); rest = np.setdiff1d(np.arange(N_CLIENTS), gset)
                    m = U[gset].mean(0); mn = m / m.norm()
                    cm += [float(mn @ Un[i]) for i in gset]
                    if len(rest):
                        cn += [float(mn @ Un[j]) for j in rest]
                    lab = int(bias[gset].mean(0).argmax())
                    lm += [lab == dom[i] for i in gset]
                    if len(rest):
                        ln += [lab == dom[j] for j in rest]
                out.append(dict(seed=seed, stage=stage, n=n, cos_member=float(np.mean(cm)), cos_nonmember=float(np.mean(cn)) if cn else float('nan'),
                                label_member=float(np.mean(lm)), label_nonmember=float(np.mean(ln)) if ln else float('nan')))
        log(f'[X2-B] seed {seed} done')
    return out

def run(cfg, log):
    out = cfg['out'] / 'x2_radio'; out.mkdir(parents=True, exist_ok=True)
    rows = part_a(cfg, log)
    expo = part_b(cfg, log)
    write_json(out / 'results.json', dict(part_a=rows, part_b=expo))

    def agg(conf, s2, tn, f):
        return float(np.mean([r[f] for r in rows if r['conf'] == conf and r['sigma2'] == s2 and r['target'] == tn]))
    s2s = sorted(set(r['sigma2'] for r in rows), reverse=True)
    floor = max(agg('local', s2, tn, 'unaff_dis') for s2 in s2s for tn in ['weakest', 'median'])
    L = ['# X2 — 무선 제어 수준 격리와 shard 크기에 따른 노출', '',
         f'N=20, K={K}, {cfg["T"]}라운드, seed {len(cfg["seeds"])}개. MIMO ZF 는 수신 안테나 {M_ANT}개, 정적 Rayleigh x 장기 진폭.', '',
         '## (A) 삭제 대상의 채널 영향이 다른 shard 에 남는가', '',
         '"미삭제 shard 불일치"는 삭제하지 않은 shard 의 모델이, u 가 있을 때(실제로 쓰는 모델)와 u 없이 무선 제어까지 다시 계산했을 때(reference) '
         '예측이 다른 test 표본 비율이다. SISA 는 이 shard 들을 다시 학습하지 않으므로, 이 값이 곧 SISA 가 남기는 흔적이다. '
         '"난수 변동"은 같은 reference 를 다른 난수로 학습했을 때의 불일치다.', '']
    tb = []
    for conf in ['local', 'common', 'zf']:
        for s2 in s2s:
            for tn in ['weakest', 'median']:
                tb.append([CONF_KO[conf], f'{round(10 * math.log10(1 / s2))}dB', '최약 채널' if tn == 'weakest' else '중간 채널',
                           fmt(agg(conf, s2, tn, 'noise_ratio'), 3), pct(agg(conf, s2, tn, 'unaff_dis')).lstrip('+'),
                           fmt(agg(conf, s2, tn, 'unaff_rel'), 4), pct(agg(conf, s2, tn, 'yard_dis')).lstrip('+'),
                           pp(agg(conf, s2, tn, 'unaff_acc_src') - agg(conf, s2, tn, 'unaff_acc_ref')),
                           pct(agg(conf, s2, tn, 'ens_dis_sisa_ref')).lstrip('+'), pp(agg(conf, s2, tn, 'acc_sisa') - agg(conf, s2, tn, 'acc_ref'))])
    L.append(table(['무선 제어', 'SNR', '삭제 대상', '미삭제 shard 잡음 비(u 있음/없음)', '미삭제 shard 불일치', '파라미터 상대 차이',
                    '난수 변동', '미삭제 shard 정확도 차이', '앙상블 불일치(SISA-ref)', '앙상블 정확도 차이'], tb))
    verd = []
    for conf in ['common', 'zf']:
        for s2 in s2s:
            v = agg(conf, s2, 'weakest', 'unaff_dis'); y = agg(conf, s2, 'weakest', 'yard_dis')
            verd.append(f'- {CONF_KO[conf]}, {round(10 * math.log10(1 / s2))}dB, 최약 채널 삭제: 미삭제 shard 불일치 {pct(v).lstrip("+")} '
                        f'(음성 대조 바닥 {pct(floor).lstrip("+")}, 난수 변동 {pct(y).lstrip("+")})')
    L += ['', '판정 기준: shard별 전력 정렬(음성 대조)의 불일치가 측정 바닥이다. 공통 scale·ZF 의 불일치가 바닥보다 크면 채널을 통한 영향이 존재하고, '
          '난수 변동과 견줄 만하면 실질적이다.', ''] + verd
    L += ['', '## (B) shard 크기 n 과 개별 노출', '',
          '서버가 보는 집계(n명 평균 update)와 개별 member update 의 cosine, 그리고 집계의 마지막 층 bias 로 member 의 주 label 을 맞히는 비율. '
          '비교: 집계에 속하지 않은 client(같은 값이면 개인이 아니라 모집단 정보만 드러난 것).', '']
    tb = []
    for stage in ['init', 'mid']:
        for n in sorted(set(e['n'] for e in expo)):
            ee = [e for e in expo if e['stage'] == stage and e['n'] == n]
            tb.append(['초기 모델' if stage == 'init' else '중간 모델', n, fmt(np.mean([e['cos_member'] for e in ee]), 3),
                       fmt(np.nanmean([e['cos_nonmember'] for e in ee]), 3), pct(np.mean([e['label_member'] for e in ee])).lstrip('+'),
                       pct(np.nanmean([e['label_nonmember'] for e in ee])).lstrip('+')])
    L.append(table(['모델', 'n', 'cos(집계, member)', 'cos(집계, 비member)', 'member 주 label 적중', '비member 주 label 적중'], tb))
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(name='X2', verdicts=[])
