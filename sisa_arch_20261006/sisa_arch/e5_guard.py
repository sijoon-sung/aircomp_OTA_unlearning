"""E5 — ⑤ guard 대역과 주파수 배치: shard 사이 누설이 삭제 후 남는 영향을 만드는가.

물리 모형: SISO OFDM 64 부반송파. shard 마다 부반송파 9개를 쓰고, 한 OFDM 심볼에 update 좌표 9개를 싣는다.
client 별 잔여 CFO(부반송파 간격으로 정규화, e x U[-1,1], 고정)가 부반송파 간 간섭(ICI)을 만든다.
블록 중심 공통 위상은 보상했다고 가정한다. 채널은 주파수 평탄, 진폭·위상은 송신 전에 역보상한다.
client i(shard j) 의 신호가 shard g 수신기에 들어오는 계수는 (h_min,j / (n_g h_min,g)) Re(L_i[g 부반송파, j 부반송파]) 이다.

바꾸는 것: CFO 범위 e {0, 0.01, 0.05}, 배치 {붙임 guard 0, 붙임 guard 2, 붙임 guard 8, 교차 배치}.
음성 대조: e=0, 그리고 shard 사이 항만 0 으로 만든 oracle.
삭제 대상: 백도어 client 0 (자기 표본의 80% 에 3x3 트리거, label 0).
측정: (a) 교차 누설 에너지 비, (b) 삭제하지 않은 shard 가 삭제 대상 유무에 따라 달라지는 정도(예측 불일치, 파라미터 상대 차이),
      (c) SISA 삭제 후 앙상블 트리거 성공률(ASR) - reference ASR, (d) guard 에 쓴 부반송파 비율.
기준: 같은 reference 를 다른 난수 tape 로 학습했을 때의 차이(학습 난수 변동).
"""
import math
import numpy as np
import torch
from .common import load_split, channels, partition, shard_of, init_vector, write_json, key, N_CLIENTS
from .trainer import LocalTrainer, Evaluator, noise_vec, disagreement, class_metrics
from . import costmodel
from .report import table, fmt, pct, pp, verdict_line

K = 4; NFFT = 64; WIDTH = 9; U_DEL = 0

def ici(eps):
    n = np.arange(NFFT)
    ph = np.exp(1j * 2 * np.pi * eps * n / NFFT) * np.exp(-1j * np.pi * eps * (NFFT - 1) / NFFT)
    F = np.exp(-2j * np.pi * np.outer(n, n) / NFFT) / np.sqrt(NFFT)
    return F @ np.diag(ph) @ F.conj().T

def layout(name):
    if name.startswith('contig_g'):
        g = int(name[len('contig_g'):])
        return [[4 + s * (WIDTH + g) + k for k in range(WIDTH)] for s in range(K)]
    if name == 'interleaved':
        return [[4 + s + K * k for k in range(WIDTH)] for s in range(K)]
    raise ValueError(name)

LAYOUT_KO = {'contig_g0': '붙임 guard 0', 'contig_g2': '붙임 guard 2', 'contig_g8': '붙임 guard 8', 'interleaved': '교차 배치'}

def configs(quick):
    cf = [('e0', 0.0, 'contig_g0', False)]
    es = [0.05] if quick else [0.01, 0.05]
    lays = ['contig_g0', 'contig_g8'] if quick else ['contig_g0', 'contig_g2', 'contig_g8', 'interleaved']
    for e in es:
        for lay in lays:
            cf.append((f'e{e:g}_{lay}', e, lay, False))
    cf.append(('oracle', 0.05, 'contig_g0', True))
    return cf

def reb(seed, groups, e, lay):
    """[K(수신 shard), N(client), 9, 9] 실수 블록. client 의 송신 부반송파는 자기 shard 의 것."""
    tones = layout(lay); out = np.zeros((K, N_CLIENTS, WIDTH, WIDTH))
    for i in range(N_CLIENTS):
        j = shard_of(groups, i)
        if e == 0:   # 정확히 단위 행렬 / 0 (부동소수 잔여 제거)
            out[j, i] = np.eye(WIDTH); continue
        cfo = e * np.random.default_rng(key(seed, 'cfo', i)).uniform(-1, 1)
        Li = ici(cfo)
        for g in range(K):
            out[g, i] = np.real(Li[np.ix_(tones[g], tones[j])])
    return out

def leak_ratio(R, groups):
    own = cross = 0.0
    for i in range(N_CLIENTS):
        j = shard_of(groups, i)
        own += (R[j, i] ** 2).sum(); cross += sum((R[g, i] ** 2).sum() for g in range(K) if g != j)
    return cross / own

def run_systems(tr, systems, h, seed, T, D, log):
    """systems: list of dict(members={g: [clients]}, R=tensor[K,N,9,9], cross, salt). 반환 W [S, K, D]."""
    dev = tr.device; S = math.ceil(D / WIDTH); pad = S * WIDTH - D
    w0 = systems[0]['w0']
    W = w0.to(dev)[None, None, :].repeat(len(systems), K, 1).clone()
    for t in range(T):
        pairs = [(si, g, i) for si, s in enumerate(systems) for g, mem in s['members'].items() for i in mem]
        Wst = torch.stack([W[si, g] for si, g, _ in pairs])
        Uall, _ = tr.updates(Wst, [p[2] for p in pairs], t, [systems[p[0]]['salt'] for p in pairs])
        pos = {}
        for k, (si, g, i) in enumerate(pairs):
            pos.setdefault(si, []).append((k, g, i))
        for si, s in enumerate(systems):
            lst = pos.get(si, [])
            if not lst:
                continue
            act = {g: mem for g, mem in s['members'].items() if mem}
            hmin = {g: float(h[t, mem].min()) for g, mem in act.items()}
            nn = {g: len(mem) for g, mem in act.items()}
            ks = [k for k, _, _ in lst]; cl = [i for _, _, i in lst]; src_sh = [g for _, g, _ in lst]
            U = Uall[ks]
            U = torch.nn.functional.pad(U, (0, pad)).view(len(ks), S, WIDTH)
            gs = sorted(act)
            Rb = s['R'][:, cl]                                       # [K, m, 9, 9]
            for gr in gs:
                if s['cross']:
                    bs = list(range(len(ks)))
                else:      # 교차 항이 정확히 0 인 조건은 자기 shard 항만 더한다(음성 대조의 산술을 동일하게)
                    bs = [b for b in range(len(ks)) if src_sh[b] == gr]
                if not bs:
                    continue
                cvec = torch.tensor([1.0 / nn[gr] if src_sh[b] == gr else hmin[src_sh[b]] / (nn[gr] * hmin[gr]) for b in bs], device=dev)
                M = cvec[:, None, None] * Rb[gr, bs]
                agg = torch.einsum('akl,asl->sk', M, U[bs]).reshape(S * WIDTH)[:D]
                var, _ = costmodel.noise_var(nn[gr], h[t, act[gr]], 'R1', sigma2=.01)
                W[si, gr] += agg + noise_vec(seed, gr, t, s['salt'], D, dev) * math.sqrt(var)
        if (t + 1) % max(1, T // 4) == 0:
            log(f'  round {t + 1}/{T}  pairs={len(pairs)}')
    return W

def run(cfg, log):
    out = cfg['out'] / 'e5_guard'; out.mkdir(parents=True, exist_ok=True)
    T = cfg['T']; rows = []; yard = []; info = {}
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device'], backdoor_client=U_DEL)
        base, h = channels(seed, T)
        groups = partition('random_split', range(N_CLIENTS), base, K, seed); cu = shard_of(groups, U_DEL)
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device']); D = tr.D
        ev = Evaluator(tr, data, with_trigger=True)
        full = {g: list(groups[g]) for g in range(K)}
        minus = {g: [i for i in groups[g] if i != U_DEL] for g in range(K)}
        rep = {g: (minus[g] if g == cu else []) for g in range(K)}
        systems, names = [], []
        for nm, e, lay, orc in configs(cfg['quick']):
            Rn = reb(seed, groups, e, lay); Rt = torch.tensor(Rn, dtype=torch.float32, device=cfg['device'])
            info[nm] = dict(e=e, layout=lay, oracle=orc, leak_ratio=0.0 if orc else leak_ratio(Rn, groups),
                            guard_overhead=(int(lay[len('contig_g'):]) * (K - 1) / (WIDTH * K + int(lay[len('contig_g'):]) * (K - 1))) if lay.startswith('contig') else 0.0)
            for role, mem in [('src', full), ('ref', minus), ('rep', rep)]:
                systems.append(dict(members=mem, R=Rt, cross=(not orc and e > 0), salt='', w0=w0)); names.append((nm, role))
            if nm == 'e0':
                systems.append(dict(members=minus, R=Rt, cross=(not orc and e > 0), salt='alt', w0=w0)); names.append((nm, 'alt'))
        log(f'[E5] seed {seed}: systems={len(systems)}, 삭제 대상 shard={cu}')
        W = run_systems(tr, systems, h, seed, T, D, log)
        ix = {n: k for k, n in enumerate(names)}
        P = ev.probs(W.reshape(-1, D))
        def pr(si, g):
            return {k: v[si * K + g] for k, v in P.items()}
        for nm, e, lay, orc in configs(cfg['quick']):
            s_, r_, p_ = ix[(nm, 'src')], ix[(nm, 'ref')], ix[(nm, 'rep')]
            src_e = ev.ensemble([pr(s_, g) for g in range(K)])
            ref_e = ev.ensemble([pr(r_, g) for g in range(K)])
            sisa_e = ev.ensemble([pr(p_, g) if g == cu else pr(s_, g) for g in range(K)])
            unaff = [g for g in range(K) if g != cu]
            dis = [disagreement(P['test'][s_ * K + g], P['test'][r_ * K + g]) for g in unaff]
            rel = [float((W[s_, g] - W[r_, g]).norm() / W[r_, g].norm()) for g in unaff]
            asr_u = [float((P['trig'][s_ * K + g].argmax(1) == 0).float().mean() - (P['trig'][r_ * K + g].argmax(1) == 0).float().mean()) for g in unaff]
            asr_shard_src = float((P['trig'][s_ * K + cu].argmax(1) == 0).float().mean())
            asr_shard_rep = float((P['trig'][p_ * K + cu].argmax(1) == 0).float().mean())
            rows.append(dict(seed=seed, config=nm, e=e, layout=lay, oracle=orc, unaffected_disagreement=float(np.mean(dis)),
                             unaffected_rel_param=float(np.mean(rel)), unaffected_asr_diff=float(np.mean(asr_u)),
                             asr_src=src_e['asr'], asr_sisa=sisa_e['asr'], asr_ref=ref_e['asr'], acc_sisa=sisa_e['test'], acc_ref=ref_e['test'],
                             asr_backdoor_shard_src=asr_shard_src, asr_backdoor_shard_replay=asr_shard_rep,
                             sisa_ref_disagreement=disagreement(sisa_e['_ptest'], ref_e['_ptest'])))
        r_, a_ = ix[('e0', 'ref')], ix[('e0', 'alt')]
        yard.append(dict(seed=seed, unaffected_disagreement=float(np.mean([disagreement(P['test'][r_ * K + g], P['test'][a_ * K + g]) for g in range(K) if g != cu])),
                         unaffected_rel_param=float(np.mean([float((W[r_, g] - W[a_, g]).norm() / W[a_, g].norm()) for g in range(K) if g != cu])),
                         asr_diff=abs(ev.ensemble([pr(r_, g) for g in range(K)])['asr'] - ev.ensemble([pr(a_, g) for g in range(K)])['asr'])))
        log(f'[E5] seed {seed} done')
    write_json(out / 'results.json', dict(configs=info, rows=rows, yardstick=yard))

    # ---------------- 판정
    cfgs = configs(cfg['quick'])
    def m(nm, f):
        return float(np.mean([r[f] for r in rows if r['config'] == nm]))
    neg_rows = [r for r in rows if r['config'] in ('e0', 'oracle')]
    neg_level = max([r['unaffected_rel_param'] for r in neg_rows] or [0.0])
    neg_ok = all(r['unaffected_rel_param'] < 1e-3 and r['unaffected_disagreement'] <= 1e-3 for r in neg_rows)
    ymean = float(np.mean([y['unaffected_disagreement'] for y in yard])); yrel = float(np.mean([y['unaffected_rel_param'] for y in yard]))
    yasr = float(np.mean([y['asr_diff'] for y in yard]))
    worst = max((nm for nm, e, lay, orc in cfgs if not orc and e > 0), key=lambda n: m(n, 'unaffected_disagreement'))
    h5a = m(worst, 'unaffected_disagreement') > ymean or abs(m(worst, 'asr_sisa') - m(worst, 'asr_ref')) > yasr
    e_hi = max(e for _, e, _, _ in cfgs)
    guard_names = [f'e{e_hi:g}_contig_g{g}' for g in ([0, 8] if cfg['quick'] else [0, 2, 8])]
    mono = []
    for seed in cfg['seeds']:
        v = [next(r['unaffected_rel_param'] for r in rows if r['seed'] == seed and r['config'] == n) for n in guard_names]
        mono.append(all(a > b for a, b in zip(v, v[1:])))
    h5b = bool(np.mean(mono) >= .8)
    L = ['# E5 — guard 대역과 주파수 배치', '',
         f'SISO OFDM {NFFT} 부반송파, shard 당 {WIDTH}개, K={K}, {T}라운드, seed {len(cfg["seeds"])}개. 삭제 대상은 백도어 client {U_DEL}. '
         '명목 20dB 1회 전송.', '',
         '## 조건별 결과 (seed 평균)', '',
         table(['조건', 'CFO 범위', '배치', '교차 누설 에너지 비', 'guard 비율', '미삭제 shard 예측 불일치', '미삭제 shard 파라미터 상대 차이',
                '미삭제 shard ASR 차이', 'ASR: 삭제 전 / SISA / reference', '백도어 shard ASR 삭제 전→replay'],
               [[nm, e, ('oracle(교차 항 0)' if orc else LAYOUT_KO[lay]), fmt(info[nm]['leak_ratio']), pct(info[nm]['guard_overhead']).lstrip('+'),
                 pct(m(nm, 'unaffected_disagreement')).lstrip('+'), fmt(m(nm, 'unaffected_rel_param')), pp(m(nm, 'unaffected_asr_diff')),
                 f"{m(nm, 'asr_src'):.4f} / {m(nm, 'asr_sisa'):.4f} / {m(nm, 'asr_ref'):.4f}",
                 f"{m(nm, 'asr_backdoor_shard_src'):.4f} → {m(nm, 'asr_backdoor_shard_replay'):.4f}"] for nm, e, lay, orc in cfgs]), '',
         f'학습 난수 변동 기준(같은 reference, 다른 tape): 미삭제 shard 예측 불일치 {pct(ymean).lstrip("+")}, 파라미터 상대 차이 {fmt(yrel)}, '
         f'앙상블 ASR 차이 {pp(yasr)}.', '',
         '## 판정', '',
         verdict_line('음성 대조(e=0, oracle)에서 미삭제 shard 변화가 부동소수 오차 수준', neg_ok, f'파라미터 상대 차이 최대 {fmt(neg_level)} (기준 1e-3). 이 값보다 작은 효과는 구분할 수 없다'),
         verdict_line('H5a 누설로 남는 삭제 대상의 영향이 학습 난수 변동보다 크다', h5a,
                      f'가장 큰 조건 {worst}: 불일치 {pct(m(worst, "unaffected_disagreement")).lstrip("+")} vs 기준 {pct(ymean).lstrip("+")}, '
                      f'ASR 차이 {pp(m(worst, "asr_sisa") - m(worst, "asr_ref"))} vs 기준 {pp(yasr)}'),
          verdict_line('H5b guard 를 넓히면 미삭제 shard 변화가 단조 감소', h5b, f'seed 별 단조 감소 비율 {fmt(float(np.mean(mono)), 2)} (기준 0.8)'), '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(name='E5', verdicts=[('음성 대조', neg_ok), ('H5a', h5a), ('H5b', h5b)])
