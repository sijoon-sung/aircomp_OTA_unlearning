"""P0-A — 코드로 분리하는 다중 shard AirComp 에서 shard 간 간섭은 얼마나 생기고, 그것이 언러닝에 흔적을 남기는가.

설정: shard 4개가 같은 자원에 동시에 보내고, 각 shard 는 자기 코드를 붙인다. 서버는 합을 받아 코드로 shard 합을 분리한다.
      전력 정렬은 shard 안에서 한다 (일반적인 다중 그룹 AirComp 와 같음). 코드 종류·길이·칩 타이밍 오차를 바꾼다.
측정:
  간섭: 분리된 shard 수신값에서 다른 shard 에서 샌 성분의 에너지 / 자기 shard 성분 에너지 (매 라운드 실제 값의 평균)
  흔적: 삭제 대상 u 를 지울 때 SISA 는 u 의 shard 만 재학습한다. 삭제하지 않은 shard 의 모델이
        (u 가 있던 원래 학습) vs (u 없이 같은 코드 시스템으로 처음부터 학습) 에서 얼마나 다른가 = 간섭을 통해 u 가 남긴 흔적
  기준: 같은 reference 를 다른 잡음·배치 난수로 학습했을 때의 차이(학습 난수 변동)
"""
import math
import numpy as np
import torch
from .fl import ShardJob, load_split, channels, partition, shard_of, init_vector, key, N_CLIENTS, LocalTrainer, Evaluator, disagreement, write_json
from .cdma import CdmaConfig, run_rounds_cdma
from .exp_p0_problem import fmt, pct, pp, table

K = 4
CONFIGS = [('walsh', 4, 0.0), ('walsh', 4, 0.1), ('walsh', 4, 0.3), ('pn', 16, 0.0), ('pn', 64, 0.0), ('pn', 16, 0.1)]
QUICK = [('walsh', 4, 0.1), ('pn', 16, 0.0)]

def cname(c):
    return f'{c[0]}{c[1]}_d{c[2]:g}'

def run(cfg, log):
    out = cfg['out'] / 'p0a_cdma'; out.mkdir(parents=True, exist_ok=True)
    T = cfg['T']; confs = QUICK if cfg['quick'] else CONFIGS; s2s = [1.0] if cfg['quick'] else [1.0, .01]
    rows = []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device']); base, h = channels(seed, T)
        groups = partition('random_split', range(N_CLIENTS), base, K, seed)
        order = np.argsort(base); targets = {'weakest': int(order[0]), 'median': int(order[N_CLIENTS // 2])}
        unit_delay = np.array([np.random.default_rng(key(seed, 'chipdelay', i)).uniform() for i in range(N_CLIENTS)])
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device']); ev = Evaluator(tr, data)
        # 지연은 설정마다 scale 이 다르므로 설정별로 따로 실행
        for c in confs:
            delays = unit_delay * c[2]; jobs = []; sysconf = {}
            for s2 in s2s:
                cc = CdmaConfig(sigma2=s2, code=c[0], L=c[1], delay_max=c[2]); tag = f'{s2}'
                sysconf[f'{tag}|src'] = cc
                for k, g in enumerate(groups):
                    jobs.append(ShardJob(f'{tag}|src|{k}', list(g), w0, T, system=f'{tag}|src', tape=k))
                for tn, u in targets.items():
                    cu = shard_of(groups, u)
                    for sysn, salt in [('ref', ''), ('alt', 'alt')]:
                        sysconf[f'{tag}|{sysn}|{tn}'] = cc
                        for k, g in enumerate(groups):
                            jobs.append(ShardJob(f'{tag}|{sysn}|{tn}|{k}', [i for i in g if i != u], w0, T, system=f'{tag}|{sysn}|{tn}', tape=k, salt=salt))
                    sysconf[f'{tag}|rep|{tn}'] = cc
                    jobs.append(ShardJob(f'{tag}|rep|{tn}', [i for i in groups[cu] if i != u], w0, T, system=f'{tag}|rep|{tn}', tape=cu))
            log(f'[P0-A] seed {seed} {cname(c)}: shard jobs={len(jobs)}')
            W, leds, diags = run_rounds_cdma(tr, jobs, h, seed, lambda n: sysconf[n], delays, log, max(1, T // 4))
            P = ev.probs(W); ix = {j.name: k for k, j in enumerate(jobs)}
            def pr(n): return {kk: v[ix[n]] for kk, v in P.items()}
            for s2 in s2s:
                tag = f'{s2}'
                src = {k: pr(f'{tag}|src|{k}') for k in range(K)}
                sir = [diags[ix[f'{tag}|src|{k}']] for k in range(K)]
                leak_ratio = float(np.mean([d['leak_energy'] / max(d['own_energy'], 1e-30) for d in sir]))
                noise_ratio = float(np.mean([d['noise_energy'] / max(d['own_energy'], 1e-30) for d in sir]))
                leak_noise = float(np.mean([d['leak_energy'] / max(d['noise_energy'], 1e-30) for d in sir]))
                for tn, u in targets.items():
                    cu = shard_of(groups, u); un = [k for k in range(K) if k != cu]
                    ref = {k: pr(f'{tag}|ref|{tn}|{k}') for k in range(K)}; alt = {k: pr(f'{tag}|alt|{tn}|{k}') for k in range(K)}
                    rep = pr(f'{tag}|rep|{tn}')
                    dis = float(np.mean([disagreement(src[k]['test'], ref[k]['test']) for k in un]))
                    yard = float(np.mean([disagreement(ref[k]['test'], alt[k]['test']) for k in un]))
                    rel = float(np.mean([float((W[ix[f'{tag}|src|{k}']] - W[ix[f'{tag}|ref|{tn}|{k}']]).norm() / W[ix[f'{tag}|ref|{tn}|{k}']].norm()) for k in un]))
                    acc_un_src = float(np.mean([float((src[k]['test'].argmax(1) == data['ty']).float().mean()) for k in un]))
                    acc_un_ref = float(np.mean([float((ref[k]['test'].argmax(1) == data['ty']).float().mean()) for k in un]))
                    sisa = ev.ensemble([rep if k == cu else src[k] for k in range(K)]); refe = ev.ensemble(list(ref.values()))
                    rows.append(dict(seed=seed, config=cname(c), code=c[0], L=c[1], delay_max=c[2], sigma2=s2, target=tn, client=u,
                                     leak_to_own=leak_ratio, noise_to_own=noise_ratio, leak_to_noise=leak_noise, unaff_dis=dis, unaff_rel=rel, yard_dis=yard,
                                     unaff_acc_src=acc_un_src, unaff_acc_ref=acc_un_ref, ens_dis_sisa_ref=disagreement(sisa['_ptest'], refe['_ptest']),
                                     acc_sisa=sisa['test'], acc_ref=refe['test'], chips_per_symbol=leds[ix[f'{tag}|src|0']].ul_symbols / max(1, leds[ix[f'{tag}|src|0']].rounds) / tr.D))
        log(f'[P0-A] seed {seed} done')
    write_json(out / 'results.json', dict(rows=rows))
    L = ['# P0-A — 코드 분리 다중 shard AirComp 의 간섭과 언러닝 흔적', '',
         f'shard {K}개가 같은 자원에 동시에 보내고 서버가 코드로 분리한다. 전력 정렬은 shard 안. {cfg["T"]}라운드, seed {len(cfg["seeds"])}개. '
         '칩 타이밍 오차는 client 마다 U[0, 상한] (정적).', '',
         '"간섭/자기 신호": 분리된 shard 수신값에서 다른 shard 에서 샌 성분 에너지 ÷ 자기 shard 성분 에너지 (학습 전체 평균). '
         '"미삭제 shard 불일치": 삭제하지 않은 shard 모델의 u 있을 때 vs u 없이 처음부터 학습했을 때 예측 불일치 = 간섭으로 남은 u 의 흔적. '
         '"난수 변동": 같은 reference 를 다른 난수로 학습한 불일치.', '']
    tb = []
    for c in confs:
        for s2 in s2s:
            for tn in ['weakest', 'median']:
                sel = [r for r in rows if r['config'] == cname(c) and r['sigma2'] == s2 and r['target'] == tn]
                m = lambda f: float(np.mean([r[f] for r in sel]))
                tb.append([c[0], c[1], c[2], f'{round(10 * math.log10(1 / s2))}dB', '최약' if tn == 'weakest' else '중간', fmt(m('leak_to_own'), 4),
                           fmt(m('noise_to_own'), 4), fmt(m('leak_to_noise'), 4), pct(m('unaff_dis')), fmt(m('unaff_rel'), 4), pct(m('yard_dis')),
                           pp(m('unaff_acc_src') - m('unaff_acc_ref')), pct(m('ens_dis_sisa_ref')), pp(m('acc_sisa') - m('acc_ref'))])
    L.append(table(['코드', '길이 L', '칩 오차 상한', 'SNR', '삭제 대상', '간섭/자기 신호', '잡음/자기 신호', '간섭/잡음', '미삭제 shard 불일치', '파라미터 상대 차이',
                    '난수 변동', '미삭제 shard 정확도 차이', '앙상블 불일치(SISA-ref)', '앙상블 정확도 차이'], tb))
    L += ['', '읽는 법: walsh·칩 오차 0 은 완전 직교라 간섭 0 (측정 바닥). 칩 오차나 PN 코드에서 간섭/자기 신호가 0 보다 크면 shard 가 섞이고, '
          '그때 미삭제 shard 불일치가 바닥보다 크면 데이터 격리만으로는 삭제 대상의 흔적이 다른 shard 에 남는다는 뜻이다.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(name='P0-A')
