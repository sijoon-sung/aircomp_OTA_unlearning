"""sharding (A, E) — shard 수 K 의 양면: 조를 나누면 정확도가 얼마나 떨어지고, 기기 쪽 추론·내려받기 비용은 얼마나 느는가.

(E) SISA 의 본래 비용: 조 모델 하나는 데이터의 1/K 만 보므로 약하고, 앙상블이 그것을 얼마나 메우는지는 K 와 학습량에 달렸다.
    지금까지의 실험은 라운드당 로컬 step 2 (160 라운드 = 약 1/3 epoch) 라 모델이 덜 배운 상태였고, 그래서 K 를 늘려도 정확도가
    안 떨어져 보였고 잡음 eps 가 정확도를 올리는 이상한 결과도 있었다. 여기서는 로컬 step 을 늘려 제대로 배운 상태에서 다시 잰다.
(A) 중앙 SISA 는 서버가 K 개 모델을 들고 예측한다. FL 에서는 기기가 예측하므로 모델 K 개를 내려받고 K 번 돌려야 한다.
    내려받기 비트 = K × 32 × D, 예측 연산 = K 배. 삭제 뒤에는 바뀐 조 모델 하나를 모든 기기에 다시 보낸다 (N × 32 × D, K 와 무관).
설정: 무작위 배정, K in {1,2,4,5,10}, 채널 없이 평균에 잡음만 (ideal) — 잡음 없음 / eps. 로컬 step in {2 (지금까지), 10}.
측정: 앙상블 test 정확도, 조 모델 하나의 평균 정확도, 앙상블 − 하나 (앙상블이 메우는 양), 잡음이 정확도에 주는 변화, 추론 내려받기·연산.
"""
import numpy as np
from . import seed_context
from ..channel import partition
from ..config import N_CLIENTS, LOCAL_STEPS
from ..fl import Shard, train
from ..radio import RadioConfig
from ..util import write_json, fmt, pct, pp, table, select, mean

NAME = 'sharding'
KS = [1, 2, 4, 5, 10]
STEPS = [LOCAL_STEPS, 10]

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    T, eps = cfg['T'], cfg['eps']
    Ks = [1, 4] if cfg['quick'] else KS
    steps_list = [LOCAL_STEPS, 4] if cfg['quick'] else STEPS
    noises = [0.0, eps]
    rows = []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed); D = c.tr.D
        parts = {K: (partition('random_split', range(N_CLIENTS), c.base, K, seed) if K > 1 else [list(range(N_CLIENTS))]) for K in Ks}
        for steps in steps_list:
            shards = []
            for K in Ks:
                for nz in noises:
                    shards += [Shard(f'{K}|{nz}|{k}', g, f'{K}|{nz}', slot=k, T=T) for k, g in enumerate(parts[K])]
            c.tr.steps = steps
            log(f'[sharding] seed {seed}, 로컬 step {steps}: shard 모델 {len(shards)}개')
            try:
                W, leds, _, _ = train(c.tr, shards, c.h, c.w0, seed, lambda s: RadioConfig(mux='ideal', noise_mse=float(s.split('|')[1])), log=log, log_every=T // 4)
            finally:
                c.tr.steps = LOCAL_STEPS
            P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
            for K in Ks:
                for nz in noises:
                    ps = [c.ev.pick(P, ix[f'{K}|{nz}|{k}']) for k in range(K)]
                    single = [c.ev.acc(p) for p in ps]
                    rows.append(dict(seed=seed, steps=steps, K=K, noise=nz, acc_ens=c.ev.ensemble(ps)['test'], acc_single=float(np.mean(single)),
                                     acc_single_max=float(max(single)), dl_infer_bits=K * 32 * D, infer_passes=K, dl_delete_bits=N_CLIENTS * 32 * D,
                                     local_calls=sum(leds[ix[f'{K}|{nz}|{k}']].compute_calls for k in range(K))))
            del W, P
        log(f'[sharding] seed {seed} 끝')
    write_json(out / 'results.json', dict(rows=rows, setting=dict(Ks=Ks, steps=steps_list, noises=noises, T=T, seeds=list(cfg['seeds']))))
    L = ['# sharding (A, E) — shard 수 K 에 따른 정확도와 기기 쪽 추론 비용', '',
         f'무작위 배정, 채널 없이 평균에 잡음만, {T}라운드, seed {len(cfg["seeds"])}개. 로컬 step {steps_list} (지금까지의 실험은 {LOCAL_STEPS}).', '',
         '열: 앙상블 = K 개 조 모델의 확률 평균. 하나 = 조 모델 하나의 평균 정확도. 잡음 효과 = eps 잡음을 더했을 때 앙상블 정확도 변화 (양수 = 좋아짐). '
         '내려받기 = 기기가 예측하려고 받아야 하는 비트 (모델 K 개). 삭제 재전송 = 삭제 뒤 바뀐 조 모델을 모든 기기에 보내는 비트.', '']
    tb = []
    for steps in steps_list:
        for K in Ks:
            a0 = select(rows, steps=steps, K=K, noise=0.0); a1 = select(rows, steps=steps, K=K, noise=eps)
            if not a0:
                continue
            tb.append([steps, K, pct(mean(a0, 'acc_ens')), pct(mean(a0, 'acc_single')), pp(mean(a0, 'acc_ens') - mean(a0, 'acc_single')),
                       pct(mean(a1, 'acc_ens')) if a1 else '-', pp(mean(a1, 'acc_ens') - mean(a0, 'acc_ens')) if a1 else '-',
                       fmt(mean(a0, 'dl_infer_bits')), K, fmt(mean(a0, 'dl_delete_bits')), fmt(mean(a0, 'local_calls'), 0)])
    L += [table(['로컬 step', 'K', '앙상블 정확도 (잡음 0)', '조 모델 하나', '앙상블 − 하나', '앙상블 정확도 (eps 잡음)', '잡음 효과',
                 '추론 내려받기 bit', '예측 연산 (배)', '삭제 재전송 bit', '학습 로컬 횟수'], tb), '',
          '## 판정', '']
    for steps in steps_list:
        r1 = select(rows, steps=steps, K=1, noise=0.0)
        for K in Ks[1:]:
            rk = select(rows, steps=steps, K=K, noise=0.0)
            if r1 and rk:
                L.append(f'- 로컬 step {steps}: K={K} 앙상블 − K=1 = {pp(mean(rk, "acc_ens") - mean(r1, "acc_ens"))}, 조 모델 하나 − K=1 = {pp(mean(rk, "acc_single") - mean(r1, "acc_ens"))}')
        r4 = select(rows, steps=steps, K=4, noise=0.0); r4n = select(rows, steps=steps, K=4, noise=eps)
        if r4 and r4n:
            L.append(f'- 로컬 step {steps}: K=4 에서 eps 잡음의 효과 = {pp(mean(r4n, "acc_ens") - mean(r4, "acc_ens"))} '
                     + ('(잡음이 도움 = 아직 덜 배운 상태)' if mean(r4n, 'acc_ens') > mean(r4, 'acc_ens') else '(잡음이 해침 = 배운 상태)'))
    L += ['', '## 읽는 법', '',
          '- 제대로 배운 상태 (로컬 step 10) 에서 K 를 늘리면 조 모델 하나의 정확도는 떨어져야 하고, 앙상블이 그것을 얼마나 메우는지가 "조 몇 개" 의 정확도 쪽 축이다.',
          '- 잡음 효과가 양수이면 아직 덜 배운 상태라는 뜻이다 (잡음이 regularizer 역할). 제대로 배운 상태에서는 음수여야 한다. 이 실험은 다른 실험의 정확도 숫자를 어떻게 읽어야 하는지 정한다.',
          '- 추론 내려받기와 예측 연산은 K 에 정비례한다. 중앙 SISA 는 이 비용이 서버에 있어 보이지 않았다.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
