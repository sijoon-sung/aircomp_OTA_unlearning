"""G0 — 정확도를 해치지 않는 집계 잡음 한계 eps* 실측.

바꾸는 것: 라운드마다 집계 update 에 더하는 잡음의 크기(벡터 전체 제곱합 = 집계 오차) eps.
고정: 무작위 배정 4 shard x 5명, 데이터, 초기값, minibatch, 잡음 방향, 라운드 수.
측정: 초기 학습 앙상블 정확도, client 삭제 4건(shard 마다 1명)의 앙상블 정확도, 2명짜리 작은 shard 단독 정확도.
판정: 잡음 0 대비 정확도 하락이 모든 seed 에서 1pp 이하인 최대 eps 를 eps* 로 둔다(낮은 단계부터 연속 통과).
"""
import numpy as np
import torch
from .common import load_split, channels, partition, init_vector, write_json, N_CLIENTS
from .trainer import LocalTrainer, Job, run_jobs, Evaluator, class_metrics, split_probs
from . import costmodel
from .report import table, fmt, pp, verdict_line

LEVELS = [0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
QUICK_LEVELS = [0, 10, 1000]
K = 4

def run(cfg, log):
    out = cfg['out'] / 'g0_noise'; out.mkdir(parents=True, exist_ok=True)
    levels = QUICK_LEVELS if cfg['quick'] else LEVELS
    T = cfg['T']; rows = []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device'])
        base, h = channels(seed, T)
        groups = partition('random_split', range(N_CLIENTS), base, K, seed)
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk'])
        w0 = init_vector(seed, cfg['device'])
        jobs, names = [], []
        for lv in levels:
            rg = {'mode': 'mse', 'eps': lv} if lv > 0 else {'mode': 'none'}
            for c, g in enumerate(groups):
                jobs.append(Job(f'src_{lv}_{c}', c, w0, {i: 0 for i in g}, 0, T, rg))
            for c, g in enumerate(groups):
                u = g[0]
                jobs.append(Job(f'del_{lv}_{c}', c, w0, {i: 0 for i in g if i != u}, 0, T, rg))
            jobs.append(Job(f'small_{lv}', 100, w0, {i: 0 for i in groups[0][:2]}, 0, T, rg))
        log(f'[G0] seed {seed}: jobs={len(jobs)}')
        W, _, rec = run_jobs(tr, jobs, h, seed, log_every=max(1, T // 4), logger=log)
        ev = Evaluator(tr, data); P = ev.probs(W); idx = {j.name: k for k, j in enumerate(jobs)}
        for lv in levels:
            src = [split_probs(P, idx[f'src_{lv}_{c}']) for c in range(K)]
            s = ev.ensemble(src)
            dels = []
            for c in range(K):
                plist = [split_probs(P, idx[f'del_{lv}_{c}']) if k == c else src[k] for k in range(K)]
                dels.append(ev.ensemble(plist)['test'])
            small_acc, _ = class_metrics(P['test'][idx[f'small_{lv}']], data['ty'])
            clip = np.mean([rec[idx[f'src_{lv}_{c}']]['clipped'] / max(1, rec[idx[f'src_{lv}_{c}']]['calls']) for c in range(K)])
            rows.append(dict(seed=seed, eps=lv, source_test=s['test'], source_dev=s['dev'],
                             source_worst_recall=s['test_worst_recall'], delete_test_mean=float(np.mean(dels)),
                             delete_tests=dels, small_test=small_acc, clip_rate=float(clip)))
        log(f'[G0] seed {seed} done: ' + ', '.join(f"eps={r['eps']}:{r['source_test']:.4f}" for r in rows if r['seed'] == seed))

    # ---------------- 판정
    def drops(field):
        d = {}
        for seed in cfg['seeds']:
            base_acc = next(r[field] for r in rows if r['seed'] == seed and r['eps'] == 0)
            for r in rows:
                if r['seed'] == seed and r['eps'] > 0:
                    d.setdefault(r['eps'], []).append(base_acc - r[field])
        return d
    dsrc, ddel, dsmall = drops('source_test'), drops('delete_test_mean'), drops('small_test')
    pos = [lv for lv in levels if lv > 0]
    def contiguous(ok):
        best = 0
        for lv in pos:
            if ok(lv):
                best = lv
            else:
                break
        return best
    eps_strict = contiguous(lambda lv: max(dsrc[lv]) <= .01 and max(ddel[lv]) <= .01)
    eps_len = contiguous(lambda lv: np.mean(dsrc[lv]) <= .01 and np.mean(ddel[lv]) <= .01)
    eps_small = contiguous(lambda lv: max(dsmall[lv]) <= .01)
    flag = 'ok'
    if eps_strict == 0:
        flag = 'none_passed'; eps_used = pos[0] / 10
    elif eps_strict == pos[-1]:
        flag = 'all_passed'; eps_used = eps_strict
    else:
        eps_used = eps_strict
    # 현실적인 조건에서 반복 전송이 필요한가
    real = []
    for snr, s2 in [(0, 1.0), (10, .1), (20, .01)]:
        for n in [1, 2, 3, 4, 5]:
            for hmin in [0.1, 0.316, 1.0]:
                m1 = float(costmodel.mse1(n, hmin, s2))
                real.append(dict(snr_db=snr, n=n, hmin=hmin, mse1=m1, R_needed=int(max(1, np.ceil(m1 / eps_used)))))
    worst0 = next(r for r in real if r['snr_db'] == 0 and r['n'] == 1 and r['hmin'] == 0.1)
    repetition_matters = worst0['R_needed'] > 1
    res = dict(levels=levels, rows=rows, eps_star=eps_strict, eps_star_lenient=eps_len, eps_small_shard=eps_small,
               eps_used=eps_used, flag=flag, realistic=real, repetition_matters=repetition_matters)
    write_json(out / 'results.json', res)
    write_json(cfg['out'] / 'eps_star.json', dict(eps_used=eps_used, eps_star=eps_strict, flag=flag,
                                                   repetition_matters=repetition_matters))

    # ---------------- 보고서
    L = ['# G0 — 정확도를 해치지 않는 집계 잡음 한계', '',
         f'seed {len(cfg["seeds"])}개, {T}라운드, 무작위 배정 {K} shard. 집계 오차 eps 는 라운드마다 집계 update 에 더한 잡음 벡터의 제곱합 기대값이다. '
         '잡음 방향은 모든 eps 에서 같고 크기만 다르다.', '',
         '## 잡음 0 대비 정확도 하락 (양수 = 나빠짐)', '']
    trs = []
    for lv in pos:
        trs.append([fmt(lv, 0), pp(np.mean(dsrc[lv])), pp(max(dsrc[lv])), pp(np.mean(ddel[lv])), pp(max(ddel[lv])),
                    pp(np.mean(dsmall[lv])), pp(max(dsmall[lv]))])
    L.append(table(['eps', '초기 학습 평균', '초기 학습 최대', '삭제 평균', '삭제 최대', '2명 shard 평균', '2명 shard 최대'], trs))
    L += ['', '## 판정', '',
          verdict_line('eps*(모든 seed 1pp 이하)', eps_strict > 0, f'eps* = {fmt(eps_strict, 0)} '
                       f'(평균 기준 {fmt(eps_len, 0)}, 2명 shard 기준 {fmt(eps_small, 0)}). 이후 실험은 eps = {fmt(eps_used)} 를 쓴다.'),
          f'- 상태 표시: `{flag}` (all_passed 면 실제 한계는 시험한 최대값보다 크다는 뜻이다. none_passed 면 가장 작은 단계도 실패해 그 1/10 을 썼다.)', '',
          '## 현실적인 조건에서 1회 전송 집계 오차와 필요한 반복 수 (eps 기준)', '',
          table(['명목 SNR', 'n', 'h_min', '1회 집계 오차', '필요 반복 R'],
                [[f"{r['snr_db']}dB", r['n'], r['hmin'], fmt(r['mse1']), r['R_needed']] for r in real if r['n'] in (1, 2, 5)]), '',
          ('가장 나쁜 경우(0dB, n=1, h_min=0.1)에서도 반복이 필요 없다. 그래서 이 설정에서는 채널 사용량 지표가 조건 사이에 같게 나오고, '
           '비용 차이는 송신 에너지와 지연으로만 드러난다. E2~E4 보고서는 이 점을 반영해 읽어야 한다.' if not repetition_matters else
           '현실적인 조건 일부에서 반복 전송이 필요하다. 그래서 채널 사용량 지표에도 배정·일정의 차이가 드러날 수 있다.'), '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(name='G0', eps_star=eps_strict, eps_used=eps_used, flag=flag, repetition_matters=repetition_matters,
                verdicts=[('eps* 존재', eps_strict > 0)])
