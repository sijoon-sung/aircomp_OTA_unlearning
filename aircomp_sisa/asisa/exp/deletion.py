"""같은 자원을 쓰는 shard 묶음에서 삭제 대상 u 가 다른 shard 에 남기는 흔적을 재는 공통 절차.

system 하나 (tag) 마다 다음 학습 경로를 만든다.
  src       원래 학습 (모든 shard)
  alt       같은 system 을 다른 배치·잡음 난수로 (학습 난수 변동의 기준선)
  ref|u     u 없이 처음부터 같은 system 으로 (src 와 배치·잡음이 같다)
  rep|u     삭제 후 SISA 재학습: u 의 shard 만 단독으로
측정 (u 의 shard 가 아닌 shard 마다)
  rel       파라미터 차이 ||W_src - W_ref|| / ||W_ref||  : u 가 남긴 흔적의 크기
  rel_yard  ||W_src - W_alt|| / ||W_alt||               : 같은 양의 학습 난수 기준선
  trace     예측 불일치 (src vs ref), yard (src vs alt). 작은 흔적에도 몇 % 로 포화되므로 참고용.
  u 간섭/자기 신호, 전체 간섭/자기 신호, 수신 크기 비 beta_k / beta_u의shard, 반복 수
"""
import numpy as np
from ..fl import Shard, per_round
from ..trainer import disagreement

def tv(pa, pb):
    """test 표본별 확률 분포의 total variation 거리 평균 (예측 불일치보다 덜 포화된다)."""
    return float(0.5 * (pa - pb).abs().sum(1).mean())

def add_system(shards, radio, track, tag, groups, slot_of, targets, rc, T):
    """groups: shard 별 client 목록 (빈 shard 는 건너뜀), slot_of: shard -> 코드 번호, targets: 이름 -> client."""
    live = [k for k, g in enumerate(groups) if g]
    for kind, salt in [('src', ''), ('alt', 'alt')]:
        radio[f'{tag}|{kind}'] = rc
        shards += [Shard(f'{tag}|{kind}|{k}', groups[k], f'{tag}|{kind}', slot=slot_of[k], salt=salt, T=T) for k in live]
    track[f'{tag}|src'] = tuple(targets.values())
    for tn, u in targets.items():
        cu = next(k for k in live if u in groups[k])
        radio[f'{tag}|ref|{tn}'] = rc; radio[f'{tag}|rep|{tn}'] = rc
        shards += [Shard(f'{tag}|ref|{tn}|{k}', [i for i in groups[k] if i != u], f'{tag}|ref|{tn}', slot=slot_of[k], T=T)
                   for k in live if [i for i in groups[k] if i != u]]
        shards.append(Shard(f'{tag}|rep|{tn}', [i for i in groups[cu] if i != u], f'{tag}|rep|{tn}', slot=slot_of[cu], T=T))

def measure(c, W, P, leds, diags, ix, tag, groups, slot_of, targets, hmin, focus=None, extra=None):
    """반환: (shard 별 행 목록, system 행)."""
    extra = extra or {}
    live = [k for k, g in enumerate(groups) if g]
    pick = lambda n: c.ev.pick(P, ix[n])
    src = {k: pick(f'{tag}|src|{k}') for k in live}; alt = {k: pick(f'{tag}|alt|{k}') for k in live}
    dg = {k: diags[ix[f'{tag}|src|{k}']] for k in live}; ld = {k: leds[ix[f'{tag}|src|{k}']] for k in live}
    beta = {k: per_round(dg[k], 'beta') for k in live}; nr = max(1, dg[live[0]]['rounds'])
    sysrow = dict(extra, energy=sum(l.energy for l in ld.values()) / nr, ul=ld[live[0]].ul_symbols / nr, slots=per_round(dg[live[0]], 'slots'),
                  mse=float(np.mean([l.mse_sum / max(1, l.rounds) for l in ld.values()])), beta_spread=max(beta.values()) / min(beta.values()),
                  max_leak_to_own=max(dg[k]['leak_energy'] / max(dg[k]['own_energy'], 1e-30) for k in live),
                  acc_shard=float(np.mean([c.ev.acc(p) for p in src.values()])), acc_ens=c.ev.ensemble(list(src.values()))['test'],
                  max_power_ratio=max(l.max_power_ratio for l in ld.values()), sizes=[len(g) for g in groups])
    ae = c.ev.ensemble(list(alt.values())); se = c.ev.ensemble(list(src.values()))
    sysrow['ens_yard_dis'] = disagreement(se['ptest'], ae['ptest']); sysrow['ens_yard_tv'] = tv(se['ptest'], ae['ptest'])
    rows = []
    for tn, u in targets.items():
        cu = next(k for k in live if u in groups[k])
        ref = {k: pick(f'{tag}|ref|{tn}|{k}') for k in live if f'{tag}|ref|{tn}|{k}' in ix}
        sisa = c.ev.ensemble([pick(f'{tag}|rep|{tn}') if k == cu else src[k] for k in live]); full = c.ev.ensemble(list(ref.values()))
        lr = leds[ix[f'{tag}|rep|{tn}']]
        sysrow[f'ens_dis_{tn}'] = disagreement(sisa['ptest'], full['ptest'])
        sysrow[f'ens_tv_{tn}'] = tv(sisa['ptest'], full['ptest'])
        sysrow[f'del_energy_{tn}'] = lr.energy; sysrow[f'del_compute_{tn}'] = lr.compute_calls; sysrow[f'del_acc_{tn}'] = sisa['test']
        for k in live:
            if k == cu or k not in ref:
                continue
            d = dg[k]; own = max(d['own_energy'], 1e-30)
            ws, wr, wa = W[ix[f'{tag}|src|{k}']], W[ix[f'{tag}|ref|{tn}|{k}']], W[ix[f'{tag}|alt|{k}']]
            rows.append(dict(extra, target=tn, client=u, shard=k, affected=cu, code_index=slot_of[k],
                             role='focus' if focus is not None and k == focus.get(tn) else 'other',
                             beta_ratio=beta[k] / beta[cu], hmin_ratio=hmin[cu] / hmin[k],
                             own_energy=per_round(d, 'own_energy'), uleak_energy=d['u_leak'].get(u, 0.0) / max(1, d['rounds']),
                             leak_to_own=d['leak_energy'] / own, uleak_to_own=d['u_leak'].get(u, 0.0) / own, noise_to_own=d['noise_energy'] / own,
                             repeats=per_round(d, 'repeats'), rel=float((ws - wr).norm() / wr.norm()), rel_yard=float((ws - wa).norm() / wa.norm()),
                             trace=disagreement(src[k]['test'], ref[k]['test']), yard=disagreement(src[k]['test'], alt[k]['test']),
                             acc_src=c.ev.acc(src[k]), acc_ref=c.ev.acc(ref[k])))
    return rows, sysrow
