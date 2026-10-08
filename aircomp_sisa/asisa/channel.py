"""채널과 shard 배정 규칙."""
import math
import numpy as np
from .config import N_CLIENTS
from .util import key

def channels(seed, T, fading='mild'):
    """client 채널 진폭. 장기 진폭 10^(U[-20,0]/20) x 라운드별 페이딩. 반환: base [N], h [T, N].
    fading='mild'     U[0.85, 1.15] (지금까지의 실험)
    fading='rayleigh' |CN(0,1)| (E|g|^2 = 1, 깊은 골이 있음. 기존 AirComp 논문의 블록 페이딩)"""
    base = 10 ** (np.random.default_rng(key(seed, 'slow_channel')).uniform(-20, 0, N_CLIENTS) / 20)
    if fading == 'mild':
        fad = np.stack([np.random.default_rng(key(seed, t, 'fading')).uniform(.85, 1.15, N_CLIENTS) for t in range(T)])
    elif fading == 'rayleigh':
        fad = np.stack([np.random.default_rng(key(seed, t, 'rayleigh')).rayleigh(1 / np.sqrt(2), N_CLIENTS) for t in range(T)])
    else:
        raise ValueError(fading)
    return base, base[None, :] * fad

RULES = ['random_split', 'rank_chunk', 'rank_rr', 'bins', 'hash']
RULE_KO = {'random_split': '무작위', 'rank_chunk': '채널 정렬 자르기', 'rank_rr': '채널 돌려 담기',
           'bins': '채널 고정 구간', 'hash': 'ID 해시'}

def partition(rule, ids, base, K, seed):
    """client 집합 ids 를 K 개 shard 로 나눈다. 반환: 길이 K 의 client 목록 리스트 (빈 shard 허용).
    random_split·rank_chunk·rank_rr 는 집단 전체를 보고 정하고, bins·hash 는 각 client 의 자기 정보만으로 정한다
    (그래서 한 명을 지워도 다른 사람의 소속이 바뀌지 않는다)."""
    ids = sorted(int(i) for i in ids)
    if rule == 'random_split':
        order = list(np.random.default_rng(key(seed, 'routing')).permutation(ids))
        return [sorted(map(int, g)) for g in np.array_split(order, K)]
    if rule in ('rank_chunk', 'rank_rr'):
        order = sorted(ids, key=lambda i: (base[i], i))         # 약한 채널부터
        if rule == 'rank_chunk':
            return [sorted(map(int, g)) for g in np.array_split(order, K)]
        groups = [[] for _ in range(K)]
        for r, i in enumerate(order):
            groups[r % K].append(i)
        return [sorted(g) for g in groups]
    if rule == 'bins':                                          # U[-20,0] dB 의 등확률 구간 (공개 상수)
        groups = [[] for _ in range(K)]
        for i in ids:
            groups[min(K - 1, max(0, int((20 * math.log10(base[i]) + 20) // (20 / K))))].append(i)
        return groups
    if rule == 'hash':
        groups = [[] for _ in range(K)]
        for i in ids:
            groups[key(seed, i, 'hash') % K].append(i)
        return groups
    raise ValueError(rule)

def shard_of(groups, u):
    return next(k for k, g in enumerate(groups) if u in g)
