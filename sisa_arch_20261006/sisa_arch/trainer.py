"""배치 로컬 학습(vmap) + shard 별 AirComp 집계 + 평가.

같은 seed 안에서는 (client, 라운드, local step) 의 minibatch 와 (shard tape, 라운드) 의 잡음 방향이
모든 조건에서 같다. 그래서 조건 사이 차이는 바꾼 변수에서만 나온다.
여러 작업(job: shard 모델 하나의 학습 경로)을 한 번에 묶어 GPU 에서 동시에 계산한다.
"""
from dataclasses import dataclass, field
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from .common import Net, key, LR, LOCAL_STEPS, BATCH, CLIP
from . import costmodel

class LocalTrainer:
    def __init__(self, data, seed, device, chunk=128, salt=''):
        self.data, self.seed, self.device, self.chunk, self.salt = data, seed, device, chunk, salt
        self.net = Net().to(device)
        self.names = [n for n, _ in self.net.named_parameters()]
        self.shapes = [p.shape for _, p in self.net.named_parameters()]
        self.numels = [p.numel() for _, p in self.net.named_parameters()]
        self.D = sum(self.numels)
        self._idx_cache = {}
        net = self.net
        def loss(params, x, y):
            return F.cross_entropy(functional_call(net, params, (x,)), y)
        self._vgrad = vmap(grad(loss))
        def fwd(params, x):
            return functional_call(net, params, (x,)).softmax(-1)
        self._vfwd = vmap(fwd, in_dims=(0, None))

    def unflat(self, W):
        out = {}; off = 0
        for n, s, k in zip(self.names, self.shapes, self.numels):
            out[n] = W[:, off:off + k].reshape(W.shape[0], *s); off += k
        return out

    def flatg(self, g):
        return torch.cat([g[n].reshape(g[n].shape[0], -1) for n in self.names], 1)

    def batch_idx(self, i, t, step, salt=None):
        salt = self.salt if salt is None else salt
        k = (i, t, step, salt)
        if k not in self._idx_cache:
            pool = self.data['pools'][i]
            r = np.random.default_rng(key(self.seed, i, t, step, 'batch', salt)).integers(len(pool), size=BATCH)
            self._idx_cache[k] = pool[r]
            if len(self._idx_cache) > 200000:
                self._idx_cache.clear()
        return self._idx_cache[k]

    def updates(self, W, clients, t, salts=None):
        """W: [P, D] 각 pair 의 시작 모델, clients: 길이 P. 반환: clip 된 update [P, D], clip 전 norm [P].
        GPU 메모리 부족·cuDNN 내부 오류가 나면 chunk 를 절반으로 줄여 다시 계산한다(계산 결과는 chunk 와 무관)."""
        while True:
            try:
                return self._updates(W, clients, t, salts)
            except RuntimeError as ex:
                if not _recoverable(ex) or self.chunk <= 4:
                    raise
                self.chunk //= 2
                _free()
                print(f'[경고] GPU 메모리/cuDNN 오류로 chunk 를 {self.chunk} 로 줄여 다시 계산합니다: {str(ex)[:80]}', flush=True)

    def _updates(self, W, clients, t, salts=None):
        P = W.shape[0]; out = torch.empty_like(W); norms = torch.empty(P, device=W.device)
        for s in range(0, P, self.chunk):
            e = min(P, s + self.chunk); p = W[s:e]
            for step in range(LOCAL_STEPS):
                idx = np.stack([self.batch_idx(clients[j], t, step, None if salts is None else salts[j]) for j in range(s, e)])
                idx = torch.as_tensor(idx, device=W.device)
                g = self.flatg(self._vgrad(self.unflat(p), self.data['x'][idx], self.data['y'][idx]))
                p = p - LR * g
            u = p - W[s:e]; nrm = u.norm(dim=1)
            out[s:e] = u * torch.clamp(CLIP / nrm.clamp_min(1e-30), max=1.0)[:, None]
            norms[s:e] = nrm
        return out, norms

    @torch.no_grad()
    def predict(self, W, x, chunk_models=16, chunk_x=500):
        """W: [J, D] -> 확률 [J, len(x), 10]"""
        while True:
            try:
                outs = []
                for s in range(0, W.shape[0], chunk_models):
                    Wc = self.unflat(W[s:s + chunk_models])
                    outs.append(torch.cat([self._vfwd(Wc, xb) for xb in x.split(chunk_x)], 1))
                return torch.cat(outs, 0)
            except RuntimeError as ex:
                if not _recoverable(ex) or chunk_models <= 1:
                    raise
                chunk_models = max(1, chunk_models // 2); chunk_x = max(50, chunk_x // 2)
                _free()


def _recoverable(ex):
    m = str(ex).lower()
    return 'out of memory' in m or 'cudnn' in m or 'cublas' in m


def _free():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class Job:
    name: str
    tape: int                 # 잡음 방향 키(보통 shard 번호)
    w0: torch.Tensor          # 시작 모델 [D]
    entry: dict               # {client: 최초 참여 라운드}
    t0: int = 0
    T: int = 160
    regime: dict = field(default_factory=lambda: {'mode': 'none'})
    save_at: tuple = ()       # 이 라운드 시작 직전 모델 저장
    salt: str = ''            # 대조용 다른 난수 tape


def noise_vec(seed, tape, t, salt, D, device):
    g = torch.Generator(device=device).manual_seed(key(seed, tape, t, 'noise', salt))
    return torch.randn(D, generator=g, device=device)


def run_jobs(tr, jobs, h, seed, log_every=0, logger=None):
    """jobs 를 라운드 단위로 함께 진행. 반환: 최종 모델 [J, D], 체크포인트 list[dict], 비용 기록 list[dict]."""
    dev = tr.device
    W = torch.stack([j.w0.to(dev).clone() for j in jobs])
    t_lo = min(j.t0 for j in jobs); t_hi = max(j.T for j in jobs)
    ck = [dict() for _ in jobs]
    rec = [dict(R=0.0, ul_uses=0.0, total_re=0.0, energy=0.0, ul_time=0.0, dl_time=0.0, mse_sum=0.0,
                rounds=0, calls=0, clipped=0) for _ in jobs]
    members = [sorted(j.entry) for j in jobs]
    for t in range(t_lo, t_hi):
        for ji, j in enumerate(jobs):
            if t in j.save_at and j.t0 <= t:
                ck[ji][t] = W[ji].clone()
        pairs = [(ji, i) for ji, j in enumerate(jobs) if j.t0 <= t < j.T for i in members[ji] if j.entry[i] <= t]
        if not pairs:
            continue
        jidx = torch.tensor([p[0] for p in pairs], device=dev)
        clients = [p[1] for p in pairs]; salts = [jobs[p[0]].salt for p in pairs]
        U, nrm = tr.updates(W[jidx], clients, t, salts)
        sums = torch.zeros_like(W).index_add_(0, jidx, U)
        cnt = torch.zeros(len(jobs), device=dev).index_add_(0, jidx, torch.ones(len(pairs), device=dev))
        clip_flag = (nrm > CLIP).float()
        for ji in sorted(set(p[0] for p in pairs)):
            j = jobs[ji]; act = [i for i in members[ji] if j.entry[i] <= t]
            if 'var_fn' in j.regime:   # 무선 제어를 외부에서 정하는 경우 (X2: 공통 scale, MIMO ZF)
                var = float(j.regime['var_fn'](t, act))
                info = dict(R=1.0, ul_uses=0.0, total_re=0.0, energy=float('nan'), ul_time=0.0, dl_time=0.0, mse=var * tr.D)
            else:
                var, info = costmodel.noise_var(len(act), h[t, act], **_rg(j.regime))
            upd = sums[ji] / cnt[ji]
            if var > 0:
                upd = upd + noise_vec(seed, j.tape, t, j.salt, tr.D, dev) * math.sqrt(var)
            W[ji] += upd
            r = rec[ji]
            for k in ('R', 'ul_uses', 'total_re', 'energy', 'ul_time', 'dl_time'):
                if not math.isnan(info[k]):
                    r[k] += info[k]
            r['mse_sum'] += info['mse']; r['rounds'] += 1; r['calls'] += len(act)
        for p, f in zip(pairs, clip_flag.tolist()):
            rec[p[0]]['clipped'] += int(f)
        if log_every and logger and (t + 1) % log_every == 0:
            logger(f'  round {t + 1}/{t_hi}  pairs={len(pairs)}')
    for ji, j in enumerate(jobs):
        if j.T in j.save_at:
            ck[ji][j.T] = W[ji].clone()
        r = rec[ji]; r['mean_mse'] = r['mse_sum'] / max(1, r['rounds'])
    assert torch.isfinite(W).all(), 'non-finite model'
    return W, ck, rec


def _rg(regime):
    return dict(mode=regime['mode'], sigma2=regime.get('sigma2', 0.0), eps=regime.get('eps'),
                bw=regime.get('bw', 1.0), power=regime.get('power', 'symbol'))

# ---------------------------------------------------------------- 평가
def class_metrics(prob, y):
    pred = prob.argmax(1)
    acc = float((pred == y).float().mean())
    rec = [float((pred[y == c] == c).float().mean()) for c in range(10) if (y == c).any()]
    return acc, min(rec)

class Evaluator:
    """모델 묶음의 test/dev 확률을 한 번에 구하고 앙상블 지표를 계산."""
    def __init__(self, tr, data, with_trigger=False):
        self.tr, self.data, self.with_trigger = tr, data, with_trigger
        self.xdev = data['x'][data['dev']]; self.ydev = data['y'][data['dev']]
    def probs(self, W):
        p = {'test': self.tr.predict(W, self.data['tx']), 'dev': self.tr.predict(W, self.xdev)}
        if self.with_trigger:
            p['trig'] = self.tr.predict(W, self.data['trig_x'])
        return p
    def ensemble(self, plist):
        """plist: 모델별 확률 dict 의 리스트 -> 평균 앙상블 지표."""
        out = {}
        pt = torch.stack([p['test'] for p in plist]).mean(0)
        out['test'], out['test_worst_recall'] = class_metrics(pt, self.data['ty'])
        pd = torch.stack([p['dev'] for p in plist]).mean(0)
        out['dev'], _ = class_metrics(pd, self.ydev)
        if self.with_trigger:
            ptr = torch.stack([p['trig'] for p in plist]).mean(0)
            out['asr'] = float((ptr.argmax(1) == 0).float().mean())
        out['_ptest'] = pt
        return out

def disagreement(pa, pb):
    return float((pa.argmax(1) != pb.argmax(1)).float().mean())

def kl(pa, pb):
    """KL(pa || pb) 평균"""
    return float((pa * (pa.clamp_min(1e-12).log() - pb.clamp_min(1e-12).log())).sum(1).mean())

def split_probs(P, i):
    return {k: v[i] for k, v in P.items()}
