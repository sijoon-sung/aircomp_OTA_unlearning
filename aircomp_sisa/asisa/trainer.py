"""로컬 학습과 평가.

여러 shard 모델의 로컬 SGD 를 torch.func.vmap 으로 한 번에 계산한다.
minibatch 는 (seed, client, 라운드, step, salt) 로 정해지므로, salt 가 같으면 어느 조건에서든 같은 배치를 쓴다.
"""
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from .config import LR, LOCAL_STEPS, BATCH, CLIP
from .model import Net
from .util import key

def _recoverable(ex):
    m = str(ex).lower()
    return 'out of memory' in m or 'cudnn' in m or 'cublas' in m

def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class LocalTrainer:
    def __init__(self, data, seed, device, chunk=64):
        self.data, self.seed, self.device, self.chunk = data, seed, device, chunk
        self.steps = LOCAL_STEPS          # 라운드당 로컬 step (sharding 실험이 잠시 바꿈)
        net = Net().to(device)
        self.names = [n for n, _ in net.named_parameters()]
        self.shapes = [p.shape for _, p in net.named_parameters()]
        self.numels = [p.numel() for _, p in net.named_parameters()]
        self.D = sum(self.numels)
        self._batches = {}
        self._vgrad = vmap(grad(lambda p, x, y: F.cross_entropy(functional_call(net, p, (x,)), y)))
        self._vfwd = vmap(lambda p, x: functional_call(net, p, (x,)).softmax(-1), in_dims=(0, None))

    def unflat(self, W):
        out, off = {}, 0
        for n, s, k in zip(self.names, self.shapes, self.numels):
            out[n] = W[:, off:off + k].reshape(W.shape[0], *s); off += k
        return out

    def flat(self, g):
        return torch.cat([g[n].reshape(g[n].shape[0], -1) for n in self.names], 1)

    def batch(self, i, t, step, salt):
        k = (i, t, step, salt)
        if k not in self._batches:
            pool = self.data['pools'][i]
            self._batches[k] = pool[np.random.default_rng(key(self.seed, i, t, step, 'batch', salt)).integers(len(pool), size=BATCH)]
            if len(self._batches) > 200000:
                self._batches.clear()
        return self._batches[k]

    def updates(self, W, clients, t, salts):
        """W: [P, D] 각 (모델, client) 쌍의 시작 모델. 반환: L2 <= CLIP 로 자른 update [P, D].
        GPU 메모리 부족이나 cuDNN 오류가 나면 chunk 를 절반으로 줄여 다시 계산한다 (결과는 chunk 와 무관)."""
        while True:
            try:
                return self._updates(W, clients, t, salts)
            except RuntimeError as ex:
                if not _recoverable(ex) or self.chunk <= 4:
                    raise
                self.chunk //= 2; _free()
                print(f'[경고] GPU 메모리/cuDNN 오류로 chunk 를 {self.chunk} 로 줄입니다: {str(ex)[:80]}', flush=True)

    def _updates(self, W, clients, t, salts):
        out = torch.empty_like(W)
        for s in range(0, W.shape[0], self.chunk):
            e = min(W.shape[0], s + self.chunk); p = W[s:e]
            for step in range(self.steps):
                idx = torch.as_tensor(np.stack([self.batch(clients[j], t, step, salts[j]) for j in range(s, e)]), device=W.device)
                p = p - LR * self.flat(self._vgrad(self.unflat(p), self.data['x'][idx], self.data['y'][idx]))
            u = p - W[s:e]
            out[s:e] = u * torch.clamp(CLIP / u.norm(dim=1).clamp_min(1e-30), max=1.0)[:, None]
        return out

    @torch.no_grad()
    def predict(self, W, x, chunk_models=16, chunk_x=500):
        """W: [J, D] -> 확률 [J, len(x), 10]"""
        while True:
            try:
                return torch.cat([torch.cat([self._vfwd(self.unflat(W[s:s + chunk_models]), xb) for xb in x.split(chunk_x)], 1)
                                  for s in range(0, W.shape[0], chunk_models)], 0)
            except RuntimeError as ex:
                if not _recoverable(ex) or chunk_models <= 1:
                    raise
                chunk_models = max(1, chunk_models // 2); chunk_x = max(50, chunk_x // 2); _free()

class Evaluator:
    """모델 묶음의 test/dev 확률과 앙상블(확률 평균) 정확도."""
    def __init__(self, tr, data):
        self.tr, self.data = tr, data
        self.xdev = data['x'][data['dev']]; self.ydev = data['y'][data['dev']]

    def probs(self, W):
        return {'test': self.tr.predict(W, self.data['tx']), 'dev': self.tr.predict(W, self.xdev)}

    def pick(self, P, j):
        return {k: v[j] for k, v in P.items()}

    def acc(self, p):
        return float((p['test'].argmax(1) == self.data['ty']).float().mean())

    def ensemble(self, plist):
        pt = torch.stack([p['test'] for p in plist]).mean(0)
        pd = torch.stack([p['dev'] for p in plist]).mean(0)
        return dict(test=float((pt.argmax(1) == self.data['ty']).float().mean()), dev=float((pd.argmax(1) == self.ydev).float().mean()), ptest=pt)

def disagreement(pa, pb):
    """두 모델의 예측 label 이 다른 test 표본 비율."""
    return float((pa.argmax(1) != pb.argmax(1)).float().mean())
