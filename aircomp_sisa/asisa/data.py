"""데이터: FashionMNIST(또는 MNIST)를 20 client 에 Dirichlet 비율로 나눈다.

이전 실험(sisa_arch_20261006, aircomp_sim_20261007)과 같은 seed 에서 같은 분할이 나오는 절차다.
"""
import gzip
from pathlib import Path
import numpy as np
import torch
from .config import N_CLIENTS, DIRICHLET, MIN_SAMPLES

IDX = {'x': 'train-images-idx3-ubyte', 'y': 'train-labels-idx1-ubyte', 'xt': 't10k-images-idx3-ubyte', 'yt': 't10k-labels-idx1-ubyte'}
FOLDER = {'fashionmnist': 'FashionMNIST', 'mnist': 'MNIST'}

def ensure_dataset(data_dir, dataset='fashionmnist'):
    """data_dir/<FOLDER>/raw 에 IDX 파일이 없으면 torchvision 으로 내려받는다(약 30MB). 반환: raw 폴더."""
    raw = Path(data_dir) / FOLDER[dataset] / 'raw'
    if not all((raw / f).exists() or (raw / (f + '.gz')).exists() for f in IDX.values()):
        import torchvision
        cls = torchvision.datasets.FashionMNIST if dataset == 'fashionmnist' else torchvision.datasets.MNIST
        cls(str(data_dir), train=True, download=True); cls(str(data_dir), train=False, download=True)
    return raw

def _read(raw, kind):
    p = Path(raw) / IDX[kind]
    b = p.read_bytes() if p.exists() else gzip.open(str(p) + '.gz', 'rb').read()
    if kind in ('x', 'xt'):
        return np.frombuffer(b, dtype=np.uint8, offset=16).copy().reshape(-1, 1, 28, 28)
    return np.frombuffer(b, dtype=np.uint8, offset=8).copy().astype(np.int64)

_CACHE = {}
def _arrays(raw):
    if str(raw) not in _CACHE:
        _CACHE[str(raw)] = [_read(raw, k) for k in ['x', 'y', 'xt', 'yt']]
    return _CACHE[str(raw)]

def load_split(raw, seed, device):
    """class 별 train 1200 / dev 200 을 뽑고, train 을 client 20명에게 class 마다 Dirichlet(0.5) 비율로 나눈다.
    반환 dict: x, y (client 표본 + dev), pools (client 별 인덱스), dev, tx, ty (test), hist (client 별 label 수, 평가용)."""
    x, y, xt, yt = _arrays(raw)
    rng = np.random.default_rng(seed)
    tr, dev = [], []
    for c in range(10):
        ids = rng.permutation(np.flatnonzero(y == c)); tr.extend(ids[:1200]); dev.extend(ids[1200:1400])
    tr = np.array(tr); dev = np.array(dev)
    for _ in range(10000):
        pools = [[] for _ in range(N_CLIENTS)]
        for c in range(10):
            ids = rng.permutation(tr[y[tr] == c]); f = rng.dirichlet(np.full(N_CLIENTS, DIRICHLET))
            for i, part in enumerate(np.split(ids, (np.cumsum(f)[:-1] * len(ids)).astype(int))):
                pools[i].extend(part)
        if min(map(len, pools)) >= MIN_SAMPLES:
            break
    else:
        raise RuntimeError('분할 실패')
    pools = [np.array(p, dtype=np.int64) for p in pools]
    order = np.concatenate([*pools, dev]); xx = x[order]; yy = y[order]
    off = np.cumsum([0] + [len(p) for p in pools])
    local = [np.arange(off[i], off[i + 1]) for i in range(N_CLIENTS)]
    gx = lambda a: torch.tensor(a, device=device, dtype=torch.float32).div_(255).sub_(.5).div_(.5)
    return dict(x=gx(xx), y=torch.tensor(yy, device=device), pools=local,
                dev=torch.tensor(np.arange(off[-1], off[-1] + len(dev)), device=device),
                tx=gx(xt), ty=torch.tensor(yt, device=device),
                hist=np.stack([np.bincount(yy[p], minlength=10) for p in local]), seed=seed)
