"""공통 요소: 결정적 난수 키, 데이터 분할, 모델, 채널, shard 배정 규칙, 참여 일정.

이전 실험(research_20261003_sisa_aircomp, aircomp_*_20261003)과 같은 seed 에서 같은 분할·채널이
나오도록 절차를 그대로 옮겼다. 외부 경로 의존성은 없다.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import gzip, hashlib, json, math, platform, socket, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

N_CLIENTS = 20
LR = 0.05
LOCAL_STEPS = 2
BATCH = 64
CLIP = 1.0          # client update L2 clipping C
P_MAX = 1.0         # per-symbol 평균 송신 전력 상한(정규화)
ALPHA = 0.5         # Dirichlet
SLOT_SPACING = 20   # 단계적 참여 간격(160 라운드 기준)
BIN_EDGES_DB = None  # 채널 구간 경계는 K 에서 계산

# ---------------------------------------------------------------- 기본 유틸
def key(*args):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, args)).encode()).digest()[:8], 'little') % (2**62)

def setup_torch():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def pick_device(name='auto'):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_json_default)

def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, torch.Tensor): return o.detach().cpu().tolist()
    raise TypeError(type(o))

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

class Logger:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def __call__(self, *msg):
        line = time.strftime('%H:%M:%S ') + ' '.join(str(m) for m in msg)
        print(line, flush=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')

# ---------------------------------------------------------------- 데이터
IDX = {'x': 'train-images-idx3-ubyte', 'y': 'train-labels-idx1-ubyte',
       'xt': 't10k-images-idx3-ubyte', 'yt': 't10k-labels-idx1-ubyte'}
DATASET_DIR = {'fashionmnist': 'FashionMNIST', 'mnist': 'MNIST'}

def _read_idx(raw, kind):
    p = Path(raw) / IDX[kind]
    b = p.read_bytes() if p.exists() else gzip.open(str(p) + '.gz', 'rb').read()
    if kind in ('x', 'xt'):
        return np.frombuffer(b, dtype=np.uint8, offset=16).copy().reshape(-1, 1, 28, 28)
    return np.frombuffer(b, dtype=np.uint8, offset=8).copy().astype(np.int64)

def ensure_dataset(data_dir, dataset='fashionmnist'):
    """data_dir/FashionMNIST/raw 에 IDX 파일이 없으면 torchvision 으로 내려받는다(약 30MB)."""
    raw = Path(data_dir) / DATASET_DIR[dataset] / 'raw'
    have = all((raw / IDX[k]).exists() or (raw / (IDX[k] + '.gz')).exists() for k in IDX)
    if not have:
        import torchvision
        cls = torchvision.datasets.FashionMNIST if dataset == 'fashionmnist' else torchvision.datasets.MNIST
        cls(str(data_dir), train=True, download=True)
        cls(str(data_dir), train=False, download=True)
    return raw

_RAW_CACHE = {}
def _raw_arrays(raw):
    raw = str(raw)
    if raw not in _RAW_CACHE:
        _RAW_CACHE[raw] = [_read_idx(raw, k) for k in ['x', 'y', 'xt', 'yt']]
    return _RAW_CACHE[raw]

def load_split(raw, seed, device, backdoor_client=None):
    """core.load_data 와 같은 절차: class 별 train 1200 / dev 200, 20 clients Dirichlet(0.5), 최소 60개.
    backdoor_client 가 주어지면 그 client 표본의 80% 에 3x3 트리거를 넣고 label 0 으로 바꾼다."""
    x, y, xt, yt = _raw_arrays(raw)
    rng = np.random.default_rng(seed)
    tr, dev = [], []
    for c in range(10):
        ids = rng.permutation(np.flatnonzero(y == c)); tr.extend(ids[:1200]); dev.extend(ids[1200:1400])
    tr = np.array(tr); dev = np.array(dev)
    for _ in range(10000):
        pools = [[] for _ in range(N_CLIENTS)]
        for c in range(10):
            ids = rng.permutation(tr[y[tr] == c]); f = rng.dirichlet(np.full(N_CLIENTS, ALPHA))
            for i, part in enumerate(np.split(ids, (np.cumsum(f)[:-1] * len(ids)).astype(int))):
                pools[i].extend(part)
        if min(map(len, pools)) >= 60:
            break
    else:
        raise RuntimeError('partition infeasible')
    pools = [np.array(p, dtype=np.int64) for p in pools]
    original = np.concatenate([*pools, dev]); xx = x[original].copy(); yy = y[original].copy()
    offsets = np.cumsum([0] + [len(p) for p in pools])
    localp = [np.arange(offsets[i], offsets[i + 1]) for i in range(N_CLIENTS)]
    devlocal = np.arange(offsets[-1], offsets[-1] + len(dev))
    hist = np.stack([np.bincount(yy[p], minlength=10) for p in localp])  # 평가용(배정에는 쓰지 않음)
    poisoned = np.array([], dtype=np.int64)
    if backdoor_client is not None:
        lp = localp[backdoor_client]
        poisoned = rng.permutation(lp)[:int(.8 * len(lp))]
        xx[poisoned, :, -4:-1, -4:-1] = 255; yy[poisoned] = 0
    def gx(a):
        return torch.tensor(a, device=device, dtype=torch.float32).div_(255).sub_(.5).div_(.5)
    tx = gx(xt); ty = torch.tensor(yt, device=device)
    trig_ids = np.flatnonzero(yt != 0)
    trig = tx[trig_ids].clone(); trig[:, :, -4:-1, -4:-1] = 1.0
    return dict(x=gx(xx), y=torch.tensor(yy, device=device), pools=localp,
                dev=torch.tensor(devlocal, device=device), tx=tx, ty=ty, trig_x=trig,
                hist=hist, poisoned=poisoned, seed=seed, backdoor_client=backdoor_client)

# ---------------------------------------------------------------- 모델 (core.Net 과 동일, 38,282 파라미터)
class DeterministicPool(nn.Module):
    def __init__(self):
        super().__init__()
        for n in [7, 8]:
            a = torch.zeros(4, n)
            for i in range(4):
                lo = math.floor(i * n / 4); hi = math.ceil((i + 1) * n / 4); a[i, lo:hi] = 1 / (hi - lo)
            self.register_buffer('pool' + str(n), a, persistent=False)
    def forward(self, x):
        a = getattr(self, 'pool' + str(x.shape[-1]))
        return torch.matmul(torch.matmul(a, x), a.T)

class Net(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), DeterministicPool(),
                                 nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(self, x):
        return self.net(x)

def init_vector(seed, device):
    """모든 shard 가 같은 초기값을 쓴다(이전 실험의 shared initial vector 와 같은 방식)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(key(seed, 'init') % 2**31)
        m = Net()
    return torch.cat([p.detach().flatten() for p in m.parameters()]).to(device)

D_PARAMS = 38282

# ---------------------------------------------------------------- 채널
def slow_channel(seed):
    rng = np.random.default_rng(key(seed, 'slow_channel'))
    return 10 ** (rng.uniform(-20, 0, N_CLIENTS) / 20)

def channels(seed, T):
    """장기 진폭 10^(U[-20,0]/20) x 라운드별 U[.85,1.15]. 이전 Engine 과 같은 키."""
    base = slow_channel(seed)
    fad = np.stack([np.random.default_rng(key(seed, t, 'fading')).uniform(.85, 1.15, N_CLIENTS) for t in range(T)])
    return base, base[None, :] * fad

# ---------------------------------------------------------------- shard 배정 규칙
RULES = ['random_split', 'rank_chunk', 'rank_rr', 'bins', 'hash']
RULE_KO = {'random_split': '무작위 순열 후 자르기', 'rank_chunk': '채널 순위로 정렬 후 자르기',
           'rank_rr': '채널 순위 돌려 담기', 'bins': '채널 고정 구간', 'hash': 'ID 해시 무작위'}

def partition(rule, ids, base_h, K, seed):
    """ids(현재 client 집합)를 K 개 shard 로 나눈다. 반환: 길이 K 의 리스트(빈 shard 허용).
    stable 규칙(bins, hash)은 client 하나의 소속이 자기 정보로만 정해진다."""
    ids = sorted(int(i) for i in ids)
    if rule == 'random_split':
        order = list(np.random.default_rng(key(seed, 'routing')).permutation(ids))
        return [sorted(map(int, g)) for g in np.array_split(order, K)]
    if rule == 'rank_chunk':   # 약한 채널 shard 가 0 번
        order = sorted(ids, key=lambda i: (base_h[i], i))
        return [sorted(map(int, g)) for g in np.array_split(order, K)]
    if rule == 'rank_rr':      # 채널 순위대로 0,1,..,K-1,0,1,.. 에 담아 shard 간 채널을 고르게
        order = sorted(ids, key=lambda i: (base_h[i], i))
        groups = [[] for _ in range(K)]
        for r, i in enumerate(order):
            groups[r % K].append(i)
        return [sorted(g) for g in groups]
    if rule == 'bins':         # 채널 분포 U[-20,0] dB 의 등확률 구간(공개 상수)
        groups = [[] for _ in range(K)]
        for i in ids:
            db = 20 * math.log10(base_h[i])
            groups[min(K - 1, max(0, int((db + 20) // (20 / K))))].append(i)
        return groups
    if rule == 'hash':
        groups = [[] for _ in range(K)]
        for i in ids:
            groups[key(seed, i, 'hash') % K].append(i)
        return groups
    raise ValueError(rule)

def shard_of(groups, u):
    return next(c for c, g in enumerate(groups) if u in g)

# ---------------------------------------------------------------- 참여 일정
ORDERS = ['all0', 'random', 'strong', 'weak', 'stable_thr']
ORDER_KO = {'all0': '전원 동시 참여', 'random': '무작위 순서', 'strong': '강한 채널 먼저',
            'weak': '약한 채널 먼저(반대 대조)', 'stable_thr': '자기 채널 구간으로 시점 결정'}

def slot_spacing(T, spacing=SLOT_SPACING):
    return max(1, round(spacing * T / 160))

def entry_schedule(group, base_h, order, seed, c, T, spacing=SLOT_SPACING):
    """shard 하나의 {client: 최초 참여 라운드}. 순위 기반은 [0,0,s,2s,3s,..]."""
    group = list(group); sp = slot_spacing(T, spacing)
    if order == 'all0':
        return {i: 0 for i in group}
    if order == 'stable_thr':  # -5dB 이상 0, -10 이상 s, -15 이상 2s, 그 밑 3s
        out = {}
        for i in group:
            db = 20 * math.log10(base_h[i])
            out[i] = 0 if db >= -5 else sp if db >= -10 else 2 * sp if db >= -15 else 3 * sp
        return out
    if order == 'random':
        seq = list(np.random.default_rng(key(seed, 'entry', c)).permutation(group))
    elif order == 'strong':
        seq = sorted(group, key=lambda i: (-base_h[i], i))
    elif order == 'weak':
        seq = sorted(group, key=lambda i: (base_h[i], i))
    else:
        raise ValueError(order)
    return {int(i): (0 if k < 2 else sp * (k - 1)) for k, i in enumerate(seq)}

# ---------------------------------------------------------------- 실행 환경 기록·전력 측정
def environment(device):
    info = {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
            'platform': platform.platform(), 'host': socket.gethostname(), 'device': device}
    if device.startswith('cuda'):
        info['gpu'] = torch.cuda.get_device_name(0)
    return info

class PowerMeter:
    """nvidia-smi 가 있으면 5초마다 GPU 보드 전력을 적분한다. 없으면 조용히 건너뛴다."""
    def __init__(self, interval=5.0):
        self.interval = interval; self.samples = []; self._stop = threading.Event(); self.ok = False
    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=10).stdout
                self.samples.append((time.time(), float(out.strip().splitlines()[0])))
                self.ok = True
            except Exception:
                pass
            self._stop.wait(self.interval)
    def __enter__(self):
        self.t0 = time.time(); self.th = threading.Thread(target=self._poll, daemon=True); self.th.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self.th.join(timeout=15); self.t1 = time.time()
    def report(self):
        wh = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            wh += (p0 + p1) / 2 * (t1 - t0) / 3600
        return {'seconds': self.t1 - self.t0, 'gpu_board_Wh': wh if self.ok else None,
                'power_samples': len(self.samples)}
