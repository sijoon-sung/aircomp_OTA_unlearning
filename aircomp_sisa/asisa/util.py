"""난수 키, 입출력, 기록, 보고서 표."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib, json, math, platform, socket, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
import torch

def key(*args):
    """인자들로 정해지는 결정적 정수 시드. 같은 인자면 어느 실험에서든 같은 난수가 나온다."""
    return int.from_bytes(hashlib.sha256('|'.join(map(str, args)).encode()).digest()[:8], 'little') % (2 ** 62)

def setup_torch():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def pick_device(name='auto'):
    return ('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name

def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, torch.Tensor): return o.detach().cpu().tolist()
    raise TypeError(type(o))

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default), encoding='utf-8')

class Logger:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def __call__(self, *msg):
        line = time.strftime('%H:%M:%S ') + ' '.join(str(m) for m in msg)
        print(line, flush=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')

def environment(device):
    info = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
                platform=platform.platform(), host=socket.gethostname(), device=device)
    if device.startswith('cuda'):
        info['gpu'] = torch.cuda.get_device_name(0)
    return info

class PowerMeter:
    """nvidia-smi 가 있으면 5초마다 GPU 보드 전력을 적분한다. 없으면 건너뛴다."""
    def __init__(self, interval=5.0):
        self.interval = interval; self.samples = []; self._stop = threading.Event()
    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=10).stdout
                self.samples.append((time.time(), float(out.strip().splitlines()[0])))
            except Exception:
                pass
            self._stop.wait(self.interval)
    def __enter__(self):
        self.t0 = time.time(); self.th = threading.Thread(target=self._poll, daemon=True); self.th.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self.th.join(timeout=15); self.t1 = time.time()
    def report(self):
        wh = sum((p0 + p1) / 2 * (t1 - t0) / 3600 for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]))
        return dict(seconds=self.t1 - self.t0, gpu_board_Wh=wh if self.samples else None)

# ---------------------------------------------------------------- 보고서 표
def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)): return '-'
    if isinstance(v, (int, np.integer)): return f'{int(v):,}'
    if abs(v) >= 1e6 or (v != 0 and abs(v) < 1e-3): return f'{v:.3e}'
    return f'{v:,.{nd}f}'

def pct(v):
    return '-' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{100 * v:.2f}%'

def pp(v):
    return '-' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{100 * v:+.2f}pp'

def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] +
                     ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])

def select(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]

def mean(rows, field):
    v = [r[field] for r in rows]
    return float(np.mean(v)) if v else float('nan')

def paired(rows, field, base, a, b, keys=('seed',)):
    """base 조건 안에서 a 와 b 를 keys 가 같은 것끼리 짝지은 차이 (a - b). 반환: 평균, 표준오차, 짝 수."""
    A = {tuple(r[k] for k in keys): r[field] for r in select(rows, **base, **a)}
    B = {tuple(r[k] for k in keys): r[field] for r in select(rows, **base, **b)}
    d = [A[s] - B[s] for s in A if s in B]
    if not d:
        return float('nan'), float('nan'), 0
    se = float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float('nan')
    return float(np.mean(d)), se, len(d)

def pm(m, se, f=pct):
    return f'{f(m)} ± {f(2 * se) if not math.isnan(se) else "-"}'
