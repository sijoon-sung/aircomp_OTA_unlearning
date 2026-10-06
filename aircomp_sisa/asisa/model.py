"""모델: 작은 CNN (파라미터 38,282개). 이전 실험과 같은 구조·초기값."""
import math
import torch
from torch import nn
from .util import key

class DeterministicPool(nn.Module):
    """7x7, 8x8 -> 4x4 평균 풀링을 행렬곱으로 (GPU 비결정성 회피)."""
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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), DeterministicPool(),
                                 nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(self, x):
        return self.net(x)

def init_vector(seed, device):
    """모든 shard 가 같은 초기 파라미터 벡터에서 시작한다."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(key(seed, 'init') % 2 ** 31)
        m = Net()
    return torch.cat([p.detach().flatten() for p in m.parameters()]).to(device)
