"""Exact channel-partitioned convolution and backward, without a low-rank sketch."""
from pathlib import Path
import json
import torch
import torch.nn.functional as F

torch.manual_seed(1928)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
dev, dt = 'cuda', torch.float64
x = torch.randn(3, 8, 9, 9, device=dev, dtype=dt, requires_grad=True)
w = torch.randn(12, 8, 3, 3, device=dev, dtype=dt, requires_grad=True)
b = torch.randn(12, device=dev, dtype=dt, requires_grad=True)
full = F.relu(F.conv2d(x, w, b, padding=1))
partials = [F.conv2d(x[:, 2*k:2*k+2], w[:, 2*k:2*k+2], padding=1) for k in range(4)]
summed = F.relu(sum(partials)+b[None, :, None, None])
probe = torch.randn_like(full)
g_full = torch.autograd.grad((full*probe).sum(), [x,w,b], retain_graph=True)
g_sum = torch.autograd.grad((summed*probe).sum(), [x,w,b])
forward_error = float((full-summed).detach().abs().max())
backward_error = max(float((a-c).abs().max()) for a,c in zip(g_full,g_sum))
assert forward_error < 1e-11 and backward_error < 1e-11
# Two different channel assignments have the same unweighted sum but distinct weighted output.
lost_identity = dict(channel_features=[[1.,1.],[2.,0.]], raw_sums=[2.,2.],
                     weights=[1.,2.], correct_outputs=[3.,2.])
obj = dict(passed=True, gpu=torch.cuda.get_device_name(0), forward_max_error=forward_error,
           backward_max_error=backward_error, input_channels=8, output_channels=12, devices=4,
           scope='one dense convolution partitioned across input channels; exact CSI/no noise; activation after sum',
           nonlinear_order_example=dict(partials=[1.,-1.], relu_after_sum=0., sum_of_relus=1.),
           raw_channel_sum_counterexample=lost_identity,
           limitation='Does not assert arbitrary independently nonlinear submodels sum to the original dense network.')
Path(__file__).with_name('channel_sum_validation.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
print(json.dumps(obj))
