"""Reviewed upstream functions, loaded without running upstream setup/import trees.

CuReNU: official ICLR 2026 supplement; Orthogonal CNN: pinned official GitHub.
FedQUIT below is an explicitly labelled PyTorch port, not upstream TensorFlow execution.
"""
from pathlib import Path
import ast, types
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
CURE=ROOT/'vendor/CuReNU_supplement/code'

def selected(path, names, namespace):
    source=path.read_text(encoding='utf-8')
    tree=ast.parse(source,filename=str(path))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names), (path,names)
    code=compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec')
    exec(code,namespace)
    return namespace

base={'torch':torch,'nn':nn,'F':F,'np':np,'partial':partial}
ConvNet2=selected(CURE/'settings/models/convnet.py',['ConvNet2'],dict(base))['ConvNet2']
orth_dist=selected(ROOT/'vendor/Orthogonal-Convolutional-Neural-Networks/imagenet/utils.py',
                   ['orth_dist'],dict(base))['orth_dist']
cubic_ns=selected(CURE/'helper/cubic_func.py',
    ['compose_param_vector','decompose_param_vector','compute_loss','gradient'],dict(base))
cubic=types.SimpleNamespace(**{k:v for k,v in cubic_ns.items() if callable(v)})
utilities=types.SimpleNamespace(clear_cache=lambda:None,
                               convert_torch_to_numpy=lambda x:x.detach().cpu().numpy())
sto_ns=selected(CURE/'helper/sto_cubic_func.py',['gd_cubic_subsolver','hvp','hvp_func'],
               dict(base,cubic_func=cubic,utils=utilities))
official_subsolver=sto_ns['gd_cubic_subsolver']
official_hvp_func=sto_ns['hvp_func']

def fedquit_teacher(logits, labels, temperature=1.):
    """Port ModelFedQuitLogitDynamic: dynamic_v=True, dynamic_type='min'."""
    z=logits.detach().clone()
    z[torch.arange(len(z),device=z.device),labels]=z.min(dim=1).values
    return F.softmax(z/temperature,dim=1)

def kd_loss(logits, target, temperature=1.):
    return F.kl_div(F.log_softmax(logits/temperature,dim=1),target,
                    reduction='batchmean')*temperature**2
