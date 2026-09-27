"""Independent mathematical checks before any model training."""
import contextlib, io, math, hashlib
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,DEVICE,write,rng,ota,Cost
from vendor_adapters import orth_dist,fedquit_teacher,kd_loss,official_subsolver

def run():
    assert torch.cuda.is_available()
    torch.manual_seed(270)
    dtype=torch.float64;out={'device':torch.cuda.get_device_name(0),'precision':'float64','training_started':False}
    x=torch.randn(17,3,device=DEVICE,dtype=dtype);y=torch.arange(17,device=DEVICE)%2
    w=torch.randn(8,device=DEVICE,dtype=dtype,requires_grad=True)*.2
    def objective(t,ix):return F.cross_entropy(torch.tanh(x[ix]@t[:6].reshape(3,2)+t[6:]),y[ix])
    ids=[torch.arange(0,5,device=DEVICE),torch.arange(5,17,device=DEVICE)]
    allids=torch.arange(17,device=DEVICE);weights=[5/17,12/17]
    v=torch.randn(8,device=DEVICE,dtype=dtype)
    grad=lambda t,ix:torch.autograd.functional.jacobian(lambda a:objective(a,ix),t)
    localg=sum(p*grad(w,ix) for p,ix in zip(weights,ids))
    globalg=grad(w,allids)
    hv=sum(p*torch.autograd.functional.hvp(lambda t:objective(t,ix),w,v)[1] for p,ix in zip(weights,ids))
    h=torch.autograd.functional.hessian(lambda t:objective(t,allids),w)
    fd=(grad(w+1e-5*v,allids)-grad(w-1e-5*v,allids))/(2e-5)
    out['hvp']={'gradient_aggregation_max_error':float((localg-globalg).abs().max()),
                'hvp_dense_max_error':float((hv-h@v).abs().max()),'hvp_fd_max_error':float((hv-fd).abs().max())}
    assert max(out['hvp'].values())<1e-8

    a=torch.randn(5,5,device=DEVICE,dtype=dtype);h=a.T@a+.3*torch.eye(5,device=DEVICE,dtype=dtype)
    b=torch.randn(5,device=DEVICE,dtype=dtype);w0=torch.randn_like(b)
    wr=torch.linalg.solve(h,b);corrected=w0-torch.linalg.solve(h,h@w0-b)
    out['quadratic_exact_error']=float((wr-corrected).abs().max());assert out['quadratic_exact_error']<1e-12
    s=torch.randn(5,device=DEVICE,dtype=dtype,requires_grad=True);g=torch.randn_like(s);M=5.
    loss=g@s+.5*s@h@s+M*s.norm()**3/6
    actual=torch.autograd.grad(loss,s)[0];expected=g+h@s+M/2*s.norm()*s
    out['cubic_coefficient_error']=float((actual-expected).abs().max());assert out['cubic_coefficient_error']<1e-12
    out['paper_code_note']='M||delta||^3/6 differentiates to (M/2)||delta||delta; official helper uses M/2. Pseudocode line uses L, so code convention retained.'
    # Execute the actual downloaded subsolver on a convex problem, without its import tree.
    param=torch.nn.Parameter(torch.zeros(5,device=DEVICE,dtype=dtype));gg=torch.ones_like(param)*.1
    cb=lambda ts:(None,(h@ts[0],))
    np.random.seed(270)
    with contextlib.redirect_stdout(io.StringIO()):
        step=official_subsolver(None,(param,),cb,(gg,),M=5.,num_steps=5,learning_rate=.01,sigma=0.,device=DEVICE)
    step=torch.as_tensor(step,device=DEVICE)
    q=float(gg@step+.5*step@h@step+5/6*step.norm()**3)
    out['official_cubic_step_model_decrease']=q;assert q<0 and torch.isfinite(step).all()

    kernel=torch.randn(8,1,3,3,device=DEVICE,dtype=dtype)
    matrix=kernel.flatten(1);ours=(matrix@matrix.T-torch.eye(8,device=DEVICE)).square().sum()
    upstream=orth_dist(kernel).square()
    out['orthogonal_regularizer_difference']=float((ours-upstream).abs());assert torch.allclose(ours,upstream,atol=1e-10)
    out['orthogonal_feature_counterexample']={'kernel_inner_product':0.,'feature_covariance':.8}
    vals=[torch.tensor([[1.,2.],[3.,1.]],device=DEVICE,dtype=dtype),torch.tensor([[9.,4.]],device=DEVICE,dtype=dtype)]
    weighted=sum(z.sum(0) for z in vals)/3
    out['unequal_count_statistic_error']=float((weighted-torch.cat(vals).mean(0)).abs().max())
    out['unweighted_local_means_bias']=float((sum(z.mean(0) for z in vals)/2-weighted).norm())
    assert out['unequal_count_statistic_error']<1e-12 and out['unweighted_local_means_bias']>0

    logits=torch.randn(9,10,device=DEVICE,dtype=dtype);labels=torch.arange(9,device=DEVICE)
    target=fedquit_teacher(logits,labels)
    z=logits.cpu().numpy().copy();z[np.arange(9),np.arange(9)]=z.min(1)
    e=np.exp(z-z.max(1,keepdims=True));p=e/e.sum(1,keepdims=True)
    out['fedquit_numpy_port_error']=float((target-torch.tensor(p,device=DEVICE)).abs().max())
    student=logits.detach().clone().requires_grad_()
    kg=torch.autograd.grad(kd_loss(student,target),student)[0]
    out['kl_gradient_error']=float((kg-(student.softmax(1)-target)/9).abs().max())
    orig=logits.softmax(1);zero_student=logits.detach().clone().requires_grad_()
    zero=torch.autograd.grad(kd_loss(zero_student,orig),zero_student)[0]
    out['original_teacher_retain_kd_grad_norm']=float(zero.norm())
    out['privacy_counterexample']='At student=old teacher all retained KD gradients are zero; the mixed aggregate is only the target gradient times its known coefficient.'
    assert max(out['fedquit_numpy_port_error'],out['kl_gradient_error'],out['original_teacher_retain_kd_grad_norm'])<1e-12

    vs=[torch.randn(100,device=DEVICE,dtype=dtype) for _ in range(3)];weights=[.1,.3,.6]
    received=ota(vs,weights,[0,1,2],None,None)
    exact=sum(w*t for w,t in zip(weights,vs))
    out['mac_noiseless_max_error']=float((received-exact).abs().max());assert out['mac_noiseless_max_error']<1e-12
    out['independent_teacher_condition']='With target-independent initialization, per-client random streams and no teacher feedback, excluding teacher A leaves exactly the target-free KD objective. Student optimization must restart independently of A.'
    out['passed']=True
    write(ROOT/'results/math_validation.json',out)
    print(__import__('json').dumps(out,indent=2))

if __name__=='__main__':run()
