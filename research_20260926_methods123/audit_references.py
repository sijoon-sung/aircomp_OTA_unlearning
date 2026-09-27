from common import *
from run_experiments import get_training,statistic_vectors,prune
import json

assert not (RESULTS/'reference_audit.json').exists()
records=[]
for seed in [271,272,273]:
    data=load_data(seed)
    for ortho in [False,True]:
        src,sm=get_training(data,seed,150,ortho=ortho)
        ref,rm=get_training(data,seed,150,ortho=ortho,exclude=True)
        alt,am=get_training(data,seed,150,ortho=ortho,exclude=True,alternate=True)
        x=data['x'][data['forget']]
        q=probabilities(ref,x);q2=probabilities(alt,x);p0=probabilities(src,x)
        base=(float(js(p0,q).mean())+float(js(p0,q2).mean()))/2
        models={'no_op':src}
        if ortho:
            vs,_=statistic_vectors(src,data);models['noiseless_prune']=prune(src,sum(vs))
        else:
            for name in ['curenus_exact','curenus_ota20_r1','curenus_ota20_r4']:
                models[name]=load_model(ROOT/'checkpoints'/f'{name}_seed{seed}.pt')
        rows=[]
        for name,m in models.items():
            p=probabilities(m,x);j1=float(js(p,q).mean());j2=float(js(p,q2).mean())
            # Exact full retained empirical CE and its gradient; evaluator only.
            gsum=torch.zeros_like(flat(m));loss_sum=0.;n=0
            for ix in data['retained'].split(256):
                loss=F.cross_entropy(m(data['x'][ix]),data['y'][ix]);gg=torch.autograd.grad(loss,tuple(m.parameters()))
                gsum+=flatten(gg).detach()*len(ix);loss_sum+=float(loss.detach())*len(ix);n+=len(ix)
            rows.append({'method':name,'js_to_reference1':j1,'js_to_reference2':j2,
                         'mean_js_ratio_to_noop':(j1+j2)/2/max(base,1e-12),
                         'retained_ce':loss_sum/n,'retained_gradient_norm':float(gsum.norm()/n)})
        records.append({'seed':seed,'ortho':ortho,'reference_pairwise_js':float(js(q,q2).mean()),
                        'noop_mean_js_to_references':base,'rows':rows,'alternate_training_seconds':am['wall_seconds']})
        write(RESULTS/'reference_audit_in_progress.json',records)
        log(audit_seed=seed,ortho=ortho,reference_js=records[-1]['reference_pairwise_js'],rows=rows)
write(RESULTS/'reference_audit.json',records)
log(status='reference_audit_complete')
