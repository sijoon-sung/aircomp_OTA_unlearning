"""Bounded, sequential CUDA client-unlearning feasibility experiments."""
import argparse, contextlib, io, json, time, math, hashlib
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from vendor_adapters import official_subsolver, official_hvp_func, fedquit_teacher, kd_loss

CHECKPOINTS=ROOT/'checkpoints'; CHECKPOINTS.mkdir(exist_ok=True)
SEEDS=[271,272,273]

def record_result(path,rows):write(path,rows)

def get_training(data,seed,rounds,ortho=False,exclude=False,alternate=False):
    tag=f'seed{seed}_'+('ortho' if ortho else 'plain')+('_retain' if exclude else '_all')+('_alternate' if alternate else '')
    cp=CHECKPOINTS/(tag+'.pt');meta=RESULTS/(tag+'_training.json')
    if cp.exists() and meta.exists():return load_model(cp),json.loads(meta.read_text())
    m,cost,seconds=train_fl(data,seed,rounds,ortho,exclude,sampling_offset=500000 if alternate else 0)
    info={'tag':tag,'seed':seed,'rounds':rounds,'ortho':ortho,'exclude_client0':exclude,
          'cost':cost,'wall_seconds':seconds,'validation_accuracy':accuracy(m,data['x'][data['val']],data['y'][data['val']]),
          'parameter_sha256':vec_hash(flat(m)),'parameters':flat(m).numel()}
    save_model(cp,m);write(meta,info);return m,info

def grads(model,bx,by,create=False):
    return torch.autograd.grad(F.cross_entropy(model(bx),by),tuple(model.parameters()),create_graph=create)

def hvp_local(model,bx,by,v):
    grad=grads(model,bx,by,create=True)
    product=sum((g*vv).sum() for g,vv in zip(grad,split_params(model,v)))
    return flatten(torch.autograd.grad(product,tuple(model.parameters()))).detach()

def run_hvp(source,data,seed,snr,repeats):
    model=clone_model(source);model.eval();active=list(range(1,10));cost=Cost(20 if snr is None else snr)
    ng=rng(seed+500);streams={i:rng(seed*5000+i) for i in active};traces=[]
    start=time.perf_counter();dim=flat(model).numel()
    for outer in range(10):
        cost.digital(dim*32)
        gb=[batch(data,i,streams[i],16) for i in active]
        hb=[batch(data,i,streams[i],8) for i in active]
        gg=ota([flatten(grads(model,*b)).detach() for b in gb],[1/9]*9,active,snr,ng,repeats,cost)
        cost.c['client_gradient_samples']+=144
        calls=0
        def aggregate_hvp(tuples):
            nonlocal calls
            v=flatten(tuples).to(DEVICE,dtype=torch.float32);cost.digital(dim*32)
            result=ota([hvp_local(model,*b,v) for b in hb],[1/9]*9,active,snr,ng,repeats,cost)
            cost.c['client_hvp_samples']+=72;calls+=1
            return None,split_params(model,result)
        np.random.seed(seed*100+outer)
        # Actual downloaded official solver; stdout retained as a local provenance log.
        captured=io.StringIO()
        with contextlib.redirect_stdout(captured):
            step=official_subsolver(model,tuple(model.parameters()),aggregate_hvp,split_params(model,gg),
                M=5.,num_steps=5,learning_rate=.01,sigma=20.,device=DEVICE)
        step=torch.as_tensor(step,device=DEVICE,dtype=torch.float32)
        assert torch.isfinite(step).all(),'Cubic step diverged'
        assign(model,flat(model)+step)
        traces.append({'outer':outer,'gradient_norm':float(gg.norm()),'step_norm':float(step.norm()),'hvp_calls':calls,
                       'official_solver_log':captured.getvalue()})
    cost.digital(dim*32);sync()
    return model,cost.report(),time.perf_counter()-start,traces

def retained_sgd(source,data,seed):
    model=clone_model(source);active=list(range(1,10));cost=Cost();streams={i:rng(seed*5000+i) for i in active}
    start=time.perf_counter();d=flat(model).numel()
    for step in range(70):
        cost.digital(d*32)
        gs=[flatten(grads(model,*batch(data,i,streams[i],16))).detach() for i in active]
        avg=ota(gs,[1/9]*9,active,None,None,cost=cost)
        assign(model,flat(model)-.01*avg);cost.c['client_gradient_samples']+=144
    cost.digital(d*32);sync()
    return model,cost.report(),time.perf_counter()-start

def stage1(seed,rounds):
    path=RESULTS/f'method1_seed{seed}.json'
    if path.exists() and json.loads(path.read_text()).get('complete'):return
    data=load_data(seed);source,training=get_training(data,seed,rounds);ref,refinfo=get_training(data,seed,rounds,exclude=True)
    base,sp=evaluate(source,ref,data);rows=[{'method':'no_op','metrics':base}]
    ref_metrics,_=evaluate(ref,ref,data,sp);rows.append({'method':'target_free_retrain','metrics':ref_metrics,'cost':refinfo['cost'],'wall_seconds':refinfo['wall_seconds']})
    if seed==271:
        alt,altmeta=get_training(data,seed,rounds,exclude=True,alternate=True)
        metrics,_=evaluate(alt,ref,data,sp);rows.append({'method':'independent_sampling_retrain','metrics':metrics,'cost':altmeta['cost']})
    cases=[('curenus_exact',None,1),('curenus_ota20_r1',20,1),('curenus_ota20_r4',20,4),('curenus_ota30_r1',30,1)]
    for name,snr,rep in cases:
        model,cost,seconds,traces=run_hvp(source,data,seed,snr,rep)
        metrics,_=evaluate(model,ref,data,sp)
        row={'method':name,'metrics':metrics,'cost':cost,'wall_seconds':seconds,'trace':traces}
        rows.append(row);save_model(CHECKPOINTS/f'{name}_seed{seed}.pt',model)
        write(path,{'seed':seed,'training':training,'rows':rows,'complete':False})
        log(stage=1,seed=seed,method=name,forget_js_ratio=metrics['forget_normalized_js'],test_accuracy=metrics['test_accuracy'],seconds=seconds)
    model,cost,seconds=retained_sgd(source,data,seed);metrics,_=evaluate(model,ref,data,sp)
    rows.append({'method':'retained_sgd70','metrics':metrics,'cost':cost,'wall_seconds':seconds})
    write(path,{'seed':seed,'training':training,'reference_training':refinfo,'rows':rows,'complete':True})
    log(stage=1,seed=seed,status='complete')

@torch.no_grad()
def activations(model,x):
    a=F.relu(model.conv1(x));b=F.relu(model.conv2(model.pool(a)))
    return torch.cat([a.amax((2,3)),b.amax((2,3))],dim=1)

@torch.no_grad()
def statistic_vectors(model,data):
    vectors=[];moments=[]
    for i,ix in enumerate(data['pools']):
        a=torch.cat([activations(model,xx) for xx in data['x'][ix].split(128)])
        moments.append(a)
        vectors.append(a.mean(0)*(1 if i==0 else -1/9))
    all_a=torch.cat(moments);z=all_a-all_a.mean(0)
    cov=z.T@z/(len(z)-1);std=cov.diag().sqrt().clamp_min(1e-8)
    corr=cov/(std[:,None]*std[None,:]);off=corr[~torch.eye(24,device=DEVICE,dtype=torch.bool)]
    return vectors,{'feature_abs_correlation':float(off.abs().mean()),'kernel_gram_error':float(ortho_loss(model).detach())}

def pruning_indices(score):
    return torch.cat([score[:8].topk(2).indices,8+score[8:].topk(4).indices])

def boundary_indices(score):
    out=[]
    for start,end,k in [(0,8,2),(8,24,4)]:
        v=score[start:end];order=v.sort(descending=True).values;boundary=(order[k-1]+order[k])/2
        out.append(start+(v-boundary).abs().topk(k,largest=False).indices)
    return torch.cat(out)

def prune(model,score):
    result=clone_model(model)
    with torch.no_grad():
        for layer,s,k in [(result.conv1,score[:8],2),(result.conv2,score[8:],4)]:
            indices=s.topk(k).indices
            strength=torch.maximum(torch.full((k,),.5,device=DEVICE),1-torch.arange(1,k+1,device=DEVICE)/k)
            layer.weight[indices]*=(1-strength)[:,None,None,None]
            layer.bias[indices]*=(1-strength)
    return result

def mask_cost_start(snr):
    cost=Cost(snr);cost.c['metadata_bits']+=32*10 # local counts
    cost.c['activation_forward_samples']=12000
    return cost

def stat_observation(vectors,seed,snr,mode,matched_cost=None):
    g=rng(seed);clients=list(range(10));cost=mask_cost_start(snr)
    if mode=='boundary':
        first=ota(vectors,[1]*10,clients,snr,g,4,cost)
        idx=boundary_indices(first);cost.digital(6*5)
        second=ota([v[idx] for v in vectors],[1]*10,clients,snr,g,16,cost)
        # Coordinate subset uses its own admissible normalization; combine by inverse noise variance.
        h=torch.tensor(CHANNELS,device=DEVICE)
        scale1=torch.stack([v.square().mean().sqrt()/h[i] for i,v in enumerate(vectors)]).max()
        scale2=torch.stack([v[idx].square().mean().sqrt()/h[i] for i,v in enumerate(vectors)]).max()
        v1=scale1.square()/4;v2=scale2.square()/16
        score=first.clone();score[idx]=(v2*first[idx]+v1*second)/(v1+v2).clamp_min(1e-20)
    else:
        repeats=8
        if mode=='matched':
            # Final mask + one packet control; spend the remainder on uniform integer repeats.
            base=mask_cost_start(snr);base.packet(24,10,0);base.digital(6*(5+32))
            repeats=max(1,math.floor((matched_cost-base.total())/(1.125*24)))
        score=ota(vectors,[1]*10,clients,snr,g,repeats,cost)
    cost.digital(6*(5+32)) # selected coordinates and strengths
    return score,cost

def stage2(seed,rounds):
    path=RESULTS/f'method2_seed{seed}.json'
    if path.exists() and json.loads(path.read_text()).get('complete'):return
    data=load_data(seed);source,training=get_training(data,seed,rounds,ortho=True)
    ref,refinfo=get_training(data,seed,rounds,ortho=True,exclude=True)
    baseline,sp=evaluate(source,ref,data);vectors,diagnostic=statistic_vectors(source,data)
    exact=sum(vectors);truth=pruning_indices(exact);ideal=prune(source,exact)
    idealp=probabilities(ideal,data['x'][data['forget']])
    cost=mask_cost_start(20);cost.packet(24,10,1);cost.digital(6*37)
    im,_=evaluate(ideal,ref,data,sp)
    rows=[{'method':'no_op','metrics':baseline},{'method':'ortho_noiseless_prune','metrics':im,'cost':cost.report()}]
    rmet,_=evaluate(ref,ref,data,sp);rows.append({'method':'target_free_ortho_retrain','metrics':rmet,'cost':refinfo['cost']})
    randomscore=torch.randn(24,device=DEVICE,generator=rng(seed+11000));rm=prune(source,randomscore)
    rmet,_=evaluate(rm,ref,data,sp);rows.append({'method':'random_mask','metrics':rmet})
    plain,plainmeta=get_training(data,seed,rounds);plainref,_=get_training(data,seed,rounds,exclude=True)
    pv,pdiag=statistic_vectors(plain,data);_,pp=evaluate(plain,plainref,data)
    pm=prune(plain,sum(pv));pmet,_=evaluate(pm,plainref,data,pp)
    rows.append({'method':'no_ortho_noiseless_prune','metrics':pmet,'diagnostic':pdiag})
    for snr in [10,20]:
        for noise_seed in range(4):
            seednoise=seed*100+noise_seed
            bs,bc=stat_observation(vectors,seednoise,snr,'boundary')
            for mode in ['uniform','boundary','matched']:
                start=time.perf_counter()
                if mode=='boundary':score,co=bs,bc
                else:score,co=stat_observation(vectors,seednoise,snr,mode,bc.total())
                model=prune(source,score);metrics,probs=evaluate(model,ref,data,sp)
                idx=pruning_indices(score);wrong=int((~torch.isin(idx,truth)).sum())
                row={'method':f'ota{snr}_{mode}','noise_seed':noise_seed,'metrics':metrics,'cost':co.report(),
                     'mask_wrong_count':wrong,'selection_regret':float(exact[truth].sum()-exact[idx].sum()),
                     'score_mse':float((score-exact).square().mean()),
                     'forget_js_to_noiseless_prune':float(js(probs['forget'],idealp).mean()),
                     'post_selection_eval_seconds':time.perf_counter()-start}
                rows.append(row)
            write(path,{'seed':seed,'training':training,'reference_training':refinfo,'diagnostic':diagnostic,'rows':rows,'complete':False})
        log(stage=2,seed=seed,snr=snr,status='communication_cases_complete')
    write(path,{'seed':seed,'training':training,'reference_training':refinfo,'diagnostic':diagnostic,
                'true_scores':exact.cpu().tolist(),'rows':rows,'complete':True})
    log(stage=2,seed=seed,status='complete')

def mixed_kd(source,data,seed,snr):
    model=clone_model(source);teacher=clone_model(source);teacher.eval();active=list(range(10));cost=Cost()
    streams={i:rng(seed*8000+i) for i in active};ng=rng(seed+333);start=time.perf_counter();initial=None;dim=flat(model).numel()
    for step in range(100):
        cost.digital(dim*32);gs=[]
        for i in active:
            bx,by=batch(data,i,streams[i],32)
            if i==0:
                with torch.no_grad():target=fedquit_teacher(teacher(bx),by)
                loss=kd_loss(model(bx),target);cost.c['teacher_forward_samples']+=32
            else:loss=F.cross_entropy(model(bx),by)
            gs.append(flatten(torch.autograd.grad(loss,tuple(model.parameters()))).detach())
        if step==0:initial={'target_gradient_norm':float(gs[0].norm()),'retained_aggregate_norm':float(sum(gs[1:]).norm()/9)}
        avg=ota(gs,[.5]+[.5/9]*9,active,snr,ng,4 if snr else 1,cost)
        assign(model,flat(model)-.01*avg);cost.c['client_gradient_samples']+=320
    cost.digital(dim*32);sync()
    return model,cost.report(),time.perf_counter()-start,initial

def train_teacher(data,seed,i,force=False):
    cp=CHECKPOINTS/f'teacher_seed{seed}_client{i}.pt';mp=RESULTS/f'teacher_seed{seed}_client{i}.json'
    if not force and cp.exists() and mp.exists():return load_model(cp),json.loads(mp.read_text())
    model=new_model(seed);model.train();gen=rng(seed*9000+i);start=time.perf_counter()
    for _ in range(500):
        bx,by=batch(data,i,gen,64);g=grads(model,bx,by)
        with torch.no_grad():
            for p,gg in zip(model.parameters(),g):p.add_(gg,alpha=-.03)
    sync();meta={'client':i,'seed':seed,'steps':500,'gradient_samples':32000,'wall_seconds':time.perf_counter()-start,
                'parameter_sha256':vec_hash(flat(model))}
    if not force:save_model(cp,model);write(mp,meta)
    return model,meta

def simplex(v):
    u=v.sort(dim=1,descending=True).values
    cssv=u.cumsum(1)-1;j=torch.arange(1,v.shape[1]+1,device=DEVICE,dtype=v.dtype)
    valid=u-cssv/j>0;rho=valid.sum(1)-1
    theta=cssv.gather(1,rho[:,None])/(rho[:,None]+1)
    return (v-theta).clamp_min(0)

def aggregate_predictions(qs,active,seed,snr=None,repeats=1):
    cost=Cost();cost.digital(qs[0].shape[0]*32) # public query IDs, shared public data preinstalled
    cost.c['teacher_forward_samples']=len(active)*qs[0].shape[0]
    agg=ota([qs[i].flatten() for i in active],[1/len(active)]*len(active),active,snr,rng(seed+555),repeats,cost)
    result=simplex(agg.reshape_as(qs[0]))
    cost.c['metadata_bits']+=len(active)*32 # surviving teacher weights/counts
    return result,cost

def distill(data,seed,targets):
    model=new_model(seed);model.train();g=rng(seed+444);x=data['x'][data['public']];start=time.perf_counter()
    for _ in range(600):
        ix=torch.randint(len(x),(64,),device=DEVICE,generator=g)
        loss=kd_loss(model(x[ix]),targets[ix],2.)
        grad=torch.autograd.grad(loss,tuple(model.parameters()))
        with torch.no_grad():
            for p,v in zip(model.parameters(),grad):p.add_(v,alpha=-.05)
    sync();return model,time.perf_counter()-start

def stage3(seed,rounds):
    path=RESULTS/f'method3_seed{seed}.json'
    if path.exists() and json.loads(path.read_text()).get('complete'):return
    data=load_data(seed);source,training=get_training(data,seed,rounds);fedref,fedrefinfo=get_training(data,seed,rounds,exclude=True)
    base,sp=evaluate(source,fedref,data);rows=[{'family':'fedavg','method':'no_op','metrics':base}]
    for name,snr in [('fedquit_ce_exact',None),('fedquit_ce_ota20_r4',20)]:
        m,cost,seconds,initial=mixed_kd(source,data,seed,snr);metrics,_=evaluate(m,fedref,data,sp)
        rows.append({'family':'fedavg','method':name,'metrics':metrics,'cost':cost,'wall_seconds':seconds,'initial_gradients':initial})
        save_model(CHECKPOINTS/f'{name}_seed{seed}.pt',m)
        log(stage=3,seed=seed,method=name,forget_js_ratio=metrics['forget_normalized_js'],test_accuracy=metrics['test_accuracy'])
    oldq=probabilities(source,data['x'][data['public']],2.)
    m,seconds=distill(data,seed,oldq);metrics,_=evaluate(m,fedref,data,sp)
    rows.append({'family':'fedavg','method':'fresh_student_old_global_teacher','metrics':metrics,'wall_seconds':seconds,
                 'student_train_samples':38400,'scope':'contaminated-teacher diagnostic, public queries'})
    qs=[];teacher_meta=[]
    for i in range(10):
        teacher,meta=train_teacher(data,seed,i);teacher_meta.append(meta)
        qs.append(probabilities(teacher,data['x'][data['public']],2.))
        if i%3==0:log(stage=3,seed=seed,trained_teacher=i)
    qall,allcost=aggregate_predictions(qs,list(range(10)),seed)
    original,originalsecs=distill(data,seed,qall)
    qret,rcost=aggregate_predictions(qs,list(range(1,10)),seed)
    reference,refsecs=distill(data,seed,qret)
    origmet,osp=evaluate(original,reference,data)
    rows.append({'family':'independent_teachers','method':'no_op_ensemble_student','metrics':origmet,
                 'cost':allcost.report(),'wall_seconds':originalsecs})
    exactmet,_=evaluate(reference,reference,data,osp)
    rcost.c['student_train_samples']+=38400;rcost.digital(flat(reference).numel()*32)
    rows.append({'family':'independent_teachers','method':'exclude_fresh_noiseless','metrics':exactmet,
                'cost':rcost.report(),'wall_seconds':refsecs,'reference_kind':'target-free independent-teacher KD algorithm'})
    save_model(CHECKPOINTS/f'independent_reference_seed{seed}.pt',reference)
    replay=None
    if seed==271:
        replayqs=[qs[0]];checks=[]
        for i in range(1,10):
            teacher,meta=train_teacher(data,seed,i,force=True)
            checks.append({'client':i,'equal_sha256':meta['parameter_sha256']==teacher_meta[i]['parameter_sha256'],
                            'wall_seconds':meta['wall_seconds']})
            replayqs.append(probabilities(teacher,data['x'][data['public']],2.))
        rtargets,_=aggregate_predictions(replayqs,list(range(1,10)),seed)
        remade,rsecs=distill(data,seed,rtargets)
        replay={'teachers':checks,'all_teacher_hashes_equal':all(c['equal_sha256'] for c in checks),
                'student_parameter_max_error':float((flat(remade)-flat(reference)).abs().max()),
                'student_replay_seconds':rsecs,'public_target_max_error':float((rtargets-qret).abs().max())}
        assert replay['all_teacher_hashes_equal'] and replay['student_parameter_max_error']<1e-6
    for rep in [1,4]:
        target,cost=aggregate_predictions(qs,list(range(1,10)),seed,20,rep)
        model,seconds=distill(data,seed,target);cost.c['student_train_samples']+=38400;cost.digital(flat(model).numel()*32)
        metrics,_=evaluate(model,reference,data,osp)
        compared,_=evaluate(model,fedref,data)
        rows.append({'family':'independent_teachers','method':f'exclude_fresh_ota20_r{rep}',
            'metrics':metrics,'fedavg_reference_comparison':compared,'cost':cost.report(),'wall_seconds':seconds,
            'query_target_mse':float((target-qret).square().mean())})
        save_model(CHECKPOINTS/f'exclude_fresh_ota20_r{rep}_seed{seed}.pt',model)
        log(stage=3,seed=seed,method=f'exclude_fresh_ota20_r{rep}',forget_js_ratio=metrics['forget_normalized_js'],test_accuracy=metrics['test_accuracy'])
    # Evaluate clean structural reference against FedAvg only as a different algorithm, never as its deletion oracle.
    cross,_=evaluate(reference,fedref,data)
    write(path,{'seed':seed,'training':training,'rows':rows,'teacher_preparation':teacher_meta,'fresh_replay':replay,
                'structural_reference_vs_fedavg_retrain':cross,'complete':True})
    log(stage=3,seed=seed,status='complete')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='all',choices=['1','2','3','all']);parser.add_argument('--seed',type=int)
    args=parser.parse_args();assert json.loads((RESULTS/'math_validation.json').read_text())['passed']
    calibration=json.loads((RESULTS/'calibration.json').read_text());assert calibration['quality_gate_passed']
    seeds=[args.seed] if args.seed is not None else SEEDS
    write(RESULTS/'run_environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'pid':os.getpid(),'one_worker':True,'rounds':calibration['chosen_rounds'],'seeds':seeds,
        'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL_KO.md').read_bytes()).hexdigest()})
    for stage,func in [('1',stage1),('2',stage2),('3',stage3)]:
        if args.stage not in ['all',stage]:continue
        for seed in seeds:func(seed,calibration['chosen_rounds'])
    log(status='all_requested_runs_complete')

if __name__=='__main__':main()
