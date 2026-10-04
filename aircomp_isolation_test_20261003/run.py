from pathlib import Path
import numpy as np
import json,time,hashlib
R=Path(__file__).resolve().parent
def save(name,v): (R/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def rel(a,b):return float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(b)))
def main():
 if (R/'results.json').exists():raise RuntimeError('overwrite forbidden')
 start=time.perf_counter();rows=[];states={};inputs={};D=9;T=100;cp=20;lr=.04
 for seed in range(41001,41021):
  rng=np.random.default_rng(seed);teacher=rng.normal(size=8)
  X=np.concatenate([rng.normal(size=(3,4,64,8))+rng.normal(scale=.5,size=(3,4,1,8)),np.ones((3,4,64,1))],axis=-1)
  y=X[...,:8]@teacher+rng.normal(scale=.5,size=(3,4,1))+rng.normal(scale=.1,size=(3,4,64))
  Xt=np.c_[rng.normal(size=(2048,8)),np.ones(2048)];yt=Xt[:,:8]@teacher
  H=np.einsum('gcnp,gcnq->gcpq',X,X)/64;v=np.einsum('gcnp,gcn->gcp',X,y)/64
  reg=np.diag([.02]*8+[0]);baseH=H[:,:3].mean(1)+reg;basev=v[:,:3].mean(1)
  allH=H.mean(1)+reg;allv=v.mean(1)
  noise=rng.normal(size=(T,3,D));err=rng.normal(size=(3,3))
  inputs[f's{seed}_H']=H;inputs[f's{seed}_v']=v;inputs[f's{seed}_noise']=noise;inputs[f's{seed}_CSIerror']=err;inputs[f's{seed}_Xtest']=Xt;inputs[f's{seed}_ytest']=yt
  def train(h,b,B,N,w,begin,end,only=None):
   w=w.copy()
   for t in range(begin,end):
    grad=np.einsum('gij,gj->gi',h,w)-b
    if only is not None:
     mask=np.zeros_like(grad);mask[only]=grad[only];grad=mask
    step=B@grad+N[t]
    if only is None:w-=lr*step
    else:w[only]-=lr*step[only]
   assert np.isfinite(w).all()
   return w
  def mse(w):return float(np.mean((Xt@w.mean(0)-yt)**2))
  for rho in [0.,.01,.1,.3]:
   A=np.eye(3)+rho*(np.ones((3,3))-np.eye(3))
   for sigma in [0.,.01]:
    for method in ['orthogonal','mixed','zf_exact','zf_csi_error']:
     W=np.eye(3) if method in ['orthogonal','mixed'] else np.linalg.inv(A+(.02*err if method=='zf_csi_error' else 0))
     B=np.eye(3) if method=='orthogonal' else W@A
     N=np.einsum('ij,tjd->tid',W,noise)*sigma
     assert np.linalg.cond(A)<10 and np.linalg.cond(A+.02*err)<10
     checkpoint=train(baseH,basev,B,N,np.zeros((3,D)),0,cp)
     source=train(allH,allv,B,N,checkpoint,cp,T)
     prefix=f'{seed}_{rho}_{sigma}_{method}'
     inputs[prefix+'_B']=B;inputs[prefix+'_W']=W
     states[prefix+'_checkpoint']=checkpoint;states[prefix+'_source']=source
     for g in range(3):
      dh=allH.copy();dv=allv.copy();dh[g]=baseH[g];dv[g]=basev[g]
      full=train(dh,dv,B,N,checkpoint,cp,T)
      partial=train(dh,dv,B,N,checkpoint,cp,T,only=g)
      partial[np.arange(3)!=g]=source[np.arange(3)!=g]
      sham=train(allH,allv,B,N,checkpoint,cp,T,only=g);sham[np.arange(3)!=g]=source[np.arange(3)!=g]
      unaffected=[i for i in range(3) if i!=g]
      propagation=rel(full[unaffected],source[unaffected]);gap=rel(partial,full)
      key=prefix+f'_delete{g}'
      states[key+'_full']=full;states[key+'_partial']=partial;states[key+'_sham']=sham
      rows.append(dict(key=key,seed=seed,rho=rho,sigma=sigma,method=method,deleted_shard=g,unaffected_change=propagation,partial_full_gap=gap,sham_source_gap=rel(sham,source),source_MSE=mse(source),full_MSE=mse(full),partial_MSE=mse(partial),offdiagonal_norm=float(np.linalg.norm(B-np.diag(np.diag(B)))),noise_variance_gain=float(np.trace(W@W.T)/3),full_replay_uses=(T-cp)*D*(3 if method=='orthogonal' else 1),partial_replay_uses=(T-cp)*D,full_gradient_calls=(T-cp)*3,partial_gradient_calls=T-cp))
  print(f'{seed} completed',flush=True)
 summaries=[]
 for method in ['orthogonal','mixed','zf_exact','zf_csi_error']:
  for rho in [0.,.01,.1,.3]:
   subset=[r for r in rows if r['method']==method and r['rho']==rho]
   summaries.append(dict(method=method,rho=rho,n=len(subset),propagation_cases=sum(r['unaffected_change']>1e-9 for r in subset),replay_matches=sum(r['partial_full_gap']<=1e-9 for r in subset),mean_unaffected_change=float(np.mean([r['unaffected_change'] for r in subset])),max_replay_gap=max(r['partial_full_gap'] for r in subset),mean_replay_gap=float(np.mean([r['partial_full_gap'] for r in subset])),mean_sham_gap=float(np.mean([r['sham_source_gap'] for r in subset])),mean_full_MSE=float(np.mean([r['full_MSE'] for r in subset])),mean_partial_MSE=float(np.mean([r['partial_MSE'] for r in subset])),mean_noise_gain=float(np.mean([r['noise_variance_gain'] for r in subset]))))
 np.savez_compressed(R/'raw_states.npz',**states);np.savez_compressed(R/'scenario_inputs.npz',**inputs)
 cost=dict(seconds=time.perf_counter()-start,GPU=False,paid_cost=0,energy_measured=False,source_runs=640,full_replays=1920,partial_replays=1920,sham_replays=1920,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prereg_sha256=hashlib.sha256((R/'PREREG.txt').read_bytes()).hexdigest())
 save('results.json',dict(summary=summaries,rows=rows,cost=cost));print(json.dumps(dict(summary=summaries,cost=cost)),flush=True)
if __name__=='__main__':main()
