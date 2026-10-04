from pathlib import Path
import numpy as np,json,time,hashlib
P=Path(__file__).resolve().parent;OLD=P.parent/'01_기존_독립성_실험'/'scenario_inputs.npz'
def rel(a,b):return float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(b)))
def main():
 if (P/'results.json').exists():raise RuntimeError('no overwrite')
 t=time.perf_counter();old=np.load(OLD);rows=[];arrays={};maxwave=0.;D=9;T=100
 for seed in range(41001,41021):
  H=old[f's{seed}_H'].reshape(12,D,D)+np.diag([.02]*8+[0]);v=old[f's{seed}_v'].reshape(12,D)
  xt=old[f's{seed}_Xtest'];yt=old[f's{seed}_ytest'];rng=np.random.default_rng(seed+80000);offset=rng.uniform(-1,1,12)
  noise=(rng.normal(size=(T,64))+1j*rng.normal(size=(T,64)))/np.sqrt(2);nf=np.fft.fft(noise,axis=1,norm='ortho')
  for name,a in [('H',H),('v',v),('Xt',xt),('yt',yt),('offset',offset),('noise',noise)]:arrays[f's{seed}_{name}']=a
  for guard in [0,4]:
   tones=np.array([np.arange(4+g*(9+guard),13+g*(9+guard)) for g in range(3)]);toneflat=tones.ravel()
   for eps in [0.,.01,.05,.1]:
    raw=np.zeros((27,108));phases=[]
    for i in range(12):
     freq=np.zeros((64,9),complex);freq[tones[i//4],np.arange(9)]=1
     phase=np.exp(2j*np.pi*eps*offset[i]*(np.arange(64)-31.5)/64);phases.append(phase)
     block=np.fft.fft(np.fft.ifft(freq,axis=0,norm='ortho')*phase[:,None],axis=0,norm='ortho')[toneflat].real
     raw[:,i*9:(i+1)*9]=block
    # independent full composite waveform check on random real client coordinates
    probe=rng.normal(size=(12,9));wave=np.zeros(64,complex)
    for i in range(12):
     f=np.zeros(64,complex);f[tones[i//4]]=probe[i];wave+=np.fft.ifft(f,norm='ortho')*phases[i]
    maxwave=max(maxwave,float(np.max(np.abs(raw@probe.ravel()-np.fft.fft(wave,norm='ortho')[toneflat].real))))
    cross=raw.copy()
    for i in range(12):cross[(i//4)*9:(i//4+1)*9,i*9:(i+1)*9]=0
    ratio=float(np.sum(cross**2)/np.sum(raw**2))
    for mode in ['ofdm','oracle_no_cross']:
     C=raw if mode=='ofdm' else raw-cross
     for sigma in [0.,.01]:
      key=f'{seed}_{guard}_{eps}_{mode}_{sigma}';N=nf[:,toneflat].real*sigma
      arrays[key+'_C']=C;arrays[key+'_N']=N
      def system(active,only=None):
       counts=np.array([sum(i//4==g for i in active) for g in range(3)],float);scale=np.repeat(np.where(counts>0,1/np.maximum(counts,1),0),9)
       L=np.zeros((27,27));b=np.zeros(27)
       for i in active:
        q=C[:,i*9:(i+1)*9]*scale[:,None];L[:,(i//4)*9:(i//4+1)*9]+=q@H[i];b+=q@v[i]
       return np.eye(27)-.04*L,.04*b,N*scale
      def train(active,only=None):
       M,b,n=system(active);w=np.zeros(27)
       for step in range(T):w=M@w+b-.04*n[step]
       assert np.isfinite(w).all();return w.reshape(3,9)
      def mse(w):return float(np.mean((xt@w.mean(0)-yt)**2))
      source=train(list(range(12)));arrays[key+'_source']=source
      for g in range(3):
       target=4*g+3;active=[i for i in range(12) if i!=target];full=train(active)
       local=train([i for i in active if i//4==g]);sham=train(list(range(4*g,4*g+4)))
       others=np.arange(3)!=g;local[others]=source[others];sham[others]=source[others]
       rid=key+f'_delete{g}';arrays[rid+'_full']=full;arrays[rid+'_local']=local;arrays[rid+'_sham']=sham
       delta=rel(full[others],source[others]);gap=rel(local,full)
       rows.append(dict(key=rid,seed=seed,guard=guard,eps=eps,mode=mode,sigma=sigma,deleted_shard=g,cross_energy_ratio=ratio if mode=='ofdm' else 0,unaffected_change=delta,replay_gap=gap,sham_gap=rel(sham,source),source_MSE=mse(source),full_MSE=mse(full),local_MSE=mse(local),prediction_RMS=float(np.sqrt(np.mean((xt@(local.mean(0)-full.mean(0)))**2)))))
  print(f'{seed} complete',flush=True)
 summary=[]
 for mode in ['ofdm','oracle_no_cross']:
  for guard in [0,4]:
   for eps in [0.,.01,.05,.1]:
    a=[r for r in rows if r['mode']==mode and r['guard']==guard and r['eps']==eps]
    summary.append(dict(mode=mode,guard=guard,eps=eps,n=len(a),propagation=sum(r['unaffected_change']>1e-9 for r in a),matches=sum(r['replay_gap']<=1e-9 for r in a),over_1e4=sum(r['unaffected_change']>1e-4 for r in a),over_1e3=sum(r['unaffected_change']>1e-3 for r in a),mean_cross_energy=float(np.mean([r['cross_energy_ratio'] for r in a])),mean_unaffected=float(np.mean([r['unaffected_change'] for r in a])),mean_gap=float(np.mean([r['replay_gap'] for r in a])),max_gap=max(r['replay_gap'] for r in a),mean_sham=float(np.mean([r['sham_gap'] for r in a])),mean_full_MSE=float(np.mean([r['full_MSE'] for r in a])),mean_local_MSE=float(np.mean([r['local_MSE'] for r in a])),prediction_RMS=float(np.mean([r['prediction_RMS'] for r in a]))))
 assert maxwave<1e-10
 np.savez_compressed(P/'raw.npz',**arrays)
 out=dict(summary=summary,rows=rows,audit_waveform_max_error=maxwave,cost=dict(seconds=time.perf_counter()-t,source=640,full=1920,local=1920,sham=1920,GPU=False,paid_cost=0,energy_measured=False),hashes={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['run.py','PREREG.txt']})
 (P/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(summary=summary,cost=out['cost'])),flush=True)
if __name__=='__main__':main()
