import numpy as np,json,time,math
from pathlib import Path
P=Path(__file__).resolve().parent
def main():
 if (P/'results.json').exists():raise RuntimeError('no overwrite')
 t=time.perf_counter();rows=[];arrays={};maxerr=0.;maxpower=0
 for seed in range(51001,51031):
  for M in [4,8,9,12,16]:
   for corr in [0.,.8]:
    rng=np.random.default_rng(seed*100+M);cov=corr**np.abs(np.arange(M)[:,None]-np.arange(M)[None,:]);L=np.linalg.cholesky(cov)
    H=L@(rng.normal(size=(M,12))+1j*rng.normal(size=(M,12)))/np.sqrt(2)
    H*=10**(rng.uniform(-10,0,size=12)/20)
    prefix=f'{seed}_{M}_{corr}';arrays[prefix+'_H']=H;zfs=[];tds=[];zs=[];ts=[]
    for g in range(3):
     own=np.arange(4*g,4*g+4);other=np.array([i for i in range(12) if i not in own]);C=H[:,other].conj().T
     _,sv,Vh=np.linalg.svd(C,full_matrices=True);rank=int(np.sum(sv>1e-10));Q=Vh.conj().T[:,rank:]
     def choose(Q,extra=None):
      if Q.shape[1]==0:return None
      D=Q.shape[1];F=Q.conj().T@H[:,own];_,V=np.linalg.eigh(F@F.conj().T)
      cand=np.c_[F,V[:,-1:],rng.normal(size=(D,128))+1j*rng.normal(size=(D,128))]
      cand=Q@cand
      if extra is not None:cand=np.c_[cand,extra]
      cand/=np.linalg.norm(cand,axis=0)
      gains=np.min(np.abs(cand.conj().T@H[:,own]),axis=1);j=int(np.argmax(gains));w=cand[:,j];eta=1/(4*gains[j]);b=1/(4*eta*(w.conj()@H[:,own]));return w,eta,b
     z=choose(Q);d=choose(np.eye(M),None if z is None else z[0])
     for label,entry in [('zf',z),('td',d)]:
      if entry is None:continue
      w,eta,b=entry;coef=eta*(w.conj()@H[:,own])*b;maxerr=max(maxerr,float(np.max(np.abs(coef-.25))));maxpower=max(maxpower,float(np.max(np.abs(b))))
      if label=='zf':maxerr=max(maxerr,float(np.max(np.abs(w.conj()@H[:,other]))))
      arrays[prefix+f'_{g}_{label}_w']=w;arrays[prefix+f'_{g}_{label}_b']=b
      repeats=max(1,math.ceil(.01*eta**2/.001));(zfs if label=='zf' else tds).append(float(eta**2));(zs if label=='zf' else ts).append(repeats)
    feasible=len(zfs)==3
    rows.append(dict(seed=seed,M=M,correlation=corr,feasible=feasible,zf_noise_coeff=zfs,td_noise_coeff=tds,zf_repeats=zs,td_repeats=ts,zf_uses=max(zs) if feasible else None,td_uses=sum(ts),noise_ratio=float(np.mean(zfs)/np.mean(tds)) if feasible else None,uses_ratio=max(zs)/sum(ts) if feasible else None))
 summaries=[]
 for M in [4,8,9,12,16]:
  for c in [0.,.8]:
   a=[r for r in rows if r['M']==M and r['correlation']==c];b=[r for r in a if r['feasible']]
   summaries.append(dict(M=M,correlation=c,feasible=len(b),n=len(a),mean_noise_ratio=float(np.mean([r['noise_ratio'] for r in b])) if b else None,mean_uses_ratio=float(np.mean([r['uses_ratio'] for r in b])) if b else None,slower_count=sum(r['uses_ratio']>1 for r in b)))
 assert maxerr<1e-9 and maxpower<=1+1e-9
 np.savez_compressed(P/'channels_receivers.npz',**arrays)
 out=dict(summary=summaries,rows=rows,audit=dict(max_coefficient_error=maxerr,max_precoder_amplitude=maxpower,status='passed'),cost=dict(seconds=time.perf_counter()-t,GPU=False,paid_cost=0,energy_measured=False,channel_scenarios=300))
 (P/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
 table=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in [s['M'],s['correlation'],str(s['feasible'])+'/30','—' if s['mean_noise_ratio'] is None else f"{s['mean_noise_ratio']:.2f}",'—' if s['mean_uses_ratio'] is None else f"{s['mean_uses_ratio']:.2f}",s['slower_count']])+'</tr>' for s in summaries)
 (P/'report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>client 채널의 ZF 비용</title><style>body{{font-family:Malgun Gothic;max-width:1100px;margin:40px auto;line-height:1.8}}td,th{{border:1px solid #ccd;padding:12px}}table{{border-collapse:collapse}}</style><h1>client별 채널에서 SISA 분리의 비용</h1><p><b>가설:</b> shard 외 client를 제거하는 ZF에는 자유도·잡음 비용이 든다.<br><b>독립변수:</b> 안테나 수·공간상관·ZF/시간분리.<br><b>종속변수:</b> 분리 가능성·잡음계수·목표 MSE 반복 수·channel uses.<br><b>통제:</b> 12 clients, 고정3shards, 완전CSI, 동일 전력한도·채널.</p><table><tr><th>안테나 M</th><th>공간상관</th><th>분리 가능</th><th>잡음계수 ZF/TD</th><th>채널사용 ZF/TD</th><th>ZF 더 느린 seed</th></tr>{table}</table><p>각 값은 seed별 비율의 평균. M≤8에서는 다른8client를 모두 nulling하는 비영 수신 벡터가 없어 이 특정 scalar 송신·선형 수신 모형에서 불가능하다. 다른 부호화/다중안테나 client/시간확장까지 불가능하다는 뜻은 아니다. M=9부터 가능해도 작은 유효 gain으로 잡음이 증폭할 수 있다.</p><p>정확한 ZF는 가능해질 수 있지만 비용이 항상 작은 것은 아니다. 후보 수신기는 투영 채널·주성분·128 random 후보 중 선택하며 전역 최적해가 아니다. 시간분리에 ZF 후보를 포함했다. 딥러닝·실제 파형·CSI 추정·보안·총 RF 에너지는 검증하지 않았다. TD는3shards 순차, ZF는 동시 전송 후 필요한 반복 최대값으로 계산. pilot/control 비용 제외.</p><p>수치 검증: 계수 오차 {maxerr:.3e}, 최대 송신계수 크기 {maxpower:.6f}. CPU 실행 {out['cost']['seconds']:.2f}초, GPU0, 유료비용0, 에너지 미측정. 기존 독립성 실험은 별도 복사본이며 이번에 재실행하지 않았다.</p><p><a href="PREREG.txt">사전 정의</a> · <a href="results.json">원자료 수치</a> · <a href="channels_receivers.npz">채널·수신기</a> · <a href="run.py">코드</a></p>''',encoding='utf-8')
if __name__=='__main__':main()
