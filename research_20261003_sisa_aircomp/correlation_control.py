"""Finite synthetic coupling experiment; original experiment and outputs unchanged."""
import json, math, os, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
import experiment as E

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'correlation_control_v1'
LEVEL = 0.
OriginalEngine = E.Engine

def standard(x):
    x = x-x.mean()
    return x/np.sqrt(np.mean(x*x))

def rankdata(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts)-(counts-1)/2.)[inverse]

def js(a,b):
    a=np.maximum(np.asarray(a,dtype=float),1e-30); b=np.maximum(np.asarray(b,dtype=float),1e-30)
    a=a/a.sum(axis=-1,keepdims=True); b=b/b.sum(axis=-1,keepdims=True); m=(a+b)/2
    return .5*np.sum(a*np.log(a/m)+b*np.log(b/m),axis=-1)

def spearman(a,b):
    a=standard(rankdata(a)); b=standard(rankdata(b))
    return float(np.mean(a*b))

def channel_design(hist, seed, level):
    _,_,vt=np.linalg.svd(hist-hist.mean(0),full_matrices=False)
    pc=vt[0].copy()
    if pc[np.argmax(np.abs(pc))]<0: pc=-pc
    score=standard(hist@pc)
    rng=np.random.default_rng(E.key(seed,'coupling'))
    e=standard(rng.normal(size=20)); e-=score*np.mean(e*score); e=standard(e)
    latent=level*score+math.sqrt(1-level**2)*e
    values=np.sort(rng.uniform(-20,0,20))
    db=np.empty(20); db[np.argsort(latent,kind='stable')]=values
    return db, score, pc

def association(hist, db, seed):
    distance=np.sqrt(np.maximum(0,js(hist[:,None,:],hist[None,:,:])))
    ij=np.triu_indices(20,1); x=distance[ij]
    rho=spearman(x,np.abs(db[:,None]-db[None,:])[ij])
    rng=np.random.default_rng(E.key(seed,'mantel'))
    null=[]
    for _ in range(1999):
        p=rng.permutation(db); null.append(spearman(x,np.abs(p[:,None]-p[None,:])[ij]))
    p=(1+sum(abs(r)>=abs(rho) for r in null))/2000
    return dict(rho=rho,p=p,permutations=1999,null=null)

class CoupledEngine(OriginalEngine):
    def __init__(self,data,seed,K,method,rounds,stress=False):
        super().__init__(data,seed,K,method,rounds,stress)
        counts=np.stack([torch.bincount(data['y'][p],minlength=10).cpu().numpy() for p in data['pools']])
        hist=counts/counts.sum(1,keepdims=True)
        db,score,pc=channel_design(hist,seed,LEVEL)
        self.base_h=10**(db/20)
        order=np.argsort(db,kind='stable') if method.startswith('channel') else np.random.default_rng(E.key(seed,'routing')).permutation(20)
        self.groups=[list(map(int,g)) for g in np.array_split(order,K)]
        self.affected=next(c for c,g in enumerate(self.groups) if 0 in g)
        self.h=np.stack([self.base_h*np.random.default_rng(E.key(seed,t,'fading')).uniform(.85,1.15,20) for t in range(rounds)])
        self.hist=hist

def metrics(prob,y):
    pred=prob.argmax(1); rec=[]; f1=[]
    for c in range(10):
        tp=np.sum((pred==c)&(y==c)); fn=np.sum((pred!=c)&(y==c)); fp=np.sum((pred==c)&(y!=c))
        rec.append(float(tp/(tp+fn))); f1.append(float(2*tp/max(1,2*tp+fn+fp)))
    return dict(accuracy=float(np.mean(pred==y)),macro_f1=float(np.mean(f1)),worst_recall=min(rec),class_recall=rec)

def extra(row,hist):
    folder=OUT/row['case']; cfg=json.loads((folder/'config.json').read_text())
    y=np.load(folder/'labels.npz')['test']; target=hist.mean(0)
    divergence=[float(js(hist[g].mean(0),target)) for g in cfg['groups']]
    retained=[i for i in cfg['groups'][cfg['affected']] if i!=0]
    result=dict(level=LEVEL,shard_js=divergence,mean_shard_js=float(np.mean(divergence)),max_shard_js=max(divergence),
        deleted_shard_js=float(js(hist[retained].mean(0),hist[1:].mean(0))),
        source=metrics(np.load(folder/'source_probabilities.npz')['test'],y),
        deleted=metrics(np.load(folder/'replay_probabilities.npz')['test'],y))
    E.dump(folder/'extra_metrics.json',result)
    return dict(**row,extra=result)

def finalize(rows, associations, cost):
    # BH over all nine synthetic design checks, not independent pair observations.
    order=np.argsort([a['p'] for a in associations]); running=1.
    for j in range(len(order)-1,-1,-1):
        idx=int(order[j]); running=min(running,associations[idx]['p']*len(order)/(j+1)); associations[idx]['q_bh']=running
    E.dump(OUT/'association_summary.json',[{k:v for k,v in a.items() if k!='null'} for a in associations])
    comparisons=[]; summary=[]
    for level in [0.,.5,1.]:
        pairs=[]
        for seed in [10701,10702,10703]:
            a=next(r for r in rows if r['seed']==seed and r['extra']['level']==level and r['method']=='random_norm')
            b=next(r for r in rows if r['seed']==seed and r['extra']['level']==level and r['method']=='channel_norm')
            p=dict(seed=seed,level=level,accuracy_delta_pp=100*(b['test_accuracy']-a['test_accuracy']),
                worst_recall_delta_pp=100*(b['extra']['deleted']['worst_recall']-a['extra']['deleted']['worst_recall']),
                total_re_saving=1-b['delete_cost']['total_re']/a['delete_cost']['total_re'],
                ul_saving=1-b['delete_cost']['ul_reals']/a['delete_cost']['ul_reals'],
                source_re_saving=1-b['source_cost']['total_re']/a['source_cost']['total_re'],
                shard_js_delta=b['extra']['mean_shard_js']-a['extra']['mean_shard_js'],
                random_accuracy=a['test_accuracy'],channel_accuracy=b['test_accuracy'])
            pairs.append(p); comparisons.append(p)
        avg={k:float(np.mean([p[k] for p in pairs])) for k in pairs[0] if k not in ['seed','level']}
        successes=sum(p['total_re_saving']>0 for p in pairs)
        summary.append(dict(level=level,**avg,improved_seeds=successes,practical_pass=bool(avg['total_re_saving']>=.1 and successes>=2 and avg['accuracy_delta_pp']>=-2 and all(r['replay_exact'] for r in rows))))
    E.dump(OUT/'comparisons.json',comparisons); E.dump(OUT/'summary.json',summary)
    lines=['# 데이터–채널 연결 통제 실험 결과','', '18개 신규 GPU 실험 완료. 현실의 상관관계가 아니라 합성 연결에 대한 민감도 실험이다.', '',
           '조건: FashionMNIST train12000/test10000, client20, Dirichlet .5, K4, 240 rounds, 3 seeds. Random vs channel-sorted, 삭제 client0. 목표 MSE1e-4, 전체 FP32 DL, 이상적 직교 shard 분리.', '',
           '|연결 계수|Random 정확도|Channel 정확도|정확도 차이 pp|최저 recall 차이 pp|삭제 UL 절감|삭제 total RE 절감|초기 total RE 절감|대표성 JS 차이|개선 seed|실용 기준|',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for s in summary:
        lines.append(f"|{s['level']}|{s['random_accuracy']*100:.2f}%|{s['channel_accuracy']*100:.2f}%|{s['accuracy_delta_pp']:+.2f}|{s['worst_recall_delta_pp']:+.2f}|{s['ul_saving']*100:.2f}%|{s['total_re_saving']*100:.2f}%|{s['source_re_saving']*100:.2f}%|{s['shard_js_delta']:+.5f}|{s['improved_seeds']}/3|{s['practical_pass']}|")
    lines += ['', '절감은各seed에서 1-channel/random을 계산한 산술평균이며, 음수는 비용 증가다. JS 차이 양수는 채널 배정의 대표성 악화다. 최저 recall은 방법별 최저 클래스이며 동일 클래스를 뜻하지 않는다.', '',
              '|seed|연결 계수|JS-distance–channel-distance rho|permutation p|BH q|','|---|---|---|---|---|']
    for a in associations: lines.append(f"|{a['seed']}|{a['level']}|{a['rho']:.4f}|{a['p']:.4f}|{a['q_bh']:.4f}|")
    lines += ['',f"Exact replay: {sum(r['replay_exact'] for r in rows)}/{len(rows)}. 같은 SISA 알고리즘·삭제 없는 client들과 동일 난수로 전체 재학습한 reference와 bitwise 일치. 일반적인 DP/정보비노출 보장을 뜻하지 않는다.",
              '', '## 총 실험 비용', '```json',json.dumps(cost,ensure_ascii=False,indent=2),'```','',
              'GPU board energy는 시스템 보드 샘플 적분이며 RF 통신 Joule이 아니다. 외부 유료 API/클라우드 사용 0. 삭제 shard는 4 retained clients, reference는19: local calls 절감 78.95%. 단일 전역모델 AirComp와 비교하지 않았다.', '',
              '## 한계와 다음 판단',
              '- 합성 PC1 연결 계수와 실제 거리 기반 상관은 다르다. 실제 상관 존재 여부는 paired 실측 데이터 없이는 미확인이다.',
              '- 3 seeds, 20 clients, 단일 dataset/K/삭제client/고정 라운드. 모든 클래스가 균형인 test이며 local class imbalance만 존재한다. 학습 수렴을 확증하지 않았다.',
              '- 대표성·정확도·비용의 seed별 차이를 함께 본다. 일부 조건에서만 이득이면 보편적인 채널순 grouping을 채택하지 않는다.',
              '- 고정 MSE에서 자원량을 바꿨다. 고정 무선 예산에서 통신 오차가 정확도에 주는 영향은 후속 검증 대상이다.',
              '- 원자료: 각 case config/partition/labels/source 및 reference 및 replay models/probabilities/ledgers/result/extra_metrics. association_s*.json에 histogram·채널·순열 null 보존. comparisons.json에 모든 seed별 차이.',
              '- 코드 correlation_control.py, 기반 experiment.py 및 core.py, 사전기록 CORRELATION_CONTROL_PREREG.md; 실행 코드 해시는 environment.json.', '']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    global LEVEL
    OUT.mkdir(exist_ok=False)
    inv=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True)
    foreign=[s for s in inv.splitlines()[1:] if s.strip() and 'ChatGPT.exe' not in s and not s.strip().startswith(str(os.getpid())+',')]
    if foreign:
        E.dump(OUT/'blocked.json',dict(foreign=foreign)); raise RuntimeError('Other compute job active')
    E.dump(OUT/'environment.json',dict(gpu=inv,torch=torch.__version__,code_sha256=E.sha(__file__),base_sha256=E.sha(ROOT/'experiment.py'),core_sha256=E.sha(E.BASE/'research_20260929_pdf_ota_unlearning/core.py'),prereg_sha256=E.sha(ROOT/'CORRELATION_CONTROL_PREREG.md')))
    E.Engine=CoupledEngine
    rows=[]; associations=[]; began=time.perf_counter(); error=None
    with E.Power() as meter:
        try:
            E.radio_audit(OUT)
            for seed in [10701,10702,10703]:
                data=E.load_data('FashionMNIST',.5,seed)
                counts=np.stack([torch.bincount(data['y'][p],minlength=10).cpu().numpy() for p in data['pools']]); hist=counts/counts.sum(1,keepdims=True)
                del data
                for LEVEL in [0.,.5,1.]:
                    db,score,pc=channel_design(hist,seed,LEVEL)
                    a=dict(seed=seed,level=LEVEL,**association(hist,db,seed))
                    associations.append(a)
                    E.dump(OUT/f'association_s{seed}_l{LEVEL}.json',dict(**a,histogram=hist.tolist(),db=db.tolist(),pc=pc.tolist(),score=score.tolist()))
                    for method in ['random_norm','channel_norm']:
                        row=E.one_case(OUT,f'coupling{int(LEVEL*100)}',seed,4,method,240)
                        rows.append(extra(row,hist))
            E.dump(OUT/'results.json',rows)
        except BaseException:
            error=traceback.format_exc(); E.dump(OUT/'failure.json',dict(error=error,completed_cases=len(rows)))
        finally:
            torch.cuda.synchronize()
    cost=dict(seconds=time.perf_counter()-began,**meter.report(),completed_cases=len(rows),local_calls=sum(r[p]['local_calls'] for r in rows for p in ['source_cost','reference_cost','delete_cost']),error=error)
    E.dump(OUT/'cost_total.json',cost)
    if error: raise RuntimeError(error)
    finalize(rows,associations,cost)
    E.dump(OUT/'complete.json',dict(cases=len(rows),all_exact=all(r['replay_exact'] for r in rows)))
    E.log(event='all_complete',summary=json.loads((OUT/'summary.json').read_text()),cost=cost)

if __name__=='__main__': main()
