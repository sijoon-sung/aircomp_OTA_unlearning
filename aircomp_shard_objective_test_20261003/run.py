import math, random, itertools, json, gzip, time, hashlib, html
from pathlib import Path
ROOT=Path(__file__).resolve().parent
D=38282
def js(a,b):
    m=[(x+y)/2 for x,y in zip(a,b)]
    return sum(.5*x*math.log(x/z) if x else 0 for x,z in zip(a,m))+sum(.5*y*math.log(y/z) if y else 0 for y,z in zip(b,m))
def partitions(items):
    if not items:
        yield ();return
    a=items[0]
    for rest in itertools.combinations(items[1:],3):
        g=(a,)+rest;s=set(g)
        for tail in partitions(tuple(i for i in items if i not in s)):
            yield (g,)+tail
def mean(v): return sum(v)/len(v)
def write(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    if (ROOT/'results.json').exists(): raise RuntimeError('Refuse overwrite')
    start=time.perf_counter();parts=list(partitions(tuple(range(12))))
    assert len(parts)==5775 and len(set(parts))==5775
    groups=list(itertools.combinations(range(12),4));gi={g:i for i,g in enumerate(groups)}
    pi=[tuple(gi[g] for g in p) for p in parts]
    write('partitions.json',parts)
    results=[];scenarios=[]
    with gzip.open(ROOT/'per_partition.json.gz','wt',encoding='utf-8') as raw:
      for seed in range(31001,31031):
        rng=random.Random(seed)
        hist=[]
        for i in range(12):
            v=[rng.gammavariate(.5,1) for _ in range(10)];hist.append([x/sum(v) for x in v])
        glob=[mean([v[j] for v in hist]) for j in range(10)]
        u=[rng.random() for _ in range(12)]
        for scenario,width,coupled in [('narrow_independent',3,False),('wide_independent',20,False),('wide_coupled',20,True)]:
            db=[-width+width*x for x in u]
            if coupled:
                order=sorted(range(12),key=lambda i:hist[i][0]);vals=sorted(db)
                db=[vals[order.index(i)] for i in range(12)]
            h=[10**(x/20) for x in db]
            gs=[]
            for g in groups:
                rep=js([mean([hist[i][j] for i in g]) for j in range(10)],glob)
                source=max(1,math.ceil(1/(16*min(h[i] for i in g)**2)-1e-12))
                dels=[];drep=[]
                for removed in g:
                    retained=[i for i in g if i!=removed]
                    dels.append(max(1,math.ceil(1/(9*min(h[i] for i in retained)**2)-1e-12)))
                    drep.append(js([mean([hist[i][j] for i in retained]) for j in range(10)],glob))
                gs.append((rep,source,sum(dels),max(dels),sum(drep)))
            reps=[mean([gs[j][0] for j in p]) for p in pi]
            threshold=sorted(reps)[math.ceil(.2*len(reps))-1]
            source=[sum(gs[j][1] for j in p) for p in pi]
            deletion=[sum(gs[j][2] for j in p) for p in pi] # /12 expected
            worst=[max(gs[j][3] for j in p) for p in pi]
            after=[sum(gs[j][4] for j in p)/12 for p in pi]
            eligible=[i for i,v in enumerate(reps) if v<=threshold+1e-14]
            smin=min(source[i] for i in eligible);dmin=min(deletion[i] for i in eligible)
            ix=min(eligible,key=lambda i:(source[i],deletion[i],i))
            dx=min(eligible,key=lambda i:(deletion[i],source[i],i))
            jx=min(eligible,key=lambda i:(source[i]*dmin+deletion[i]*smin,deletion[i],source[i],i))
            overlap=any(source[i]==smin and deletion[i]==dmin for i in eligible)
            # frozen cost negative control: uniform client deletion selects each shard with probability 1/3
            nullerr=max(abs(sum(gs[j][1]*4 for j in p)/12-source[k]/3) for k,p in enumerate(pi))
            assert nullerr<1e-12
            def metrics(i):
                return dict(partition_index=i,groups=parts[i],initial_UL_RE=source[i]*D,expected_deletion_UL_RE=deletion[i]*D/12,worst_deletion_UL_RE=worst[i]*D,mean_JS=reps[i],deleted_shard_JS=after[i])
            gain=100*(1-deletion[jx]/deletion[ix]);over=100*(source[jx]/source[ix]-1)
            row=dict(seed=seed,scenario=scenario,feasible=len(eligible),JS_threshold=threshold,strict_conflict=not overlap,null_max_error=nullerr,initial=metrics(ix),deletion=metrics(dx),joint=metrics(jx),joint_deletion_saving_pct=gain,joint_initial_increase_pct=over,practical_pass=gain>=5-1e-10 and over<=10+1e-10)
            results.append(row);scenarios.append(dict(seed=seed,scenario=scenario,h=h,channel_db=db,histograms=hist))
            raw.write(json.dumps(dict(seed=seed,scenario=scenario,source_repeat_sum=source,deletion_repeat_sum=deletion,worst_repeat=worst,mean_JS=reps,deleted_shard_JS=after,eligible_indices=eligible))+'\n')
        print(f'seed {seed}: completed 3 scenarios',flush=True)
    summaries=[]
    for scenario in ['narrow_independent','wide_independent','wide_coupled']:
        rows=[r for r in results if r['scenario']==scenario]
        summaries.append(dict(scenario=scenario,n=len(rows),strict_conflicts=sum(r['strict_conflict'] for r in rows),practical_passes=sum(r['practical_pass'] for r in rows),mean_joint_saving=mean([r['joint_deletion_saving_pct'] for r in rows]),mean_initial_increase=mean([r['joint_initial_increase_pct'] for r in rows]),mean_deletion_only_saving=mean([100*(1-r['deletion']['expected_deletion_UL_RE']/r['initial']['expected_deletion_UL_RE']) for r in rows]),mean_worst_change_pct=mean([100*(r['joint']['worst_deletion_UL_RE']/r['initial']['worst_deletion_UL_RE']-1) for r in rows]),mean_deleted_JS_change=mean([r['joint']['deleted_shard_JS']-r['initial']['deleted_shard_JS'] for r in rows])))
    cost=dict(wall_seconds=time.perf_counter()-start,source_training_runs=0,GPU_used=False,paid_external_cost=0,partitions_per_scenario=5775,scenarios=90,partition_evaluations=90*5775,single_deletion_evaluations=90*5775*12,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prereg_sha256=hashlib.sha256((ROOT/'PREREG.txt').read_bytes()).hexdigest())
    write('results.json',dict(summary=summaries,results=results,cost=cost));write('scenarios.json',scenarios)
    table=''.join('<tr>'+''.join(f'<td>{x}</td>' for x in [s['scenario'],f"{s['strict_conflicts']}/30",f"{s['mean_joint_saving']:.2f}%",f"{s['mean_initial_increase']:.2f}%",f"{s['practical_passes']}/30",f"{s['mean_worst_change_pct']:+.2f}%",f"{s['mean_deleted_JS_change']:+.5f}"])+'</tr>' for s in summaries)
    content=f'''<!doctype html><meta charset="utf-8"><title>초기학습 vs 삭제 목적 — 열거 실험</title><style>body{{font-family:Malgun Gothic,sans-serif;max-width:1250px;margin:40px auto;line-height:1.8;color:#203040}}td,th{{border:1px solid #ccd6df;padding:10px}}table{{border-collapse:collapse}}b{{color:#124c80}}</style><h1>초기학습 목적과 삭제 목적은 다른 shard를 만드는가?</h1><p><b>실행 완료 · CPU 전수 열거 · 모델 학습 없음</b></p><p><b>가설:</b> 삭제 후 AirComp 비용 변화가 두 목적의 최적 배정을 다르게 만든다.<br><b>독립변수:</b> 배정 목적(initial/deletion/joint), 채널 폭, 합성 데이터-채널 연결.<br><b>종속변수:</b> 최적해 충돌, 기대·최악 삭제 UL RE, 초기 UL RE, 데이터 대표성(JS).<br><b>통제:</b> 12 clients / K=3 / 4명씩 / 균일 단일 삭제 / 동일 대표성 후보 / 동일 rounds.</p><table><tr><th>환경</th><th>엄격한 목적 충돌</th><th>joint 평균 삭제 절감</th><th>평균 초기비용 증가</th><th>실용 기준 통과</th><th>최악 삭제 비용 변화</th><th>삭제 shard JS 변화</th></tr>{table}</table><p>절감/증가는 initial 최적 배정 대비 seed별 비율의 평균. 실용 기준은 삭제 UL 5% 이상 절감·초기 UL 증가 10% 이하를 20/30 seeds 이상 달성. 양의 JS 변화는 삭제 후 대표성 악화. 엄격한 충돌은 두 목적의 최소값을 동시에 달성하는 해가 없다는 뜻이며 동점 선택 차이를 제외했다.</p><p><b>음성 대조:</b> 삭제 후 비용을 원래 shard 비용으로 고정하면 모든 배정에서 기대 삭제 비용=초기 비용/3 검증 통과.</p><p><b>해석 범위:</b> 이는 동일 목표 집계 MSE를 만족하는 반복 전송의 UL 자원 모형이다. 정확도·총 통신비용·실측 지연·에너지·프라이버시·기존 clustering 대비 우월성·신규성을 검증하지 않았다. 후보 partition의 평균 JS 하위 20%를 공통으로 허용했지만 정확도는 보장하지 않는다. 고정 동일 rounds이므로 초기/삭제 RE는 한 라운드 정규화 값이다.</p><p><b>총비용:</b> {cost['wall_seconds']:.2f}초, CPU 사용, GPU/학습 실행 0, 외부 유료 비용 0. 에너지 미측정. {cost['partition_evaluations']:,}개 배정-환경 조합, {cost['single_deletion_evaluations']:,}개 단일 삭제 평가.</p><p><a href="PREREG.txt">사전기록</a> · <a href="results.json">전체 결과</a> · <a href="scenarios.json">채널·데이터</a> · <a href="partitions.json">전체 배정</a> · <a href="per_partition.json.gz">배정별 비용 원자료</a> · <a href="run.py">재현 코드</a></p>'''
    (ROOT/'report.html').write_text(content,encoding='utf-8');print(json.dumps(dict(summary=summaries,cost=cost),ensure_ascii=False),flush=True)
if __name__=='__main__': main()
