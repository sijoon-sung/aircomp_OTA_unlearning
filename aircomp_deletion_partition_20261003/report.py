from pathlib import Path
import json,time,html
import numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,v):
    with (R/name).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
started=time.perf_counter();records=read(R/'results.json');resource=read(R/'resource.json');audit_events=0;maxpower=0
fixed=read(R/'fixed160_results.json');extra=read(R/'fixed160_resource.json');fixedaudit=read(R/'fixed160_audit.json')
fixedsummary=[]
searchaudit=[]
for seed in [62001,62002,62003]:
    route=read(R/f'{seed}_routing.json');scale=np.array(route['scales'])
    aa=route['methods']['train_joint'];bb=route['methods']['delete_joint']
    a_score=np.array(aa['score'])/scale;b_score=np.array(bb['score'])/scale
    searchaudit.append({'seed':seed,'same_groups':aa['groups']==bb['groups'],'delete_objective_of_train':float(.5*(a_score[0]+a_score[2])),
        'delete_objective_of_delete':float(.5*(b_score[0]+b_score[2])),
        'delete_search_worse_than_known_train':bool(b_score[0]+b_score[2]>a_score[0]+a_score[2]+1e-12)})
for method in ['random','channel','data','train_joint','delete_joint']:
    rr=[r for r in fixed if r['method']==method]
    assert len(rr)==60
    for r in rr:
        c=r['metrics']['cost'];n=r['size_before']-1;D=38282
        assert c['ul_reals']==160*D and c['local_calls']==160*n
        assert c['total_re']==160*(D+8*n+(32*D+64*n+64)/2)
    fixedsummary.append({'method':method,'test':float(np.mean([r['metrics']['test'] for r in rr])),
        'mean_re':float(np.mean([r['metrics']['cost']['total_re'] for r in rr])),
        'mean_ul':float(np.mean([r['metrics']['cost']['ul_reals'] for r in rr])),
        'mean_calls':float(np.mean([r['metrics']['cost']['local_calls'] for r in rr]))})
for r in records:
    D=r['D'];seed=r['seed'];method=r['method'];route=read(R/f'{seed}_routing.json');h=np.array(route['h']);hist=np.array(route['histograms'])
    groups=r['groups'];assert sorted(i for g in groups for i in g)==list(range(20));assert all(3<=len(g)<=8 for g in groups)
    q=float(np.mean([1/(len(g)**2*min(h[g])**2) for g in groups]));dq=float(sum(1/((len(g)-1)**2*min(h[[i for i in g if i!=u]])**2) for g in groups for u in g)/20)
    assert abs(q-r['routing_score'][1])<1e-9 and abs(dq-r['routing_score'][2])<1e-9
    for p in [R/f'{seed}_{method}_source.json',*R.glob(f'{seed}_{method}_delete_*.json')]:
        obj=read(p);u=obj.get('summary',{}).get('client');cost=0;events=obj['events']
        for e in events:
            expected=[i for i in groups[e['shard']] if i!=u];assert e['ids']==expected and e['repeats']==1
            n=len(expected);assert e['ul_reals']==D
            assert abs(e['expected_mse']-.01/(n*n*e['min_h']**2))<1e-10
            assert e['max_power']<=1.00001;maxpower=max(maxpower,e['max_power'])
            re=D+8*n+(32*D+64*n+64)/2;assert e['total_re']==re
            cost+=re;audit_events+=1
        if u is not None:
            selected=obj['summary']['selected'];T=selected['round'];chosen=[e for e in events if e['round']<=T]
            assert len(chosen)==T and selected['cost']['total_re']==sum(e['total_re'] for e in chosen)
            assert obj['summary']['matched']==any(v['dev']>=obj['summary']['target_dev'] for v in obj['curve'])
            if obj['summary']['matched']:assert T==next(v['round'] for v in obj['curve'] if v['dev']>=obj['summary']['target_dev'])
        else:assert len(events)==640 and obj['cost']['total_re']==cost+3840
summary=[];methods=['random','channel','data','train_joint','delete_joint']
for method in methods:
    rr=[r for r in records if r['method']==method];dd=[d for r in rr for d in r['deletions']]
    costs=np.array([d['selected']['cost']['total_re'] for d in dd]);rounds=[d['selected']['round'] for d in dd]
    summary.append({'method':method,'matched':sum(d['matched'] for d in dd),'total':len(dd),'mean_rounds_or_cap':float(np.mean(rounds)),
        'mean_re_or_lowerbound':float(costs.mean()),'median_re_or_lowerbound':float(np.median(costs)),'worst_re_or_lowerbound':float(costs.max()),
        'source_test':float(np.mean([r['source_metrics']['test'] for r in rr])),'delete_test':float(np.mean([d['selected']['test'] for d in dd])),
        'mean_source_re':float(np.mean([r['source_cost']['total_re'] for r in rr])),
        'mean_delete_calls':float(np.mean([d['selected']['cost']['local_calls'] for d in dd])),
        'source_noise_proxy':float(np.mean([r['routing_score'][1] for r in rr])),
        'delete_noise_proxy':float(np.mean([r['routing_score'][2] for r in rr])),
        'sizes':[r['sizes'] for r in rr]})
by={s['method']:s for s in summary};a=by['train_joint'];b=by['delete_joint']
comparison={'source_test_delta_pp':100*(b['source_test']-a['source_test']),'delete_test_delta_pp':100*(b['delete_test']-a['delete_test']),
            'delete_re_saving_pct_or_cap_comparison':100*(1-b['mean_re_or_lowerbound']/a['mean_re_or_lowerbound']),
            'both_all_matched':a['matched']==b['matched']==60,
            'delete_proxy_improvement_pct':100*(1-b['delete_noise_proxy']/a['delete_noise_proxy'])}
comparison['candidate_all_matched']=b['matched']==60
comparison['saving_is_lower_bound']=b['matched']==60 and a['matched']<60
comparison['candidate_pass']=bool(comparison['candidate_all_matched'] and comparison['source_test_delta_pp']>=-1 and comparison['delete_test_delta_pp']>=-1 and comparison['delete_re_saving_pct_or_cap_comparison']>=10)
paired=[]
for seed in [62001,62002,62003]:
    aa=next(r for r in records if r['seed']==seed and r['method']=='train_joint');bb=next(r for r in records if r['seed']==seed and r['method']=='delete_joint')
    for x,y in zip(aa['deletions'],bb['deletions']):
        paired.append({'seed':seed,'client':x['client'],'both_matched':x['matched'] and y['matched'],'train_rounds':x['selected']['round'],'delete_rounds':y['selected']['round'],
            'train_re':x['selected']['cost']['total_re'],'delete_re':y['selected']['cost']['total_re'],'test_delta_pp':100*(y['selected']['test']-x['selected']['test'])})
comparison['paired_faster']=sum(p['both_matched'] and p['delete_rounds']<p['train_rounds'] for p in paired)
comparison['paired_equal']=sum(p['both_matched'] and p['delete_rounds']==p['train_rounds'] for p in paired)
comparison['paired_slower']=sum(p['both_matched'] and p['delete_rounds']>p['train_rounds'] for p in paired)
ref=[d['reference_maxabs'] for r in records for d in r['deletions'] if d['reference_maxabs'] is not None]
audit={'passed':True,'events':audit_events,'max_power':maxpower,'reference_cases':len(ref),'reference_maxabs':max(ref),'unchanged_cases':sum(d['unchanged_source'] for r in records for d in r['deletions']),'seconds':time.perf_counter()-started}
write('analysis.json',{'summary':summary,'comparison':comparison,'paired':paired,'fixed160_summary':fixedsummary,'search_audit':searchaudit});write('audit.json',audit)
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+str(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
result_table=table(['배정','도달/60','최초 test','삭제후 test','삭제 rounds 평균¹','삭제 RE 평균¹','최대 RE¹','삭제 local calls'],[[s['method'],f"{s['matched']}/60",f"{100*s['source_test']:.2f}%",f"{100*s['delete_test']:.2f}%",f"{s['mean_rounds_or_cap']:.1f}",f"{s['mean_re_or_lowerbound']/1e6:.2f}M",f"{s['worst_re_or_lowerbound']/1e6:.2f}M",f"{s['mean_delete_calls']:.1f}"] for s in summary])
score_table=table(['방법','3seed shard 크기','학습전 q','삭제후 q'],[[s['method'],str(s['sizes']),f"{s['source_noise_proxy']:.3f}",f"{s['delete_noise_proxy']:.3f}"] for s in summary])
seed_table=table(['seed','방법','shard크기','source test','도달','삭제round 평균¹','삭제test'],[[r['seed'],r['method'],r['sizes'],f"{100*r['source_metrics']['test']:.2f}%",sum(d['matched'] for d in r['deletions']),f"{np.mean([d['selected']['round'] for d in r['deletions']]):.1f}",f"{100*np.mean([d['selected']['test'] for d in r['deletions']]):.2f}%"] for r in records])
individual=table(['seed','client','둘다도달','학습중심 rounds','삭제중심 rounds','test차이(%p)'],[[p['seed'],p['client'],p['both_matched'],p['train_rounds'],p['delete_rounds'],f"{p['test_delta_pp']:+.2f}"] for p in paired])
source_slots=4*160;modelbytes=records[0]['model_and_initial_bytes'];D=records[0]['D']
judgment='사전 실용후보 기준 충족' if comparison['candidate_pass'] else '사전 실용후보 기준 미달'
fixedtable=table(['배정','삭제160 test','UL uses','평균 삭제 RE','평균 local calls'],[[r['method'],f"{100*r['test']:.2f}%",f"{r['mean_ul']:,.0f}",f"{r['mean_re']/1e6:.3f}M",f"{r['mean_calls']:.1f}"] for r in fixedsummary])
body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>삭제 후 집계를 고려한 AirComp shard 배정</title><style>body{{font-family:Segoe UI,Malgun Gothic,sans-serif;margin:36px auto;max-width:1280px;padding:0 24px;color:#213047;line-height:1.75}}h1{{font-size:29px}}h2{{margin-top:32px;font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5;margin:18px 0}}td,th{{padding:10px;border:1px solid #dce3ed;text-align:left}}th{{background:#edf3fc}}.box{{padding:22px;border-left:5px solid #3273dc;background:#f0f6ff}}.note{{background:#fff7e6;padding:16px}}a{{color:#175cd3}}code{{background:#f1f3f5;padding:2px 4px}}</style>
<h1>삭제 후 남는 집합까지 고려한 고정 shard 배정</h1>
<p class="box"><b>가설</b> 동일 슬롯 예산에서 삭제후 집계품질을 고려하면 실제 삭제 통신이 감소한다.<br><b>독립변수</b> 5가지 배정 목적함수 · <b>종속변수</b> 정확도/삭제후 목표도달 통신/연산/집계오차 · <b>통제</b> 20clients, S=4, source160rounds, R=1, 전력1 · <b>상태</b> 15학습조건·300개별삭제·15재현검증 완료</p>
<h2>먼저: 원래 학습과 같은160round 삭제 재학습</h2><p>300개 삭제 모두160round 결과를 확인했다. 이 조건에서는 <b>모든 배정의 analog uplink 사용량이 동일</b>하다. 배정은 집계 잡음·정확도·로컬 연산·pilot/control 비용을 바꾼다. 고정round에서 정확 삭제 통신이 크게 줄었다고 주장할 수 없다.</p>{fixedtable}
<p class="note">실행 중 평가 정의를 보완했다. 원160round 학습에서 조기중단 삭제모델이 같은160round 삭제제외 모델과 같다는 보장은 없다. 따라서 조기 정확도 도달 비용을 exact unlearning 절감으로 표현하지 않는다. 원사전등록은 보존하고 <a href="AMENDMENT_FIXED160.txt">수정 기록</a>에 이유와 추가 실행 범위를 공개했다.</p>
<h2>보조 지표 — 정확도 도달 비용: {judgment}</h2><p>delete_joint 대 train_joint: 최초 test {comparison['source_test_delta_pp']:+.2f}%p, 조기선택 삭제후 test {comparison['delete_test_delta_pp']:+.2f}%p. 목표정확도 도달 통신 비교 {comparison['delete_re_saving_pct_or_cap_comparison']:.2f}% 절감 방향(음수면 증가). 양쪽 전원 목표달성: {comparison['both_all_matched']}. 동일 client paired 비교 빠름/같음/느림: {comparison['paired_faster']}/{comparison['paired_equal']}/{comparison['paired_slower']}.</p>
<p class="note">¹ 목표 미도달 건은240round에서 중단. 이 경우 표의 평균·최대 비용은 달성 비용이 아닌 실행상한을 넣은 하한 요약이다. 후보가 전원 도달하고 비교군만 미도달한 경우 비용 절감은 보수적 하한으로 해석 가능하며, 후보에도 미도달이 있으면 전체 달성비용 절감을 주장하지 않는다. 현재 절감 하한 여부: {comparison['saving_is_lower_bound']}. 정확도 허용폭1%p, 측정간격40round이므로 엄밀한 동일정확도/정밀 지연 측정은 아님.</p>
{result_table}
<h2>AirCluster와 무엇이 다른가</h2>
{table(['항목','AirCluster 원문','이번 실행'],[['목표','개인화된 cluster 모델 학습','삭제 가능한 독립 shard 모델 및 삭제 비용'],['배정','학습중 local loss 기반 선택','학습 전 고정; 삭제 후 집합을 score에 반영'],['무선','MIMO ZF + sketching, 동시 분리','SISO 순차4blocks, 채널역보상, R=1'],['삭제','삭제 절차를 제안한 논문 아님','모든client 개별삭제 및 affected shard 재학습'],['모델평가','사용자/그룹별 모델 품질','공통 test에서4모델 ensemble'],['이 실험의 의미','전체 AirCluster 성능 재현/비교는 미실행','동일 PHY 아래 배정목적의 효과를 분리']])}
<p>AirCluster와 다른 목표를 정했다는 사실만으로 신규성이 입증되지는 않는다. <b>이 보고서는 AirCluster보다 우수하다는 결과가 아니다.</b> AirCluster 자원분리 기전을 새로운 제안으로 주장하지 않는다. <a href="https://basakguler.github.io/SG_TWC_2023.pdf">AirCluster 원문</a> · <a href="https://iqua.ece.toronto.edu/papers/ningxinsu-infocom23.pdf">KNOT 원문</a>.</p>
<h2>문제와 알고리즘</h2><p>삭제확률 예측 없음. 모든client를 한 번씩 삭제한 균일 평균을 평가한다. 정확한 목표는 평균 실제 재학습 비용이지만, 학습 전에 이를 알 수 없으므로 본 후보는 <b>삭제 후 집계잡음 대리값</b>을 최적화한다. 대리값 개선을 실제 비용 절감이라고 부르지 않는다.</p>
<p><code>q(G)=1/(|G|² min(h_i)²)</code><br><code>Q_train=mean_shards q(G)</code><br><code>Q_delete=mean_clients q(G(u) without u)</code><br><code>D_data=mean_shards JS(mean client histogram, global mean histogram)</code></p>
{table(['방법','최소화 항'],[['random','balanced random5/5/5/5'],['channel','Q_train'],['data','D_data'],['train_joint','0.5 normalized D_data + 0.5 normalized Q_train'],['delete_joint','0.5 normalized D_data + 0.5 normalized Q_delete']])}
<p>정규화 분모는 각seed의 random배정. 동일2시작점(random/채널순), 최대15회 best improvement 이동·교환, shard3~8명. 미래 fading/검증정확도/삭제결과를 검색에 사용하지 않음. labels 전체 공개 대신client별10-class count 요약과 base채널 사용. 검색 휴리스틱이며 전역최적보장 없음.</p>{score_table}
<p class="note">검색 한계도 확인했다. 알려진 train_joint 배정을 delete 목적함수에 대입했을 때 delete_joint 검색결과보다 좋은 seed 수: {sum(x['delete_search_worse_than_known_train'] for x in searchaudit)}/3. 즉 동일2시작점·15회 탐색이 더 좋은 기존해를 놓칠 수 있다. 이번 결과는 현재 휴리스틱의 결과이며 삭제목적의 전역최적해 비교가 아니다. 결과를 보고 재검색하지 않았으며, 후속 버전에서는 train_joint 배정을 후보에 반드시 포함하는 수정이 필요하다.</p>
<h2>상세 실험 조건</h2>
{table(['구분','설정'],[['데이터','FashionMNIST train12000 / dev2000 / test10000; Dirichlet .5'],['반복','seeds62001,62002,62003; 각seed 동일 partition/초기화/채널 비교'],['모델',f'CNN {D:,} parameters, 같은초기벡터를 모든shard에 적용'],['로컬학습','2SGD steps × batch64, lr.05, momentum0, L2 update clip1'],['shard','4개; 각3~8명; 전원round0 참여; 이동없음'],['최초학습','모든방법160rounds, 각round4blocks'],['무선','flat SISO, 완전CSI/동기화, P=1평균symbol전력, sigma²=.01'],['채널','h=10^(Uniform[-20,0]/20), 매round×Uniform(.85,1.15), 데이터와독립'],['송신','R=1; 한block은D real channel uses; OFDM/MIMO/간섭없음'],['집계','equal-client평균; beta=1/(n*h_min*sqrt(D)); MSE=.01/(n²h_min²)'],['삭제','20명 전부 각자독립삭제; affected shard초기화재학습; 다른모델보존'],['정확도목표','각seed/client의random삭제160round dev−1%p; test선택금지'],['삭제중단','40round간격 최초dev목표도달, 최대240; random160까지기준측정'],['정확삭제범위','고정 공개배정/일정 조건부 모델검증; metadata삭제보장 아님']])}
<h2>자원 예산을 어떻게 맞췄나</h2><p>모든 source는 {source_slots}blocks={source_slots*D:,} analog uses. 삭제는1round당1block만 점유하며 남은3blocks를 공짜 반복전송에 사용하지 않는다. 한round가4slot고정프레임이면 삭제대기시간은4D×rounds, 점유량은D×rounds로 구분. 이번 표는 점유자원 비용이다.</p>
<p>DL32D bits/shard broadcast, pilots8n uses, control64n+64 bits. RE=UL+pilots+(DL+control)/2. DL6bits/use도 원자료에 기록. 실제 RF초/Joule 아님. 초기metadata7680bits(10class counts+count+basechannel) 동일허용; 추가privacy보장없음. source시행별최종4모델+shared초기상태 {modelbytes/1024:.1f}KiB, 메타데이터별도. 실험snapshot파일은 서비스저장량이 아님.</p>
<h2>seed별 결과</h2>{seed_table}<details><summary>train_joint vs delete_joint 전체60쌍</summary>{individual}</details>
<h2>검증·한계·다음 판단</h2><p>물리계수/전력/고정R/통신계산 {audit_events:,}events 검산 통과. source모델불변300건, client0각조건 독립재학습15건 최대parameter오차 {audit['reference_maxabs']}. 전체4shards 독립재실행을300번한것은아님. source배정이data를쓰므로 retained routing정보까지제거한unconditional exactness는주장하지않음. AirComp는암호학적secure aggregation이아님.</p>
<p>3seed와한데이터셋/작은CNN, 고정명목20dB, 이상적CSI에한정. 실제dataset-channel상관·순차다중삭제·실제RF·AirCluster MIMO·독립noise분포보장 미검증. 모든방법의초기학습량은같지만초기품질은다를수있음. ensemble평가라타shard정확도가삭제후목표도달에영향을주므로source품질을함께표시. threshold표본오차/40round격자/3seed불확실성을남긴다.</p>
<p>다음판단: 사전기준충족이면 learned-cost proxy 및 추가seed/물리계층검증을검토; 미달이면 actual학습곡선과q대리값의괴리를분석하고이번설계의실패로보고. 결과를보고가중치나seed를바꿔성공으로만들지않음. 후속실험은이번실행에포함하지않음.</p>
<p>fixed160 보완: 기존curve160을 재사용하거나 조기 종료모델에서160까지 연장. client0각조건 초기화 reference15건 추가, metric error 최대 {max(a['metric_error'] for a in fixedaudit):.2g}. 원자료 <a href="fixed160_results.json">300개 fixed160 결과</a> 및 <a href="fixed160_audit.json">검증</a>.</p>
<h2>실제 실행 비용과 원자료</h2><p>기존 실행 {resource['seconds']/60:.2f}분 + fixed160 보완 {extra['seconds']/60:.2f}분 = {(resource['seconds']+extra['seconds'])/60:.2f}분; GPU보드 합계 {resource['gpu_board_Wh']+extra['gpu_board_Wh']:.3f}Wh(5초샘플, 호스트전체전력아님); local training calls 합계 {resource['local_calls']+extra['local_calls']:,}. 외부결제0원, 전기요금미산정. 비용감사 {audit['seconds']:.2f}초 별도.</p>
<p><a href="PREREG.txt">사전등록 상세조건</a> · <a href="results.json">15조건·300삭제 요약</a> · <a href="analysis.json">비교분석·paired결과</a> · <a href="audit.json">검산</a> · <a href="resource.json">GPU비용</a> · <a href="run.py">실행코드</a> · <a href="report.py">분석코드</a> · <a href="provenance.json">코드SHA256</a></p><p>각seed_method_source.json 및 seed_method_delete_client.json에 전체집계events와학습곡선; seed_routing.json에배정·score·검색trace·사용metadata 저장.</p></html>'''
with (R/'report.html').open('x',encoding='utf-8') as f:f.write(body)
print(json.dumps({'summary':summary,'comparison':comparison,'fixed160':fixedsummary,'audit':audit,'resource':{k:v for k,v in resource.items() if k!='samples'},'extra_resource':{k:v for k,v in extra.items() if k!='samples'}},ensure_ascii=False,indent=2))
