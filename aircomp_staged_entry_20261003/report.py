from pathlib import Path
import json,math,html
import numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
rows=read(R/'results.json');resource=read(R/'resource.json');audits=[]
for row in rows:
    seed,name=row['seed'],row['policy'];source=read(R/f'{seed}_{name}_source.json');D=row['D']
    for p in [R/f'{seed}_{name}_source.json',*R.glob(f'{seed}_{name}_delete_*.json')]:
        obj=read(p);de=obj.get('summary',{}).get('deleted');events=obj['events'];total=0
        for e in events:
            expected=[i for j,i in enumerate(source['groups'][e['c']]) if e['t']>=row['entry'][j] and i!=de]
            assert e['ids']==expected
            n=len(expected);repeat=max(1,math.ceil(.01/(n*n*e['min_h']**2*1e-4)))
            assert repeat==e['repeats']
            re=repeat*D+8*n+(64*n+64)/2+32*D/2
            assert abs(re-e['total_re'])<1e-6
            assert e['expected_mse']<=1e-4*(1+1e-10)
            assert e['max_client_symbol_power']<=1.00001
            total+=re
        if de is not None:
            assert len(events)==row['selected']['round']-obj['summary']['entry_round']
            assert abs(total-obj['summary']['cost']['total_re'])<1e-6
        audits.append({'file':p.name,'events_checked':len(events)})
summary=[]
for name in ['all','mild','late']:
    rr=[r for r in rows if r['policy']==name];base=[r for r in rows if r['policy']=='all']
    initial=np.mean([r['selected']['cost']['total_re'] for r in rr]);delete=np.mean([r['expected_delete_re'] for r in rr]);bi=np.mean([r['selected']['cost']['total_re'] for r in base]);bd=np.mean([r['expected_delete_re'] for r in base])
    pre=np.mean([r['selected']['test']-b['selected']['test'] for r,b in zip(rr,base)])
    post=np.mean([r['mean_deleted_test']-b['mean_deleted_test'] for r,b in zip(rr,base)])
    saving=1-delete/bd;match=all(r['matched'] for r in rr)
    summary.append({'policy':name,'matched':match,'rounds':[r['selected']['round'] for r in rr],'test_delta_pp':100*pre,'post_delete_delta_pp':100*post,
        'initial_re':float(initial),'delete_re':float(delete),'initial_increase_pct':100*(initial/bi-1),'delete_saving_pct':100*saving,
        'one_delete_total_change_pct':100*((initial+delete)/(bi+bd)-1),'break_even_requests':max(0,float((initial-bi)/(bd-delete))) if delete<bd else None,
        'candidate':bool(name!='all' and match and pre>=-.01 and post>=-.01 and saving>=.1)})
maxerr=max(d['maxabs_reference'] for r in rows for d in r['deletions'])
rankdiag=[]
for name in ['all','mild','late']:
    rr=[r for r in rows if r['policy']==name]
    for j in range(5):
        dd=[r['deletions'][j] for r in rr]
        rankdiag.append({'policy':name,'position':j+1,'entry':rr[0]['entry'][j],
            'mean_rounds':float(np.mean([d['replay_rounds'] for d in dd])),
            'mean_re':float(np.mean([d['cost']['total_re'] for d in dd])),
            'mean_local_calls':float(np.mean([d['cost']['local_calls'] for d in dd]))})
audit={'passed':True,'paired_replay_cases':45,'maxabs_reference':maxerr,'checks':audits,'scope':'same-seed affected-shard from-scratch reference; unchanged shards reused, not independent four-shard retraining'}
for name,obj in [('audit.json',audit),('analysis.json',summary),('participation_diagnostic.json',rankdiag)]:
    with (R/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
tr=''.join(f"<tr><td>{s['policy']}</td><td>{s['rounds']}</td><td>{s['test_delta_pp']:+.2f}</td><td>{s['post_delete_delta_pp']:+.2f}</td><td>{s['initial_increase_pct']:+.1f}%</td><td>{s['delete_saving_pct']:.1f}%</td><td>{s['one_delete_total_change_pct']:+.1f}%</td><td>{'후보 기준 충족' if s['candidate'] else ('기준선' if s['policy']=='all' else '후보 기준 미달')}</td></tr>" for s in summary)
detail=''.join(f"<tr><td>{r['seed']}</td><td>{r['policy']}</td><td>{100*r['at160']['test']:.2f}%</td><td>{r['selected']['round']}</td><td>{100*r['selected']['dev']:.2f}%</td><td>{100*r['selected']['test']:.2f}%</td><td>{100*r['mean_deleted_test']:.2f}%</td><td>{r['expected_delete_re']/1e6:.2f}M</td><td>{r['checkpoint_bytes']/1024:.1f} KiB</td></tr>" for r in rows)
deletion=''.join(f"<tr><td>{r['seed']}/{r['policy']}</td><td>{d['deleted']}</td><td>{d['entry_round']}</td><td>{d['replay_rounds']}</td><td>{100*d['test']:.2f}%</td><td>{d['cost']['total_re']/1e6:.2f}M</td><td>{d['maxabs_reference']:.2g}</td></tr>" for r in rows for d in r['deletions'])
decisions=''.join(f"<li>{s['policy']}: 평균 삭제 비용 {s['delete_saving_pct']:.1f}% 절감, 최초 통신 {s['initial_increase_pct']:+.1f}%, 삭제 후 정확도 차이 {s['post_delete_delta_pp']:+.2f}%p. 선형 비용 모형 손익분기 요청 수: {s['break_even_requests']}.</li>" for s in summary[1:])
ranktable=''.join(f"<tr><td>{r['policy']}</td><td>{r['position']}</td><td>{r['entry']}</td><td>{r['mean_rounds']:.1f}</td><td>{r['mean_re']/1e6:.2f}M</td><td>{r['mean_local_calls']:.1f}</td></tr>" for r in rankdiag)
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>단계적 참여: SISA·AirComp 실험</title><style>body{{font-family:Segoe UI,Malgun Gothic,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#1d2939;line-height:1.65}}table{{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}}th,td{{border:1px solid #d0d5dd;padding:9px;text-align:left}}th{{background:#edf3fb}}.box{{padding:20px;background:#eef5ff;border-radius:12px}}h1{{font-size:28px}}a{{color:#175cd3}}</style>
<h1>클라이언트 참여 시점을 늦추면 삭제 replay가 싸지는가?</h1>
<p class="box"><b>가설</b> 정확도 손실 1%p 이내에서 삭제 통신 10% 이상 절감 · <b>독립변수</b> client 최초 참여 시점 · <b>종속변수</b> 정확도/최초·삭제 통신/저장량 · <b>통제</b> K=4·고정 배정·동일 모델/데이터/채널 · <b>상태</b> 학습9조건, 삭제45건 및 각각의 초기화 reference 실행 완료</p>
<p>FashionMNIST 실제 이미지, train12000/dev2000/test10000, CNN {rows[0]['D']:,} parameters, 20clients, shard당5명, 3seeds. shard별 순차 SISO AirComp. OFDM·ZF·shard간 간섭은 사용하지 않음. 목표 잡음 MSE를 고정하고 반복 송신 수를 조절.</p>
<h2>무엇을 바꿨나</h2><p>all: [0,0,0,0,0], mild: [0,0,20,40,60], late: [0,0,40,80,120]. 같은 그룹의 고정 random 순서에 적용. 참여 후에는 계속 학습. 삭제위험·채널·데이터를 이용한 순서 최적화는 아직 하지 않았음.</p>
<p>기준선160라운드 dev 정확도−1%p를 처음 만족하는 시점을 160~240 사이 20라운드 간격으로 선택. test는 선택에 사용하지 않음. 이는 엄밀히 같은 정확도가 아닌 <b>1%p 허용 비교</b>. 정책별240까지 탐색에 사용한 실제 실행비용도 아래 자원 보고에 포함.</p>
<h2>3seed 평균 결과</h2><table><tr><th>정책</th><th>선택 rounds</th><th>학습후 test 차이(%p)</th><th>삭제후 test 차이(%p)</th><th>최초 통신 변화</th><th>삭제 통신 절감</th><th>최초+삭제1회 변화</th><th>사전 기준</th></tr>{tr}</table><ul>{decisions}</ul>
<p>통신은 DL2bits/use를 가정한 정규화 channel uses. RF 실측 시간이나 에너지 아님. 손익분기는 동일한 단일삭제 평균을 선형 반복한 계획식이며 연속삭제 실행결과가 아님. 삭제는 shard0의5명을 각각 삭제한 값으로 전체20명의 평균이 아님.</p>
<h2>참여 순서에 따른 삭제 비용</h2><p>전체 평균만으로는 초기 참여자와 후기 참여자의 차이를 놓칠 수 있음. 아래는 동일 위치의 client를 삭제한 3seed 평균. 집계 잡음은 C²σ²/(n²h_min²R)에 비례하므로, 소수 참여자 특히 삭제 후 한 명만 남는 초기 구간은 같은 목표 오차를 맞추기 위한 반복 송신이 늘어날 수 있음. 이는 이 실험의 fixed-MSE·채널반전 가정에 의존하며 모든 AirComp에 대한 결론이 아님.</p><table><tr><th>정책</th><th>그룹내 위치</th><th>최초참여</th><th>replay rounds</th><th>삭제 RE</th><th>로컬 학습 calls</th></tr>{ranktable}</table>
<h2>seed별 결과와 저장량</h2><table><tr><th>seed</th><th>정책</th><th>160 test</th><th>선택 round</th><th>dev</th><th>test</th><th>삭제후 평균 test</th><th>평균 삭제 RE</th><th>4shards checkpoints</th></tr>{detail}</table>
<p>저장량은 초기 상태를 포함한 서로 다른 진입 시점의 FP32 weights. 별도로 최종4모델 {rows[0]['final_model_bytes']/1024:.1f} KiB 및 배정/일정 메타데이터 필요. SGD momentum 없음. 실제 실험 states.pt는 평가용 snapshot과 삭제 결과까지 포함하므로 서비스 저장량과 다름.</p>
<h2>삭제와 replay 검증</h2><p>각 client 최초참여 직전 checkpoint로 복원하여 재학습. 해당 client를 처음부터 제외한 동일 일정의 affected-shard reference를 독립 실행. 45/45 maxabs≤1e-6, 최대 {maxerr:.3g}. 나머지 shard는 독립성이 고정되어 재사용. 고정난수 수치 일치이며 신규 잡음에서의 분포적 보장이나 개인정보 보호 검증이 아님.</p>
<details><summary>삭제45건 상세</summary><table><tr><th>seed/정책</th><th>삭제client</th><th>최초참여</th><th>replay rounds</th><th>test</th><th>RE</th><th>reference 오차</th></tr>{deletion}</table></details>
<h2>실행 비용·한계·다음 판단</h2><p>학습/검증 실행 {resource['seconds']/60:.2f}분, local training calls {resource['local_calls']:,} (각2steps), 집계 {resource['aggregate_rounds']:,}회. GPU 보드 전력 샘플 적분 {resource['gpu_board_Wh']:.3f} Wh, 호스트 전체 전력 아님. 외부 결제0, 전기요금 미산정.</p>
<p>3seed·한 데이터셋·작은 CNN·합성 채널·완전 CSI. 채널과 데이터의 현실 상관관계 검증 아님. 초기 소수 client 학습 편향과 초기 낮은 참여자 수의 반복송신 증가를 포함. 이 실험은 일반적인 단계적 참여의 타당성 선별이지 AirComp 특화 신규성 입증이 아님. 사전 기준 미달이면 추가 sweep 없이 중단하고 원인을 보고. 기준 충족 시에만 data/채널을 고려한 순서 최적화의 필요성을 후속 검토.</p>
<p><a href="PREREG.txt">실행 전 가설/조건</a> · <a href="results.json">결과 원자료</a> · <a href="analysis.json">비교 수치</a> · <a href="audit.json">독립 비용 감사</a> · <a href="resource.json">자원 로그</a> · <a href="run.py">실행 코드</a> · <a href="provenance.json">코드 SHA256</a></p></html>'''
with (R/'report.html').open('x',encoding='utf-8') as f:f.write(page)
print(json.dumps({'summary':summary,'resource':{k:v for k,v in resource.items() if k!='samples'},'maxerr':maxerr},ensure_ascii=False,indent=2))
