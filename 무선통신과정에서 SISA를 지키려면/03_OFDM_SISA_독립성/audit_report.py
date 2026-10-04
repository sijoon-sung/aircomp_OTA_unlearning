from pathlib import Path
import numpy as np,json,time
P=Path(__file__).resolve().parent;t=time.perf_counter();out=json.loads((P/'results.json').read_text(encoding='utf-8'));a=np.load(P/'raw.npz');maxerr=0.
def solve(M,b,N):
 power=np.eye(27);v=np.zeros(27)
 for n in N[::-1]:v+=power@(b-.04*n);power=power@M
 return v.reshape(3,9)
for r in out['rows']:
 prefix=r['key'].rsplit('_delete',1)[0];seed=r['seed'];g=r['deleted_shard'];C=a[prefix+'_C'];N=a[prefix+'_N'];H=a[f's{seed}_H'];v=a[f's{seed}_v']
 def reference(active):
  counts=np.bincount([i//4 for i in active],minlength=3);scale=np.repeat(1/np.maximum(counts,1),9);G=np.zeros((108,27));b=np.zeros(108)
  for i in active:G[i*9:(i+1)*9,(i//4)*9:(i//4+1)*9]=H[i];b[i*9:(i+1)*9]=v[i]
  K=C*scale[:,None];return solve(np.eye(27)-.04*K@G,.04*K@b,N*scale)
 full=reference([i for i in range(12) if i!=4*g+3]);maxerr=max(maxerr,float(np.max(np.abs(full-a[r['key']+'_full']))))
 if g==0:maxerr=max(maxerr,float(np.max(np.abs(reference(list(range(12)))-a[prefix+'_source']))))
 assert np.array_equal(a[r['key']+'_local'][np.arange(3)!=g],a[prefix+'_source'][np.arange(3)!=g])
 if r['mode']=='oracle_no_cross' or r['eps']==0:assert r['unaffected_change']<=1e-9 and r['replay_gap']<=1e-9
assert maxerr<1e-10
audit=dict(status='passed',full_references_recomputed=1920,source_recomputed=640,max_affine_error=maxerr,waveform_error=out['audit_waveform_max_error'],seconds=time.perf_counter()-t)
(P/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
tr=''
for s in out['summary']:
 if s['mode']!='ofdm':continue
 tr+='<tr>'+''.join(f'<td>{v}</td>' for v in [s['guard'],s['eps'],f"{s['propagation']}/{s['n']}",f"{100*s['mean_unaffected']:.6f}%",f"{100*s['mean_gap']:.6f}%",f"{100*s['max_gap']:.6f}%",f"{s['mean_full_MSE']:.8f}",f"{s['mean_local_MSE']:.8f}"])+'</tr>'
doc=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SISA 학습과 OFDM AirComp 독립성</title><style>body{{font-family:Malgun Gothic,sans-serif;max-width:1250px;margin:35px auto;padding:15px;line-height:1.8;color:#213345}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd6df;padding:10px}}th,.note{{background:#edf3f7}}.note{{padding:18px}}h2{{margin-top:28px}}</style><h1>SISA로 분리해 학습해도 OFDM 잔여 간섭으로 독립성이 깨지는가?</h1><p><b>실행 완료 · 합성 회귀 학습+OFDM IFFT/FFT 기반 모형 · 실제 무선 장비 실험 아님</b></p>
<p class="note"><b>가설:</b> 고정된 독립 shard라도 다른 shard 신호가 수신되면 삭제 영향이 다른 모델로 전파된다.<br><b>독립변수:</b> residual CFO 범위0/.01/.05/.1, guard0/4tones, 잡음0/.01, OFDM/교차간섭만 제거한oracle.<br><b>종속변수:</b> 미삭제 모델 변화·부분/전체 재학습 차이·test MSE.<br><b>통제:</b>12clients/3shards, 모델·데이터·초기화·100rounds·paired noise.</p>
<h2>결론</h2><p><b>이상적 동기화(CFO0)는240/240 삭제 비교에서 격리 유지.</b> 잡음이 있어도 유지됐다. 따라서 일반 AirComp가 항상 SISA를 깨뜨린다고 결론낼 수 없다.<br><b>CFO가 있는 OFDM은720/720에서 미삭제 shard 모델이 변했다.</b> 변화 크기는 아래 표와 같으며 유의미한 공격 위험이나 큰 성능 손실이라고 자동 해석하지 않는다.<br><b>교차shard 항만 제거한oracle는960/960에서 격리 유지.</b> 같은 shard 내부의 CFO 왜곡을 그대로 둬도 유지되어, 영향 전파의 원인이 교차shard 간섭임을 이 모형에서 분리했다.</p>
<h2>주요 수치</h2><p>CFO 범위 e는 client별 오차가 부반송파 간격의 ±e 이내라는 뜻이다. 예를 들어 .01은 ±1% 조건. 실제 배치에서 이 값이 일반적이라는 주장은 하지 않는다. 각 행120건=20seeds×2noise×3삭제. 모델변화(%)는 norm(diff)/max(1,norm(reference))×100이며 정확도 하락률이 아니다.</p><div style="overflow:auto"><table><tr><th>guard</th><th>CFO 범위</th><th>미삭제 shard 변화</th><th>평균 미삭제 모델 변화</th><th>평균 부분/전체 모델오차</th><th>최대 부분/전체 오차</th><th>전체 test MSE</th><th>부분 test MSE</th></tr>{tr}</table></div>
<h2>SISA와 비교 기준의 정의</h2><p>client는 처음부터 끝까지 고정shard에 속하고 자기shard모델로만 로컬 gradient를 계산한다. 모델 간 averaging/지식전달은 없다. 최종 평가에서만 세 모델의 예측을 평균한다. 모든 client가 round0부터 참여하므로 안전한 rollback은 초기화다. 중간 checkpoint로 전체client를 삭제할 수 있다고 가정하지 않았다.</p><p><b>full reference:</b> client한명을 삭제하고 모든shard를 동일통신규칙·동일난수로 처음부터 재학습.<br><b>local replay:</b> 해당shard만 재학습, 나머지는 원래최종모델유지·송신하지않음.<br><b>주지표:</b> full reference에서 미삭제shard가 원래모델과 달라지는지. local-only에서는 다른shard의송신도사라지므로 replay오차만으로 삭제영향을판정하지 않는다. 삭제없는sham local replay도results.json에모두기록했다.</p>
<h2>물리 모형과 한계</h2><p>SISO OFDM64tones, 각shard9tones의서로다른주파수사용, 같은shard의4clients만같은tone에동시송신. client별IFFT신호에CFO위상회전을적용한뒤합산·FFT한다. 직접파형합산과행렬표현의일치검증을수행했다. CP8은flat채널·완전timing에서정상제거한유효블록을계산하여결과에영향없음. multipath·CP초과지연·PA clipping·pilot·CFO추적은미구현. common phase는보상하고잔여CFO의ICI만남긴조건이다.</p><p>초기데이터는이전합성회귀조건과같은20seeds를사용한다. CFO와noise는새로운고정난수. 본결과는신경망·실측일반성·분포적exact-unlearning의불가능성·보안/DP보장이아니다. 모델상태변화가0이아니라는것과실용적으로큰삭제실패는구분한다. oracle은원인분석도구이지제안알고리즘이아니다. 새로운방법의신규성이나비용우위는측정하지않았다.</p>
<h2>판정·다음 판단·비용</h2><p>사전수치격리기준1e-9로가설은위조건에서지지. 다만CFO0에서는실패가없고작은CFO의차이는작다. 사용할수있는주장은 “이 실험의 잔여 CFO 조건에서는 주파수 분리만으로 학습 영향 격리가 유지되지 않았다”이다. “일반 AirComp에서는 SISA 불가능”으로 확대하지 않는다. 실용성 판단에는실제잔여CFO분포와신경망·허용삭제오차의별도검증이필요하다.</p><p>CPU 실행 {out['cost']['seconds']:.2f}초 + 검증 {audit['seconds']:.2f}초. source640회, full/local/sham각1920회. GPU0, 외부유료비용0, 에너지미측정. 1920 full reference·640 source를별도affine전개로재검산: 최대오차{maxerr:.3e}, 파형비교오차{out['audit_waveform_max_error']:.3e}. 실패없음.</p><p><a href="PREREG.txt">사전정의</a> · <a href="results.json">모든수치·sham결과</a> · <a href="raw.npz">모델·채널·입력원자료</a> · <a href="audit.json">검증</a> · <a href="run.py">코드</a></p></html>'''
(P/'report.html').write_text(doc,encoding='utf-8')
print(json.dumps(audit));print(json.dumps([s for s in out['summary'] if s['mode']=='ofdm']))
