from pathlib import Path
import json,datetime
R=Path(__file__).resolve().parent;O=R.parent/'aircomp_mechanism_summary_20261004';O.mkdir(exist_ok=True)
rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(R.glob('*_summary.json'))]
def table(head,rows):return '<table><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</table>'
body=[]
for r in rows:
    ds=r['deletions'];q='q60';ok=[d for d in ds if d['stops'][q] is not None]
    body.append([r['seed'],r['method'],r['noise'],str(sum(d['stops']['q65'] is not None for d in ds))+'/20',str(len(ok))+'/20',round(sum(d['stops'][q] for d in ok)/20,1) if len(ok)==20 else '전체 완료 아님',f"{100*sum(d['test']['fixed160']['accuracy'] for d in ds)/20:.2f}%"])
now=datetime.datetime.now().isoformat(timespec='seconds')
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>실험 중간 기록 — GPU 대기</title><style>body{{font-family:Malgun Gothic,Segoe UI,sans-serif;max-width:1100px;margin:35px auto;line-height:1.8;padding:20px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd4df;padding:9px}}th{{background:#edf3fa}}.box{{background:#fff4df;padding:20px}}</style>
<p class="box"><b>가설:</b> 동일 크기에서 데이터 배정과 잡음은 삭제 학습에 다른 경로로 영향 | <b>독립변수:</b> 배정 3개 × 잡음 2개 | <b>종속변수:</b> 정확도·완료율·삭제 통신량 | <b>통제:</b> N20/K4/각 5명/같은 SGD | <b>상태:</b> {len(rows)}/18조건, {20*len(rows)}/360삭제 완료. 나머지는 GPU 자원 대기.</p>
<h1>중간 기록: {now}</h1><p>실험 중 다른 GPU 작업이 시작되어 본 실험 PID6380만 일시 중단했습니다. 기존 결과는 보존했고 실험 조건은 변경하지 않았습니다. 이 페이지는 중간 시점의 기록이며 최종 완료 보고서가 아닙니다.</p>
{table(['seed','배정','잡음','65% 완료','60% 완료','60% 전체 완료 평균round','삭제 후 고정160 ensemble 정확도'],body)}
<p>완료율은 shard 자체의 공개 검증 정확도에 대한 것입니다. 20라운드마다 검증하여 목표를 연속 두 번 넘는 최초 시점을 완료로 정의합니다. 초기학습과 삭제재학습 모두 같은 규칙을 사용합니다. 200라운드 안에 못 넘은 경우를 완료 비용으로 계산하지 않습니다. 삭제 후 데이터 배제와 SISA 독립성의 실패를 의미하지 않습니다.</p>
<p>첫 seed의 60% 기준에서 data 배정은 평균142round, channel은156round로 모두20삭제 완료했습니다. 통신량은 라운드마다38,282 real uses이므로 data가8.97% 적습니다. 이 하나의 seed 결과를 전체 결론으로 일반화하지 않습니다.</p>
<p>무잡음 seed64001/client14 삭제의 shard dev160: channel70.80%→59.45%, data72.55%→67.80%. 설명용 사후 사례이며 데이터 대표성 점수 하나의 인과효과로 단정하지 않습니다.</p>
<p>알고리즘 초안: 같은 크기의 후보 생성 → 초기·각client삭제 학습 완료 확인 → 가능한 후보 중 실측 평균삭제UL 최소 선택 → 학습 전에 배정 고정. 준비 단계에서 후보마다 모든 삭제를 실험하는 비용이 발생합니다. 저렴한 온라인 알고리즘의 성능은 아직 검증하지 않았습니다.</p>
<p><a href="../aircomp_mechanism_20261004/PREREG.txt">사전 가설·조건·판정기준</a> · <a href="../aircomp_mechanism_20261004/GPU_RESOURCE_NOTE.txt">자원 중단 기록</a> · <a href="../aircomp_mechanism_20261004/select_partition.py">실측 비용 선택 코드</a></p>
<p>비용은 아직 누적 중입니다. 다른 작업과 공유한 GPU 전력을 이 실험 단독 전력으로 귀속하지 않습니다. 완료 후 전체 벽시계 시간, 중단 시간, 실행량, 독점 관측 에너지를 분리해서 보고합니다.</p></html>'''
with (O/f'중간_{len(rows)}조건.html').open('x',encoding='utf-8') as f:f.write(page)
print(O/f'중간_{len(rows)}조건.html')
