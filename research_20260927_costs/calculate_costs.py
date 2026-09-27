"""Stdlib-only lifecycle cost audit. Run from any directory; inputs are sibling archives."""
from pathlib import Path
import json, math, statistics as st, hashlib
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
OLD=BASE/'research_20260926_versioned/results';PART=BASE/'research_20260927_orthogonality/results'
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
D=63562;MODEL_BITS=D*32;CP=1.125;QUERY_BITS=2000*28*28*8


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rate(snr):return .5*math.log2(1+10**(snr/10))
def bit_uses(bits,snr=20):return CP*bits/rate(snr)
def components(c,snr=20):
    r=rate(snr)
    x=dict(analog_payload=c.get('ul_real',c.get('ul_payload_real',0.)),
        digital_uplink=c.get('digital_ul_bits',0.)/r,downlink=c.get('dl_bits',0.)/r,
        metadata=c.get('metadata_bits',0.)/r,pilots=c.get('pilot_real',0.))
    x['cp_overhead']=(CP-1)*sum(x.values());x['total']=sum(x.values())
    assert abs(x['total']-c['total_real_uses'])<1e-6,(x,c)
    return x


def compute():
    records=[];timings={};prepare={}
    # source preparation after initial construction, including final deployed model.
    dsrecs=[load(OLD/f'DS_seed{s}.json') for s in [202609263,202609264,202609265]]
    dsraw=dsrecs[0]['preparation_cost'];components(dsraw)
    prepare['DS']=dict(legacy=dsraw['total_real_uses'],additional_bits=2*32+128,
        completed=dsraw['total_real_uses']+bit_uses(2*32+128),query_bits=0,
        communication_note='ideal sufficient-statistic source construction; not noisy end-to-end source training')
    for fam,folder,seeds,tag in [('Sketch',PART,[381,382,383],'source'),('OG',OLD,[371,372,373],'ortho_all')]:
        rr=[load(folder/f'seed{s}_{tag}_training.json') for s in seeds]
        for r in rr:components(r['cost'])
        packets=150 if fam=='Sketch' else 300
        old=st.mean(r['cost']['total_real_uses'] for r in rr)
        addon=MODEL_BITS+packets*32+128
        prepare[fam]=dict(legacy=old,additional_bits=addon,completed=old+bit_uses(addon),query_bits=0,
                          gradient_samples=384000,simulation_seconds_mean=st.mean(r['seconds'] for r in rr))
    kd=[load(OLD/f'RTDfp64_seed{s}.json') for s in [371,372,373]]
    for r in kd:components(r['preparation'])
    old=st.mean(r['preparation']['total_real_uses'] for r in kd)
    prepare['RTD']=dict(legacy=old,additional_bits=10*32+32+128,completed=old+bit_uses(10*32+32+128),
        query_bits=QUERY_BITS,public_query_cold_extra=bit_uses(QUERY_BITS),teacher_gradient_samples=320000,
        source_student_gradient_samples=38400,source_teacher_forward_samples=20000,
        teacher_simulation_seconds_sum_mean=st.mean(sum(t['seconds'] for t in r['teachers']) for r in kd),
        teacher_simulation_seconds_max_mean=st.mean(max(t['seconds'] for t in r['teachers']) for r in kd),
        source_student_simulation_seconds_mean=st.mean(r['source_student_seconds'] for r in kd))

    def add(label,fam,rows,snr,initial_model_bits,scale_bits=0,gradient_samples=0,forward_samples=0,student_samples=0):
        costs=[components(r['cost'],snr) for r in rows]
        comp={k:st.mean(x[k] for x in costs) for k in costs[0]}
        missing_bits=128+128+scale_bits
        cold=comp['total']+bit_uses(missing_bits,snr)
        warm=cold-bit_uses(initial_model_bits,snr)
        pc=prepare[fam]['completed'];query=prepare[fam].get('query_bits',0)
        # deployment begins at20dB; source model is cached at deletion regardless of deletion SNR.
        digital=(comp['digital_uplink']+comp['downlink']+comp['metadata'])*CP+bit_uses(missing_bits,snr)
        rec=dict(label=label,family=fam,snr_db=snr,legacy_deletion=comp['total'],components=comp,
            added_control_bits=256,added_receive_scale_bits=scale_bits,
            deletion_cold=cold,deletion_cached_source=warm,source_preparation=pc,
            public_query_cold_extra=bit_uses(query),lifecycle_cached_query=pc+warm,
            lifecycle_uncached_query=pc+warm+bit_uses(query),
            gradient_samples=gradient_samples,forward_samples=forward_samples,student_samples=student_samples,
            transmission_seconds_at_1M=cold/1e6,transmission_seconds_at_10M=cold/1e7,
            digital_retry_sensitivity={str(p):cold+digital*(1/(1-p)-1) for p in [0,.01,.05]},
            source_initial_model_rebroadcast_bits=initial_model_bits)
        records.append(rec)
    for method,version,label in [('DS-Air','Original','DS Original'),('DS-Air','V1','DS V1'),('DS-Air-subspace48','V1','DS V1 subspace48')]:
        rows=[r for d in dsrecs for r in d['rows'] if r['method']==method and r['version']==version and r.get('peak') is True]
        assert len(rows)==48
        add(label,'DS',rows,20,650*32,scale_bits=(5 if '48' in label else 6)*32)
    og=[load(OLD/f'OG_seed{s}.json') for s in [371,372,373]]
    for version in ['Original','V1']:
        rows=[r for d in og for r in d['rows'] if r['method']=='OG-Air' and r['version']==version]
        assert len(rows)==12
        add('OG '+version,'OG',rows,20,MODEL_BITS,32,gradient_samples=12000 if version=='V1' else 0,
            forward_samples=12000 if version=='Original' else 0)
    for snr in [20,40]:
        rr=[load(PART/f'seed{s}_complete.json') for s in [381,382,383]]
        for beta in [1.,.5]:
            rows=[r for d in rr for r in d['rows'] if r['mode']==f'ota{snr}' and r['beta']==beta and
                  r['normalization']=='target_norm' and r['step']==1]
            assert len(rows)==12
            add(f'Sketch beta{beta:g} {snr}dB','Sketch',rows,snr,MODEL_BITS,gradient_samples=2560)
    for version in ['Original','V1']:
        for reps in [1,4]:
            rows=[r for d in kd for r in d['rows'] if r['method']=='RTD-Air' and r['version']==version and r['extra']['repeats']==reps]
            assert len(rows)==12
            add(f'RTD {version} r{reps}','RTD',rows,20,0,forward_samples=18000,student_samples=38400)
            timings[f'RTD {version} r{reps}']=st.mean(r['extra']['student_seconds'] for r in rows)
    # Optional deployment-only cache scenarios: do not alter quality claims.
    cache=[]
    for rec in records:
        if rec['family']=='OG':
            cache.append(dict(label=rec['label']+' cached model + gate-only delivery',
                deletion=rec['deletion_cached_source']-bit_uses(MODEL_BITS-24*32),
                status='communication accounting only; same known gate operator, not a new successful deletion experiment'))
    conv_macs=26*26*8*9+11*11*16*8*9+1936*32+32*10
    assert conv_macs==250336
    compute_info=dict(conv_forward_MACs_per_sample=conv_macs,model_parameters=D,
        warning='MAC counts omit activations, pooling, softmax, backward and normalization; not complete training FLOPs',
        DS_initial_private_encoding_MACs=10000*784*64,
        DS_initial_dense_H_MACs=10000*65*65,DS_initial_dense_C_MACs=10000*65*10,
        DS_cached_residual_MACs=9*65*65*10,DS_client_basis_transform_MACs=9*65*65*10,
        DS_inverse_scalings_V1=9*65*10,DS_server_lift_MACs=65*65*10,
        DS_recompute_if_local_stats_not_cached=dict(private_forward_samples=9000,
            encoder_MACs=9000*784*64,H_MACs=9000*65*65,C_MACs=9000*65*10),
        DS_subspace_H_projection_MACs=9*(65*65*48+48*65*48),
        DS_subspace_residual_projection_MACs=9*48*65*10,
        DS_subspace_basis_transform_MACs=9*48*48*10,
        Sketch_SRHT_additions_all_clients=10*65536*16,Sketch_SRHT_sign_multiplications_all_clients=10*65536,
        solver_complexity={'DS_server_eigendecomposition':'O(d^3), d=65 or48; actual FLOPs not profiled',
                           'Sketch_server_SVD':'O(m*k^2), m=2048,k=9; workspace not included'},
        source_training_forward_MAC_component=dict(Sketch=384000*conv_macs,OG=384000*conv_macs,
            RTD_teachers=320000*conv_macs,RTD_student=38400*conv_macs),
        deletion_forward_MAC_component=dict(Sketch=2560*conv_macs,RTD=(18000+38400)*conv_macs))
    storage=dict(unit='bytes, tensor payload only; not process or peak GPU memory',
        CNN_model_fp32=D*4,CNN_one_gradient_fp32=D*4,
        DS_encoder_fp32=784*64*4,DS_head_fp32=65*10*4,
        DS_local_dense_H_and_C_fp64=(65*65+65*10)*8,
        DS_local_upper_H_and_C_fp64=(65*66//2+65*10)*8,
        DS_local_feature_cache1000_fp64=1000*65*8,
        DS_server_full_basis_fp32=65*65*4,
        Sketch_client_wire_sketch16bit=2048*2,Sketch_server_10_decoded_sketches_fp32=10*2048*4,
        Sketch_client_SRHT_single_padded_buffer_fp32=65536*4,
        RTD_one_local_teacher_fp32=D*4,RTD_all_10_teachers_distributed_fp32=10*D*4,
        RTD_public_images_uint8=2000*28*28,RTD_public_images_fp32=2000*28*28*4,
        RTD_local_probability_cache_fp64=2000*10*8,
        RTD_server_retained_target_fp32=2000*10*4)
    # References are offline research expenses, never a required cost of the deployed unlearner.
    refs=[]
    for seed in [381,382,383]:
        for ref in [0,1]:
            r=load(PART/f'seed{seed}_reference{ref}_training.json')
            refs.append(dict(seed=seed,reference=ref,cost=r['cost']['total_real_uses'],seconds=r['seconds']))
    checks=dict(passed=True,variant_count=len(records),legacy_ledgers_recomputed=True,
        cached_source_deduction_checked=all(abs(r['deletion_cold']-r['deletion_cached_source']-
            bit_uses(r['source_initial_model_rebroadcast_bits'],r['snr_db']))<1e-6 for r in records),
        no_negative_costs=all(r['deletion_cached_source']>0 for r in records),
        query_bits=QUERY_BITS,query_cost_20db=bit_uses(QUERY_BITS))
    assert checks['cached_source_deduction_checked'] and checks['no_negative_costs']
    out=dict(assumptions_sha256=hashlib.sha256((ROOT/'ASSUMPTIONS_KO.md').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),preparation=prepare,
        records=records,optional_cache_scenarios=cache,compute=compute_info,storage=storage,
        existing_KD_simulator_seconds=timings,offline_reference_audit=refs,verification=checks)
    write(OUT/'costs.json',out)
    lines=['**①②③ 총비용 보완 계산 — 2026-09-27**','',
        '기존 삭제 통신비를 성분별로 다시 계산하고, 제어·수신scale·초기준비·최종모델배포·공개query 비용을 보완했다. 재학습 기준모델과 비교 실험 sweep은 연구용 비용으로 분리했다. 비용만 수정했으며 기존 성능 결과를 새 설정에서 얻었다고 주장하지 않는다.','',
        '단위는 real channel-use equivalent. 기본20dB goodput3.3291bit/real use,CP1.125다. 디지털 패킷·실제RF 지연·전력 소비의 실측 결과가 아니다. Cold 삭제는 source 모델을 다시 받는 경우, cached 삭제는 이미 배포된 source를 사용하는 경우다.','',
        '**총비용표**','','| 방식 | 기존 삭제 | 보완 cold 삭제 | source cache 삭제 | 초기 준비 완료 | 준비+삭제1회 | query 미보유 시 전체 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append('|'+r['label']+'|'+ '|'.join(f'{r[k]:,.0f}' for k in ['legacy_deletion','deletion_cold','deletion_cached_source',
            'source_preparation','lifecycle_cached_query','lifecycle_uncached_query'])+'|')
    lines+=['','준비+삭제1회는 준비 완료 모델이 cache된 조건으로 계산해 source 모델을 두 번 배포하는 비용을 중복하지 않았다.40dB 행의 초기준비는 기존20dB 설정이고 삭제 구간만40dB다. 이종 family는 모델·학습규칙·품질이 달라 비용 숫자만으로 성능 대비 우열을 정할 수 없다.','',
        '**기존 삭제 비용의 구성: CP를 제외한 항목과 CP 추가분**','',
        '| 방식 | analog UL | digital UL | DL | metadata | pilot | CP 추가 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append('|'+r['label']+'|'+ '|'.join(f"{r['components'][k]:,.1f}" for k in ['analog_payload','digital_uplink','downlink','metadata','pilots','cp_overhead'])+'|')
    lines+=['','**추가 항목의 근거**','',
        '- 요청128bit UL+공통지시128bit DL은 이번에 명시한 프로토콜 가정이다. 실제 패킷표준 overhead라고 주장하지 않는다.',
        '- DS 수신scale은Hessian1packet+보정5block(48차원은4block)의32bit를 추가했다. OG는1packet32bit다. Sketch의 기존계수DL에는scale32bit가 이미 있고,RTD query-ID뒤32bit는scale slot으로 명시했다.',
        '- FedAvg/OG 초기준비는 마지막완성모델DL과 학습packet별scale방송을 추가했다. RTD 준비는누락된norm/scale metadata와공개초기seed를추가했다.',
        '- 공개query 이미지는RTD에만 필요하다. 이미지cache가없으면12,544,000bit,20dB에서4,238,976 real uses가 최초1회 추가된다. 현재 표의query-ID전송비는 이미별도포함돼있다.',
        '- 초기학습은기존의이상적집계source를사용한실험이다. 추가한통신예산이있다는이유로noisy end-to-end source학습을검증했다고하지않는다.',
        '- 회복학습은0회다. Checkpoint및평가서버로의업로드는실제연구관리비용으로서프로토콜에포함하지않으며,본GitHub배포파일크기는별도manifest에기록한다.','',
        '**계산량: gradient/forward sample 수와 MAC**','','| 단계 | 계산량 |','|---|---|',
        '|②-B 초기 FedAvg|384,000 gradient-samples /150rounds|',
        '|②-A 초기 orthogonal FedAvg|384,000 gradient-samples /300 OTA phases;orthogonal 정규화 추가연산 별도|',
        '|②-B 삭제1회|2,560 gradient-samples;SRHT10,485,760 additions;서버2048×9 SVD|',
        '|③ teacher 사전학습|320,000 gradient-samples 합계|',
        '|③ 최초student|20,000 teacher forward-samples+38,400 student gradient-samples|',
        '|③ 삭제1회|18,000 teacher forward-samples+38,400 student gradient-samples|',
        '|① 초기통계|private encoder501,760,000 MAC+H42,250,000 MAC+C6,500,000 MAC|',
        '|① cache된 통계로삭제|residual380,250 MAC+client basis변환380,250 MAC+server lift42,250 MAC;V1추가division5,850개|',
        '|① 통계가없을때추가|잔존9000개 encoder451,584,000 MAC+H38,025,000 MAC+C5,850,000 MAC|',
        '', 'ConvNet2 forward만250,336 MAC/sample이다. backward·pool·activation·loss·정규화를포함한완전한training FLOPs로환산하지않았다.① eigendecomposition은O(d³),② SVD는O(2048×9²)로표시하고미측정상수를붙이지않았다.','',
        '**저장 비용**','','| 항목 | tensor bytes |','|---|---:|']
    for key,value in storage.items():
        if isinstance(value,int):lines.append(f'|{key}|{value:,}|')
    lines+=['','Prediction cache를보존하면③의삭제시teacher forward18000회를줄일수있지만,로컬당FP64확률160000bytes가추가된다. 이cache최적화로성능을다시실험했다고하지않는다. Byte표는tensor payload로서allocator,activation,optimizer workspace를포함한peakRAM이아니다.','',
        '**가정에 따른 시간 환산과 재전송 민감도**','','| 방식 | 1M real/s | 10M real/s | digital PER1% 총비용 | PER5% 총비용 |','|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"|{r['label']}|{r['transmission_seconds_at_1M']:.4f}s|{r['transmission_seconds_at_10M']:.4f}s|{r['digital_retry_sensitivity']['0.01']:,.0f}|{r['digital_retry_sensitivity']['0.05']:,.0f}|")
    lines+=['','시간표는cold삭제통신만의환산이다. MHz대역폭,실제시스템지연과동일하지않다. 디지털만기대1/(1-p) 재전송을적용하며analog재시도는포함하지않는다. 실제end-to-end latency에는각phase의max client계산,server계산,동기화및왕복지연이필요하다. 기존한GPU에서순차실행한client시간합을그대로실서비스지연으로쓰지않는다.','',
        '전력W가측정되지않아J·원화비용은확정하지않았다. 단일링크활성전력1W라는예시에서는그링크통신시간1초당1J이지만,단말전체·서버·GPU·RF증폭기에너지의합은아니다. CSI오류,CFO,ACK,동기화재시도,실제coding overhead는미측정이다.','',
        '**측정된 기존 simulator 시간**','','| 항목 | 평균 초 |','|---|---:|',
        f"|②-B source 학습|{prepare['Sketch']['simulation_seconds_mean']:.4f}|",
        f"|②-A source 학습|{prepare['OG']['simulation_seconds_mean']:.4f}|",
        f"|③ teacher10명 시간합|{prepare['RTD']['teacher_simulation_seconds_sum_mean']:.4f}|",
        f"|③ source student KD|{prepare['RTD']['source_student_simulation_seconds_mean']:.4f}|"]
    for label,sec in timings.items():lines.append(f'|{label} student KD|{sec:.4f}|')
    lines+=['','DS 전체seed시간과②의삭제trajectory시간에는평가·다른noise·reference연산이섞여있어개별삭제계산시간으로재사용하지않았다.','',
        '**추가 cache 시나리오: 성능개선 주장과 분리**','','| 조건 | 삭제 real uses |','|---|---:|']
    for c in cache:lines.append(f"|{c['label']}|{c['deletion']:,.0f}|")
    lines+=['','OG는source가cache돼있고gate만적용할수있다면24개gate배포로비용을크게줄일수있다. 그러나OG Original/V1은삭제품질에서보류된방법이므로통신절감만으로재채택하지않는다.','',
        '**검증**','',
        f"기존13개variant ledger를재계산했고총합오차<1e-6을확인했다. Source중복차감과query bit수를검증했다. 계산은표준Python만필요하다: `python research_20260927_costs/calculate_costs.py`.",
        '', '자세한가정은ASSUMPTIONS_KO.md,모든성분·원자료연결은results/costs.json에있다.']
    (ROOT/'COST_REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(verification=checks,records=[{k:r[k] for k in ['label','deletion_cold','deletion_cached_source','lifecycle_uncached_query']} for r in records]),ensure_ascii=False,indent=2))


if __name__=='__main__':compute()
