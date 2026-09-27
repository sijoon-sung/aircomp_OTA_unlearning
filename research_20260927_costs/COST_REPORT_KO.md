**①②③ 총비용 보완 계산 — 2026-09-27**

기존 삭제 통신비를 성분별로 다시 계산하고, 제어·수신scale·초기준비·최종모델배포·공개query 비용을 보완했다. 재학습 기준모델과 비교 실험 sweep은 연구용 비용으로 분리했다. 비용만 수정했으며 기존 성능 결과를 새 설정에서 얻었다고 주장하지 않는다.

단위는 real channel-use equivalent. 기본20dB goodput3.3291bit/real use,CP1.125다. 디지털 패킷·실제RF 지연·전력 소비의 실측 결과가 아니다. Cold 삭제는 source 모델을 다시 받는 경우, cached 삭제는 이미 배포된 source를 사용하는 경우다.

**총비용표**

| 방식 | 기존 삭제 | 보완 cold 삭제 | source cache 삭제 | 초기 준비 완료 | 준비+삭제1회 | query 미보유 시 전체 |
|---|---:|---:|---:|---:|---:|---:|
|DS Original|143,940|144,091|137,062|10,645|147,708|147,708|
|DS V1|144,643|144,794|137,765|10,645|148,410|148,410|
|DS V1 subspace48|88,173|88,314|81,285|10,645|91,930|91,930|
|OG Original|1,375,097|1,375,194|687,853|228,404,647|229,092,500|229,092,500|
|OG V1|1,375,097|1,375,194|687,853|228,404,647|229,092,500|229,092,500|
|Sketch beta1 20dB|1,557,477|1,557,564|870,223|114,546,016|115,416,238|115,416,238|
|Sketch beta0.5 20dB|1,557,477|1,557,564|870,223|114,546,016|115,416,238|115,416,238|
|Sketch beta1 40dB|816,134|816,178|471,768|114,546,016|115,017,784|115,017,784|
|Sketch beta0.5 40dB|816,134|816,178|471,768|114,546,016|115,017,784|115,017,784|
|RTD Original r1|731,755|731,842|731,842|731,840|1,463,682|5,702,658|
|RTD Original r4|799,255|799,342|799,342|731,840|1,531,182|5,770,158|
|RTD V1 r1|729,505|729,592|729,592|731,840|1,461,432|5,700,408|
|RTD V1 r4|790,255|790,342|790,342|731,840|1,522,182|5,761,158|

준비+삭제1회는 준비 완료 모델이 cache된 조건으로 계산해 source 모델을 두 번 배포하는 비용을 중복하지 않았다.40dB 행의 초기준비는 기존20dB 설정이고 삭제 구간만40dB다. 이종 family는 모델·학습규칙·품질이 달라 비용 숫자만으로 성능 대비 우열을 정할 수 없다.

**기존 삭제 비용의 구성: CP를 제외한 항목과 CP 추가분**

| 방식 | analog UL | digital UL | DL | metadata | pilot | CP 추가 |
|---|---:|---:|---:|---:|---:|---:|
|DS Original|73,840.0|0.0|53,155.4|519.1|432.0|15,993.3|
|DS V1|73,840.0|0.0|53,780.2|519.1|432.0|16,071.4|
|DS V1 subspace48|42,432.0|0.0|35,151.8|432.5|360.0|9,797.0|
|OG Original|192.0|0.0|1,221,940.2|96.1|80.0|152,788.5|
|OG V1|192.0|0.0|1,221,940.2|96.1|80.0|152,788.5|
|Sketch beta1 20dB|63,562.0|98,525.0|1,222,055.5|201.9|80.0|173,053.0|
|Sketch beta0.5 20dB|63,562.0|98,525.0|1,222,055.5|201.9|80.0|173,053.0|
|Sketch beta1 40dB|63,562.0|49,368.4|612,341.3|101.1|80.0|90,681.6|
|Sketch beta0.5 40dB|63,562.0|49,368.4|612,341.3|101.1|80.0|90,681.6|
|RTD Original r1|20,000.0|0.0|630,204.1|173.0|72.0|81,306.1|
|RTD Original r4|80,000.0|0.0|630,204.1|173.0|72.0|88,806.1|
|RTD V1 r1|18,000.0|0.0|630,204.1|173.0|72.0|81,056.1|
|RTD V1 r4|72,000.0|0.0|630,204.1|173.0|72.0|87,806.1|

**추가 항목의 근거**

- 요청128bit UL+공통지시128bit DL은 이번에 명시한 프로토콜 가정이다. 실제 패킷표준 overhead라고 주장하지 않는다.
- DS 수신scale은Hessian1packet+보정5block(48차원은4block)의32bit를 추가했다. OG는1packet32bit다. Sketch의 기존계수DL에는scale32bit가 이미 있고,RTD query-ID뒤32bit는scale slot으로 명시했다.
- FedAvg/OG 초기준비는 마지막완성모델DL과 학습packet별scale방송을 추가했다. RTD 준비는누락된norm/scale metadata와공개초기seed를추가했다.
- 공개query 이미지는RTD에만 필요하다. 이미지cache가없으면12,544,000bit,20dB에서4,238,976 real uses가 최초1회 추가된다. 현재 표의query-ID전송비는 이미별도포함돼있다.
- 초기학습은기존의이상적집계source를사용한실험이다. 추가한통신예산이있다는이유로noisy end-to-end source학습을검증했다고하지않는다.
- 회복학습은0회다. Checkpoint및평가서버로의업로드는실제연구관리비용으로서프로토콜에포함하지않으며,본GitHub배포파일크기는별도manifest에기록한다.

**계산량: gradient/forward sample 수와 MAC**

| 단계 | 계산량 |
|---|---|
|②-B 초기 FedAvg|384,000 gradient-samples /150rounds|
|②-A 초기 orthogonal FedAvg|384,000 gradient-samples /300 OTA phases;orthogonal 정규화 추가연산 별도|
|②-B 삭제1회|2,560 gradient-samples;SRHT10,485,760 additions;서버2048×9 SVD|
|③ teacher 사전학습|320,000 gradient-samples 합계|
|③ 최초student|20,000 teacher forward-samples+38,400 student gradient-samples|
|③ 삭제1회|18,000 teacher forward-samples+38,400 student gradient-samples|
|① 초기통계|private encoder501,760,000 MAC+H42,250,000 MAC+C6,500,000 MAC|
|① cache된 통계로삭제|residual380,250 MAC+client basis변환380,250 MAC+server lift42,250 MAC;V1추가division5,850개|
|① 통계가없을때추가|잔존9000개 encoder451,584,000 MAC+H38,025,000 MAC+C5,850,000 MAC|

ConvNet2 forward만250,336 MAC/sample이다. backward·pool·activation·loss·정규화를포함한완전한training FLOPs로환산하지않았다.① eigendecomposition은O(d³),② SVD는O(2048×9²)로표시하고미측정상수를붙이지않았다.

**저장 비용**

| 항목 | tensor bytes |
|---|---:|
|CNN_model_fp32|254,248|
|CNN_one_gradient_fp32|254,248|
|DS_encoder_fp32|200,704|
|DS_head_fp32|2,600|
|DS_local_dense_H_and_C_fp64|39,000|
|DS_local_upper_H_and_C_fp64|22,360|
|DS_local_feature_cache1000_fp64|520,000|
|DS_server_full_basis_fp32|16,900|
|Sketch_client_wire_sketch16bit|4,096|
|Sketch_server_10_decoded_sketches_fp32|81,920|
|Sketch_client_SRHT_single_padded_buffer_fp32|262,144|
|RTD_one_local_teacher_fp32|254,248|
|RTD_all_10_teachers_distributed_fp32|2,542,480|
|RTD_public_images_uint8|1,568,000|
|RTD_public_images_fp32|6,272,000|
|RTD_local_probability_cache_fp64|160,000|
|RTD_server_retained_target_fp32|80,000|

Prediction cache를보존하면③의삭제시teacher forward18000회를줄일수있지만,로컬당FP64확률160000bytes가추가된다. 이cache최적화로성능을다시실험했다고하지않는다. Byte표는tensor payload로서allocator,activation,optimizer workspace를포함한peakRAM이아니다.

**가정에 따른 시간 환산과 재전송 민감도**

| 방식 | 1M real/s | 10M real/s | digital PER1% 총비용 | PER5% 총비용 |
|---|---:|---:|---:|---:|
|DS Original|0.1441s|0.0144s|144,703|147,277|
|DS V1|0.1448s|0.0145s|145,413|148,017|
|DS V1 subspace48|0.0883s|0.0088s|88,720|90,428|
|OG Original|1.3752s|0.1375s|1,389,082|1,447,557|
|OG V1|1.3752s|0.1375s|1,389,082|1,447,557|
|Sketch beta1 20dB|1.5576s|0.1558s|1,572,574|1,635,773|
|Sketch beta0.5 20dB|1.5576s|0.1558s|1,572,574|1,635,773|
|Sketch beta1 40dB|0.8162s|0.0816s|823,699|855,366|
|Sketch beta0.5 40dB|0.8162s|0.0816s|823,699|855,366|
|RTD Original r1|0.7318s|0.0732s|739,006|769,171|
|RTD Original r4|0.7993s|0.0799s|806,506|836,671|
|RTD V1 r1|0.7296s|0.0730s|736,756|766,921|
|RTD V1 r4|0.7903s|0.0790s|797,506|827,671|

시간표는cold삭제통신만의환산이다. MHz대역폭,실제시스템지연과동일하지않다. 디지털만기대1/(1-p) 재전송을적용하며analog재시도는포함하지않는다. 실제end-to-end latency에는각phase의max client계산,server계산,동기화및왕복지연이필요하다. 기존한GPU에서순차실행한client시간합을그대로실서비스지연으로쓰지않는다.

전력W가측정되지않아J·원화비용은확정하지않았다. 단일링크활성전력1W라는예시에서는그링크통신시간1초당1J이지만,단말전체·서버·GPU·RF증폭기에너지의합은아니다. CSI오류,CFO,ACK,동기화재시도,실제coding overhead는미측정이다.

**측정된 기존 simulator 시간**

| 항목 | 평균 초 |
|---|---:|
|②-B source 학습|13.2431|
|②-A source 학습|29.5223|
|③ teacher10명 시간합|12.0463|
|③ source student KD|1.1226|
|RTD Original r1 student KD|1.3487|
|RTD Original r4 student KD|1.3596|
|RTD V1 r1 student KD|1.3539|
|RTD V1 r4 student KD|1.3439|

DS 전체seed시간과②의삭제trajectory시간에는평가·다른noise·reference연산이섞여있어개별삭제계산시간으로재사용하지않았다.

**추가 cache 시나리오: 성능개선 주장과 분리**

| 조건 | 삭제 real uses |
|---|---:|
|OG Original cached model + gate-only delivery|771|
|OG V1 cached model + gate-only delivery|771|

OG는source가cache돼있고gate만적용할수있다면24개gate배포로비용을크게줄일수있다. 그러나OG Original/V1은삭제품질에서보류된방법이므로통신절감만으로재채택하지않는다.

**검증**

기존13개variant ledger를재계산했고총합오차<1e-6을확인했다. Source중복차감과query bit수를검증했다. 계산은표준Python만필요하다: `python research_20260927_costs/calculate_costs.py`.

자세한가정은ASSUMPTIONS_KO.md,모든성분·원자료연결은results/costs.json에있다.
