# AirComp shard·SNR·프라이버시 실험: 최종 보고

2026-10-03. 주57조건과 사후 진단8조건을 모두 실행했다. 71개의 삭제 reference/replay 비교(주63+진단8), 284개 checkpoint 비교에서 고정 routing의 동일 SISA 전체 재학습과 bitwise 일치했다. Preflight1조건은 별도다. 기존 실험의 원자료를 변경하지 않았다.

**판단:** AirComp 집계 오차와 데이터 분포를 함께 고려한 배정은 이번 주 실험에서 무작위 배정보다 우월하지 않았다. 다만 주 실험에 과도한 clipping 제약이 있었고, 이를 완화한 사후 진단의 한 조건에서는 목표 정확도 도달 자원을 약25% 줄이는 후보 결과가 나왔다. 독립 seed에서 확인한 통신 절감으로 주장하지 않는다. 강한 프라이버시와 정확도를 동시에 확보하는 데에는 실패했다.

## 실행 조건

- FashionMNIST train12000/dev2000/test10000, client20, Dirichlet .5, CNN38282 parameters, local SGD2x64, lr.05, 160 rounds.
- K2/4/5, 명목 SNR0/10/20dB, random/channel/joint 배정. K4는3seeds, K2/5는screen1seed. R1 기본, 일부R4 비교. 초기학습과 같은SISA 전체 reference, 해당 shard replay를 별도 실행.
- joint는 MSE를 결정하는 shard bottleneck channel 및 가능한 단일 삭제 후 bottleneck을 평가하고, label histogram JS 대표성 항과 .5/.5로 결합한 client 교환 heuristic이다. 신규성·최적성·SNR별 최적 배정은 주장하지 않는다.
- 완전CSI·동기화·위상보상을 가정한 SISO AirComp. shard 간 직교 자원을 비용에 포함한다. 실제 무선 trace나 MIMO 공간분리는 실험하지 않았다.

## 주 실험 결과: K4, 3 seeds 평균, C0.1

|명목 SNR|무작위 정확도|채널순 정확도|공동 배정 정확도|
|---|---|---|---|
|0dB|66.58%|65.14%|65.54%|
|10dB|66.77%|64.92%|65.55%|
|20dB|66.76%|64.98%|65.39%|

고정 K/R/round에서는 세 방법의 전송 자원 수가 같다. K4의 초기학습+한 번 삭제 비용은 모두520,818,240 RE였다. SNR0→20dB에서 MSE는1/100로 줄지만 정확도는 일관되게 증가하지 않았다. 공동 배정은 무작위보다 평균1.04~1.36pp 낮았다. 사전 목표 dev70%에 주 실험 후보들이 도달하지 못하여 목표 정확도에서의10% 통신 절감 기준은 미통과/비교불가로 기록했다. 이 결과만으로 grouping 방향 자체를 기각하지 않는다.

반복 R1→R4는 MSE를1/4로 줄였지만 총 초기학습+삭제 RE는17.64% 증가했다. 대응6쌍의 정확도 변화는−0.29~+0.14pp, 평균−0.01pp였다. 조건부 epsilon 상한은 약4배가 되었다. 반복 전송의 비용과 privacy 손실을 제외한 무료 정확도 개선으로 해석하면 안 된다.

## 발견한 학습 제약과 사후 진단

주 실험 natural 조건에서 source update의91.28~95.50%가 C0.1 clipping을 받았다. 이를 확인한 뒤 별도 기록을 남기고8개 진단만 추가했다. 원래 experiment.py에서 사용했던 C1.0을 비교값으로 사용했으며, 주 실험의 조건이나 성공기준을 소급 변경하지 않았다.

잡음이 거의 없는 수치 기준(명목120dB, 현실 무선 배치 주장 아님)에서 C0.1→C1.0 변경 시 test66.35→71.44%(+5.09pp), dev+6.35pp, clipping94.22→0%였다. 학습 제약이 정확도를 낮춘다는 진단 기준(test/dev 각각2pp 개선)을 통과했다. 한 seed의 진단이다.

### 목표 dev70%에서 관측한 통신 절감 후보

추가 진단의 seed10801/K4/SNR0/C1.0에서,40라운드 간격으로 저장한 모델 중 처음 dev70%에 도달한 지점은 다음과 같다.

|배정|라운드|dev 정확도|test 정확도|초기학습+삭제 RE|조건부 epsilon 상한|
|---|---|---|---|---|---|
|무작위|160|73.05%|72.66%|520,818,240|724,920|
|채널순|160|72.10%|71.67%|520,818,240|2,226,892|
|공동 배정|120|70.25%|70.40%|390,614,640|806,379|

공동 배정은 같은 **최소 dev 목표**에서 약25% 자원을 덜 사용했다. 최종 정확도가 같다는 뜻은 아니다. 이 지점의 test 정확도는 무작위보다2.26pp 낮고, epsilon 상한은 더 크다.40라운드 간격의 거친 관측과 사후 한seed 결과이므로 정밀한 시간단축이나 확인된 일반적 개선으로 표현하지 않는다.20dB에서는 세 방법 모두160라운드에서 처음 목표에 도달해 관측된 절감은 없었다.

## 프라이버시 결과

프라이버시는 공개·고정된 client별 label histogram/count/CSI/routing을 조건으로 client 하나의 feature 데이터 전체를 교체하는 **조건부 client DP 상한**이다. histogram·client 참여 여부·위치 자체 보호나 실제 공격 성공률을 측정한 것은 아니다. δ=1e-5, clip 민감도2C/m와 Gaussian zCDP 합성을 사용했다. 현재 update norm은 송신하지 않았다. source+한 삭제 replay의 공개 transcript를 합산했고, offline full reference는 공개하지 않는 것으로 계산했다.

- 자연 잡음만으로 계산한 epsilon 상한은 매우 커서 실용적인 강한 보호를 입증하지 못했다. 상한이 크다는 사실을 곧바로 실제 공격 성공의 증명으로 해석하지 않는다.
- 전력을 낮춰 epsilon8을 맞춘 조건의 정확도는6.34~6.52%, epsilon100은4.19~10.03%였다. 이번20-client/고차원/매round 전체참여/보수적 client 단위 조건에서는 정확도와 강한 조건부 privacy를 동시에 확보하지 못했다.
- 이 이상적인 full-power alignment 모델에서는 잡음과 민감도의1/m 항이 상쇄된다. shard 크기만으로 privacy가 자동으로 강해지는 것은 아니며, grouping에 따른 bottleneck channel·참여 round·전력 제어를 같이 봐야 한다.
- 이 검증은 고정 routing에 대한 모델 재학습 동등성이다. 공개된 histogram/routing 자체의 삭제나 삭제 후 재배정 절차 전체의 동등성을 검증하지 않았다.

## SISA 자체의 절감과 추가 배정 효과

대표 K4 조건에서 해당 shard만 삭제 재학습하면 **같은4-shard SISA 전체 재학습 대비** 통신 자원 약75.00%, local training 호출78.95%를 줄였다. 이는 SISA 범위 제한의 이득이다. 단일 전역 AirComp 모델 대비 절감으로 표현하지 않는다. 공동 배정의 추가 통신 이득은 위의 별도 사후25% 후보와 구분한다.

## 다음 판단

SISA 구조와 AirComp에 맞는 shard 배정 방향은 유지할 근거가 있다. 하지만 현재 .5/.5 MSE/JS heuristic을 우수한 방법으로 확정할 근거는 부족하다. 먼저 과도한 clipping을 피하고 학습량을 확보한 조건에서 여러 seed와 삭제 대상에 대해 목표 정확도 도달 비용을 확인해야 한다. SNR별 배정과 수신 구조의 차별화는 그 뒤 검토할 후보이며 이번에 최적화하지 않았다. 실용적 privacy는 보장 단위/공개 정보 조건을 명확히 한 별도 설계가 필요하다. 이번 실험은 여기서 종료한다.

## 총비용·원자료

주 실험+진단+preflight 실행시간1881.38초(31.36분), client local training 호출475,212회(각2SGD steps), GPU board 측정 적분 약83.98Wh, 외부 유료 서비스 사용0. 전력은5초 주기로 GPU board를 측정했으며 host 전체/RF 통신 에너지/정확한 전기요금이 아니다. 짧은preflight는 적분 구간이 없어 실제 소비0으로 해석하면 안 된다. 분석·그림 생성 시간은 각analysis_cost.json에 별도 기록했다. 실행 오류 없이 모두 완료했다.

- 주 실험 원자료: ../snr_privacy_v1/의 모델·partition·label·config·round ledger·checkpoint metric·privacy 합성.
- 진단 원자료: ../snr_clip_diagnostic_v1/C0.1 및 C1.0. comparisons.json, near_noiseless_diagnostic.json, combined_cost.json.
- 주 상세 결과: REPORT_KO.md, all_final_metrics.json, repetition_pairs.json, matched_accuracy_privacy.json, equal_total_budget.json, deletion_target_check.json, audit.json.
- 사전기록: ../SNR_PRIVACY_PREREG.md, ../SNR_CLIP_DIAGNOSTIC_AMENDMENT.md. 모델 설명: ../SNR_PRIVACY_MODEL_KO.md.
- 코드: ../snr_privacy.py, ../snr_privacy_analyze.py, ../snr_clip_diagnostic.py, ../snr_diagnostic_report.py. 실행 당시 hash는 각 environment.json.
