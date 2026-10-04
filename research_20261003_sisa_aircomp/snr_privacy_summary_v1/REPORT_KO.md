# SISA / AirComp SNR·shard·privacy 실험 결과

57개 source 조건, 63개 deletion reference/replay 비교 완료. 252개 체크포인트에서 고정 shard 배정의 동일 SISA reference와 exact 일치. 실측 무선 데이터가 아닌 시뮬레이션이다. 공개 histogram/routing 자체를 삭제하거나 삭제 후 배정 알고리즘을 다시 실행하는 절차의 동일성까지 검증한 것은 아니다.

FashionMNIST train12000/dev2000/test10000, client20, Dirichlet .5, local2x64 SGD lr.05, clip.1, 160 rounds. SNR=Pmax/sigma2, Pmax1, R1 기본. K2/5는 screen1seed, K4는3seeds. 원자료와 사전기록을 함께 확인한다.

**프라이버시 범위:** 공개·고정된 client별 클래스 분포/크기/CSI/routing 조건에서 client의 모든 feature를 교체하는 conditional DP 상한. 클래스 분포/참여 여부 보호, 악성 수신기, 공모, 여러 안테나로 개별 신호 분리, 반복 삭제 요청 전체의 보장은 아니다. delta1e-5. 매우 큰 epsilon은 유용한 프라이버시 보장으로 해석하지 않는다.

## K4 고정 라운드/SNR 비교 (3 seeds 평균)
|SNR|배정|정확도|최저 클래스 recall|조건부 epsilon 상한|집계 MSE|source+delete RE|
|---|---|---|---|---|---|---|
|0|channel|65.14%|5.27%|4.23e+06|0.0149|520818240|
|0|joint|65.54%|3.77%|1.45e+06|0.0179|520818240|
|0|random|66.58%|8.10%|4.66e+05|0.0256|520818240|
|10|channel|64.92%|6.00%|4.22e+07|0.00149|520818240|
|10|joint|65.55%|3.73%|1.45e+07|0.00179|520818240|
|10|random|66.77%|8.70%|4.63e+06|0.00256|520818240|
|20|channel|64.98%|5.67%|4.21e+08|0.000149|520818240|
|20|joint|65.39%|3.43%|1.45e+08|0.000179|520818240|
|20|random|66.76%|8.97%|4.62e+07|0.000256|520818240|

같은 K/R/라운드에서는 배정별 통신 자원 수가 같다. 좋은 grouping만으로 송신 symbol 수가 줄었다고 주장하지 않는다.

## Joint의 목표 dev accuracy 70% 도달 비용 판정
|SNR|비교 기준|평균 최종 정확도 차이 pp|모든 seed에서 두 방식 도달|평균 자원 절감|개선 seed|실용 기준|
|---|---|---|---|---|---|---|
|0|random|-1.04|False|N/A|0/3|False|
|0|channel|+0.40|False|N/A|0/3|False|
|10|random|-1.22|False|N/A|0/3|False|
|10|channel|+0.63|False|N/A|0/3|False|
|20|random|-1.36|False|N/A|0/3|False|
|20|channel|+0.41|False|N/A|0/3|False|

실용 기준: 정확도 평균 손실2pp 이내, 같은 dev 목표에 대한 총 RE 평균10% 이상 절감 및2/3seed 이상 개선. 사전등록한 전체3seed 판정이며 screen/confirm 차이는 원자료에 명시된다. N/A는 실패/미도달을 삭제한 평균이 아니라 비교 불가다.

## 같은 정확도·조건부 privacy 목표에서 grid 탐색
|epsilon 상한|실현 가능한 seed/SNR/method 셀|전체 셀|
|---|---|---|
|8|0|27|
|100|0|27|
|100000|0|27|
|10000000|0|27|

각셀의 최저 source+delete RE, 선택K/R/round, dev와test는 matched_accuracy_privacy.json. 이 유한grid에서 미실현이라는 뜻이며 불가능성 증명은 아니다. seed10801만 K2/5와privacy-cap 후보가 있어 seed간 탐색공간이 다르므로 최저비용을 단순 pooled 비교하지 않는다.

## 프라이버시 제한 전력 제어 (screen seed, K4/SNR10)
|배정|목표epsilon|실제누적epsilon|정확도|
|---|---|---|---|
|random|8|8.000|6.34%|
|channel|8|8.000|6.39%|
|joint|8|8.000|6.52%|
|random|100|100.000|10.03%|
|channel|100|100.000|4.19%|
|joint|100|100.000|9.43%|

## R1에서 R4 반복 전송으로 변경 (confirmation 2 seeds)
|seed|배정|정확도 차이 pp|MSE 비율|epsilon 비율|총 RE 비율|
|---|---|---|---|---|---|
|10802|random|+0.11|0.250|3.993|1.176|
|10802|channel|+0.02|0.250|3.998|1.176|
|10802|joint|-0.16|0.250|3.996|1.176|
|10803|random|+0.12|0.250|3.992|1.176|
|10803|channel|+0.14|0.250|3.998|1.176|
|10803|joint|-0.29|0.250|3.997|1.176|

## 세부자료와 한계
- all_final_metrics.json:57개 전체 조건. summary.json:조건별 평균/범위. paired_grouping.json:seed별 차이. equal_total_budget.json:K2/R1/160라운드 상당의 동일 총 예산에서 가능한 최신checkpoint 비교.
- deletion_target_check.json:screen K4/SNR10에서client0/7/14 삭제. 각source+한삭제 요청의 별도 시나리오이며 요청3개를 순차 공개하는 privacy budget이 아니다.
- natural R1과 R4 비교는 confirmation 2 seeds만 대응 비교해야 한다. 반복은 UL/energy와 privacy composition을 증가시킨다. R4의 MSE 감소와 정확도 변화 및 epsilon 증가를 함께 확인한다.
- 고정 clip/고정 전력 한도, 이상적 SISO 정렬, 완전 CSI, orthogonal shard transmission. MIMO 동시 분리나 실제 무선 간섭을 평가하지 않았다. 고정 라운드이며 수렴 보장이 없다.
- 잡음 공분산과 민감도가 모두 shard 크기에 의존하므로, 크기가 커진다고 자동으로 privacy가 강해지지 않는다. 높은 epsilon 상한은 강한 보호를 입증하지 못한다는 뜻이며, 상한 차이는 실제 공격 성공률의 측정이 아니다.
- 전체 데이터에 대한 client DP를 원하면 공개 label histogram도 보호하는 별도 protocol이 필요하다. raw histogram 기반 routing의 조건부 범위를 숨기지 않는다.

## 비용
```json
{
  "seconds": 1633.3291813998949,
  "gpu_board_Wh": 73.48497380491264,
  "sampling_seconds": 5,
  "external_payment": 0,
  "scope": "whole GPU board; not whole host, electricity tariff unavailable",
  "completed_deletions": 63,
  "source_cases": 57,
  "local_calls": 420000.0,
  "error": null,
  "ledger_count": 183,
  "preflight_seconds": 1.7904397000093013,
  "preflight_gpu_board_Wh": 0.0
}
```

GPU board Wh는 GPU 보드 샘플 적분이며 host 전체/통신 RF Joule이 아니다. 외부 유료 서비스0. 정규화 signal energy는 각 ledger에 별도로 보존한다. 실패 시 failure.json도 확인한다.

재현: ../snr_privacy.py, ../snr_privacy_analyze.py, ../SNR_PRIVACY_PREREG.md. 원자료 ../snr_privacy_v1/{case}/config,partition,labels,source와reference/replay models·ledgers·curves. 코드 hash는 environment.json. 기존 실험 결과는 변경하지 않았다.
