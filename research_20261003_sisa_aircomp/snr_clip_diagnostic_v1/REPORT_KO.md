# SNR 실험의 clipping 진단과 최종 판단

주57조건을 대체하지 않는 사후 진단8조건이다. 주 실험에서91~95.5% clipping과 dev70% 미도달을 확인한 뒤 별도 사전기록을 남기고 실행했다. seed10801 하나만 사용했으므로 확인실험3seeds 결과와 합쳐 평균하지 않는다.

|SNR|배정|C0.1 정확도|C1 정확도|차이 pp|C0.1 clipping|C1 clipping|
|---|---|---|---|---|---|---|
|0|random|66.22%|72.66%|+6.44|94.25%|0.00%|
|0|channel|65.15%|71.67%|+6.52|93.88%|0.00%|
|0|joint|66.19%|72.27%|+6.08|92.97%|0.00%|
|20|random|66.38%|71.49%|+5.11|94.22%|0.00%|
|20|channel|64.75%|70.34%|+5.59|93.91%|0.00%|
|20|joint|66.30%|70.98%|+4.68|92.97%|0.00%|

## 잡음을 거의 없앤 수치 기준
120dB/random에서 C0.1 정확도66.35%, C1 정확도71.44%. Test 차이+5.09pp, dev 차이+6.35pp. 진단 기준(test/dev 모두2pp 이상 개선) 통과=True.

120dB는 현실적인 무선 SNR 주장이 아니라 작은 잡음의 수치 기준이다. 0/20dB에서 C를 바꾸면 clipping뿐 아니라 송신 alignment와 잡음 규모도 함께 바뀐다. 따라서 이 두 조건만으로 clipping의 순수 인과효과를 주장하지 않는다.

## 해석
- 주 실험의 실패/미도달 판정은 유지한다. 현재 조건부 client DP 상한은 자연잡음만으로 실용적인 보호를 입증하지 못하고, 프라이버시 제한 조건의 정확도 결과도 함께 읽어야 한다.
- 진단에서 정확도가 회복된다면, 기본 실험의 작은 SNR 정확도 변화만으로 AirComp grouping 방향을 기각하지 않는다. 후속 확인은 calibration한 clipping/충분한 학습량에서 grouping과 SNR을 다시 비교하는 것이다. 아직 실행하지 않았다.
- 공동 배정은 고정된 .5/.5의 MSE/JS heuristic이며 SNR별 최적화나 최적성/신규성을 주장하지 않는다. 크기 자체의 privacy 효과와 bottleneck channel 효과도 구분해야 한다.
- 삭제 후 정확성은 고정 routing의 동일 SISA reference와 비교했다. 공개된 histogram/routing 자체의 삭제를 검증한 것은 아니다.

## 전체 실행 비용 (주 실험+진단+preflight)
```json
{
  "experiment_seconds": 1881.3776241999585,
  "gpu_board_Wh": 83.98151548141308,
  "local_calls": 475212.0,
  "external_payment": 0,
  "primary_source_cases": 57,
  "diagnostic_source_cases": 8,
  "primary_deletions": 63,
  "diagnostic_deletions": 8,
  "preflight_cases": 1,
  "energy_note": "Approximate sampled whole-board integral; preflight shorter than 5s has no integration interval, not zero actual energy"
}
```

GPU board Wh는 host 전체나 통신 RF Joule이 아니다. 각 조건의 통신 RE/정규화 signal energy는 ledger에 별도로 있다. 외부 유료 서비스0. 주 실험 보고서: ../snr_privacy_summary_v1/REPORT_KO.md. 원자료: ../snr_privacy_v1/ 및 이 폴더의 C0.1/C1.0 하위폴더. 비교 수치 comparisons.json, near_noiseless_diagnostic.json, 전체비용 combined_cost.json.
