# 먼저 상관관계를 확인: 기존 자료의 탐색 분석

2026-10-03. 사용자 지시에 따라 신규 배정 알고리즘 설계보다 관계 검증을 우선한다. 기존 실험 결과를 이미 관측했으므로 이 분석은 사후 탐색이며 새 독립 확인 실험이 아니다. 기존 원자료는 수정하지 않는다. GPU 재학습은 이 단계에서 수행하지 않는다.

## 서로 다른 세 질문

1. shard당 client 수와 앙상블 정확도/삭제 통신 비용: seed10601, random_norm, K1/2/4/5의4점을 비교한다. 다른 seed에는 K2/5가 없으므로 pooled correlation으로 반복 측정 수를 부풀리지 않는다. Pearson/Spearman을 기술하되 유의성·일반성을 주장하지 않는다.
2. 같은 크기의 shard에서 데이터 대표성 및 channel bottleneck과 결과: K4 random_norm/channel_norm의 seed10601,10611,10612,10613 네 쌍을 비교한다. seed10601은 선택에 쓰였으므로 held-out seed3쌍을 별도로 요약. class histogram과 client 균등평균 목표분포 사이 JS divergence, shard 평균/최악 JS, 삭제 후 해당 shard JS, retained min channel gain, 삭제 UL/totalRE/accuracy를 기록. 단일 deletionclient0이며 source grouping도 바뀌므로 causal 효과로 해석하지 않는다.
3. client 데이터 분포와 채널 특성 자체가 관련되는가: 각 seed20 clients의 class histogram 간 JS distance와 long-term channel dB 간 절대거리의 Spearman correlation을 계산. 190개 pair가 독립 표본인 것처럼 p-value를 계산하지 않고, client channel label을 순열하는 Mantel-style permutation 1999회로 exploratory 양측 p를 계산. primary4seeds에 BH 보정, stress1seed는 분리. class 번호를 연속 수치로 취급한 단순 Pearson은 사용하지 않는다.

## 해석 범위와 기준

main은 채널을 데이터와 독립 생성했고 stress는 evaluator가 dominant class 순서와 채널 순서를 연결했다. 따라서 이 자료로 현실의 client 데이터와 실제 무선 채널의 상관 여부를 판정할 수 없다. permutation은 simulator 점검이며 empirical deployment evidence가 아니다.

관계 후보는 기술적 |rho|>=.5를 참고하되, 4점 크기 sweep/3개 독립 확인 seed로 채택하지 않는다. 특히 이전 실험은 목표 집계 MSE를 일정하게 유지하도록 반복 전송량을 변경했으므로 채널 변화가 정확도보다 비용으로 드러나도록 설계되어 있었다. 채널-정확도 관계에 대한 fixed-MSE와 fixed-radio-budget 실험을 구분해야 한다.

모든 결과·차이 없음·반대 방향을 보고한다. CPU 시간, 추가 GPU 학습0, 원자료/분석 코드 위치를 기록한다. 상관을 확인하지 못한 결과를 상관이 없다는 증명으로 쓰지 않는다.

## 사용자 명확화와 후속 실행

사용자는 우선 대상을 client 데이터 분포와 무선 채널 특성의 관계로 명확히 했다. 위 기존 자료 사후분석은 실행 완료로 간주하지 않는다. 후속 실험 요청으로 별도 사전기록 CORRELATION_CONTROL_PREREG.md에 따라 18개 신규 합성 연결 통제 실험을 실행했다. 결과는 correlation_control_v1/RESULT_INTERPRETATION_KO.md 및 REPORT_KO.md에 있다. 실제 데이터–채널 상관의 존재는 여전히 미확인이다.
