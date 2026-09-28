# OTA-FL 문헌과 후보의 겹침

2026-09-28 확인. 아래 ‘본문 확인’은 명시한 절/식/결론을 확인했다는 뜻이며 모든 증명과 구현을 재현했다는 뜻이 아니다. 논문 주장과 우리의 해석을 구분했다. 외부 PDF를 저장소에 재배포하지 않는다.

## 기초: 정렬·전력·참여

**1. Federated Learning via Over-the-Air Computation** — Yang, Jiang, Shi, Ding. 2018 preprint, TWC 2020. [원문](https://arxiv.org/pdf/1812.11750)

- 초록/문제: 다수 client의 빠른 모델 합산.
- 본문/결론 확인: MSE 조건 아래 참여 수를 늘리는 device selection·수신 beamforming, sparse/low-rank DC formulation. 결론은 완전 CSI 가정의 확장을 남긴다.
- 우리 해석: ‘더 많은 client + 낮은 집계 MSE’나 beamforming 최적화 자체는 기여가 아니다. 채널/선택이 학습에 미치는 구체적 차이가 필요하다.

**2. Broadband Analog Aggregation for Low-Latency Federated Edge Learning** — Zhu, Wang, Huang. 2018 preprint, TWC 2020. [원문](https://arxiv.org/pdf/1812.11494)

- 초록/본문/결론 확인: broadband analog 합산, truncated channel inversion, 거리 기반 scheduling. SNR–truncation 및 신뢰도–참여량 절충을 분석한다.
- 우리 해석: weak client를 빼서 빨라지는 방식과 정확도 절충은 출발점부터 존재한다. 새 후보는 추가로 생기는 오차 원인을 명확히 해야 한다.

**3. Over-the-Air Federated Learning from Heterogeneous Data (COTAF)** — 2020 preprint, TSP 2021. [원문](https://arxiv.org/pdf/2009.12787)

- 초록/본문/결론 확인: 시간에 따라 precoding/수신 scaling을 바꿔 유효 잡음을 완화한다. Heterogeneous local SGD를 다루며, convex 분석과 CIFAR-10 CNN 실험을 구분한다.
- 한계/우리 해석: CNN 실험의 잡음 이점은 모든 nonconvex 모델에 대한 증명이 아니다. ‘무선 잡음이 학습을 돕는다’는 주장도 이미 있으므로 새 기여는 제어 기준과 적용 조건이어야 한다.

**4. Transmission Power Control for Over-the-Air Federated Averaging at Network Edge** — Cao, Zhu, Xu, Cui. 2021 preprint. [원문](https://arxiv.org/pdf/2111.05719)

- 본문 확인: Theorem 1, Eq. (20)–(21)에 aggregation bias와 MSE를 명시하고 power/denoising factor로 표현한다. 여러 라운드의 오류 기여도까지 반영한다.
- 우리 해석: 현재 후보에 가장 중요한 견제선이다. ‘편향 항을 추가한 전력 제어’만으로는 부족하다. 분할·배정·타 조각 간섭을 추가했을 때도 기존 식의 단순한 적용이면 신규성은 약하다.

## 비용·CSI·새 프로토콜

**5. Over-the-Air Federated Learning with Retransmissions (Extended Version)** — Hellström, Fodor, Fischione. 2021 preprint; 관련 journal 후속은 TWC 2023. [원문](https://arxiv.org/pdf/2111.10267)

- 초록/본문/결론 확인: static channel에서 반복 송신 power control, loss bound, 반복 수 선택 heuristic.
- 한계: 본문에서 static channel 확장을 별도 과제로 남긴다. Gradient element의 IID 및 normalization 단순화도 분석에 사용한다.
- 우리 해석: 횟수만 늘려 MSE를 줄인 결과는 새 기여가 아니다. 같은 총 통신 예산에서 비교해야 한다.

**6. Over-the-Air Computing with Imperfect CSI: Design and Performance Optimization** — Evgenidis 등의 연구, TWC 2024. [기관 기록](https://cris.fau.de/publications/324822441/) · [저자 preprint](https://d197for5662m48.cloudfront.net/documents/publicationstatus/168095/preprint_pdf/1e806e81b01536545d3c1fb046efa678.pdf)

- 초록/본문/결론 확인: CSI magnitude/phase 오차를 고려한 MSE 최적화와 pilot 재측정. Eq. (53)–(55)의 RPC는 재측정 비용과 MSE를 함께 평가한다.
- 우리 해석: ‘pilot를 적응적으로 더 보내고 overhead와 trade-off’도 이미 구체적으로 연구됐다. 기존의 열린 문제를 지금까지 미해결이라고 전제하지 않는다.

**7. Non-Coherent Over-the-Air Federated Learning: Protocol, Convergence, and Device Scheduling** — Wen 등, 2026-09-08 preprint. [원문](https://arxiv.org/html/2609.08312v1)

- 초록/본문 확인: noncoherent detection, binary dither, error feedback, device scheduling/power control. Small-scale instantaneous CSI는 필요 없지만 large-scale gain은 사용한다.
- 조건: Eq. (6) 등의 이론은 학습 궤적에 적응하지 않는 scheduling sequence로 조건을 명확히 한다. Optimized scheduler에서는 전원이 candidate local update를 계산하고 두 scalar를 보고한 뒤 선택된다.
- 우리 해석: ‘완전 CSI 불필요’나 ‘error feedback 추가’만으로 새 연구가 되지 않는다. 전원 사전 계산의 비용은 후속 문제 후보지만, 과거 norm 기반 scheduling과도 비교해야 한다. 최신 preprint이므로 확립된 최종 journal 결과처럼 다루지 않는다.

## 분할·그룹화: 현재 후보와 가장 가까움

**8. SegOTA: Accelerating Over-the-Air Federated Learning with Segmented Transmission** — Zhang, Dong, Liang, Afana, Ahmed. WiOpt 2025. [원문](https://arxiv.org/pdf/2504.09745)

- 확인 범위: 전체 구조, Eq. (3)–(5), Assumption 1–2, 그룹/beam 최적화, 실험 조건과 결론.
- 본문: spherical k-means로 채널 그룹을 만들고 송수신 빔을 조정한다. 그룹별 조각의 동시 전송으로 uplink 시간을 줄인다. Known CSI와 error-free downlink를 가정한다.
- 결론/한계: 실험은 균등 random 데이터 배분의 MNIST이며, client는 전체 local model을 학습한다. 따라서 전송량 감소를 client 계산량 감소라고 해석할 수 없다.
- 후보 차이: 그룹별 effective weighting을 목표 FedAvg와 비교하고, 이를 공간 분리 및 조각 표본 분산과 함께 설계하는 문제. 원문의 gradient-divergence 항이 있으므로 ‘이론에서 이질성 무시’라고 주장하지 않는다. 공식 코드는 이번 검색에서 확인하지 못했으며 이는 코드 부재의 증명이 아니다.

**9. Balancing Energy Efficiency and Distributional Robustness in Over-the-Air Federated Learning** — Badi 등. 2023 preprint, ICMLCN 2024. [원문](https://arxiv.org/html/2312.14638v1)

- 초록/본문/결론 확인: channel-dependent PMF와 distributional-robustness PMF를 결합한다. Eq. (7)–(9)의 tuning factor가 좋은 채널 선호 강도를 조정한다.
- 후보와 겹침: channel–data 절충이라는 큰 이야기는 이미 존재한다.
- 남는 차이: 선택한 clients가 같은 전체 벡터를 합산하는 방식과, 서로 다른 조각을 동시에 보내 생기는 off-diagonal interference는 다르다. 다르다는 사실만으로 기여가 충분한 것은 아니다.

**10. Air-FedGA: A Grouping Asynchronous Federated Learning Mechanism Exploiting Over-The-Air Computation** — IPDPS 2025. 후속 **Asynchronous Federated Learning Over Non-IID Data via Over-the-Air Computation**, IEEE TON 2026, DOI 10.1109/TON.2025.3641928. [conference 저자 PDF](https://qianpiao.github.io/files/Air-FedGA_A_Grouping_Asynchronous_Federated_Learning_Mechanism_Exploiting_Over-The-Air_Computation.pdf) · [journal 저자 PDF](https://qianpiao.github.io/files/Asynchronous_Federated_Learning_Over_Non-IID_Data_via_Over-the-Air_Computation.pdf)

- 확인 범위: 초록, 구조 설명, grouping/비교 실험 절. 전체 증명 재현은 하지 않았다.
- 그룹 안 동기 OTA, 그룹 사이 비동기 갱신. 이질적인 계산 시간, 그룹 데이터 분포, staleness, MSE를 다룬다.
- 후보와 겹침: 데이터 분포를 고려한 그룹화가 이미 있다. 차이는 ‘그룹’이라는 용어가 아니라 **같은 시간에 다른 parameter blocks를 합산하는 구조**에서 나와야 한다. Label histogram을 사용하는 비교군에는 추가 정보/통신 비용을 명시한다.

**11. Robust and Efficient Average Consensus with Non-Coherent Over-the-Air Aggregation** — Deng, Chen, Larsson. 2025 preprint. [원문 링크](https://arxiv.org/abs/2504.05729)

- 확인 범위: 이번에는 공식 초록. Noncoherent interference 아래 평균 consensus의 bias를 다루고 power/receive scaling을 결합한다.
- 판정: 연산자 norm 기반 robust 설계도 넓은 선행 영역이 있으므로 새로운 spectral-norm 이론으로 선전하지 않는다. 최종 novelty 검토 때 본문의 관련 최적화식 대조가 추가로 필요하다.

## 후보 순위

1. **분할 OTA의 그룹 대표성–공간 간섭 공동 설계:** 문제를 수식으로 좁힐 수 있고 작은 검증이 가능. 첫 검증 후보. 단순 계수 보정은 이미 내부 비교에서만 이기므로 채택하지 않음.
2. **Noncoherent OTA의 사전 local-compute/제어비:** 실제 비용 문제는 명확하지만 일반적인 과거 norm 기반 선택과 겹칠 위험이 높아 예비 후보로만 기록.
3. **재전송/pilot 조절, CSI-free+EF, 잡음 활용 일반론:** 해당 조합과 trade-off 자체로는 새 후보에 올리지 않음.

SFL은 별도 문서에서 구조의 타당성과 계산·통신 비용을 비교한다. 이번 조사에 포함되지 않은 모든 OTA 논문까지 신규성 검증을 마쳤다는 뜻은 아니다.
