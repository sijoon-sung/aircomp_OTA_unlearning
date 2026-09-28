# 폭·채널 AirComp와 겹치는 선행, 이번 후보의 위치

검토일 2026-09-28. 검색에서 동일 표현을 발견하지 못한 것은 신규성 증명이 아니다. 아래 논문들의 문제·연산·송수신 조건을 구분한다. 본 실험은 공식 알고리즘의 재현 또는 SOTA 비교가 아니다.

## 1. Krouka 등: Communication-Efficient Split Learning Based on Analog Communication and Over the Air Aggregation

[저자 arXiv 원문](https://arxiv.org/html/2106.00999v1), [IEEE 출판 페이지](https://ieeexplore.ieee.org/document/9685045/). 2021 preprint. 초록·시스템식·실험 및 결론을 확인했다.

- 초록/문제: 다수 장치의 split inference에서 digital 개별 feature 전송이 만드는 대역폭 병목.
- 본문 해결: 장치 측에 선형 변환층을 두고, 서버 첫 층에 필요한 가중 합을 analog 중첩으로 계산한다. 송신 전력 정규화와 채널 보상, 공통 scale이 등장한다.
- 결론의 범위: 장치 수가 증가할 때 inference payload 비용상 이점. Offline 학습된 모델의 remote inference가 중심이다.
- 우리의 해석/한계: 독립적인 이미지들의 training loss를 합산해도 된다는 결과가 아니다. Projection 뒤 공중합산, 차원 선택, 동시 전송 자체는 이번 신규성에서 제외한다.

## 2. Yang 등: Over-the-Air Split Learning with MIMO-Based Neural Network and Constellation-Based Activation

[저자 원문, MLSP 2022](https://arxiv.org/abs/2210.03914). 초록·forward/backward 수식·변조 activation과 실험 부분 확인.

- 문제: SL의 층 사이 전송 비용과 아날로그 송신 구현.
- 방법: precoder–MIMO channel–combiner를 하나의 신경망 선형 연산으로 사용한다. 채널 reciprocity로 backward를 구현하며 constellation 기반 activation을 제안한다.
- 결과의 범위: 제안 MIMO 구조에서 학습 가능성을 수치적으로 보인다.
- 우리의 해석/한계: 채널 상호성·동기화·변조/학습 가정을 고려해야 한다. ‘무선 채널로 NN 선형층을 만들었다’, ‘OTA backward가 된다’는 주장도 이미 선행에 있다. 우리 common digital backward는 이 방법의 재현이 아니다.

## 3. Yang 등: Over-the-Air Split Machine Learning in Wireless MIMO Networks

[저자 전체 원문](https://arxiv.org/html/2210.04742v2), JSAC 2023, DOI 10.1109/JSAC.2023.3242701. 초록·II/III/IV·V·결론을 확인했다. 가장 직접적인 channel 연산 선행이다.

- 초록: MIMO 무선 연산을 NN 층으로 이용하여 split ML 통신과 계산을 결합한다.
- 본문: W=Σ C_kᴴHP_k와 rank 조건을 제시하고, reciprocity 기반 noisy backward를 구성한다. III-A는 송신 power normalization과 수신 BN, backward scale과 Adam을 다룬다. Algorithm 1은 covariance/SVD 기반 보조 gradient도 포함한다. IV-A는 convolution/channel 변환으로 확장한다.
- 결과/결론: complex ResNet/VGG의 CIFAR-10 실험, static/quasi-static 채널에서 제안 구현을 평가한다.
- 우리의 해석/한계: 단순 width/channel 분할과 power normalization을 최초 제안할 수 없다. 다만 여기서 시험한 ‘SISO multiuser의 공통 min-scale 아래, activation-dependent 수신 잡음 분산의 기대손실 gradient를 scalar trace로 보정’과는 목적·구조가 다르다. 이것이 충분한 신규성이라는 결론은 아직 아니다. 기존 논문의 오류를 발견했다는 주장도 하지 않는다.

## 4. Wen 등: Task-Oriented Over-the-Air Computation for Multi-Device Edge AI

[저자 원문](https://arxiv.org/html/2211.01255v1). 초록·시스템 모델·목적함수/해법·결론 확인.

- 문제: 평균 aggregation MSE를 최소화해도 feature별 분류 중요도가 달라 정확도가 최적이 아닐 수 있다.
- 방법: discriminant gain을 목표로 transmit precoding과 receive beamforming을 함께 설계하고 SCA로 푼다. 다중 센서 feature를 OTA로 모으는 split inference이다.
- 결론: 사람 동작 인식 실험으로 task-oriented 설계의 이점을 보인다.
- 우리의 해석/한계: ‘중요 feature의 잡음만 작게 만들자’ 또는 ‘정확도를 고려한 전력배분’만으로 신규성을 주장하기 어렵다. 본 실험의 online gradient 보정과는 다르지만 향후 비교해야 할 통신 설계 계열이다.

## 5. Oh 등: Communication-Efficient Split Learning via Adaptive Feature-Wise Compression (SplitFC)

[저자 전체 원문](https://arxiv.org/html/2307.10805v2). 초록·방법·통신식·실험·결론 확인.

- 문제: feature와 gradient를 매번 보내는 SL의 통신량.
- 방법: feature별 dispersion에 따라 dropout과 양자화 수준을 조정한다. Forward에서 남긴 feature의 backward만 보내고, 양자화 오차 분석으로 bit 배분을 정한다.
- 결론: 이미지 분류 실험에서 두 압축 전략의 결합을 평가한다.
- 우리의 해석/한계: 압축에 따른 통신–정확도 trade-off 자체는 선행이다. 이번 Latent4/8은 동일 모델의 단순 양자화 대조군이며 **SplitFC라는 이름을 붙이거나 공식 구현 성능으로 보고하지 않는다**. 논문 수준 경쟁력을 주장하려면 adaptive 방법과 추가 비교가 필요하다.

## 6. Zheng 등: Improving Convergence for Semi-Federated Learning: An Energy-Efficient Approach by Manipulating Over-the-Air Distortion

[저자 원문](https://arxiv.org/html/2506.21893v1), 2025 preprint. 초록·도입·방법 구조 확인.

- 문제: SemiFL에서 gradient OTA 왜곡을 무조건 억제하는 전력 비용.
- 방법: 학습 구간을 나눠 OTA distortion을 조절하고 수렴 및 energy 관점에서 자원을 배분한다.
- 초록/결론의 주장: noisy aggregation을 학습 상태에 따라 다르게 제어하면 수렴·에너지 절충을 개선할 수 있다.
- 우리의 해석/한계: ‘무선 잡음을 학습에 활용한다’는 넓은 주장 역시 신규성이 아니다. Gradient aggregation의 distortion 제어와 activation normalization의 chain derivative는 구분해야 한다.

## 7. Gaussian gradient 이론은 기존 도구

[Rezende, Mohamed, Wierstra, ICML 2014 저자 논문](https://proceedings.mlr.press/v32/rezende14.pdf)의 Gaussian 미분 식(특히 covariance derivative)을 확인했다. Price/Bonnet identity는 오래된 결과다. 이번 제안은 이 항등식이나 Hutchinson trace 추정을 새 이론으로 주장하지 않는다. 차별화 가능성은 **개별 activation 없이 받은 aggregate에서 계산한 scalar로 송신 장치의 누락된 항을 구현하는 프로토콜**에 한정된다. 실험이 단순 regularizer보다 불리하면 이 차별점만으로 방법론 논문을 밀지 않는다.

## 8. 원문 확보가 안 된 가까운 문헌

`Federated Split Learning via Low-Rank Approximation: A Communication-Efficient Approach`, DOI [10.1109/TWC.2026.3657151](https://doi.org/10.1109/TWC.2026.3657151)는 서지 검색에서 발견했다. 이번 세션에서는 DOI 원문 접근이 실패했으며 primary full text를 확인하지 못했다. 따라서 본문의 정확한 rank 정의·AirComp 유무·정리·정량 결과를 확정하여 쓰지 않는다. 논문 제출 전 필수 대조 항목이다. 이 제목만을 근거로 이번 구조와 동일하다고 단정하지 않되, ‘저차원 SFL 최초’ 주장은 이미 검토 대상에서 제외한다.

## 9. 폭 분할에 직접 가까운 FedDCT

[Nguyen 등의 저자 원문](https://arxiv.org/html/2211.10948v2), [공식 코드](https://github.com/vinuni-vishc/fedDCT), TNSM DOI 10.1109/TNSM.2023.3314066. 동명의 Dynamic Cross-Tier 논문과 구분한다. 초록·channel-wise model division·cluster aggregation과 학습 구조를 확인했다.

- 문제: 자원이 작은 장치가 큰 CNN 전체를 학습하기 어려움.
- 방법: 모델을 작은 submodel ensemble로 나누고 cluster 내부 장치들이 협력 학습한다. 원본 데이터 공유 대신 main client가 여러 lower branch를 실행해 각각의 abstract feature를 proxy 장치에 보낸다. Proxy의 개별 gradient를 회수하고, cluster 사이 모델을 다시 집계한다.
- 결과의 범위: CIFAR 및 의료 이미지에서 메모리/정확도/round 비교를 보고한다.
- 우리의 해석/한계: ‘폭을 나눠 클라이언트 부하를 낮춘다’와 ‘cluster 단위 협력 후 federation’도 선행이다. 개별 branch feature/gradient가 필요한 절차를 그냥 합산으로 대체할 수는 없다. 이번 group CNN은 FedDCT의 co-training/ensemble 절차를 재현하지 않는다. 새로운 주장을 하려면 이 기존 구조에 더한 **OTA 송수신에서만 생기는 문제**를 분명히 해야 한다.

## 10. 채널 압축에 가까운 SL-ACC

[Lin 등의 저자 전체 원문](https://arxiv.org/html/2508.12984v1), 2025 preprint. 초록·II의 ACII/CGC 수식·III 실험·IV 결론 확인.

- 문제: 채널마다 다른 smashed-data 중요도를 무시한 동일 압축.
- 방법: 현재/과거 Shannon entropy로 채널을 평가하고, K-means 그룹별 bit 수를 정한다. Activation과 gradient 양방향에 적용한다.
- 결론: MNIST/HAM10000의 IID/non-IID에서 통신과 분류 성능을 평가한다.
- 우리의 해석/한계: 여기서 channel은 NN feature channel이다. 장치 여러 개의 신호를 무선으로 동시에 더하는 송수신 구조와 동일하지는 않다. 그래도 ‘중요 채널만 보낸다’는 후보를 새로 주장할 수 없고, 4-bit 고정 양자화만 이겼다고 이런 adaptive 방법들을 이겼다고 말할 수 없다.

추가 서지 탐색에서 `Optimizing Federated Learning Efficiency via Slimmable Neural Networks and AirComp for Resource-Heterogeneous Devices`, DOI 10.1109/JSAC.2026.3726846도 발견했다. Primary 본문 접근은 실패했다. 실제 내용/발행 상태를 이번에 확정하지 않았으며, 차후 제출 전 확인해야 할 목록에만 남긴다.

## 신규성 판정 기준

|요소|판정|
|---|---|
|폭/채널 분할, local projection, OTA 선형 합산|선행에 있음. baseline|
|저차원 latent와 digital 양자화|기존 압축 원리. baseline|
|채널 중요도/잡음에 따른 전력 조절|관련 선행 다수. 그 자체로 신규성 아님|
|Gaussian covariance derivative, HVP, trace estimator|기존 수학 도구|
|공통 min-scale의 activation-dependent noise gradient를 aggregate-only server의 scalar feedback으로 보정|이번에 좁혀 검토한 후보. 위 문헌과 문제/프로토콜 차이는 있지만 완전한 신규성 확인은 아님|
|단순 보정보다 확실한 정확도–총비용 이점|GPU 결과로 판정. 없는 경우 후보 채택 보류|

특히 ‘논문에서 depth split을 썼으니 우리는 width split이라 새롭다’고 주장하지 않는다. 선형층/conv의 채널 계산을 AirComp에 넣는 아이디어가 이미 존재한다.
