# 소규모 검증 계획과 중단 조건

상태: **미실행 계획**. 이번에 실행한 것은 `audit_segmented_weights.py`의 정확한 산술 검산뿐이다. 아래 GPU 학습이나 SegOTA 원본 재현을 완료했다고 해석하지 않는다.

## 1차: 원본 재현과 문제가 남는지 확인

- SegOTA Eq. (3)–(5), power constraint, 그룹 알고리즘을 구현하고 알려진 ideal/noise-only 사례부터 검증한다. 논문 기반 재구현과 공식 코드를 구별한다.
- Full-model/delta 전송과 normalization을 분리한다. Delta로 바꾸는 것만으로 개선되는 경우 그 효과를 새 그룹 방법의 효과로 합치지 않는다.
- 목표 FedAvg 가중치 (p_k), 실제 own-segment 계수, cross-segment leakage를 로그로 남긴다. Oracle에서 individual gradient를 읽는 진단 코드는 scheduler 입력에서 분리한다.
- 동일한 데이터 partition에 channel–data 관계만 permutation해서 비교한다. 단지 SNR과 non-IID를 동시에 더 나쁘게 만드는 실험으로 원인을 혼동하지 않는다.
- CPU/소규모 synthetic quadratic에서 원본과 단순 rescale을 검증한다. 그룹을 매 라운드 새로 만들 때 장기 기대 편향이 실제로 사라지는지도 확인한다.

이 단계에서 문제의 크기가 작거나 재현이 안 되면 GPU 규모를 늘리지 않는다.

## 고정 비교군

1. Full-model AirComp: unbiased inversion과 일반 MSE/LMMSE power/receive 설계.
2. SegOTA 원형: channel grouping + 원문의 aggregation; random/round-robin을 구분.
3. SegOTA + 단순 group mass 보정; 가능하면 within-group gain 보정도 별도.
4. 데이터 mass 균형 grouping + 동일한 통신 solver. Label-balanced grouping은 label histogram을 쓰는 별도 정보 우위 비교군.
5. 기존 weighted-MSE objective를 분할 구조에 그대로 적용한 solver.
6. 검토 후보: 전체 aggregation operator 오차와 잡음을 반영해 제한된 grouping/beam/보정 후보 중 선택.

Noise-free full FedAvg는 정확도 기준점이지 동일 비용의 통신 비교군은 아니다. 후보에서 beam solver를 개선하면 같은 solver를 적용한 ablation도 둔다.

## 작은 GPU pilot

첫 단계 통과 시에만 FashionMNIST의 작은 CNN, client 20명, seed 3개로 시작한다. 모든 방식은 동일 모델/초기화/partition/local step을 사용한다. Single GPU에서 순차 실행하고 다른 작업과 중첩하지 않는다.

- 모델은 전체 CNN을 학습한다. Frozen tiny head로 문제를 바꾸지 않는다.
- Segments (S=1,2,4), 수신 안테나 (N=4,8). 첫 pilot은 (S=2,N=4)에서 시작해 조건 수를 억제한다.
- 동일 non-IID partition을 channel group과 독립 배치/상관 배치한 두 조건. 추가 균등 IID 조건은 sanity check로 사용한다.
- 1차 SNR 20 dB, 정확 CSI. 통과하면 10 dB와 imperfect CSI 5%를 스트레스 조건으로 추가한다. 5%는 채널 NMSE 등의 정확한 정의를 미리 고정한다.
- Server가 사용하는 정보는 CSI, data count, 공개 clipping bound, 기존 metadata로 제한한다. Ground-truth gradient나 test accuracy로 라운드별 후보를 고르지 않는다.
- 후보 탐색은 작은 정해진 횟수의 group move/swap 및 (\lambda\in\{0,.25,.5,.75,1\}). 부가 연산시간을 측정한다. Oracle 선택은 상한 진단용 별도 곡선이다.

첫 pilot의 seed/validation split에서 정한 hyperparameter는 test seed 결과를 보고 재선택하지 않는다. 본 실험은 다른 seed 3개와 두 번째 dataset에서 확인하며, pilot이 실패하면 본 실험을 시작하지 않는다.

## 비용 원장

$$
T_{\rm total}=\sum_t\left(T_{\rm local,t}+T_{\rm CSI,t}+T_{\rm ctrl,t}
+T_{\rm UL,t}+T_{\rm DL,t}+T_{\rm server,t}\right).
$$

실제 overlap이 있으면 그 schedule의 critical path로 다시 계산하며 직렬 합과 혼동하지 않는다. Pilot/control을 공짜로 두지 않는다. Analog payload는 bit 수로 억지 환산하지 않고 real/complex channel uses를 명시한다.

- 주 지표: 공통 target accuracy까지 총 modeled time/energy, 같은 총 예산에서 accuracy.
- 보조: 평균/최하위 client accuracy, aggregate NMSE, own-weight bias, sampling variance, cross interference, 송신 peak violation.
- TX 에너지: $\sum_{t,k}P_{k,t}\tau_{k,t}$. 사용한 device 수를 곱하지 않은 shared airtime과 device 에너지를 구분한다.
- Local compute, downlink, CSI, group ID/scale 제어, CP/guard, solver 시간이 모두 원장 항목이다. 추정한 RF 비용과 실측 GPU wall time을 따로 표기한다.
- Fixed rounds는 진단용으로만 제시한다. 같은 payload 또는 같은 총 시간 비교를 추가한다. Target 미도달은 결과에서 제외하지 않고 미도달/검열로 보고한다.

## 채택 판단

Pilot 내부 기준으로, strongest comparable baseline 대비 동일 target accuracy까지 **총 비용 10% 이상 감소**, 또는 동일 총 예산에서 **accuracy 1 percentage point 이상 증가**를 검토 기준으로 삼는다. 이는 학회 기준이나 통계적 유의성 보장이 아니라 시간 낭비를 막기 위한 자체 기준이다. Seed별 분산과 paired difference를 함께 보고, 3개 seed만으로 보편적 우월성을 단정하지 않는다.

다음 중 하나면 확장하지 않는다.

- 일반 weighted-MSE 분할 적용이나 단순 rescale과 사실상 같은 방법/성능이다.
- Full-model OTA가 동일 총 비용에서 더 좋다.
- 극단적으로 유리한 channel–data 배치에서만 작은 이득이 있고 다른 조건의 손해가 크다.
- 제어/CSI/server 연산을 포함하면 이득이 사라진다.
- Metadata만으로 선택한 rule은 실패하고 oracle rule만 이긴다.

논문에 쓸 수 있는 주장은 이 조건을 통과한 영역의 trade-off다. 새 이름, 많은 실험 행, 평균 MSE 개선만으로 후보를 살리지 않는다.
