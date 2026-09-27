**③ RTD-Air — 독립 teacher의 예측을 공중 합산하는 언러닝**

현재 유지하는 것은 **Original:10개 확률좌표를 합산하는 방식**이다. V1:9개 contrast좌표 전송은 통신량을 조금 줄였지만 최종 student의 삭제오차 개선을 입증하지 못해 보류한다.

**왜 학습 구조를 바꾸는가**

A의 영향을 받은 기존 글로벌모델을 teacher로 사용하면, A를 제외한 데이터로 증류해도 A의 영향이 다시 전달될 수 있다. 이를 피하기 위해 RTD-Air는 처음부터 각 client의 teacher를 독립적으로 학습한다.

T_i=Train(D_i,공개초기값,client별난수)이며 teacher 학습에 글로벌모델·글로벌KD 출력의 feedback을 넣지 않는다. Student는 별도로 학습하며 teacher로 되돌려 보내지 않는다. 삭제 후에는 A의 teacher를 ensemble에서 빼고 fresh student를 만든다.

이것은 기존 임의의 FedAvg 모델을 역으로 수정하는 알고리즘이 아니다. 초기 학습 규칙을 바꿔 target-free 구조를 쉽게 구성하는 방법이다. 미리 teacher를 독립적으로 준비하는 계산·저장 비용을 포함해서 평가해야 한다.

**수식과 삭제 조건**

공개 query 집합 X_pub={x_j}에 대해 q_i(x)=softmax(T_i(x)/tau),q_R(x)=sum_{i≠A}r_iq_i(x)로 둔다. r_i는 A를 제외하고 다시 정규화한 client 가중치다. 서버는 수신한 q_hat_R로 student S_theta를 학습한다.

`min_theta mean_x tau² KL(q_hat_R(x) || softmax(S_theta(x)/tau))`.

공개 query·초기값·학습난수·잔존teacher·배포규칙이 A의 private data에서 독립적이면, A를 제외한 이 절차의 출력도 그 데이터에 의존하지 않는다. 무잡음에서 동일 난수를 사용하면 같은 target-free KD 실행과 결과가 같아질 수 있다. 분포 관점의 주장에는 모든 입력과 난수,통신정책의 독립성 조건이 필요하다.

잡음이 A와 독립적인 경우에도 인과적 독립성은 유지될 수 있지만, noisy student와 noiseless KD reference가 같다는 의미는 아니다. 기존 A포함 student의 초기값·optimizer state를 재사용하면 위 설명이 깨질 수 있어 현재코드는 fresh 초기값을 사용한다.

**실제 송수신과 계산 순서**

1. 사전: 각client가500 SGD steps,batch64,lr=.03으로 teacher를 학습한다. 공개 query2000개와 온도tau=2를 공유한다.
2. 삭제 시: 서버는 A를 참여 집합에서 제외하고 query 순서·가중치·송신조건을 잔존client에 알린다. A의 마지막 gradient 송신은 필요 없다.
3. 각 잔존 client가 동일 query에 대한 q_i를 계산한다. 현재 수치 구현은 teacher FP32 출력을 FP64로 변환하고 각 행의 합을1로 정규화한다.
4. Original: r_i q_i를 채널 보상·전력 정규화해 같은 자원에 동시 송신한다. 서버는 확률의 합을 수신하고 simplex에 투영한다.
5. 서버는 마지막에 target을 FP32로 변환해 fresh student를600 KD steps,batch64,lr=.05로 학습한다. 각 step에서 새 OTA 송신을 받는 것이 아니라, 한 번 수집한2000 query target을 재사용한다.
6. Student 전체 모델을 배포한다. 별도의 retain recovery는 없다.

```text
독립 T_1,...,T_N 사전학습
서버 → 공개query·A제외집합 → 잔존clients
잔존clients → 동일query의 확률벡터 동시OTA → 서버
서버 → target-free 확률로 fresh student KD → 모델배포
```

공중 합산은 교사 예측 단계에 있다. 삭제 가능성의 핵심은 독립 teacher 구조이고, 그 조건은 일반 디지털 통신에서도 성립한다. 따라서 ‘AirComp를 사용한다’와 ‘AirComp만의 신규성이 확립됐다’를 구분한다.

**V1:9좌표 전송의 정확한 의미**

10개 확률의 합은1이다. Q∈R^(10×9),Q^TQ=I,Q^T1=0인 공개 Helmert 기저를 사용하면

`q = (1/10)1 + Q(Q^T q)`.

Client가 Q^Tq_i를 송신하고 서버가 (1/10)1+Q y를 복원한다. 이는 무잡음에서 손실 없는 좌표 변환이며, 클래스 하나의 정보를 버린 것이 아니다. 알려진 공통 성분을 전송하지 않아 payload와 RMS/peak가 바뀐다. 잡음이 있는 복원은 simplex 투영 후 KD target으로 사용한다.

초기 FP32 구현은 미세한 연산 순서 차이로 무잡음 target도 달랐다. 공통 FP64 정규화·송수신으로 수정했고, Original/V1 모두3seed에서 무잡음 target과600step 후 모델이 reference와 정확히 같았던 기록6개를 검증했다. 이번에는 저장 teacher에서 target을 다시 계산해6개 모두 max error=0을 재확인했다. 이 조건의 수치 동치이며 모든 장치의 bitwise 재현성을 약속하는 것은 아니다.

**결과와 현재 선택**

FashionMNIST private12000,10clients,ConvNet2,seed371/372/373,noise 각4회,20dB,평균 power1/peak4다. Reference도 동일 독립 teacher-KD 알고리즘이다. 정규화 오차는 target-free KD의 forget prediction JS / 미삭제 student의 JS다.

| 방식 | 전송반복 | 삭제오차 | 총 real uses |
|---|---:|---:|---:|
|Original,10좌표|1|**.668330**|731,755|
|V1,9좌표|1|.755425|729,505|
|Original,10좌표|4|**.595757**|799,255|
|V1,9좌표|4|.877398|790,255|

1회에서 teacher 확률 MSE는3.795e-5→3.361e-5로11.4% 줄었지만, 최종 student의 주 삭제 오차는 줄지 않았다. 1회 평균 절대 JS는 V1이 소폭 낮아 어떤 지표에서도 항상 나쁘다고 말하지는 않는다. 주 평가지표인 seed별 정규화 JS,4회 조건,일관성을 함께 보면 V1을 채택할 근거가 없다.

좌표 수는10% 줄었지만 전체 모델 DL을 포함한 총비용 절감은1회0.31%,4회1.13%다. Student 학습 비용도 그대로다. Teacher 준비는 client당32000 gradient-samples,student는38400 samples/회다. Query가 cache돼 있지 않으면 최초 이미지 배포 비용12544000bit도 필요하다. 기존 Original의 test accuracy는71.97%,target-free KD reference는72.02%이며, 별도 FedAvg reference75.29%와 같은 품질이라고 주장하지 않는다.

**정보 공개와 차분 복원**

서버가 동일 query·동일 teacher 상태의 정확한 q_all과 q_R를 모두 안다면

`q_A = [q_all-(1-alpha)q_R]/alpha`.

따라서 예측을 공중 평균으로만 받아도 삭제 전후 기록을 조합하면 A의 teacher 예측을 복원할 수 있다. 이번 checkpoint 검증의 최대오차는3.34e-15 미만이었다. 실제 무선 잡음은 복원을 부정확하게 하지만 누출이 없다는 보장은 아니다. 이는 전체 parameter gradient 공개와는 다른 부분 정보 누출이며, ‘개별 teacher 출력도 전혀 알 수 없다’고 쓰면 안 된다.

배포된 새 student에서 A의 인과적 영향을 없애는 것과, 과거 통신 기록에서 A의 정보를 숨기는 것도 별개의 조건이다. 새 student를 만들어도 이미 관측한 과거 출력이 자동으로 삭제되지는 않는다.

**선행연구와 현재 판정**

[Over-the-Air Federated Distillation, VTC2024](https://ieeevtc.org/vtc2024spring/DATA/PID2024002319.pdf)은 soft prediction 공중 합산과 전력·수신기 공동 설계를 다룬다. Class별 평균 prediction과 global feedback으로 local model을 학습하는 구조여서, 우리의 동일 public query·독립 teacher 조건과 구분한다. [FedQUIT](https://arxiv.org/html/2408.07587v1)은 기존 모델로부터 quasi-competent teacher를 만들어 on-device unlearning을 수행한다. 동일한 독립 teacher 삭제 알고리즘은 아니다.

**독립 teacher 기반 Original은 유지하고,9좌표 V1은 보류한다. Soft prediction OTA 자체나 독립 teacher 제외만으로 AirComp 고유 신규성을 확정하지 않는다.**

[수치수정기록](../research_20260926_versioned/RTD_NUMERICAL_ADDENDUM_KO.md) · [실제송수신코드](../research_20260926_versioned/rtd_fp64.py) · [전체결과](../research_20260926_versioned/RESULTS_KO.md)
