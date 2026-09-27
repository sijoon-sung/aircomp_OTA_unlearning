**이름·버전·선행연구·변경 범위**

이번 이름은 프로젝트 안에서 버전을 구별하기 위한 작업명이다. 명칭이나 구성요소의 신규성을 주장하지 않는다. `Original`은 직전 연구에서 사용한 우리 기준 구현을 뜻한다. `V1`은 이번 수정이며, 예전V1–V21의V1은 `Legacy-Sketch-V1`이다.

**DS-Air — Deletion-Signal AirComp / 삭제신호 공중 합산**

Original: 고정-feature ridge의 잔존 Hessian과 residual을 수신하고 서버가 역곡률을 적용한다. V1: 같은 공통 역곡률 좌표 보정을 client가 적용하고 서버는 합산된 보정량을 받는다. 이전 단계에서 성공한 convex-head와 목적함수를 유지한다. CuReNUS 전체CNN solver로 다시 바꾸지 않는다.

해결할 문제는 aggregate 후 역곡률이 무선 잡음을 증폭하는 것이다. 무잡음 연산은 동일하므로 통신 비교 안에서 solver를 바꾸지 않는다. 송신 전력과 추가 곡률 방송 비용을 다시 계산한다. Main은 per-real-symbol 평균전력≤1,peak≤4다. 단순 HVP 합산 또는 Newton update 자체를 새 기여라고 하지 않는다.

가까운 [OTA Fed-Sophia, §III](https://arxiv.org/html/2410.07662v1)는 gradient/diagonal-curvature 집계와 서버의 scaling·clipping을 다룬다. [COTAF](https://arxiv.org/abs/2009.12787)는 OTA FL의 client precoding/server scaling을 다룬다. 이번에 검사하는 차이는 삭제 시점의 보정 residual, 공통 inverse의 적용 위치, 최종 삭제오차와 총비용이다. 기존 Newton 통신 연구와의 충분한 차별성이 확정된 것은 아니다.

정보 조건: full V1은 stationarity·H_R·보정량을 함께 알면 target gradient 역산이 가능한 한계가 있다. `V1-subspace48`은 같은 고정된 public48차원에서만 Hessian과 residual을 관측하는 별도 protocol이다. 비교 실험의 full/subspace transcript를 한 서버에 동시에 공개하는 것으로 해석하지 않는다. Rank 제한은 이 특정 full-vector 선형 복원 경로를 제한하지만 DP·부가정보 공격 안전을 증명하지 않는다. Subspace approximation error는 full retained optimum에 대해 평가한다.

**OG-Air — Orthogonal-Gate AirComp / 직교 모델의 gate 공중 보정**

Original: [FedOrtho](https://arxiv.org/html/2506.19891v2)의 kernel 정규화·activation 통계·soft pruning을 축소 CNN에 옮긴 구현이다. 공식 FedOrtho 저장소를 확보한 것은 아니며 논문 수식 재구현이다. Kernel regularizer는 검토·고정한 Orthogonal CNN 공개 코드의 실제 함수를 사용한다.

Original의 평균 activation 차이는 client별 class 비율 차이를 반영할 수 있고, 고정25%선택 및50–75%감쇠는 큰 개입이다. V1은 동일한 source 모델에서24개 channel gate의 gradient를 구해, client 제외에 따른 gradient 변화 α(g_A−g_R)를 OTA로 합산한다. 기존 regularization 강도는 유지한다. 즉 이번에 예방적 feature 독립성까지 새로 달성했다고 하지 않는다.

V1에서 γ=1+clip(.1d,−.05,.05). 이는 같은 위치에서 all-data/retained 한 gradient step의 차이에 근거한 작은 근사이며 역사적 학습 경로 전체 삭제 정리가 아니다. 결과가 no-op에 가까울 뿐이면 성공으로 분류하지 않는다. Client는 gate gradient24개만 보내고 original full parameter gradient63,562개와 구별한다. 한 번의 fixed-coordinate 측정만 사용한다.

비교용 [FedOSD](https://arxiv.org/html/2412.20200v1)는 UCE와 retained-gradient span에 대한 projection을 사용한다. 기존 검증된 `relation_protocol.py`의 UCE/SVD projection을 재사용한다. 저자 MIT code commit `41cc10635c6f2396795d0d9c6239ea181b7c9ee6`의 방향 정의에 대응하지만, 이번 `FedOSD-core10`은 local-gradient10 steps, 별도 recovery0인 adaptation이다. 원 논문 전체 성능을 반박하거나 재현했다고 하지 않는다. Exact full-gradient 비교군은 우리의 서버 정보 제한을 만족하지 않는 oracle baseline이다.

`Legacy-Sketch-V1`은 같은 UCE/projection·step 수에서 SRHT2048·16bit sketch로 계수를 계산하고 weighted OTA를 쓴다. 목표와 전송을 함께 바꾸지 않는다. 잡음 때문에 정확한 FedOSD보다 reference에 우연히 가까워질 수 있으므로, 그러한 경우를 더 정확한 projection이라고 해석하지 않는다.

**RTD-Air — Retained-Teacher Distillation over AirComp / 잔존 teacher 공중 증류**

Original은 앞서 조건부 성립을 확인한 독립 teacher 알고리즘이다. Teacher T_i는 자신의 data와 공개 초기값으로만 학습하며 global feedback이 없다. ClientA를 제외한 teacher와 fresh student를 사용한다. 비교 reference도 동일한 teacher/student 알고리즘을 처음부터 A 없이 실행한 경우다. 일반 FedAvg reference와 같다고 하지 않는다.

V1은 q∈simplex10의 알려진 공통 성분1/10을 제거한다. Helmert Q∈R^(10×9),QᵀQ=I,Qᵀ1=0를 모든 client가 공개 규칙으로 만든다. Client는Qᵀq,서버는1/10+QΣp_iQᵀq를 계산한다. Packet당10%차원 축소와 실제 송신 RMS/peak 변화가 생긴다. 동일2000query와 동일KD schedule이므로 손실함수는 바뀌지 않는다. 별도1000query변형은 approximation/비용 trade-off로 기록한다.

이는 알려진 선형좌표변환과 simplex 구조의 사용이다. AirComp 이전에 공통 성분을 보내지 않아 전력·noise가 달라지는지 비교하는 것이 이번 실험의 질문이다. 이 조합만으로 독립적인 신기술 주장을 하지 않는다. 확률이나 logits OTA·federated distillation 선행연구와는 겹치며, target-free teacher 조건과 삭제 시 비용에 초점을 맞춘다.

[FedQUIT](https://arxiv.org/html/2408.07587v1)는 기존 global teacher를 quasi-competent teacher로 바꿔 on-device unlearning을 수행하는 접근이다. 이번 외부 비교는 저자 GitHub commit `e447fa0c75c6a44b1325be5fd561da98c91f4cc2`의 logit-min 변환과 KL을 PyTorch로 옮긴 코드에 retained CE를 섞은 adaptation이다. TF 전체 pipeline 실행이 아니다. 별도의 plain FedAvg source/reference 표에 배치한다.

확률 aggregate의 삭제 전후 차분으로 target teacher prediction을 추론할 수 있다는 부분 정보 공개는 남는다. 이것은 전체 parameter gradient 공개와 다른 조건이며, DP/암호학적 비공개를 주장하지 않는다.

**재사용한 코드와 기록**

- Model: CuReNU ICLR2026 official supplementary ConvNet2. 이번①에서는 cubic solver를 사용하지 않는다.
- Kernel orthogonality: Orthogonal-Convolutional-Neural-Networks commit `aa7f56901c661a124e0cfe72eb2c9dc98045ce94`의 `orth_dist`.
- UCE/projection/sketch: 기존 read-only `relation_protocol.py`.
- Dataset split/training/evaluation 및 FedQUIT port: `research_20260926_methods123/common.py`, `vendor_adapters.py`에서 재사용.
- 신규 변경과 실행: 이 폴더 `benchmark.py`. 실행 코드와 protocol SHA256은 `results/environment.json`에 기록한다.

모든 기존 raw 결과와 체크포인트는 보존했다. 이번에는 새 seed를 사용하고 기준을 평가 후 바꾸지 않는다. 코드의 simulation process에 local data가 존재하는 것과 통신 protocol의 서버가 그 데이터를 수신하는 것을 구별한다.
