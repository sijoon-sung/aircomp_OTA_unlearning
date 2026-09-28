# OTA-FL 문제 재선정: 분할 집계의 데이터 가중치와 무선 간섭

검토일: 2026-09-28. 범위: **OTA-FL 전반, SFL 별도, 언러닝 필수 조건 제외**.

## 현재 판단

우선 검증할 문제는 **채널에 유리하게 구성한 그룹이 모델 조각을 나누어 보낼 때 생기는 집계 편향과 표본 분산**이다. 가장 가까운 출발 논문은 [SegOTA, WiOpt 2025](https://arxiv.org/abs/2504.09745)다. 해결 후보는 그룹 구성·보정 계수·수신 빔을 함께 선택해 이 오차와 간섭/잡음의 균형을 잡는 작은 설계다.

**아직 새로운 알고리즘의 우월성을 확인한 것은 아니다.** 이번 검산에서 단순 가중치 보정은 분할 집계 내부에서는 개선됐지만, 같은 payload 시간·에너지의 일반 전체 모델 LMMSE 집계를 이기지 못했다. 따라서 이 보정에 이름만 붙여 논문 후보로 채택하지 않는다. 아래는 문제의 존재를 확인한 단계와 다음 실험의 채택 조건을 구분한 기록이다.

이전 CR-Air/DS-Air 결과는 다른 연구 조건의 기록으로 보존한다. 이번 후보를 언러닝의 해결책으로 설명하지 않는다.

## 선행연구로 이미 닫힌 주장

| 주장 | 가까운 선행 | 이번 판정 |
|---|---|---|
|무선 잡음을 학습에 유용하게 활용|[COTAF](https://arxiv.org/abs/2009.12787)|점차 잡음을 줄이는 precoding과 학습 효과가 이미 있음|
|전력으로 편향과 MSE를 절충|[Cao 등의 Air-FedAvg](https://arxiv.org/abs/2111.05719)|일반적 trade-off 자체를 신규성으로 주장할 수 없음|
|반복 전송 / pilot 재측정|[AirReComp](https://arxiv.org/abs/2111.10267), [Imperfect-CSI AirComp](https://cris.fau.de/publications/324822441/)|자원 비용과 정확도 절충까지 선행에 있음|
|CSI 없는 집계 + error feedback|[NCAirFL 후속 논문, 2026-09](https://arxiv.org/abs/2609.08312)|그 결합 자체는 새 후보에서 제외|
|채널과 데이터 분포를 함께 고려|[Energy–robustness](https://arxiv.org/abs/2312.14638), [Air-FedGA](https://qianpiao.github.io/files/Asynchronous_Federated_Learning_Over_Non-IID_Data_via_Over-the-Air_Computation.pdf)|단순 중요도 sampling/데이터 기반 clustering도 신규성 부족|

본문에서 확인한 근거와 읽기 범위는 [문헌 기록](LITERATURE_KO.md)에 있다. 검색에서 동일 제목을 찾지 못한 것을 신규성 증명으로 사용하지 않았다.

## 1. 구체적인 문제가 무엇인가

SegOTA는 서로 다른 그룹이 서로 다른 모델 조각을 동시에 전송한다. Eq. (4)–(5)의 원하는 조각은 그룹 내부의 유효 채널 계수 합으로 정규화된다. 원문에는 random/round-robin 조각 배정이 모두 있고, Assumption 2에 gradient divergence가 있다. 따라서 단순 회전 배정이나 ‘non-IID가 전혀 고려되지 않았다’는 주장은 부정확하다. [원문](https://arxiv.org/pdf/2504.09745)

다음은 이 구조의 **간섭 없는 특수 경우**에 대한 자체 유도다. 그룹과 계수가 고정되고, 그룹–조각 배정을 균등 random permutation으로 정한다. 클라이언트 $k$의 그룹을 $g(k)$, 그룹 내 계수를 $r_k\ge0$, 그룹 수를 $S$라 하자. 각 그룹에서 $\sum_{k\in G_g}r_k=1$이다. 해당 조각의 local innovation을 $u_{k,m}$이라 하면

$$
\hat u_m=\sum_{k\in G_{\pi^{-1}(m)}}r_k u_{k,m}+n_m,
\qquad
\mathbb E_\pi[\hat u_m]=\sum_k\frac{r_k}{S}u_{k,m}.
$$

목표가 FedAvg의 $\sum_k p_k u_{k,m}$이면 실제 기대 가중치 $q_k=r_k/S$가 $p_k$와 달라질 수 있다. 그룹 내 equal gain일 때 $q_k=1/(S|G_{g(k)}|)$다. 즉, 동일 크기가 아닌 그룹을 똑같이 조각에 배정하면 작은 그룹의 한 client가 더 크게 반영된다.

**주의:** 실제 원문의 계수는 segment norm과 beam/power에도 의존한다. 위 식을 원문 전체에 무조건 대입하면 안 된다. 배정과 계수의 독립성이 없으면 배정별 실제 계수를 함께 평균해야 한다. 이는 원 논문 오류 주장이 아니라, 허용된 구조에서 발생할 수 있는 오차의 예다.

예를 들어 $K=4,S=2,G_1=\{1\},G_2=\{2,3,4\}$이면

$$
p=(1/4,1/4,1/4,1/4),\quad q=(1/2,1/6,1/6,1/6).
$$

Local quadratic $f_k(\theta)=\frac12(\theta-c_k)^2$, $c=(1,-1,-1,-1)$에서 목표 최적점은 $-0.5$, 이 가중치의 평균 drift 고정점은 $0$이다. 그 점의 원래 목적함수 초과손실은 $0.125$다. 이는 noisy SGD의 거의 확실한 수렴 증명이 아니다.

## 2. 단순 보정의 가능성과 한계

첫 진단용으로

$$
v_k(\lambda)=(1-\lambda)r_k+\lambda S p_k,\quad 0\le\lambda\le1
$$

를 원하는 조각의 수신 계수로 만든다. 앞의 독립성/간섭 없음 조건에서는

$$
q_k(\lambda)=(1-\lambda)q_k+\lambda p_k.
$$

따라서 $\lambda=1$에서 **기대 own-segment 계수**는 맞는다. 이것은 알려진 중요도/가중치 보정 원리이며 새 이론이라고 주장하지 않는다. Full model을 직접 배수하는 대신 $\theta_{t+1}=\theta_t+\hat u_t$ 형태의 innovation에 적용해야 한다.

AirComp에서는 계수를 공짜로 바꿀 수 있는 경우와 없는 경우를 분리해야 한다. 그룹 수신 빔 $w_g$, 채널 $h_k$, client innovation bound $C$, 송신 symbol peak $P_k$ 아래 한 가지 실현은

$$
b_k=\frac{\gamma_g v_k}{w_g^Hh_k},\qquad
0<\gamma_g\le\min_{k\in G_g}\frac{\sqrt{P_k}|w_g^Hh_k|}{C v_k}.
$$

양의 $v_k$, 0이 아닌 유효 채널을 전제로 한다. Deep fade에서 큰 보정은 $\gamma_g$를 작게 만들어 수신 잡음 $\sigma^2\|w_g\|^2/\gamma_g^2$를 키울 수 있다. 반대로 유리한 경우도 있어 항상 잡음이 커진다고 말할 수는 없다.

더 중요한 점은 다른 그룹 $j$에서 들어오는 계수가

$$
\frac{\gamma_j}{\gamma_g}v_k
\frac{w_g^Hh_k}{w_j^Hh_k},\quad k\in G_j
$$

라는 것이다. **자기 조각의 가중치를 맞춰도 타 조각 간섭이 사라지지 않는다.** 이번 counterexample에서도 $\lambda=1$이지만 작은 교차 채널이 있으면 목표 $-0.5$에 대해 수신 평균은 $-0.39$였다.

## 3. 이번에 실제로 계산한 수치

아래는 논문 재현이나 GPU/CNN 실험이 아닌 **유리수 연산으로 계산한 작은 통신 추정 예제**다. Frozen innovation 2차원, K4, 2개의 수신 안테나, 그룹별 직교 채널 방향, gain 1과 0.2, 실제 수신 잡음 분산 0.1, 모든 client symbol 전력 1이다. 제어·CSI·downlink 비용은 제외했다. 코드는 [audit_segmented_weights.py](audit_segmented_weights.py), 원장은 [math_audit.json](math_audit.json)이다.

|방식|shared real payload uses|합계 TX 에너지|좌표당 MSE|
|---|---:|---:|---:|
|보정 없는 분할, random 1회|1|4|1.438889|
|보정 없는 분할, balanced 2회|2|8|0.344444|
|완전 가중치 보정, balanced 2회|2|8|0.162500|
|이 예제의 최적 부분 보정, balanced 2회 **oracle**|2|8|0.154822|
|전체 모델, unbiased OTA|2|8|0.162500|
|전체 모델, 통상 LMMSE OTA|2|8|**0.152893**|

Balanced 2회는 **동일한 frozen update**를 서로 다른 그룹–조각 배정으로 보내 평균한 것이다. 두 번 local training한 효과나 최신 모델을 사용한 round-robin 학습 결과가 아니다. Oracle은 실제 gradient를 알아야 고를 수 있는 진단용 값이며 제안 scheduler 성능으로 보고하지 않는다. LMMSE는 unit independent source covariance로 설계한 일반 수신기를 같은 고정 correlated update에 평가했다.

계수 보정·잡음·표본 분산·에너지의 식은 일치했고 5개 검산을 통과했다. **분할 방식 내부의 개선만으로 우수한 통신 설계라는 결론을 내리면 안 된다는 결과**이기도 하다. 이 하나의 예제로 모든 분할 방식이 나쁘다고 결론내릴 수도 없다.

재현: `python research_20260928_ota_problem_review/audit_segmented_weights.py` (표준 라이브러리만 사용).

## 4. 남겨 둘 해결 후보 하나

**작업 제목: 데이터 가중치 오차를 제한하는 분할 AirComp 그룹·수신 설계.** 확정 알고리즘 명칭은 부여하지 않는다.

단순 $\lambda$ 보정에서 멈추지 않고, 채널 기반 그룹을 초기값으로 사용해 client 이동/교환과 빔/scale 후보를 적은 횟수 평가한다. 목적은 원하는 조각의 가중치와 다른 조각에서 새는 간섭을 **같은 집계 연산자**로 평가하는 것이다. Local update 원문을 서버에 올리지 않고 CSI, 데이터 수 $p_k$, 공개 clipping bound만으로 후보를 평가한다. 이 정보 제한 자체는 DP 보장이 아니다.

수학적 설계안: segment 길이를 $d$ complex coordinates라 하고, client–segment innovation을 $U\in\mathbb C^{KS\times d}$로 쌓는다. $\|U\|_F^2\le KC^2$. 배정 $\pi$마다 실제 OTA 행렬 $A_\pi\in\mathbb C^{S\times KS}$에 자기 조각과 타 조각 계수를 모두 넣고, 원하는 FedAvg 행렬을 $P$라 한다. 잡음과 update가 독립이고 CSI가 정확하면

$$
\mathbb E\|\hat U-PU\|_F^2
=\operatorname{tr}(U^H Q U)
 +d\sigma^2\sum_g\frac{\|w_g\|^2}{\gamma_g^2},\quad
Q=\mathbb E_\pi[(A_\pi-P)^H(A_\pi-P)],
$$

$$
\mathbb E\|\hat U-PU\|_F^2
\le KC^2\lambda_{\max}(Q)
 +d\sigma^2\sum_g\frac{\|w_g\|^2}{\gamma_g^2}.
$$

이는 표준 행렬 부등식으로 얻은 **1회 집계의 보수적 상한**이다. FL 수렴 정리나 새로운 일반 minimax 정리가 아니다. 설계가 update 값에 의존하지 않아야 위 조건부 기대 해석을 그대로 사용할 수 있다. $S\le4$부터 permutation을 열거하고 작은 $KS\times KS$ 행렬로 후보를 비교할 수 있다. 그룹 이동을 실제로 적용했을 때의 제어비와 서버 계산시간도 기록해야 한다.

통신 기여 후보는 **같은 $D/S$ payload 구조에서, 그룹을 바꿔 얻는 학습 대표성과 잃는 공간 분리 성능을 함께 조절하는 방법**이다. 디지털 통신에는 해당 동시 전송의 타 조각 leakage와 beam/power 제약이 그대로 존재하지 않는다. 다만 robust beamforming·중요도 보정·그룹화 각각은 기존 도구다. 이 도구의 결합만으로 신규성이 확정되지 않는다.

## 5. 논문 후보로 남기는 조건

국내 학회 목표로도 **현재는 문제/설계 후보 단계**다. 다음을 확인해야 작은 방법론 기여로 제시할 근거가 생긴다.

- 원본 SegOTA, 단순 mass 보정, 데이터 균형 grouping, 일반 weighted-MSE 설계의 분할 적용보다 동일 비용에서 유리할 것.
- 전체 모델 OTA와도 target accuracy까지의 총 비용에서 비교할 것. 조각 수만큼 빨라졌다고 계산하지 않을 것.
- 데이터와 채널의 상관이 있을 때뿐 아니라 독립일 때도 검증하고, 제안 방법이 불리한 영역을 보여 줄 것.
- $\lambda_{\max}(Q)$ 목적이 기존 weighted-MSE를 그대로 다시 쓴 것인지 대조할 것. 차이가 없다면 새 알고리즘 주장을 중단할 것.
- 전체 overhead를 넣으면 이득이 사라지거나, 원본에 간단한 rescale만 해도 같은 결과이면 확장 실험을 중단할 것.

**주장할 trade-off:** ‘잡음과 정확도’라는 일반론이 아니라 **동시 전송 조각 수/전력/제어비와, 그룹 편향·표본 분산·타 조각 간섭 때문에 추가로 필요한 학습 라운드 사이의 관계**다. 이번에는 이를 입증하는 CNN 결과가 없으므로 개선율을 제시하지 않는다.

작은 실험의 고정 비교군·종료 조건은 [실험 계획](EXPERIMENT_PLAN_KO.md), 별도 SFL 검토는 [SFL 비교](SFL_SEPARATE_KO.md)에 있다.
