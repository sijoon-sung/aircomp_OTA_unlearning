**① DS-Air — 공통 곡률로 송신 전에 보정하는 삭제신호 공중 합산**

고정 feature의 convex head에서 정확히 정의되는 삭제 보정을 무선으로 전달하는 방법이다. 현재 검증한 V1은 A의 마지막 참여를 요구하지 않는다.

**문제와 성립 조건**

공개 무작위 encoder z=phi(x)는 private data로 학습하지 않는다. 각 client i의 ridge 목적함수는 F_i(W)=||Z_iW-Y_i||_F²/(2n_i)+lambda||W||_F²/2다. H_i=Z_i^T Z_i/n_i+lambda I, C_i=Z_i^T Y_i/n_i, gradient b_i(W)=H_iW-C_i다. 삭제 후 정규화한 가중치를 r_i=n_i/sum_{j≠A}n_j로 둔다.

H_R=sum r_i H_i, C_R=sum r_i C_i, b_R=H_R W_0-C_R라 하면

`v* = H_R^{-1} b_R`, `W_R = W_0-v*`.

이는 H_R가 positive definite인 이차 목적함수에서는 임의 W_0에 대해 정확한 항등식이다. W_0의 stationarity는 삭제 항등식 자체에는 필요하지 않으며,뒤의 A-gradient 역산 관계에 필요하다. Encoder 자체가 A의 데이터로 학습된경우 head만 바꿔 전체모델에서 A를 삭제했다는 결론은 성립하지 않는다.

**실제 프로토콜: Original과V1**

1. 서버가 W_0, 잔존 참가 집합, 가중치와 공통 설정을 방송한다. A는 삭제 요청 후 데이터 계산·송신에 참여하지 않아도 된다.
2. 각 잔존 client가 H_i 및 b_i를 로컬에서 계산한다. H_i의 upper triangle을 동시 송신해서 서버가 noisy H_R를 얻는다. 코드에서는 대칭 복원 후 eigenvalue를 lambda 이상으로 제한한다.
3. 서버는 같은 추정 H_hat=Q Lambda Q^T를 두 비교군에 사용한다. 공통 Q와 전송 블록 정보를 방송한다. V1에는 Lambda도 방송한다.
4. Original: client가 Q^T b_i를 가중 동시 송신한다. 서버가 수신후 Lambda^{-1}을 적용하고 Q로 복원한다.
5. V1: client가 **같은** Lambda^{-1}Q^T b_i를 가중 동시 송신한다. 서버는 수신값을Q로 복원한다.
6. 서버가 W_U=W_0-v_hat를 계산하고 배포한다. 삭제 후 recovery는 하지 않는다.

```text
잔존clients → H_i의 가중 공중합 → 서버의 공통 Q,Lambda
서버 → Q,Lambda,반복 배정 → 잔존clients
잔존clients → r_i Lambda^{-1}Q^T b_i 동시송신 → 서버
서버 → W_0-Q·수신값
```

모든client가 공통 inverse를 사용하므로 sum r_i P b_i=P sum r_i b_i가 성립한다. **각자 H_i^{-1}b_i를 보내는 방법과 다르다.** 이번 작은 검증에서 local inverse 평균과 공통 inverse 적용값의 norm차는.6673이었다.

**AirComp에서 해결하려는 문제**

두 방법의 무잡음 계산은 같다. Hessian 추정을 고정하고 eigen 좌표의 수신잡음을 n이라 하면 Original의 보정잡음은 Q Lambda^{-1}n이고,V1은 Q n'이다. V1에서는 작은 eigenvalue로 잡음을 수신후 증폭하는 부분이 사라진다. 그러나 송신payload가 커져 전력 정규화·peak 제한으로 n'의 크기도 바뀐다. 따라서 V1이 항상 더좋다는 일반정리는 없다.

각block의 삭제보정 오차에 대한 기여로 반복수를 배정한다. Original의 비용함수는 eigenvalue 역수 제곱까지 반영하며,V1의 실제전송좌표는 이미 보정된 상태다. 구현은 정해진 총 반복예산 안에서 정수 반복수를 배정한다. 비교에는 곡률수집,basis/eigenvalue방송,pilot,metadata,source/결과모델DL을 포함한다.

HVP를 이용한다면 H_R z=sum r_iH_i z가 가능하지만,각query의 z방송·local HVP·OTA수신이 필요하다. **현재 성능표는dense65차원Hessian 기반이며 HVP one-shot 실험이 아니다.**

**전체 관측에서의 정보 문제**

전체 목적함수 F=alpha F_A+(1-alpha)F_R의 정확한 정상점 W_0라면 alpha g_A+(1-alpha)b_R=0이다. 서버가 full-rank 공통 P와 보정 v=P b_R를 정확히 알면

`g_A = -(1-alpha)/alpha * P^{-1} v`.

즉, A가 삭제 라운드에 참여하지 않아도 A의 현재 전체 gradient가 드러날 수 있다. P가 H_R^{-1}이면 P^{-1}=H_R다. 무선 잡음은 복원 오차를 만들지만 DP 보장이 아니다. 이 역산은 실제 작은 문제에서 최대오차2.36e-15로 확인했다. 그래서 full V1은 성능 비교안이며 핵심 정보 제약을 만족하는 배포안으로 판정하지 않는다.

**고정48차원 변형**

공개 seed로 정한 U∈R^(65×48)에 대해 client는 U^T H_iU와 U^Tb_i만 전달한다. 서버는 해당 공간에서 공통 inverse를 만들고 최종 보정을 U로 lift한다. 이 protocol에는 full H_R나 추가 독립 기저 query를 함께 주지 않는다. 손실은 retained full optimum에 대해 측정하므로 projection bias를 숨기지 않는다.

이번 추가 검증에서는 target과 다른 client의 선형 통계를 U의 null 방향으로 반대 변경해, source와 모든 projected payload가 같지만 target gradient는 달라지는 작은 quadratic 예를 확인했다. 특정 full-vector 선형 복원 경로에 대한 비식별성 예제일 뿐, 실제 분류 데이터의 전체 transcript·부가정보·DP를 증명한 것은 아니다.

**고정한 실험과 수치**

FashionMNIST10000개,10clients,client0 삭제, 공개 random ReLU784→64+bias, head65×10, lambda=.01. 새 seed202609263/264/265, 각16개 channel noise,20dB,평균 power1/peak4다. H의 upper triangle2145좌표를32회 보내며, 보정은5개 block의 총40회 반복 예산을 사용한다. 이 head 실험의 채널은 h_i=1로 고정했다.

| 방법 | parameter 잔여제곱오차 / 초기삭제격차 | 총 real uses |
|---|---:|---:|
|Original|.369047|143,940|
|Original,추가통신예산허용|.353435|144,525|
|V1|**.210444**|144,643|
|Gradient-MSE 배정|.383222|143,940|
|V1 고정48차원|.367611|88,173|

V1은 Original 대비43.0%, 비용 추가 허용 Original 대비40.5% 낮은 오차다.48차원은 Original과 비슷한 오차에서 비용38.7% 절감이나, 공개 회전도 도입되어 rank 감소만의 기여라고 말하지 않는다. 이전.0202→.0073은 평균 전력만 제한한 조건이다. Peak 제약 주표와 혼합하지 않는다.

**선행연구와 주장할 수 있는 범위**

[OTA Fed-Sophia](https://arxiv.org/html/2410.07662v1)는 대각 곡률·gradient 집계와 서버 scaling/clipping을 사용한다. [COTAF](https://arxiv.org/abs/2009.12787)는 송신 전처리와 수신 scaling으로 OTA 학습 잡음을 다룬다. Newton/AirComp나 전처리 자체를 신규성으로 말하지 않는다. 현재 확인한 차이는 삭제 residual에 대해 공통 inverse의 위치·전력 제약·추가 방송 비용이 최종 삭제 오차에 미치는 영향이다. 신규성 확정에는 가까운 방법과 동일 정보·전체 비용 조건의 비교가 남는다.

**현재 판정: 작은 convex-head에서 통신 개선 근거 유지. Full 방식의 privacy 조건은 미충족이며,48차원은 별도 제한 조건 후보다.**

[기존실험보고서](../research_20260926_versioned/RESULTS_KO.md) · [구현](../research_20260926_versioned/benchmark.py#L121) · [새수학검증](validation/verification.json)
