**공중 잡음을 어떻게 활용할 것인가 — NM-Air와 JS-Air**

2026-09-27. 검증 범위는 공개 고정 feature 위 convex ridge head다. NM-Air는 필요한 출력 무작위성을 공중 잡음으로 충당하는 방법이고, JS-Air는 수신 잡음의 분산을 사용해 삭제 보정량을 추정하는 방법이다. 이름은 이번 실험 구분용이며, 기존 noise matching이나 James–Stein 기법 자체의 신규성을 주장하지 않는다.

**1. 먼저 성립하지 않는 주장**

삭제 보정 후 평균 모델을 μ, 정확한 retained 모델을 W_R, 추가 잡음을 ξ라 하자. 조건부 평균0, covariance Σ이면

$$\mathbb E\|\mu+\xi-W_R\|_F^2=\|\mu-W_R\|_F^2+\operatorname{tr}\Sigma.$$

따라서 잡음을 더해 평균이 잘못된 모델의 squared distance를 줄일 수 없다. A의 과거 영향이 B/C에 전파된 사실도 noise injection만으로 역산되지 않는다. 비선형 예측지표가 특정 실행에서 좋아져도 이 등식에 대한 반례는 아니다. Noise-only 비교군을 둔 이유다.

잡음의 역할은 두 가지로 좁힌다. 하나는 **사전에 정한 randomized learning의 목표 분산을 구현**하는 것이고, 다른 하나는 **알려진 잡음 분산으로 관측된 삭제 방향의 추정 오차를 줄이는 것**이다. 전자는 출력분포 목표를 바꾸므로 기존 결정론적 목표의 성공으로 바꿔 부르지 않는다.

**2. NM-Air의 학습 및 삭제 목표**

Client i의 ridge 통계와 retained 집계를 다음과 같이 둔다.

$$H_i=Z_i^TZ_i/n_i+\lambda I,\quad C_i=Z_i^TY_i/n_i,$$
$$H_R=\sum_{i\ne A}p_iH_i,\quad C_R=\sum_{i\ne A}p_iC_i,\quad W_R=H_R^{-1}C_R.$$

Encoder는 private data로 학습하지 않는다. 초기 학습과 target-free 재학습을 모두 다음 규칙으로 정의한다.

$$\mathcal A_\tau(D)=W_D^*+\tau Z,\qquad \operatorname{vec}(Z)\sim\mathcal N(0,I).$$

Tau는 실험 전에 공개 고정한다. A를 포함한 초기 W₀도 이 규칙으로 생성한다. 학습 때는 결정론적 모델을 쓰고 삭제 평가 때만 유리한 randomized reference를 끼워 넣는 구성이 아니다. 최종 배포는 양쪽에 같은 FP32 양자화를 적용한다.

임의의 W₀에 대해 b_i=H_iW₀−C_i라 하면

$$W_0-H_R^{-1}\sum_{i\ne A}p_ib_i=W_R.$$

이 등식은 초기 W₀가 정확한 정상점일 필요가 없다. 하지만 convex fixed-feature head의 등식이며 전체 신경망의 얽힘을 푸는 정리가 아니다.

**3. 공통 곡률, 전송 payload와 전력 조건**

서버는 잔존 clients의 H_i upper triangle 공중 합산으로 Ĥ를 얻는다. Eigenvalue를 λ=.01 이상으로 제한하고 공통 Q,Λ를 방송한다. 실제 코드에서는 Q를 FP32로 전달한 후 모든 노드가 같은 QR을 적용해 공통 직교 기저 Q̄를 만든다. Λ도 FP32 roundtrip이다. 정의한 공통 보정 연산은

$$P=\bar Q\Lambda^{-1}\bar Q^T,\qquad d_{ij}=\left[p_i\Lambda^{-1}\bar Q^Tb_i\right]_j.$$

Client별 inverse를 각각 쓰지 않는다. Block j는 m_j=130좌표이고 합계5개 block이다. 송신과 평균 수신을

$$x_{ij}=\frac{d_{ij}}{a_jh_i},\qquad
\bar y_j=a_j\frac1{r_j}\sum_{\ell=1}^{r_j}\left(\sum_i h_ix_{ij}+n_{j\ell}\right)
=\sum_i d_{ij}+\eta_j$$

로 두면 n~N(0,σ²I), independent AWGN 조건에서

$$\eta_j\sim\mathcal N(0,a_j^2\sigma^2I/r_j).$$

평균전력≤1,peak amplitude²≤4를 지키려면

$$a_j\ge a_{\min,j}=\max_i\left\{\frac{\operatorname{RMS}(d_{ij})}{|h_i|},
\frac{\|d_{ij}\|_\infty}{2|h_i|}\right\}.$$

실험은 h_i=1이며 수학 검사는 서로 다른 h_i의 공중 합산도 확인했다. 개별 scalar 전력 metadata는 서버에 알려진다. 개별 full gradient를 수신하는 구조는 아니지만 이를 DP로 해석하지 않는다.

**4. 최소 반복과 잡음 일치 규칙**

목표 분산을 τ²로 만들기 위한 최소 정수 반복은

$$r_j^*=\max\left(1,\left\lceil\frac{a_{\min,j}^2\sigma^2}{\tau^2}\right\rceil\right),
\qquad a_j^*=\frac{\tau\sqrt{r_j^*}}{\sigma}.$$

이 식으로 a_j*≥a_min,j가 되어 전력 조건을 만족하며 a_j*²σ²/r_j*=τ²다. 반대로 더 작은 r에서는 a≥a_min인 모든 feasible 송신의 잡음 분산이 τ²보다 커진다. 따라서 **고정 payload·block·목표 분산·전력 제약 안에서** 반복 수가 최소다. 전체 FL·언러닝 알고리즘에 대한 전역 최적성을 주장하지 않는다.

필요한 잡음보다 자연 잡음이 작은 block은 최대 전력으로 보내지 않는다. 송신 진폭을 낮추고 공통 수신scale을 키워 물리 잡음을 목표 분산으로 사용한다. 반복 수를 줄였다는 이유로 target-free 평균을 건너뛰지 않으며, synthetic server noise를 추가하지 않는다.

```text
잔존 clients → H_i upper triangle 가중 OTA 합산
서버 → 공통 Q, Lambda
잔존 clients → block별 scalar 전력 조건
서버 → tau에 맞춘 반복 r_j 및 scale a_j
잔존 clients → 전처리된 삭제 residual 동시 송신
서버 → W_U = W0 - Q·수신합, 최종 모델 배포
```

따라서 최종 보정 payload 단계의 반복을 줄이는 방법이지, Hessian 수집과 기저 방송을 없앤 one-shot 프로토콜이 아니다.

**5. 무엇이 정확하고 무엇이 남는가**

Ĥ와 초기 모델, metadata를 고정하면 NM-Air의 배포 전 출력은

$$\operatorname{vec}(W_U)\mid\mathcal H\sim\mathcal N(\operatorname{vec}(\mu_{\mathcal H}),\tau^2I),
\quad \mu_{\mathcal H}=W_0-P(H_RW_0-C_R).$$

공통 기저가 직교이므로 block별 같은 분산은 model 좌표에서도 isotropic이다. P=H_R⁻¹인 이상적 조건에서는 μ=W_R여서 randomized retraining과 정확히 같은 분포가 된다. P가 근사이면

$$\mu-W_R=(I-PH_R)(W_0-W_R).$$

**채널 잡음으로 covariance를 맞춰도 이 평균 오차는 남는다.** 같은 covariance의 Gaussian 간에는

$$\operatorname{KL}(\mathcal N(\mu,\tau^2I)\|\mathcal N(W_R,\tau^2I))=
\frac{\|\mu-W_R\|_F^2}{2\tau^2},$$
$$\operatorname{TV}=2\Phi\left(\frac{\|\mu-W_R\|_F}{2\tau}\right)-1.$$

위 KL/TV는 곡률 관측 등으로 조건을 고정한 값이다. Ĥ의 잡음을 평균내면 출력은 Gaussian mixture일 수 있다. KL의 convexity와 TV의 mixture bound로 **조건부 값의 평균은 주변 분포 거리의 상계**가 된다. 상계가 크다는 사실만으로 실제 주변 분포가 멀다고 단정할 수 없어, 별도 sampler 구분 검사를 수행했다.

DS-fixed40처럼 block별 분산이 v_j이면 조건부 KL은

$$\frac12\left[\frac{\|\mu-W_R\|_F^2}{\tau^2}+
\sum_j m_j\left(\frac{v_j}{\tau^2}-1-\log\frac{v_j}{\tau^2}\right)\right].$$

이를 이용해 평균 오차와 covariance mismatch를 분리했다. 유한 seed의 작은 KL, 혹은 개별 pair의 TV 수식이 전체 데이터에 대한 certified unlearning 정리를 대신하지 않는다. FP32 quantization은 같은 deterministic postprocessing이므로 divergence를 키우지 않지만, 부동소수점 난수의 분포를 이상적인 연속 Gaussian과 완전히 동일하다고 주장하지 않는다.

**6. 강한 비교군과 동일 분포 조건**

Aware-fixed40은 기존 greedy 반복40을 유지하되 부족한 block만 최소 반복까지 늘리고, 수신 분산 v_j가 τ²보다 작으면 software Gaussian N(0,(τ²−v_j)I)를 더한다. 이 비교군과 NM-Air는 **조건부 평균과 covariance가 같다**. NM-Air가 더 잘 삭제하는 것이 아니라 같은 분포를 더 적은 반복으로 만든다는 비교다. Monte Carlo accuracy의 작은 차이를 성능 개선으로 주장하지 않는다.

Uniform-match는 모든 block에 max_j r_j*를 배정한다. 이 방식도 같은 출력분포를 만들지만 비용이 더 들 수 있다. Naive-add40은 이미 있는 수신 잡음에 τ Gaussian을 그대로 추가해 목표보다 큰 covariance를 만드는 ablation이다. 이 약한 비교군만 이기고 알고리즘 성공이라고 판정하지 않는다.

**7. JS-Air: 결정론적 목표를 유지하는 보조안**

관측된 correction block y_j=δ_j+ξ_j, ξ_j~N(0,v_j I)에 대해

$$\widehat\delta_j=\left[1-\frac{(m_j-2)v_j}{\|y_j\|^2}\right]_+y_j$$

를 사용한다. 새 통신 없이 y_j와 수신 분산만 필요하다. m_j≥3인 unbiased Gaussian mean 문제에서는 고전적 James–Stein risk 개선을 이용할 수 있다. 비절단 추정기에 c=(m−2)v를 넣으면 raw 대비 risk 차이는

$$\{c^2-2c(m-2)v\}\,\mathbb E[1/\|y\|^2]
=-(m-2)^2v^2\mathbb E[1/\|y\|^2]$$

로 음수다. 코드에는 보편적으로 쓰는 positive-part 형태를 적용한다. **Noisy H에서는 관측 평균 자체가 true deletion correction과 다르므로 이 정리만으로 retained optimum에 대한 개선을 보장하지 않는다.** 그 부분은 실제 새 seed 실험으로 확인했다. Nonlinear shrinkage 후 분포는 Gaussian이 아니므로 앞 절의 Gaussian KL을 이 방법에 적용하지 않는다.

기존 GS/FedOSD와 달리 client 간 직교성을 강제하지 않는다. 공통 Q는 곡률 좌표계이며 client들을 CDMA/OFDMA로 분리하는 신호 직교화가 아니다. 다만 이번 JS-Air 적용도 일반 Gaussian 추정에 쓰이는 방법이므로 AirComp 자체의 독점적 신규성을 주장할 근거는 부족하다.

**8. 잡음 보정 오차와 추가 통신**

실제 분산이 설정값의 ρ배라면 정확한 평균을 알고 있어도 covariance KL은

$$\frac d2(\rho-1-\log\rho),\qquad d=650.$$

ρ=.8/1.2이면 각각7.5217/5.7455다. High dimension에서는 작은 분산 불일치도 중요하다. Main 실험은 정확한 σ²를 알고 있다는 가정이다. Scale/tau의 제어 전송비는32bit로 계산하지만 modem의 제어 패킷 오류나 scale 양자화 오차까지 RF로 검증하지 않았다.

추가로 무신호 슬롯 N개에서 σ̂²=mean(n²)를 추정하는 경우를 진단했다. 상대 표준편차는 sqrt(2/N), 비용은 CP×N real uses다. 이때 NM-Air의 실제 covariance 비율은 σ²/σ̂²이다. 두 분산 일치 방식 모두 같은 calibration이 필요하면 양쪽 비용에 더해야 한다. NM-Air에만 새 calibration이 필요하면 그 비대칭 비용도 숨기면 안 된다. 시간적으로 안정된 잡음 분산을 여러 삭제 요청에 재사용하는 경우에는 amortized cost를 별도로 적는다.

**9. 개인정보와 주장 범위**

분포 비교는 최종 모델에 한정된다. 서버가 보유한 source 모델, 곡률, power metadata와 모든 중간 수신, 여러 삭제 요청의 조합을 지우거나 숨기는 정리가 아니다. 정확한 source stationary 관계에서 full correction으로 A의 gradient를 역산하는 기존 문제는 그대로 남는다. 독립 관측을 반복하면 잡음을 평균낼 수 있으며, 수신기의 실제 noise floor나 도청자의 채널도 통제할 수 없다. 따라서 본 실험을 client-level DP 또는 server transcript의 certified forgetting이라고 표현하지 않는다.

**관련 원문과 차이**

- [Liu & Simeone, WFLMC](https://arxiv.org/pdf/2108.07644): 공중 잡음을 Langevin sampling/프라이버시에 사용하는 선행 구조다. 수신 잡음을 고려한 추가 Gaussian과 전력 설계가 이미 있어 noise reuse 자체는 새롭지 않다. 여기서는 한 번의 ridge deletion correction과 정수 반복 수, 초기·곡률 비용까지 포함한 효과를 검사했다.
- [Zhang et al., WFALD](https://arxiv.org/html/2305.04152v2): 여러 local update와 stochastic gradient를 지원하는 Bayesian OTA 학습이다. 목표는 posterior sampling이고 이번의 고정 τ output-perturbed ridge와 동일한 알고리즘은 아니다. 부족한 SNR의 잡음을 무조건 유익하다고 해석할 수 없다는 점은 공통이다.
- [Guo et al., Certified Data Removal](https://proceedings.mlr.press/v119/guo20c/guo20c.pdf): 선형 classifier의 removal approximation과 학습 시 perturbation을 연결한다. 이번 구현은 해당 코드/증명 재현이 아니라 quadratic head와 Gaussian 출력분포의 직접 계산이다. 해당 논문의 certified-removal 결론을 가져다 붙이지 않는다.
- [James & Stein, Estimation with Quadratic Loss](https://www.stat.yale.edu/~hz68/619/Stein-1961.pdf): Gaussian mean의 quadratic risk를 줄이는 고전적 shrinkage 근거다. JS-Air는 이 기법의 적용이다.
- [DNI, Information Fusion](https://www.sciencedirect.com/science/article/abs/pii/S1566253525008589): publisher 초록에서 입력 perturbation과 후속 healing을 확인했다. Thermal channel noise 재사용과는 다르지만, 본문 전체에 대한 분석이나 재현을 완료했다고 하지 않는다.
- [CAFU, 저자 소속기관의 출판 목록](https://research.unsw.edu.au/people/dr-sayed-amir-hoseini/publications?type=journalarticles), [DOI](https://doi.org/10.1109/LWC.2026.3730846): *Channel-Aware Federated Unlearning with Gradient Perturbation in Open Radio Access Networks*라는 가까운 논문의 존재를 1차 출처에서 확인했다. IEEE 원문은 확보하지 못해 세부 정리·통신 조건과의 차이를 확정하지 않았다. 이 중복 위험이 남아 있어 NM-Air의 신규성을 확정할 수 없다.

수식의 feasibility와 실험 성공은 별도다. 현재 주 설정은 분산 matching과 반복 절감은 확인했지만 사전 언러닝·비용 기준을 동시에 통과하지 못했다. [전체 수치](RESULT_TABLES_KO.md)와 [결론 및 다음 판단](README_KO.md)을 함께 읽는다.
