# CR-Air: 전체 관측 프라이버시를 제한하는 AirComp 통계 갱신 후보

2026-09-28. CR-Air(Correlated Refresh over AirComp)는 이 저장소에서 쓰는 **가칭**이다. 세 요구를 함께 다루기 위한 후보이며 신규성 또는 실용 성능이 확정된 알고리즘 이름이 아니다. [실행 전 범위](PROTOCOL_KO.md), [검산 코드](verify_theory.py), [원결과](results/verification.json).

현재 결론은 **학습 구조를 바꾸면 삭제와 전체 관측 DP를 동시에 증명할 수 있고, 그 제약 안에서 무선 송신 에너지를 최적화할 변수가 생긴다**는 것이다. 기존 private end-to-end 모델에 사후 적용하는 해법은 아니다. 작은 실제 데이터 실험까지 통과한 상태도 아니다.

| 사용자 조건 | 이 후보에서 가능한 주장 | 아직 못 하는 주장 |
|---|---|---|
| A의 학습 영향을 실제로 제거 | 고정 encoder + 충분통계 learner에서 최종 모델이 동일 noisy learner의 retained-only 재학습 분포와 일치 | 기존 deterministic 재학습과 같은 가중치, private backbone 삭제 |
| 서버 합산에서도 A의 개별 기여 보호 | 학습·삭제 전체 두 관측에 client-level replacement DP 상한 | 정보량 0, A 신원 은닉, 악의적 추가 수신·반복 질의 방어 |
| AirComp가 핵심인 기여 | 수신 잡음·출력분포·전력·삭제/잔존 privacy의 공동 제약으로 송신계수와 반복 수 설계 | 최초 아이디어라는 주장, 일반 FL에서 수학적으로 모사 불가능하다는 주장 |

## 1. 모델을 어디까지 바꾸는가

공개 데이터만으로 정한 고정 특징 추출기 phi를 쓴다. 각 client의 head 학습 통계를 예컨대

\[
v_i=[\operatorname{vech}(X_i^TX_i/n_i),\operatorname{vec}(X_i^TY_i/n_i)],\qquad
s_i=v_i\min(1,B/\|v_i\|_2)
\]

로 둔다. B, feature/label scaling, 학습 정규화 분모는 사전 공개 값이다. private sample count에 가중할 경우에도 전체 client payload를 clipping해야 한다. 실제 norm이나 clipping multiplier는 서버에 보내지 않는다. 통계는 client가 cache한다. 삭제 요청 전후 동일 encoder, 데이터, cache를 사용한다.

클리핑한 통계의 noisy sum에서 결정적 decoder G로 모델을 만든다. 예를 들어 noisy Gram을 대칭화하고 PSD로 투영한 뒤 양의 ridge 정규화를 더해 푼다. 이 PSD 보정도 처음부터 선언한 learner의 일부다. Gram clipping/잡음 때문에 일반적인 원자료 ridge와는 목적함수/학습 규칙이 달라질 수 있다. 재학습 reference에도 정확히 같은 규칙을 적용한다. 공개된 active count에 따라 G를 바꿀 수도 있지만, 검산 코드는 고정 초기 K로 정규화한다.

서버는 처음부터 보호된 합산 통계만 받아야 한다. 먼저 noiseless 모델/통계를 알려 준 뒤 삭제 시점에 noise를 넣는 것으로는 과거 노출을 복구할 수 없다. encoder 자체에 A의 private data가 들어간 경우 본 보장은 head에만 해당하므로 사용 조건을 위반한다.

## 2. 삭제 신호를 실제로 공중 합산하는 방법

잔존 통계 합을 S_R, A의 통계를 s_A라 하자. 최초 AirComp 관측은

\[
Y_0=S_R+s_A+Z_0,\qquad Z_0\sim\mathcal N(0,\sigma^2 I).
\]

삭제 때 서버는 데이터 독립적으로 정한 beta, 채널 보상 및 scale을 방송한다. client 간 통신이나 Gram–Schmidt는 없다. **A는 -beta s_A, 잔존 i는 (1-beta)s_i를 같은 자원에서 동시 전송**한다. 채널 보상 후 서버 관측은

\[
U=(1-\beta)S_R-\beta s_A+Z_1,\quad
Z_1\sim\mathcal N(0,(1-\beta^2)\sigma^2 I),\quad Z_1\perp Z_0.
\]

서버가 할 일은 한 번의 선형 갱신과 head 재계산이다.

\[
\boxed{Y_1=\beta Y_0+U=S_R+\beta Z_0+Z_1.}
\]

따라서 Y_1의 분포는 N(S_R,sigma^2 I)다. 동일한 noisy-statistic 학습 알고리즘을 잔존 데이터에 처음부터 적용한 결과와 **최종 모델의 주변분포가 정확히 같다**. G가 비선형인 PSD 투영/역행렬이어도 같은 확률변수 분포의 후처리라 이 결론은 유지된다. 잡음이 A의 평균 영향을 지우는 것이 아니라, 신호가 그 영향을 상쇄하고 잡음이 개별 신호 추론을 제한한다.

단순히 최종 noise covariance만 맞춘 이전 NM-Air와 다르다. 이전에는 `(I-PH_R)(W0-W_R)`라는 평균 편향이 남았다. 여기서는 모델 가중치의 경로를 역추적하지 않고, 처음부터 학습에 충분한 통계 상태를 갱신해 A의 평균 항을 상쇄한다.

이것은 저장된 과거 Y0까지 A가 없던 것처럼 바꾸는 보장이 아니다. 서버가 기억한 과거 관측의 누출은 아래 DP로 제한한다. source Y0를 고정한 조건부분포가 독립 재학습과 같다는 주장도 하지 않는다. 삭제 대상은 사전에 고정한 한 client이며 adaptive requester/multiple deletions 정리는 아직 없다.

## 3. 삭제 전후 차이 공격까지 넣은 프라이버시

서버는 Y0, U 및 모든 공개 control을 본다. (Y0,U)와 (Y0,Y1)는 데이터 독립 가역 선형변환 관계다. beta<1에서 잡음의 joint covariance는

\[
\Sigma=\sigma^2\begin{bmatrix}1&\beta\\\beta&1\end{bmatrix}\otimes I.
\]

한 client 전체 데이터를 다른 데이터로 교체하는 adjacency를 사용한다. L2 clipping으로 Delta=2B. rho0=Delta^2/(2 sigma^2)라 쓰자. 같은 covariance Gaussian의 Rényi divergence는 `alpha/2 * delta_mean^T Sigma^{-1} delta_mean`이다. [zCDP의 Gaussian 보장과 변환](https://arxiv.org/abs/1605.02065)을 적용하면

\[
\rho_A=\frac{\rho_0}{1-\beta^2},\qquad
\rho_R=\frac{2\rho_0}{1+\beta},\qquad
\rho_{\rm all}=\max(\rho_A,\rho_R).
\]

A 교체는 joint mean을 `(delta_s,0)`만큼, 잔존 client 교체는 `(delta_s,delta_s)`만큼 움직인다. 이 차이가 두 식의 이유다. epsilon=rho_all+2sqrt(rho_all log(1/delta))로 (epsilon,delta)-DP 상한을 얻는다. client-level 보장이지 record-level 보장을 client-level로 바꿔 부른 것이 아니다.

- beta=0: 독립 noisy retained 재집계. rho_A=rho0, rho_R=2rho0.
- beta=.5: 둘 다 4rho0/3. 전체 client 최악 privacy를 최소화한다.
- beta가 1에 접근하면 rho_A가 발산한다. beta=1, Z1=0이면 Y0-Y1=s_A로 A가 정확히 노출된다.

예시 B=1, sigma^2=8, delta=1e-5에서는 beta=.5의 rho_all=1/3, epsilon 상한=4.2513이다. 이것을 강한 실무 프라이버시라고 자동 판정하지 않는다. 완벽 은닉/누출 0도 아니다. 더 작은 epsilon을 요구하면 noise가 더 커진다. 독립 refresh의 전체 rho=.5, epsilon 상한5.2985와 비교할 때 공통 **상한**을 만족한다는 뜻이며, A만의 rho는 .25에서 .3333으로 증가한다.

같은 신호를 r번 보낸 raw 관측도 누락하면 안 된다. 본 iid Gaussian/고정채널 모형에서는 그 평균이 payload의 충분통계이고, 평균을 뺀 나머지 관측은 데이터와 독립이다. 따라서 raw 반복까지 포함해 위 두 평균 관측으로 privacy를 분석할 수 있다. 공개 pilot/control은 데이터 독립이라는 전제다. 서버가 아는 seed의 software noise를 추가하는 것은 서버에 대한 이 증명을 대신하지 못한다.

## 4. AirComp 고유 설계는 어디에 있는가

일반 FL에서도 통계 갱신 대수 자체는 구현할 수 있다. 논문 후보 기여는 **삭제 후 분포를 유지하며 전체 관측 privacy를 제한하는 송신계수 beta와 physical power/repetition의 공동 설계**다. '잡음을 활용했다' 또는 '통계를 뺐다'는 결합만으로는 부족하다.

복소 위상 보상 후 양의 채널 gain |h_i|, real 좌표당 수신 잡음 분산 nu^2, client별 1회 vector 전체 송신 에너지 cap P_i를 가정한다. 이는 per-symbol peak cap과 구분한다. 각 client가 coefficient c_i를 가진 통계를 `x_i=c_i s_i/(a |h_i|)`로 전송한다. r회 평균 후 a를 곱하면 noise variance는 a^2 nu^2/r이다. 따라서

\[
a^2=\frac{r(1-\beta^2)\sigma^2}{\nu^2},\quad
a_A=\frac B{|h_A|\sqrt{P_A}},\quad
a_R=\max_{i\in R}\frac B{|h_i|\sqrt{P_i}}.
\]

\[
r_{\min}(\beta)=\max\!\left(1,\left\lceil
\frac{\nu^2\max(\beta^2a_A^2,(1-\beta)^2a_R^2)}{(1-\beta^2)\sigma^2}
\right\rceil\right).
\]

전체 client의 rho 상한을 rho_bar로 정하면, 허용되는 beta 구간은 다음과 같다.

\[
\beta_L=\max(0,2\rho_0/\bar\rho-1),\quad
\beta_U=\sqrt{1-\rho_0/\bar\rho}.
\]

비어 있지 않으려면 rho_bar>=4rho0/3이다. 연속 반복 비용은 beta=a_R/(a_A+a_R)에서 최소이며 이 값을 [beta_L,beta_U]에 clip하면 된다. 공개 upper bound와 채널만 사용한다.

payload 송신 에너지도 직접 계산된다. 공개 상한 A_e=B^2/|h_A|^2, R_e=B^2 sum_R 1/|h_i|^2를 쓰면

\[
E_{\rm payload}(\beta)\le\frac{\nu^2}{\sigma^2}
\frac{\beta^2 A_e+(1-\beta)^2R_e}{1-\beta^2}.
\]

실제 에너지 식은 B^2 대신 각 ||s_i||^2를 넣으면 된다. **실제 private norm을 수집해 beta를 고르지는 않는다.** 상한을 최소화하는 값은

\[
\beta_E=\operatorname{clip}_{[\beta_L,\beta_U]}
\frac{2R_e}{2R_e+A_e+\sqrt{A_e^2+4A_eR_e}}.
\]

이는 payload 에너지 상한의 최적화다. RF circuit, pilot, A의 추가 참여, control을 합친 전체 시스템 최적값이라고 하지 않는다. 전체 목적함수에는 beta=0의 A 비참여 옵션도 넣고 추가 CSI 획득비를 부담시켜야 한다. 이번 cost 예시는 fixed beta가 중심이며 최적 beta 결과는 별도 항목으로 저장했다.

## 5. 이번 GPU 검산과 비용 예시

RTX3080 FP64, synthetic statistics 및 Gaussian/채널 계산, 코드 본체 약0.44초, peak allocation 약598MB. GPU 분류 학습 실험은 아니다. 총6개 beta, Gaussian MC100000회,200개 채널 조합의 optimizer/grid 비교를 했다.

- 갱신 통계와 coupled retained reference 최대 차이3.56e-15 이하; 결정적 ridge decoder 최대 차이8.00e-15 이하.
- 같은 잡음에서 A의 통계 부호를 바꾸면 삭제 전 학습 모델의 좌표당 MSE는 .03608로 변한다. 처음부터 A가 학습에 아무 영향도 없었던 예제는 아니다. 이 변경은 clipped-statistic domain의 검산이며 실제 label 교체 실험은 아니다.
- 같은 잡음 아래 A 통계를 변경해도 최종 통계 차이3.56e-15 이하. 반면 beta=1 차이 공격은 A를1.86e-15 오차로 복원.
- joint precision 계산과 privacy 식 일치; minmax beta=.5.
- normalized covariance 최대 오차 .00922. 물리 송수신에서 update noise variance 예측6.0, MC6.00049. block 전력 제약과 energy 식 통과.

두 분포의 같음을 단일 coupling 오차/MC로 증명했다는 의미는 아니다. 증명은 2절의 선형 Gaussian 식이고, 위 수치는 그 구현 검산이다.

K=10, q=296, P=1, nu^2=.01, sigma^2=8의 **이상화된 비용 예시**:

| 채널 | 독립 refresh beta=0 | beta=.5 | 해석 |
|---|---:|---:|---|
| 모두 gain1: payload 에너지 상한 | .01125 | .0041667 | 63.0% 감소; 실제 RF joule 아님 |
| 모두 gain1: 총 삭제 real uses | 2273.96 | 2282.96 | 0.40% 증가; airtime 이득 없음 |
| 잔존 1명 gain .01: 반복 수 | 13 | 5 | weak retained 신호 부담 완화 |
| 잔존 1명 gain .01: 총 삭제 real uses | 6269.96 | 3614.96 | 42.3% 감소; 특정 예시 |
| A만 gain .01: 총 삭제 real uses | 2273.96 | 3614.96 | 59.0% 증가; 고정 beta 실패 |

에너지 감소율은 public-bound 상한끼리의 비교이며, 모든 payload norm=B이면 실제 payload energy 비율과 같다. 두 방법은 동일 final noise law 및 동일 전체 privacy cap을 만족하나 client별 실제 rho가 동일한 것은 아니다. source 준비비까지 넣은 lifecycle 값도 JSON에 포함했다. B, sigma, encoder/규칙 설치는 cache되어 있다고 가정하며 최초 encoder 배포·잡음 바닥 측정비와 실제 RF 회로 에너지는 미측정이다. gain=.01은 성공 사례를 일반화하기 위한 현실적 채널 추정이 아니라, A와 잔존의 채널 비대칭을 드러내는 stress example이다.

## 6. 가장 큰 실용 장애물: 작은 client 수의 utility

학습 통계를 K로 정규화하면 E||noise/K||^2=q sigma^2/K^2이고 ||sum(s_i)/K||<=B다. 위 예시의 q=296, K=10에서는 noise 제곱노름 기대값이 B^2의 **23.68배**다. 기존 d64의 q=2720이면217.6배다. 현재10-client 설정에서 client-level DP를 이렇게 강제하면 utility가 심하게 저하될 위험이 높다. 이 값을 정확도 하락률로 해석하지 않는다.

따라서 **현재10-client 조건에 이 후보를 곧바로 채택할 것을 권하지 않는다.** 더 작은 private head, 더 많은 clients, 또는 공개 pretrained features가 있는 적용 상황에서 우선 feasibility를 확인해야 한다. q=296의 비율은 K=100일 때 .2368, K=1000일 때 .002368이지만, 이것만으로 정확도가 좋다고 결론낼 수도 없다. client 수를 늘리면 pilot/control 비용도 늘어난다.

추가 검증은 작게 고정한다: MNIST/FashionMNIST, 공개/고정 encoder와 동일 clipped learner, K=10/100/1000, 새 seed3개. 동일 final law/동일 privacy cap의 독립 DP-AirComp 재집계와 비교하고, deterministic clean learner는 utility 손실 기준으로만 사용한다. private 데이터 없이 얻는 모델, noisy no-op도 함께 비교해야 한다. 통신 후보는 fixed beta=.5와 public-channel beta 최적화만 둔다. **실제 task utility와 전체 에너지 이득이 함께 안 나오면 이 후보를 보류하며, noise 크기를 사후로 낮춰 privacy 조건을 완화하지 않는다.** 이 후속 실험은 아직 실행하지 않았다.

## 7. 가까운 선행연구와 남길 수 있는 주장

| 선행연구 | 이미 알려진 부분 | 본 후보에서 별도로 검증할 차이 |
|---|---|---|
| [Koda et al., DP AirComp with receiver noise, 2020](https://arxiv.org/abs/2004.06337) | 수신 잡음을 privacy에 쓰고 송신 power로 조절 | 삭제 전후 공동 관측에서 deleted/retained에 다른 privacy 식, 출력 재학습분포 제약과 beta 설계 |
| [Chen et al., Upcycling Noise for Federated Unlearning, 2024](https://arxiv.org/html/2412.05529v1) | DPFL 잡음 재활용, local retraction 및 추가 noise 보정 | 정확 additive-statistic 상쇄, sum-only 물리 수신, 전체 두 관측 covariance를 직접 분석 |
| [Quan et al., Exact Federated Continual Unlearning for Ridge Heads, 2026](https://arxiv.org/html/2603.12977) | frozen encoder, additive Gram/moment, 정확 head 삭제; 통계 privacy 한계도 논의 | 이 대수를 새 알고리즘이라고 주장하지 않음. noisy 통계와 전체 관측 privacy 아래 무선 coefficient/power 설계가 후보 기여 |
| [Protecting the Undeleted in Machine Unlearning, 2026](https://arxiv.org/html/2602.16697) | 삭제 출력으로 잔존 데이터가 새로 노출되는 문제와 보안 정의 | 본 후보는 한 번의 fixed 삭제에 deleted 및 retained의 joint client-DP를 함께 제한; 그 논문의 일반 보안 정의를 해결했다는 뜻 아님 |
| [Correlated Noise Provably Beats Independent Noise, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1384616b65241ef17aad3f3bde8fd623-Abstract-Conference.html) | correlated noise로 private learning 개선 | correlated noise 자체의 신규성 주장 불가. 무선 삭제 신호·잔존 신호·물리 비용에 특화한 기여 필요 |

이 표는 가장 가까운 확인된 원문과 비교한 후보 차이다. 동일한 joint PHY 설계의 선행연구가 없음을 입증한 exhaustive novelty search가 아니다. 특히 공개 충분통계 + DP + AirComp를 단순 조합하는 수준으로 끝나면 논문 기여가 약하다. beta에 따른 삭제대상/잔존 privacy의 충돌, 채널 비대칭에 따른 최적 정책, 실제 동일 privacy/utility 비용 우위를 함께 제시해야 한다.

## 8. 보장이 깨지는 조건

정확 CSI에서의 수식이다. 초기 및 삭제 effective coefficient가 각각 a0_A,a1_A라면 삭제 후 A의 평균 잔여 항은 beta(a0_A-a1_A)s_A이다. 채널이 바뀌어도 정확 보상하면 문제가 없지만, 보상 오차가 서로 다르면 exact law는 깨진다. CSI 오차를 별도 허용오차로 증명하거나 측정해야 한다.

수신기가 여러 독립 관측을 추가로 얻거나 client별 신호를 공간적으로 분리할 수 있으면 전체 observation matrix와 noise covariance를 다시 계산해야 한다. 미측정 열잡음 하한, 악의적 서버, dropout, 반복 삭제, private-data-dependent scale, private source 모델의 사전 공개, decoder 선택에 private evaluator reference 사용은 현재 보장 밖이다. 프로토콜 밖 관측을 무시한 'AirComp이므로 privacy' 주장을 하지 않는다.

## 재현

NumPy/PyTorch와 CUDA가 있는 환경에서 저장소 root 기준:

```text
python research_20260928_protected_refresh/verify_theory.py
```

표준 라이브러리 및 NumPy/PyTorch만 쓴 신규 코드다. 외부 저자 코드를 복제하지 않았다. 기존 DS/NM 실험과 결과를 수정하지 않는다. 결과 JSON에 code/protocol SHA-256과 GPU/runtime 정보를 기록한다.
