# Cohort-Diversity DS-Air: 전원 참여를 유지하는 채널 선택

2026-09-28. [실행 전 프로토콜](PROTOCOL_KO.md)과 [결과](README_KO.md)의 수학적 배경이다. 기존 DS-Air의 ridge 삭제식과 전력 제한 송수신은 [이전 방법 문서](../research_20260928_ds_revision/METHOD_KO.md)를 따른다. 이 문서의 채널 선택은 기존 코드의 공개 고정 부분공간 U와 독립적으로 적용한다.

## 1. 바꾸는 것은 클라이언트 집합이 아니라 공유 채널

이전 실패는 깊은 fading의 한 클라이언트에 맞춰 모든 송신 신호를 줄이면서, 수신 후 복원 scale이 커지고 Hessian 잡음이 증폭되는 문제였다. 그 클라이언트를 제외하면 다음의 삭제 목표가 바뀐다.

\[
H_R=\frac1K\sum_{i\in R}H_i,\quad
b_R=\frac1K\sum_{i\in R}(H_iW_0-C_i),\quad
W_R=W_0-H_R^{-1}b_R,\qquad K=9.
\]

이번에는 R을 바꾸지 않는다. FD-L(Frequency Diversity)은 동시에 사용 가능한 L개의 공유 주파수 자원을 pilot으로 탐색하여

\[
\ell^*=\arg\max_{1\leq\ell\leq L}\min_{i\in R}|\widehat h_{i\ell}|^2
\]

를 고른다. 모든 잔존 클라이언트가 선택된 동일 자원에서 동시에 합산한다. 개인별 채널 할당이나 개별 gradient 전송이 아니다. 실제 payload나 재학습 reference로 자원을 고르지 않으며, **가장 약한 link가 덜 약한 자원을 고르는 단순 휴리스틱**이다. payload별 최적 자원 할당의 전역 최적성을 주장하지 않는다.

FD8-SubspaceV1은 이 자원 선택 이후 기존 r48 부분공간에서 H_U와 residual을 합산한다. U는 client 간 신호 분리를 위한 직교 코드가 아니라 전원이 공유하는 모델 좌표다.

## 2. Deep-fade 잡음의 발산을 피하는 조건

정확한 CSI에서 블록 payload z_ij의 평균전력1, peak amplitude2 제약에 필요한 scale은

\[
a_{ij}=\max\{\|z_{ij}\|_2/\sqrt{80},\|z_{ij}\|_\infty/2\},
\quad s_{j\ell}=\max_i a_{ij}/|h_{i\ell}|.
\]

독립 unit-mean Rayleigh에서 \(|h_{i\ell}|^2\sim\mathrm{Exp}(1)\)이다. \(m_\ell=\min_i |h_{i\ell}|^2\)는 Exp(K)이며, 선택된 최소 gain 제곱 \(M_L=\max_\ell m_\ell\)의 CDF는

\[
P(M_L\leq x)=(1-e^{-Kx})^L.
\]

따라서 L=1이면 E[1/M_L]이 발산하지만 L>=2이면 유한하다. x=0 부근에서 density가 O(x^(L-1))이므로 inverse moment 적분은 O(x^(L-2))가 되기 때문이다. L>=2일 때 닫힌형은

\[
\mathbb E[M_L^{-1}]
=-KL\sum_{j=0}^{L-1}(-1)^j {L-1\choose j}\log(j+1).
\]

| 후보 L | K=9에서 이론값 | 100,000회 GPU Monte Carlo |
|---|---:|---:|
|1|발산|유한 표본 평균으로 수렴을 주장하지 않음|
|2|12.47665|12.39845|
|4|6.11637|6.09551|
|8|4.04404|4.03388|

또한 \(s_{j\ell^*}^2\leq\max_i a_{ij}^2/M_L\)이다. 고정된 유한 payload에 대해 이 bound가 유한한 평균 잡음 전력을 준다. 구현에서는 inverse의 고유값을 lambda=.01로 제한하므로 preconditioned payload의 크기도 유한하게 제한된다. **유한한 평균 잡음 전력은 정확한 삭제 또는 E<1 보장이 아니다.** 근사 Hessian, 부분공간 잔차, 유한 반복과 CSI 오차는 여전히 남는다.

이 증명은 독립 자원과 정확한 CSI에 대한 것이다. 후보 채널이 전부 같으면 M_L=m_1이므로 아무 이득이 없다. 부분 상관 및 부정확한 CSI는 GPU stress test로 확인했으며 위 식을 그대로 보장으로 쓰지 않는다.

경로감쇠 amplitude beta_i가 다르면 iid 자원에서 m_l의 exponential rate는 \(\kappa=\sum_i\beta_i^{-2}\)다. 한 클라이언트만 beta=.2이면 kappa=33으로, 균일한 경우9보다 불리하다. 채널 diversity는 영구적인 경로손실을 없애지 않는다.

## 3. 대기 방식 CG와 미완료 처리

CG(Cohort Gate)는 미래 채널을 미리 보지 않는다. 각 coherence window에서

\[
\min_i|\widehat h_i|\geq\tau,\quad \tau=.4
\]

가 처음 성립하면 전원이 합산한다. 최대8번 실패하면 **모델을 갱신하지 않고 삭제 요청을 pending으로 반환**한다. 채널이 나쁜 client만 빼거나 이전의 좋은 window로 되돌아가는 동작은 없다. 대기 중 source/model과 데이터는 고정된다고 가정한다.

독립·정확CSI·균일 경로감쇠이면 한 번의 통과 확률과 8번 이내 완수 확률은

\[
p=e^{-K\tau^2}=.23693,\quad
P(\mathrm{complete})=1-(1-p)^8=.88505,
\quad\mathbb E[N_{\rm tries}]=\frac{1-(1-p)^8}{p}=3.73551.
\]

수학 단계의 100,000개 channel bank에서 완수율 .88533, 평균 시도3.74121이었다. 한 client의 beta=.2에서는 p=e^(-33*.16)로 작아져 8시도 완수율이 약4%가 된다. 기다리면 반드시 해결되는 구조가 아니다.

GPU 결과의 pending은 parameter E=1, test accuracy는 W0의 값이며 전체 평균과 지연에 포함한다. 낮은 active 비용이 미완료 때문에 생겼다면 통신 효율 개선으로 해석하지 않는다. 코드의 `unlearning_success` 필드는 완료했고 표본평균 E<1인지의 짧은 이름일 뿐, 공식 삭제 인증이나 모델 분포 보장이 아니다.

## 4. 탐색비와 대기비의 포함

기존 main ledger의 H/보정 payload, basis/eigen, power bound, model broadcast, pilot 및 CP를 그대로 사용한다. baseline에도 첫 채널의 CSI feedback9x32bit를 추가하여 server가 gain을 아는 조건을 맞춘다.

기존 ledger에 더하는 탐색비는 n개 자원을 확인할 때

\[
T_{\rm extra}=1.125\left(72(n-1)+\frac{288n+67}{R_d}\right),
\quad R_d=.5\log_2(1+\rho).
\]

첫 pilot은 main ledger에 이미 포함되어 중복 차감한다. 67bit는 후보/설정64bit+선택3bit다. baseline은 탐색 설정 없이 CSI feedback288bit만 추가한다. CG는 실제 시도 n, FD는 전체 후보 L을 지불한다. 모든 방식은 동일 active 예산에서 이 비용을 먼저 차감하고 남은 정수 반복을 배정한다.

20dB에서 추가비는 baseline97.32, FD8 1,368.23이며 차이는 1,270.91 real-use equivalent이다. 전체 acquisition 비용(이미 main에 포함된 첫 pilot까지)은 FD8 1,449.23이다. 비용을 감추지 않기 위해 full V1의 보정 반복 합56을 FD8-V1에서는42로, Subspace의639를 FD8-Subspace에서는625로 줄였다. 소비되지 않은 자투리 예산도 표시한다.

초기 source가 cache된 경우 실제 active 비용은 `base_ledger.total + search_cost.total`이다. Raw의 `base_ledger.cold_total/lifecycle`에도 탐색비를 더해야 한다. 집계 JSON의 `mean_cold`, `mean_lifecycle`, `completed_lifecycle`는 이 추가비를 포함한다. source 준비비는 이전과 같은 이상적 sufficient-statistics 준비 원장이며 noisy training의 실측 비용이 아니다.

CG의 coherence window 길이를 T_c라 하면 완료 시 지연은

\[
T_{\rm latency}=(n-1)T_c+T_{\rm active}-(n-1)T_{\rm probe},
\quad T_{\rm probe}=1.125(72+288/R_d).
\]

이미 앞선 window에 지불한 pilot/feedback을 마지막 window에 다시 더하지 않는다. timeout은8T_c다. T_c를 active 예산의1/4/16배로 계산했다. 1배는 전체 세션을 block-static으로 모사하기 위한 최소 radio-time 설정이며 실제 coherence time 측정값이 아니다. 처리/튜닝 지연, feedback packet error, RF 에너지는 포함하지 않았다.

FD는 동시에 사용 가능한 주파수 후보가 전제다. 획득 pilot은 직렬 비용으로 모두 계산하며 전체 세션 동안 채널이 고정된다고 가정한다. 선택하지 않은 자원은 이후 다른 사용자에게 돌아가므로 계속 점유하는 비용을 계산하지 않는다. **8개 자원을 세션 내내 독점 예약해야 하는 시스템이라면 자원 점유비가 최대 약8배가 될 수 있고, 이번 동일 active 예산 결과를 그대로 적용할 수 없다.** 정확한 OFDM waveform, coherence bandwidth, cross-band 간섭, Doppler, 재튜닝/계산 지연을 검증한 SDR 실험이 아니다.

채널 feedback은32bit로 비용을 계산했다. 동결된 GPU simulator는 선택 시 float64 값을 사용했으므로 사후에 [독립 검사](audit_feedback.py)를 추가했다. 모든10,368개의 band/threshold/timeout 결정을 FP32 값으로 다시 계산했을 때 결정 변화가0이었다. 클라이언트의 로컬 channel inversion CSI는 고정밀을 유지한다. 임의의 입력에서 양자화가 결정을 바꾸지 않는다는 정리는 아니다.

## 5. 프라이버시와 선행연구

채널 선택 자체에는 모델 gradient가 필요하지 않지만, 기존 DS의 공개 inverse/출력/source/power metadata에 대한 복원 문제는 그대로다. 정상점에서 g_A=-9b_R이므로 full 방식은 원래의 문제가 남고, U 부분공간의 projected gradient도 복원할 수 있다. 잡음을 줄이면 공격자가 더 정확하게 복원할 수도 있다. 이번 개선은 이 현상을 측정했으며 DP나 전체 transcript 보호를 주장하지 않는다.

가까운 선행연구는 다음과 같다. 아래 방법들의 전체 저자 코드를 재현한 실험은 아니다.

- [Cao et al., Optimized Power Control for Over-the-Air Federated Edge Learning](https://arxiv.org/abs/2011.05587): deep fading에서 channel inversion의 잡음 증폭을 지적하고 전력 제어를 최적화한다. Deep-fade 대응 자체를 최초라고 주장할 수 없다.
- [Hellström et al., Federated Learning Over-the-Air by Retransmissions](https://doi.org/10.1109/TWC.2023.3268742): 정적 채널의 재전송과 전력 제어, FL 성능을 분석한다. 단순 반복 전송 또는 반복 횟수 최적화만으로 신규성을 주장하지 않는다.
- [Optimal Power Control and CSI Acquisition for Over-the-Air Computation in OFDM System](https://ieeexplore.ieee.org/document/10328486/): multicarrier AirComp에서 전력 제어와 CSI 획득을 다룬다. 주파수 자원을 이용한다는 사실 자체는 신규 기여가 아니다.
- [Selective channel inversion protocol for over-the-air computation](https://doi.org/10.1016/j.icte.2025.11.016): 채널 threshold와 좋은 채널의 sensor만 전송하는 방식이다. 이번 방식은 잔존 집합 전체가 포함되는 공유 자원을 선택하거나 전체 집계를 보류한다는 제약을 둔다. 이 제약의 차이가 곧 독창성 증명은 아니다.

이번에 입증한 것은 **전원 참여를 유지한 diversity가 기존 DS의 발산하는 역채널 잡음을 완화하고, 탐색비를 포함한 고정 예산에서 실제 삭제 근사오차를 줄일 수 있다는 조건부 결과**다. Min-gain selection 및 diversity 원리는 일반 FL에도 적용된다. 국내 학회용으로도 이를 새로운 언러닝 원리라고 과장하면 안 된다. 더 강한 power-aware frequency allocation/combining 비교군, 실제 coherence/resource 가정, 모델 확장 및 전체 transcript 프라이버시는 아직 남아 있다.
