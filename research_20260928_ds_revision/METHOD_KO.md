# DS-Air 보완 실험: 수식, 송수신 절차와 한계

2026-09-28. 실행 전 고정한 [프로토콜](PROTOCOL_KO.md), [구현](run_experiment.py), [결과](README_KO.md)를 함께 읽는다. DS는 잔존 클라이언트의 deletion signal을 합산하는 방식이다. 삭제 대상 A가 마지막 라운드에 자신의 gradient를 보내는 방식이나, 신경망 전체의 학습 경로를 되돌리는 방식과는 다르다.

## 1. 정확한 삭제 방향이 존재하는 범위

공개된 고정 encoder의 출력 행렬을 Z_i, one-hot label을 Y_i라 두고, ridge head만 학습한다.

\[
F_i(W)=\frac{\|Z_iW-Y_i\|_F^2}{2n_i}+\frac{\lambda}{2}\|W\|_F^2,
\quad H_i=Z_i^TZ_i/n_i+\lambda I,\quad C_i=Z_i^TY_i/n_i.
\]

이번 실험은 동일 크기 클라이언트 10개 중 A를 삭제한다. 잔존 집합 R의 크기는 K=9이며

\[
H_R=K^{-1}\sum_{i\in R}H_i,\quad C_R=K^{-1}\sum_{i\in R}C_i,
\quad b_i=H_iW_0-C_i,\quad b_R=K^{-1}\sum_{i\in R}b_i.
\]

그러면 W_0가 정상점이 아니더라도

\[
W_R=H_R^{-1}C_R=W_0-H_R^{-1}b_R
\]

가 성립한다. 따라서 quadratic head에서는 과거 업데이트를 저장하거나 차감할 필요 없이, 잔존 데이터의 현재 residual로 삭제할 수 있다. **이 식은 private data로 학습된 encoder의 영향을 지우지 않는다.** 이번 encoder는 처음부터 공개 random feature로 고정했다. 잡음, 부분공간, 부정확한 곡률을 적용하면 실제 출력은 근사 삭제다.

서버는 H_R의 upper triangle을 OTA로 수집한다. 수신 행렬을 대칭화하고 고유값을 lambda=.01 이상으로 제한하여 공통 근사 inverse를 만든다.

\[
\widehat P=Q\widehat\Lambda^{-1}Q^T.
\]

Q/eigenvalue는 FP32로 방송하며 수신 Q를 동일한 QR로 다시 정규직교화한다. **이번 실험은 dense Hessian 수집이며 HVP 한 번으로 전체 inverse를 얻는 실험이 아니다.**

## 2. 송신 전 보정과 송신 후 보정

Original은 Q^T b_i/K를 공중 합산한 뒤 서버에서 inverse를 적용한다. V1은 같은 inverse를 각 클라이언트에서 먼저 적용하여 \(\widehat\Lambda^{-1}Q^Tb_i/K\)를 합산한다. 두 방법은 정확한 CSI에서 동일한 조건부 평균 \(W_0-\widehat P b_R\)을 가진다. 차이는 무선 잡음이 inverse를 통과하는 위치다.

길이 L=80인 블록 j의 실제 weighted payload를 z_ij라고 하자. 평균 송신전력 1, peak amplitude 2를 동시에 지키도록

\[
s_j=\max_i\frac{\max\{\|z_{ij}\|_2/\sqrt L,\|z_{ij}\|_\infty/2\}}{\widehat h_i},
\qquad x_{ij}=\frac{z_{ij}}{s_j\widehat h_i}
\]

를 쓴다. 각 클라이언트가 보낸 bound는 FP32 upward rounding하므로 양자화로 전력 한도를 초과하지 않는다. 실제 수신은

\[
\widehat z_j=\sum_i(h_i/\widehat h_i)z_{ij}+s_jn_j,
\qquad n_j\sim\mathcal N(0,I/(\rho r_j)),\quad \rho=10^{\mathrm{SNR}/10}.
\]

위상은 완전히 보상하며, CSI 오차 실험은 amplitude 오차만 넣는다. 반복 r_j는 Gaussian 평균의 등가식으로 모사했고 실제 반복평균의 분산과 별도로 대조했다.

블록에 포함되는 8개 고유방향과 출력 10개에 대해 보정 잡음 MSE 계수는

\[
c_j^{\rm Original}=\frac{10(s_j^{\rm raw})^2}{\rho}\sum_{\ell\in j}\widehat\lambda_\ell^{-2},
\qquad c_j^{\rm V1}=\frac{80(s_j^{\rm pre})^2}{\rho},
\quad V=\sum_j c_j/r_j.
\]

**V1은 scale 증가까지 포함해야 한다.** 한 좌표 또는 등방 inverse에서는 이득이 상쇄된다. 평균전력만 있는 예에서 P=diag(10,1), z=(0,1)이면 Original/V1의 잡음 risk가 50.5/1이지만, z=(1,0)이면 50.5/100이다. 따라서 전처리가 항상 유리하다는 명제는 거짓이다. 이 수치 예와 실제 peak 제약 송수신을 [수학 검사](results/math_checks.json)에 보존했다.

## 3. Switch: 이론적으로 가능한 선택과 실제 비용

각 클라이언트는 raw/pre 두 payload의 bound를 보내고, 서버는 각 블록에서 작은 c_j를 선택한다. 최종 r_j는 각 블록 1회부터 시작하여 \(c_j/[r_j(r_j+1)]\)가 최대인 블록에 한 번씩 추가한다. 동일 길이 블록에서 이는 주어진 정수 반복 합에 대한 \(\sum_j c_j/r_j\)의 최적 배정이다. 작은 예를 완전 탐색과 비교했다.

하지만 두 후보 bound를 수집하는 추가 digital 전송은 무료가 아니다. 같은 총예산에서는 Original의 총 반복 64회, V1 56회, Switch 47회가 된다(D64,20dB). 블록별 c_j가 작아도 남는 반복 수가 감소하면 최종 MSE는 커질 수 있다. 탐색 기준은 더 강한 Original/V1보다 5% 이상 개선하는 것이었으며, **실제 결과는 이 기준을 통과하지 못했다.**

또한 H/CSI를 고정한 조건부 오차는

\[
\mathbb E_n\|\widehat W-W_R\|_F^2
=\|\overline W-W_R\|_F^2+\sum_j c_j/r_j.
\]

정확한 CSI에서 첫 항은 \(\|(H_R^{-1}-\widehat P)b_R\|_F^2\)이다. Switch는 두 번째 항을 최적화할 뿐 첫 번째 항을 고치지 않는다. 여기서 'bias'는 H/channel draw에 조건부인 평균 오차다. 모든 H 잡음을 평균 낸 추정량의 통계적 bias와 동일한 뜻은 아니다.

## 4. SubspaceV1: 곡률 비용과 복원 가능한 방향을 줄이는 보완

데이터 및 삭제 대상과 독립적인 고정 seed로 U(D x r), U^TU=I, r=3D/4를 만든다. 서버가 모으는 것은 H_R 전체가 아니라

\[
H_U=U^TH_RU,\qquad b_U=U^Tb_R,
\quad W_U=W_0-UH_U^{-1}b_U
\]

이다. 실제로는 H_U도 OTA 잡음이 있는 상태에서 inverse를 추정한다. U seed를 공유하고 r x r 기저만 방송한다. 클라이언트들을 서로 직교화하거나 CDMA로 개별 식별하지 않는다. 모두 같은 U의 좌표에서 합산한다.

잡음이 없다면 d*=H_R^{-1}b_R에 대해 d_U=U H_U^{-1}U^T b_R는 H_R 내적에서 d*를 U 공간에 투영한 것이다.

\[
\|W_U-W_R\|_{H_R}^2
=\min_v\|d^*-Uv\|_{H_R}^2.
\]

따라서 방향을 충분히 포함하지 못하면 없앨 수 없는 근사오차가 남는다. 이 식은 Euclidean 오차나 삭제 성공을 자동으로 보장하지 않는다. r=D이면 full 방식으로 돌아간다. 데이터에 맞게 U를 탐색하지 않았으며, rank를 결과에 맞춰 재선택하지도 않았다.

D64에서 upper triangle은 2,080개에서 1,176개로, 방송 기저는 4,096개에서 2,304개로 감소한다. 두 운용 방식을 분리했다.

- **Matched:** 절약된 준비비를 반복에 재투자한다. 총예산은 Original과 같고 보정 반복 합은 639회다. 비용 절감 성과로 보고하면 안 된다.
- **Nominal:** H32회, 블록당 평균 8회라는 원래 규칙을 유지한다. 보정 반복 합은 48회로, 전체 삭제비가 39.8% 감소한다. 정확도 손실도 함께 비교해야 한다.

## 5. 프라이버시: 해결한 것과 해결하지 못한 것

원래 source가 전체 데이터의 정상점이고 A의 비중이 alpha=.1이면

\[
\alpha g_A+(1-\alpha)b_R=0,\qquad g_A=-9b_R.
\]

따라서 full inverse, source 및 보정 출력이 공개되면 잡음 없는 극한에서 g_A를 복원할 수 있다. 개별 gradient를 직접 수신하지 않았다는 사실만으로 이 위험이 사라지지 않는다. FP32 source의 작은 정상점 잔차는 원자료에 별도로 기록했다.

Subspace에서는 U 밖의 방향을 선형적으로 복원할 수 없는 예를 구성했다. 모든 H_i는 그대로 두고 U^Tv=0인 v에 대해 C_A를 C_A+v, 잔존 각 C_i를 C_i-v/9로 바꾸면 전체 source, projected H, projected residual, power metadata가 모두 동일하면서 g_A는 달라진다. 실험의 수치 오차는 1e-15 부근이며 metadata는 bitwise 동일했다.

이는 **주어진 선형 통계 관측 모형에서의 비식별성 예**다. 원본 분류 데이터에서 그 통계 변경이 항상 실현된다는 증명, 임의의 보조정보를 허용한 보장, DP 또는 certified unlearning이 아니다. 다음 위험은 남는다.

- U 안의 gradient는 계속 노출된다. 30dB unit 채널의 projected-gradient 복원 상대 MSE는 약 .00053이다.
- 여러 U 또는 full-H transcript를 같은 모델에 축적하면 보호되던 방향이 다시 드러날 수 있다.
- power scalar, source, 이후 모델도 공개 정보다. 데이터 복원 공격이나 외부 도청의 안전성을 검증하지 않았다.
- 특정 gradient 추정기의 오차가 커졌다는 사실만으로 프라이버시를 입증할 수 없다.

**전체 transcript에 대한 프라이버시 요구는 미해결로 판정했다.**

## 6. 비용 원장과 fading 한계

이번 비용 단위는 실제 무선 초/줄이 아닌 real channel-use equivalent다.

\[
T=1.125\left(N_H+80\sum_jr_j+N_{\rm pilot}
 +\frac{B_{\rm control}+B_Q+B_\lambda+B_U+B_{\rm final}}{0.5\log_2(1+\rho)}\right).
\]

pilot, client별 scale, 요청/지시, 반복/scale 방송, 선택 mode, 기저/고유값, 공개 U seed, 최종 모델을 포함한다. cached source 기준 삭제비, source를 다시 보내는 cold 비용, 이상적 충분통계 source 준비+삭제비를 별도로 기록했다. normalized transmit energy는 raw JSON에 있으나 송신 시간·회로 전력이 없어 에너지 효율 성과로 해석하지 않는다. 계산/메모리 전반의 모델별 프로파일링도 이번 범위 밖이다.

Rayleigh에서 \(|h|^2\sim\mathrm{Exp}(1)\)이며, cutoff 없이 완전 channel inversion을 하면

\[
\mathbb E[1/|h|^2]=\infty.
\]

독립적인 비영 payload의 scale 제곱, 특히 이번 Hessian 전송의 effective noise variance는 이 문제를 갖는다. 몇 개의 deep fade가 결과를 지배할 수 있어 **3 seed x 4 channel draw의 평균을 Rayleigh 모집단 평균의 안정적인 추정이나 보장으로 해석할 수 없다.** 이는 full 방식의 큰 seed SD와 함께 읽어야 한다. 실험에서는 어려운 draw를 삭제하거나 client를 탈락시키지 않았다.

후속 보완의 우선순위는 cutoff/재전송/대기를 도입하되 잔존 client를 제외해 삭제 목표 자체를 바꾸지 않는 설계다. 그때 대기 시간, 추가 pilot 및 전체 비용을 포함해야 한다. 이번에는 그 후속 알고리즘을 구현하거나 성공했다고 주장하지 않는다.

Digital control은 nominal SNR의 이상적인 오류 없는 전송률로 계산했다. 실제 fading packet error, 재전송, 위상/CFO 오차는 모사하지 않았고 전체 삭제 세션 동안 채널이 고정되는 가정이다. `rayleigh`와 `rayleigh_csi5`의 channel RNG가 달라 두 결과의 차이를 CSI 오차만의 인과효과로 해석할 수 없다. 각 조건 내부의 방법 비교는 같은 channel/H를 공유한다(부분공간/diagonal은 각각 필요한 크기의 H 잡음을 사용).

## 7. 선행연구와 이번 비교군의 정확한 범위

| 선행연구 | 실제 아이디어 | 이번 반영과 차이 |
|---|---|---|
|[Yang et al., Over-the-Air Federated Learning via Second-Order Optimization (2022)](https://arxiv.org/abs/2203.15488)|로컬 Newton 방향을 계산하여 OTA 집계하고 무선 자원을 설계|LocalNewton은 식(6)-(8)의 local inverse-gradient 집계를 ridge 삭제 한 step에 옮긴 비교군이다. 논문의 전체 반복학습·beamforming·client selection을 재현한 것은 아니다.|
|[Krouka et al., Communication-Efficient Federated Learning: A Second Order Newton-Type Method With Analog Over-the-Air Aggregation (2022)](https://doi.org/10.1109/TGCN.2022.3173420)|분산 quadratic 문제와 ADMM을 이용해 Newton 방향을 model-size vector 통신으로 계산|2차 정보/OTA 결합 자체의 신규성을 주장할 수 없게 하는 가까운 선행연구다. 이번 실험에서 그 ADMM 알고리즘은 구현하지 않았다.|
|[Ghalkha et al., Scalable and Resource-Efficient Second-Order Federated Learning via Over-the-Air Aggregation (2024)](https://arxiv.org/html/2410.07662v1)|diagonal curvature 추정, EMA, OTA, clipping과 channel threshold를 사용|Diagonal은 정확한 ridge 대각 원소를 보내는 단순 대조군이다. GNB 추정/EMA/clipping을 포함한 Fed-Sophia 재현이 아니므로 논문 전체보다 우수하다고 주장할 수 없다.|

StatsRetrain은 noisy H와 C를 공중 합산하여 ridge를 다시 푸는 통신 비교군이다. evaluator의 exact retained solution과 구분한다. Diagonal/LocalNewton은 곡률 비용을 절약한 만큼 남은 예산을 보정 반복에 쓰지만, 현재 이질적 client의 한 step 삭제에서는 근사 방향 bias가 남는다.

신규 실험 코드는 위 수식에 근거해 NumPy/PyTorch로 직접 구현했으며 외부 저자 코드를 복사하지 않았다. 원 논문 전체의 재현이나 모든 최신 방법과의 우열을 확인한 결과가 아니다. MNIST 파일 출처와 checksum은 [torchvision 공식 dataset 구현](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)을 참고했으며, [다운로더](download_mnist.py)에서 공식 mirror 파일의 MD5를 검증한다.

국내 학회용 주장으로도 'Hessian을 OTA로 합쳤다' 또는 'random projection을 썼다'만으로는 부족하다. 이번에 확보한 근거는 **무선 전력·peak·제어비를 포함하면 송신 전 보정의 이득이 조건부이며, 곡률 준비비와 fading 오차가 결과를 바꾼다**는 분석이다. 강한 신규 알고리즘으로 제출할 준비가 되었다고 판단하지 않는다.
