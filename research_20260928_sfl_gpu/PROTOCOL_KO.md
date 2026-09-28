# 폭·채널 분할 AirComp-SFL: 수학과 고정 실험 조건

작성일 2026-09-28. 언러닝은 이번 실험의 요구조건이 아니다. 공식 논문 코드를 재현한 결과가 아니라, 명시한 구조를 직접 구현한 GPU feasibility 실험이다. 결과는 README_KO.md와 raw JSON을 함께 확인한다.

## 1. 무엇을 합산할 수 있는가

같은 샘플 j를 처리하는 K개 장치의 채널 블록을 h_k=f_k(x_j) 또는 h_k=f_k(x_{j,k})라 하자. 전자는 동일 입력의 channel/width 분할, 후자는 같은 사건의 multi-view 입력이다. cohort가 여러 개이면 동일 역할 모델을 cohort 사이에서 FedAvg한다.

서버의 첫 선형층은 연결된 feature에 대해

$$s=[W_1\ \cdots\ W_K][h_1^T\ \cdots\ h_K^T]^T+b=\sum_k W_kh_k+b.$$

공통 q차원 기저 B와 장치별 A_k를 **처음부터 모델 구조로 정의**하면

$$W_k=BA_k,\quad u_k=A_kh_k,\quad s=B\sum_k u_k+b.$$

장치별 feature 차원 c_k와 CNN 폭이 달라도 u_k의 차원 q와 샘플 순서가 같으면 합산된다. g=∂ℓ/∂s일 때 서버가 d=B^Tg 하나를 broadcast하면

$$\nabla_{h_k}\ell=A_k^Td,\quad\nabla_{A_k}\ell=dh_k^T.$$

따라서 q차원의 공통 backward 신호로 각각 다른 폭의 branch를 학습시킬 수 있다. 비선형 활성화는 합산 **후**에 적용한다. 일반적으로 φ(Σu_k)≠Σφ(u_k)이므로 임의의 CNN 층을 그대로 대체할 수 없다. q<64는 rank 제약이다. 임의의 기존 full-rank W와 동등하다고 주장하지 않는다. 위 합산과 채널 분할 자체는 선행 기술이다.

서로 다른 이미지 x_A,x_B의 activation을 같은 위치에 합치면 일반 horizontal SFL의 minibatch loss가 보존되지 않는다. 이번 검산에서 두 독립 샘플 loss 0.002476이 잘못 합산하면 0.693147이었다. 따라서 일반 FL의 독립 데이터를 이 구조로 처리한다는 주장을 제외한다.

## 2. 실제 OTA 신호 모델

한 batch의 u_k∈R^(Bq)를 실수 두 개씩 complex symbol로 묶는다. 수신 채널 h_k, 추정 채널 ĥ_k, complex 잡음 n~CN(0,σ²I), 장치별 평균 symbol 전력 P=1이다.

$$\alpha=\min_k {|\hat h_k|\over\sqrt{\operatorname{mean}|u_k^{(c)}|^2}},\quad x_k={\alpha u_k^{(c)}\over\hat h_k},\quad \hat z={\sum_k h_kx_k+n\over\alpha}.$$

정확한 CSI이면 원하는 합+잡음이다. 각 장치 평균 전력은 1 이하이며 장치마다 서로 다른 직교 payload를 보내지 않는다. Pilot/scale 제어는 digital이며 비용에 포함된다. Norm 보고는 추가 정보를 드러내므로 합산 관측만으로 privacy나 DP를 보장하지 않는다. 순간 peak 전력, 실제 RF 동기화 오차는 보장하지 않는다.

## 3. 이번에 시험한 차별화 후보: OTA-Stein

alpha는 activation에 의존한다. 정확한 CSI에서 n=Bq로 쓰면 real-coordinate 수신 잡음 분산은

$$v(u)=\sigma^2\max_k{\|u_k\|^2\over n|h_k|^2},\quad z=\sum_k u_k+\sqrt{v(u)}\epsilon,\quad \epsilon\sim N(0,I).$$

단순 backward d=∂ℓ/∂z만 전달하면 잡음 분산의 변화가 빠진다. 고정 채널, 유일한 bottleneck b, smooth suffix, 적분/미분 교환 가능 조건에서 Gaussian Price identity로

$$\nabla_{u_k}\mathbb E\ell(z)=\mathbb E[d]+{\bf1}_{k=b}{\sigma^2\over n|h_k|^2}\mathbb E[\operatorname{tr}H_z\ell]\,u_k.$$

유도: ∂Eℓ/∂v=(1/2)E trH, ∂v/∂u_b=2σ²u_b/(n|h_b|²). 이는 **잡음이 있는 기대 손실**의 기울기이고 clean loss를 복원하거나 잡음을 제거하는 식이 아니다. Gaussian 미분 항등식 자체도 새 이론이 아니다.

서버는 수신 합산값과 자신의 suffix로 Rademacher probe r 하나를 뽑아 t=r^THr을 구한다. 전체 Hessian 행렬은 만들지 않으며 HVP 1회가 필요하다. 서버는 공통 d 외에 t(float32)와 b(2bit)를 전송한다. b장치는 자신의 u_b로 σ²t u_b/(n|h_b|²)를 더한다. 실제 RF 잡음 realization이나 개별 activation을 서버에 올릴 필요가 없다. 기존 scale 보고로 bottleneck을 선택할 수 있다는 조건이다.

비교군 **OTA-Reg**는 t=1을 고정한 동일 형식의 단순 channel-scaled L2 gradient다. 이 간단한 방법을 못 이기면 HVP 후보를 논문 기여로 채택하지 않는다. 예비 실험에서 이 비교가 필요함을 확인한 후, 시드 311–313은 고정 설정으로 수행했다. 세 시드 이후 hyperparameter를 바꿔 결과를 선별하지 않는다.

Fashion20 core 결과를 본 후 **Latent4/8-Reg** 진단 대조군을 추가했다. 같은 t=1 항을 digital split learning에 적용한다. 이 방법에는 activation-dependent 무선 noise가 없으므로 이론상 누락 gradient 보정이 아니라 regularization이다. Winner 계산에 필요한 K개 norm 보고도 비용에 포함한다. 이 사후 진단으로 단순 정규화 효과를 OTA 고유 효과로 오인하는지 확인하며, 사전 등록 실험이라고 표현하지 않는다.

주의: 정확한 기울기 식은 continuous backward와 exact CSI의 기대값 정리다. 실험의 8-bit backward, gradient clipping, Adam, CSI stress에서 그대로 unbiased하다고 주장하지 않는다. Suffix는 모든 비교군에서 SiLU이다. ReLU의 경계 기여를 a.e. Hessian만으로 처리했다고 주장하지 않는다. Tie에서는 max의 subgradient 처리가 필요하다.

## 4. 데이터·학습

- RTX 3080 / PyTorch 2.11.0+cu128, 단일 GPU 순차 실행, deterministic 옵션.
- Fashion-MNIST 원본에서 클래스별 층화 추출: train 12,000 / validation 2,000 / test 3,000. 각 시드의 원본·index hash를 JSON에 저장.
- 네 cohort, cohort당 네 sensor. Train을 label 순서 12개 shard로 나눠 cohort당 3개씩 배정하는 non-IID 조건. Cohort sample 수는 모두 3,000.
- Multi-view 실험: 같은 이미지 네 quadrant를 14×14로 나눔. Depth는 고정하고 첫 conv 폭 w∈{4,8,12}, 다음 conv 폭 2w. q∈{16,64}. HET는 [4,8,12,4].
- 실제 channel 분할 추가 실험: 같은 28×28 전체 이미지를 각 장치가 서로 다른 branch 채널로 처리. 두 번째 conv는 branch 내부 채널만 연결하는 grouped CNN이다. 모든 채널이 연결된 임의의 CNN과 동등하지 않다. Raw image의 8-bit digital broadcast 비용도 포함.
- Encoder: Conv3×3→ReLU→MaxPool2→Conv3×3→ReLU. 각 feature를 A_k로 투영하고 합산. 서버 B→bias→SiLU→Linear(64,32)→SiLU→Linear(32,10).
- 80 federated rounds, cohort당 round별 local 5 step, batch64, Adam lr0.001, gradient norm clip5. Round마다 모든 encoder/투영/head 파라미터 FedAvg. Cohort Adam moments는 유지. Server head replica는 같은 서버 안에서 합쳐 RF 비용 없음.
- Model/indices/채널 난수는 구조가 같은 비교군끼리 paired. 최종 round80 모델을 평가하며 test 기반 checkpoint 선택 없음. Validation은 10round마다 확인.
- Final test에서 독립 채널 draw 3회 평균. 통계 표준편차는 draw9개를 독립 시드로 부풀리지 않고 **학습 시드3개의 sample SD**.

## 5. 비교군과 무선 조건

- Feature8: 원래 cut feature 8bit와 feature gradient 8bit를 개별 전송. Projection은 서버.
- Latent8/4: 동일 q16 구조의 로컬 projection, latent 8/4bit 개별 UL, **공통 q16 gradient 8bit broadcast**. Digital baseline에 불필요한 K배 DL을 부과하지 않는다. SplitFC 공식 재현은 아니다.
- OTA-Q16/Q64: 위 analog 합산, 같은 digital backward. OTA-Stein/Reg는 동일 구조의 추가 보정.
- Ideal: 무선/양자화가 없는 진단. Radio 비용 0은 무선 방식 간 순위 비교에 쓰지 않는다.
- 학습: Rician K-factor5, path powers [1,.8,.6,.4], SNR P/σ²=20dB 또는10dB. Batch별 block fading.
- Stress: Rician10dB, Rayleigh20dB, CSI amplitude error0.05 (평균 NMSE0.0025). Train20에서 test10은 distribution shift이며 train10/test10과 별도로 표시.

## 6. 비용 장부

Complex channel use, 1MHz, CP factor1.125, P=1W로 시간을 환산한다. Digital rate는 min(6,log2(1+SNR|h_k|²)) bit/complex use. 오류 없는 이상적 coding을 가정한 유리한 digital 비교군이며 packet 오류·재전송은 구현하지 않는다. DL은 같은 gain, broadcast는 최악 링크 속도.

각 local step의 UL payload는 OTA Bq/2 uses, digital latent Σ_k(Bq·bits+32)/R_k, feature Σ_k(Bc_k·8+32)/R_k. 공통 gradient DL은 (Bq·8+32)/minR. Feature 방식의 DL은 각 c_k 크기. Batch IDs 14bit/sample, label4bit/sample, headers, scale reports, trace34bit, pilot을 별도 합산한다.

초기 client model 배포, 매 round cohort별 client model32bit UL과 역할별 DL broadcast, sync pilot/header를 포함한다. 데이터 source 자체의 사전 준비 비용은 multi-view에서는 제외하고, same-input channel 실험은 매 batch raw-input broadcast를 추가한다. 이 선택이 결과에 미치는 영향도 따로 제시한다.

`modeled_radio_seconds`는 **설정된 무선 모델의 직렬 총 점유시간**이다. GPU wall은 4cohort를 한 GPU에서 순차 시뮬레이션한 학습+validation 실행시간이며 실제 분산 장치 latency가 아니다. 두 값을 더해 실측 end-to-end라 하지 않는다. Client MAC은 forward conv/linear 곱셈누적 수, activation/메모리 접근/optimizer/backward를 포함한 실측 FLOP가 아니다. Trace 계산은 CUDA event로 별도 기록하되 Python/통신 실시간 비용을 대표하지 않는다.

Power ledger는 UL/DL payload/control/pilot별 송신 에너지를 합산한다. RF 회로·CPU 에너지, 외부 학습 데이터 수집, PAPR 제약은 제외한다. 시드별 전체 원장과 비용 구성 항목을 숨기지 않고 저장한다.
