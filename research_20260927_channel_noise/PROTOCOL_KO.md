**채널 잡음 활용 GPU 검증 — 실행 전 고정, 2026-09-27**

주 후보는 NM-Air(Noise-Matched AirComp Unlearning)다. 채널 잡음을 사전에 정한 randomized retraining의 출력 잡음으로 맞춘다. 기존 결정론적 재학습과의 오차도 함께 보고한다. 확률적 기준을 사용했다는 이유로 기존 point-reference의 성능이 개선되었다고 하지 않는다. 보조 비교는 JS-Air: 알려진 수신 잡음 분산으로 block별 positive-part James–Stein shrinkage를 적용한다. 기존 통계 기법의 적용이며 새로운 shrinkage 정리를 주장하지 않는다.

FashionMNIST 10,000개, 10 label-2 clients, client0 전체 삭제. 공개 random ReLU 784→64+bias, ridge lambda=.01. 새 분할 seed 202609271/272/273, encoder seed20260922. Test10,000개는 평가에만 사용한다. GPU worker 하나, FP64 계산, 전송되는 공통 basis/eigenvalue와 배포 모델은 FP32 roundtrip. Basis는 FP32 수신 후 같은 QR로 직교화한다. 이 변경으로 과거 DS 수치와 byte-identical 재현을 주장하지 않는다.

SNR10/20/30dB, 평균전력1/peak amplitude²4, 정확한CSI, h_i=1. 공통 Hessian upper triangle 2145좌표 OTA32회, seed당 독립 Hessian 잡음8개. 각 조건에서 payload 잡음128개로 parameter MSE를 평가하고 그중 사전 고정 첫4개로 test accuracy와 forget JS를 측정한다. 별도 무잡음 Hessian oracle을 동일 payload 구조로 검사하되 실제 무선 성과로 해석하지 않는다. 데이터·Hessian·payload 난수는 분리하며 paired noise 비교를 사용한다.

출력 perturbation 표준편차 tau=.001/.003/.01. 주 설정은 20dB, tau=.003, OTA Hessian32이다. A_tau(D)=ridge(D)+tau Z를 초기 학습과 retained reference 양쪽에 사용한다. 초기 공개 모델도 이 규칙으로 생성한 뒤 FP32로 배포한다. Tau를 test나 exact reference로 선택하지 않는다. Source 학습의 이상적 충분통계 집계는 기존과 같은 전제이고 source 준비비를 별도로 계산한다. 외부 기준 모델과 evaluator는 resource/stopping 결정에 쓰지 않는다.

각 correction은13×10=130차원 block5개. 잔존 client payload z_ij=(1/9)Lambda^-1 Q^T(H_i W0-C_i)의 해당 block이다. 최소 수신scale a_min,j=max_i{RMS(z_ij)/|h_i|, maxabs(z_ij)/(2|h_i|)}. 자연 잡음 분산 v_min,j=a_min,j²/SNR이다.

- NM-Air: r_j=max(1,ceil(v_min,j/tau²)), a_j=tau sqrt(r_j SNR). 실제 payload를 x_ij=z_ij/(a_j h_i)로 보낸다. r_j개 수신을 평균하면 잡음 분산이 tau²다. 추가 software noise 없음.
- Uniform-match: 모든 block에 max_j r_j를 배정하고 scale을 맞춘다. 같은 출력분포를 만드는 균등 반복 비교군.
- Aware-fixed40: 기존 MSE greedy 총40회 배정과 r_min을 componentwise max로 취한다. a_min으로 송신하고 부족한 분산만 software Gaussian으로 채운다. 같은 출력분포의 강한 비교군이다. NM-Air가 이 비교군의 삭제 품질보다 좋다고 주장할 수 없으며 비용을 비교한다.
- DS-fixed40: 같은 공통 곡률, MSE greedy 총40회, a_min, software noise 없음. 목표 randomized reference에 대한 covariance mismatch를 측정한다.
- Naive-add40: DS-fixed40 수신 후 tau Gaussian을 그대로 추가한다. 이미 존재하는 채널 잡음을 중복 계산하는 ablation.
- Noise-only: 삭제 보정 없이 W0+tau Z. No-op과 함께 잡음만으로 삭제가 되는지 확인한다.

보조 point-reference 실험은 deterministic W0, correction budget5/10/20/40에서 raw DS와 JS-Air를 비교한다. JS-Air는 수신 y_j에 max(0,1-(130-2)v_j/||y_j||²)를 곱한다. 여기에는 Gaussian-reference KL을 적용하지 않는다. 새로운 CNN/SFL/recovery 학습으로 범위를 넓히지 않는다.

주 평가: conditional Gaussian KL의 평균(주변 분포 KL의 상계), 같은 covariance일 때 conditional TV의 평균(주변 TV의 상계), 정규화 point MSE, forget JS, test accuracy. Gaussian 법칙/수식은 FP32 배포 전 기준이며 동일 quantization 이후 divergence는 증가하지 않는다. 일반 covariance 비교의 TV는 Pinsker 상계를 쓴다. Conditional upper bound가 큰 경우 실제 주변 divergence가 크다고 역으로 단정하지 않는다. 유한 seed가 전체 데이터에 대한 certified unlearning을 증명하지 않는다.

통신 단위는 real-use equivalent, R=.5log2(1+SNR), CP1.125. H32·basis65²×32bit·eigen65×32bit·block별 반복/scale32bit·client별 power metadata32bit·pilot8real/client/phase·최종 모델650×32bit·삭제요청128bit·지시128bit·tau32bit를 모두 센다. Source가 cache된 조건과 cold source 재배포650×32bit를 구분한다. 한 phase의 pilot은 반복 구간을 포괄하는 coherence 가정이다. Fixed/variable allocation 모두 같은 ledger를 사용한다. 시뮬레이션 시간·계산량·tensor bytes와 실제 RF 지연/Joule를 구분한다.

독립 검증: quadratic 삭제 항등식, 채널 합산/잡음 covariance, 정수 반복 최소성, 평균/peak 전력, Gaussian KL 공식, 잡음만 추가할 때 기대 MSE 증가, JS의 정확한 Gaussian mean 문제와 noisy-H 한계, noise variance ±20% 오보정 감도. 수학 검증을 통과한 후 GPU 데이터 실험을 실행한다. 결과가 나쁜 설정도 전부 남긴다.

탐색 판정은 실행 전에 고정한다. NM-Air의 Aware-fixed40 대비 비용 절감과 분포 동등성은 별개 항목으로 보고한다. 실용 후보 기준은 총 삭제비용5% 이상 감소, 평균 conditional TV 상계<=.1, randomized retrain 대비 test accuracy 하락<=1%p를 동시에 충족하는지다. 주 설정 실패를 다른 tau의 사후 선택으로 성공 처리하지 않는다. Point-reference 개선은 같은 비용의 raw와 shrinkage 비교 및 Pareto frontier로 별도 표시한다.

프라이버시 범위: 최종 배포 모델의 분포만 다룬다. Server가 보유한 W0, Hessian, power metadata, 중간 수신과 반복 관측 전체에 대한 DP/삭제는 보장하지 않는다. A의 gradient 역산 문제를 채널 잡음만으로 해결했다고 하지 않는다. CSI 오차, 수신기 조작, 도청자, 반송파 오차는 이번 검증 대상이 아니다.
