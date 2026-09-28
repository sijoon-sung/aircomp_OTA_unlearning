# DS-Air 보완 검증: 실행 전 고정

2026-09-28. 기존 결과를 덮어쓰지 않는 별도 실험이다. 사용자 요청에 따라 전처리 선택, 추가 데이터/모델 크기/채널 조건, 가까운 Newton 비교군, 총비용, 정보 복원 한계를 검증한다. 논문 채택이나 DP를 보장하는 실험이 아니다.

## 방법과 평가 분리

- Original: 공통 곡률 기저에서 residual을 합산한 다음 서버가 inverse 적용.
- V1: 같은 공통 inverse를 송신 전에 적용. Original과 같은 곡률 관측.
- Switch: 블록마다 두 방식의 실제 power-bound scalar를 수집하고, 최종 보정 잡음 MSE가 작은 방식을 선택. 두 후보 scalar 수집/모드 방송/고유값 방송을 비용에 포함. 평가 reference는 선택에 쓰지 않는다.
- LocalNewton: Yang et al., arXiv:2203.15488 식(6)-(8)의 local Newton 방향 합산을 ridge 삭제 한 step에 적용. 저자 전체 학습/beamforming 알고리즘 재현은 아니다. 각자의 inverse와 공통 inverse의 차이를 비교.
- Diagonal: 잔존 Hessian diagonal을 OTA 수집하고 공통 diagonal inverse로 residual 전송. dense curvature 비용 회피의 근사오차를 포함.
- StatsRetrain: 같은 noisy full Hessian과 잔존 C를 OTA 합산해 ridge를 다시 풂. 총예산을 보정/C 전송에 최대한 배정.
- SubspaceV1: 사전 공개 고정 U(D x 3D/4)에서 곡률과 residual만 합산. full-H 프로토콜과 관측을 합치지 않는다. 동일 전체 예산 안의 최대 반복 방식과 적은 비용의 nominal 방식 모두 기록한다.
- No-op, exact retained ridge는 평가 reference. 실제 통신 경쟁자가 아니다.

독립 함수들로 math -> GPU main -> verify 순서로 진행한다. 단일 CUDA worker, 결과를 보고 protocol이나 주 설정을 바꾸지 않는다. 실패/비실행 설정도 기록한다.

## 데이터와 크기

FashionMNIST와 MNIST, 각각 private10,000/test10,000. 10 label-2 clients(각1,000개), client0 전체 삭제. 새 partition seed202609281/282/283. 공개 random ReLU encoder를 고정하고, bias를 포함한 head 입력 D=32/64/128, 출력10개, ridge lambda=.01. encoder는 private data로 학습하지 않는다. 모델 폭 D-1개 feature+bias이므로 과거65차원 결과의 byte 재현이 아니다.

source W0는 전체 충분통계의 이상적 집계로 정확히 학습 후 FP32 배포. retained reference는 동일 규칙의 ridge 최적해. source 통신 준비비는 별도 ledger에 포함하되 noisy source 학습까지 했다고 주장하지 않는다. 공개 U는 데이터/삭제 client와 무관한 고정 seed로 정한다. 테스트는 선택에 사용하지 않는다.

## 채널 및 반복

SNR10/20/30dB; unit-gain AWGN, 독립 Rayleigh magnitude+정확한 위상 보상, Rayleigh+5% log-amplitude CSI 오차의 3조건. Rayleigh |h|=sqrt((a^2+b^2)/2); deep fade client를 탈락시키거나 gain을 잘라내지 않는다. CSI 오차는 h_hat=h exp(.05 z)이며 위상/CFO 오차는 이번 범위 밖. payload당 tx=z/(scale*h_hat). 평균전력<=1, peak amplitude²<=4. receive는 실제 h로 합산한다. 각 삭제 세션의 h/CSI는 모든 준비·보정 반복 동안 고정이라는 block-static 가정. 실제 RF 지연/도청/채널 조작 검증은 아니다.

각 seed/설정에서 독립 H/channel draw4개, 각 draw correction AWGN32개; utility는 고정 첫2개로 전체 test/forget 평가. 반복평균은 Gaussian 등가식으로 main을 실행하고 math 단계에서 실제 반복 송수신과 covariance를 독립 확인한다. Q/eigen/model은 FP32 roundtrip; Q는 공통 QR로 재직교화. power scalar는 FP32 upward rounding 후 전송하므로 전력 위반 방지. control 전송은 오류 없는 digital 비용 모형이며 실제 패킷 오류는 모사하지 않는다.

행당 8x10=80개 실수의 동일 길이 block. full-H upper triangle32회, correction 총8B회인 Original의 총 삭제비를 각 D/SNR의 예산으로 고정한다. V1/Switch/baselines는 각자 metadata와 basis 비용을 먼저 차감하고 남은 예산 안에서 정수 반복을 배정한다. Switch의 추가 정보는 공짜가 아니다. 잔여80symbol 미만은 사용하지 않는다. greedy는 동일 길이 block의 sum(c_j/r_j)를 최소화한다.

Subspace nominal은 H32+총8B회만 사용; matched는 full Original 총예산 이내 최대 반복. 별도 곡률 감도는 D64,20dB,정확CSI Rayleigh, 두 dataset, 같은3seed에서 H8/32/128을 비교한다. 이 감도에서 예산은 해당 H 반복수에 맞게 다시 계산하며 H를 더 수집한 결과를 공짜 개선으로 보고하지 않는다.

## 수학·정보 감사

조건부 최종 오차는 mean bias + channel variance로 분해한다. Switch 선택은 그 중 전송 잡음 항만 최소화하므로 H/CSI bias를 없앤다고 주장하지 않는다. 단일 좌표/등방 P에서는 전처리 scaling 이득이 상쇄됨을 확인한다. 서로 다른 payload/곡률에서 pre가 좋은 경우와 나쁜 경우를 모두 수학 예제로 보존한다.

full transcript의 정상점 gradient 복원을 다시 확인한다. Subspace에서는 U-null 방향으로 각 client 선형 통계를 반대 변경해 source, projected H/C, power metadata, correction 관측이 동일하지만 A의 full gradient가 다른 예를 구성한다. 이는 해당 관측 모형의 비식별성 예시이지 모든 보조정보/원본 분류 데이터에 대한 DP 증명이 아니다. full gradient와 projected gradient의 복원오차를 별도 보고한다. power scalar도 공개 정보이며 회전 U를 반복 질의하는 프로토콜은 허용하지 않는다.

## 비용과 판정

real channel-use equivalent, CP1.125, digital rate=.5log2(1+SNR). 요청/지시256bit, phase마다 client별scale32bit와 pilot8real, block별repeat32/scale32bit, Q/eigen/U seed, 선택 mode, 최종모델 broadcast를 모두 계산. source cached/cold와 초기준비+삭제를 구분. 스케줄러가 receiver에서 이용하는 것은 수신통계/metadata뿐이며 원래 local payload를 비용 산정 외의 의사결정에 직접 참조하지 않는다.

Primary: FashionMNIST,D64,20dB,Rayleigh exact CSI. secondary는 전체 grid를 빠짐없이 보고. Switch 실용 탐색 기준은 거의 같은 총비용(<=Original budget)에서 Original와 V1 중 더 좋은 비교군보다 seed평균 MSE5% 이상 감소,3seed 모두개선,test accuracy가 retained reference보다1%p 넘게 하락하지 않는 조건. 이 조건이 실패하면 새로운 강한 개선안으로 채택하지 않는다. Subspace nominal은 full Original보다 총비용20% 이상 감소하며 MSE가 Original의1.05배 이하인지 보고하되, full-gradient 비복원과 DP는 별개다.

최종 parameter MSE, mean-bias/variance, test accuracy, forget prediction JS, seed SD, paired comparator, 총비용, peak VRAM, GPU time, 최대 전력, 원자료/hash를 기록한다. 조건부 수식과 MC 차이를 검증한다. 개별 data/checkpoint는 저장소에 업로드하지 않는다.
