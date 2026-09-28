# Cohort-Diversity DS-Air: GPU 실행 전 고정

2026-09-28. 기존 `research_20260928_ds_revision`의 deep-fade 실패를 보완하는 별도 실험이다. 기존 파일은 수정하지 않는다. 목표는 잔존 클라이언트 누락 없이 공중 합산의 채널 역변환 오차를 줄이는 것이다. 전체 transcript 프라이버시나 대형 신경망 언러닝까지 해결했다고 주장하지 않는다.

## 범위와 방법

기존과 동일한 공개 random ReLU encoder/ridge, D64, lambda=.01, FashionMNIST/MNIST private10000/test10000, 10개 label-2 client 중 0 삭제. 새 partition seed202609284/285/286, 조건당 독립 channel/H draw32개, correction noise16개. utility는 매 draw 고정 첫1개의 전체 test/forget 결과. 조건별 표본 수를 늘리고 마지막 결과에 맞춰 설정을 바꾸지 않는다.

기존 Original(서버 inverse), V1(송신 전 inverse), SubspaceV1(r48,동일예산)을 새 seed에서 재실행한다. FD8-Original/FD8-V1/FD8-Subspace는 client를 선택하지 않고, 8개 공유 주파수 자원 중 min_i h_hat_i가 가장 큰 자원 하나를 선택한다. 선택된 자원에서는 잔존9개가 모두 같은 시간·주파수에서 합산한다. 개별 client 직교 전송이 아니다. FD2-V1은 후보2개의 민감도 비교다.

CG-V1/CG-Subspace는 시간 대기 대조군이다. 현재 및 이후의 coherence window를 인과적으로 탐색하여 min_i h_hat_i>=.4인 첫 window에서 전원 합산한다. 최대8번의 pilot 시도 후에도 실패하면 모델을 갱신하지 않고 deletion_pending으로 반환한다. 이전 window로 돌아가거나 미래 채널을 미리 아는 oracle은 사용하지 않는다. timeout도 전체평가에 포함하며 E=1이다. 성공조건만 따로 골라 성공률로 보고하지 않는다.

## 채널과 비용

주 조건은 두 데이터셋 각각20dB independent Rayleigh, 정확한 CSI. 보조 조건은30dB iid, 후보 간 complex 상관계수.9, 후보가 완전히 같은 정적 채널, 동일한 iid 채널에5% log-amplitude CSI 오차, client1만 amplitude.2로 감쇠되는 heterogeneous pathloss이다. 후보bank는 g_l=sqrt(c)g_common+sqrt(1-c)g_ind,l; 모든 client/band의 complex innovation은 CN(0,1). CSI 오차는 h_hat=h exp(.05z); main/CSI 조건의 실제 채널 및 AWGN seed는 공유한다. 시간 대기 대조군의 상관은 동일 bank의 equicorrelation stress model이며 Doppler/AR 채널의 재현이 아니다.

주파수 후보는 같은 시점에 이용 가능한 coherence-bandwidth 이상 떨어진 자원으로 가정한다. 전체 삭제 세션 동안 모든 후보의 채널이 고정된다. 이 가정과 band availability가 성립하지 않으면 FD 성과를 적용할 수 없다. c=1 조건도 포함하여 독립 diversity가 없으면 이득이 없어지는지 확인한다.

평균전력<=1, peak amplitude²<=4, FP32 upward bound, H32회, equal80-real blocks, 동일 정수 반복 배정을 유지한다. full-H Original 예산을 공통 active-airtime 예산으로 사용한다. 각 방식의 추가 탐색/제어 비용을 먼저 차감하고 남은 예산에서 보정 반복을 배정한다. 모든 방법의 이미 알려진 수신 channel을 이용한다는 가정을 동일하게 맞추기 위해 baseline에도 첫9개 channel gain FP32 feedback을 추가한다.

기존 ledger는 main payload phase의 pilot/control/basis/eigen/final-model/CP를 포함한다. 추가 탐색은 후보 또는 시간 시도마다 pilot9x8real, channel-feedback9x32bit, candidate list/설정64bit, 선택 결과3bit를 사용한다. 기존 main pilot과 중복되는 첫 탐색 pilot은 한번만 계산한다. baseline은 첫 CSI feedback288bit를 지불한다. FD2/FD8의 나머지 후보 pilot과 모든 gain feedback/control은 추가한다. 새 예산은 기존Original 예산+baseline CSI feedback 비용이다. CP1.125, digital rate=.5log2(1+SNR), 오류 없는 digital control 가정.

active airtime과 latency를 구분한다. FD는 모든 탐색이 선택 전에 직렬로 수행된다는 보수적 active-time 계산을 쓰며 추가 coherence wait는0이다. CG는 실패한 시도마다 한 coherence window를 기다린다. window 길이는 공통 active budget 이상인 최소 block-static 길이로 두고, sensitivity로 그1/4/16배를 보고한다. latency=(attempts-1)*window+마지막 시도 이후 실제 통신비, timeout은8*window. 실제 RF 초/주파수 대역폭을 측정한 것이 아니다. 시간 대기를 무료 통신 이득으로 주장하지 않는다.

## 수학과 평가

독립 Rayleigh에서는 m_l=min_i |h_il|² ~ Exp(K). FD-L의 M=max_l m_l의 CDF는 (1-exp(-Kx))^L이다. L=1의 E[1/M]은 발산하지만 L>=2는 유한하다. 이 성질, 실제 MAC의 전원 포함 및 전력제약, noisy-H 조건부 bias+variance, 채널선택이 payload/재학습 reference를 참조하지 않는 것을 먼저 검사한다. CG의 정확CSI iid 성공확률은 p=exp(-K*.4²), 최대8시도 완수확률1-(1-p)^8이다. timeout과 static-channel 실패를 포함한다.

GPU 이전 math 검사는 seed/구현/protocol hash를 저장한다. 결과는 parameter MSE/초기삭제gap, bias/variance, E<1 session 비율, test accuracy, forget JS, timeout/참여client수, 실제 탐색비·active비·latency·source준비비, H noise scale, 공격 복원오차, seed SD를 기록한다. 실패 timeout의 출력은 W0이지만 삭제 성공으로 계산하지 않는다. exact retained model은 evaluator 전용이다.

사전 채택 기준: FD8-V1 또는 FD8-Subspace가 두 dataset의 iid20 주 조건에서 각각 평균E<1, 모든3seed E<1, 같은 representation baseline 대비평균E30% 이상개선, retained test accuracy하락1%p 이하, active예산위반0. 조건을 충족해도 fading/pathloss/CSI 전체의 보장이나 새 알고리즘의 독창성 증명으로 해석하지 않는다. 다른논문의 재전송/자원선택은 관련 선행이며 새 contribution 주장은 별도 구분한다.
