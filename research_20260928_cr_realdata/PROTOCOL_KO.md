# CR-Air 실제 데이터 검증: 실행 전 고정

2026-09-28. 이전 CR-Air 수식 검산의 후속이다. 이 문서/코드 hash를 실행 결과에 기록하고, 결과를 본 뒤 hyperparameter/privacy 기준을 바꾸지 않는다. 기존 실험 파일은 수정하지 않는다.

## 데이터와 학습

MNIST/FashionMNIST 각 private10,000/test10,000. 3개 새 partition/noise seed202609290/291/292. 각 class1,000개를 private로 추출한다. label-2 non-IID K=10/100/1000 clients: client별1,000/100/10개, 같은 전체 private sample pool. client0 하나를 삭제한다. K가 커질 때 삭제 sample 수도 작아지는 조건이므로 K 증가에 따른 결과를 동일 삭제 규모의 비교로 해석하지 않는다.

공개 고정 random-ReLU encoder: centered pixels(784) ->15개 feature+bias, D=16. projection seed2026092890, N(0,1/784), private 데이터로 encoder를 학습하지 않는다. 강한 pretrained encoder의 평가가 아니다. local sample-mean Gram upper triangle와 feature-label moment를 이어 q=136+160=296 real 통계를 만든다. 전체 client 통계 L2 bound B=1로 clip한다. norm, clipping coefficient, 원래 통계를 서버에 별도 제공하지 않는다. 서버 결정에 테스트/evaluator reference를 쓰지 않는다.

고정 초기 K로 noisy sum을 나누고 noisy Gram의 음의 eigenvalue를0으로 PSD 투영, ridge lambda=.01을 추가한 뒤 solve. FP64 계산 후 FP32 모델 방송을 모사한다. retained 재학습에도 같은 clipping/분모/PSD/정규화를 쓴다. clean clipped retained head와 clean unclipped head는 utility 참고값이며 noisy unlearning reference와 구분한다. private 데이터 없는 zero head(class0 고정)를 공개 기준으로 사용한다.

## 방법과 프라이버시

source aggregate variance sigma^2=8, B=1, client replacement sensitivity2, rho0=.25. 삭제 전후 전체 관측 cap rho_bar=.5 (delta=1e-5, epsilon 상한5.29853). 계수와 모든 scheduling은 공개 channel/bound에만 의존한다.

- Independent: beta=0, 잔존 통계를 독립적으로 noisy AirComp 재집계한다. 같은 final law와 전체 privacy cap 아래의 비교군.
- CR-Fixed: beta=.5. A는 -beta s_A, 잔존은 (1-beta)s_i를 동시 전송한다.
- CR-Power: public bound/channel의 payload 에너지 상한을 최소화하는 beta를 [0,sqrt(.5)]에서 선택. FP64 경계의 수치오차를 피하도록 위쪽 경계는 안쪽으로 한 ULP 이동한다. 총 RF energy의 최적값은 아니다.
- No-op: 기존 noisy source 모델 유지. 삭제 통신비0이며 A의 평균 영향을 유지한다.

학습과 삭제의 실제 client 신호를 channel inversion해 합산하고, 반복 평균의 정확한 Gaussian 등가 noise를 더한다. 원시 반복 송수신 등가성은 기존 theory 검산과 새 math check로 확인한다. 공개 beta, scale 등은 FP64 제어 메시지로 계산한다. 같은 method group의 source/noise draws는 공유하되, 다른 beta의 결과가 독립 재학습과 같은 realization이라고 가정하지 않는다.

최종 통계의 retained-only ideal noisy 재학습에 대한 Gaussian KL을 정확한 mean/variance로 계산한다. 동일 최종 noise를 쓰는 coupled reference와의 수치 차이는 구현 감사이고 통계 검정/독립 proof로 부르지 않는다. 같은 noise로 A를 빈 통계로 바꿨을 때 source 및 final head의 변화를 기록한다. source가 이미 거의 변하지 않는 경우도 공개한다.

CSI 조건에서는 현재 삭제 채널의 실제 gain ratio로 잔존 통계를 새로 합산한 radio reference와의 KL도 별도로 기록한다. Independent는 이 reference와 일치하지만 ideal reference와는 오차가 있을 수 있다. retained 채널 오차와 A의 잔여 영향을 혼동하지 않는다.

한 번의 사전에 정한 client 삭제, honest-but-curious 단일 합산 수신기, 데이터 독립 control, Gaussian thermal noise, static-per-block 채널 가정이다. 모델 분포 일치와 서버가 기억하는 과거 관측의 DP는 다르다. client 신원이나 정보량0을 보장하지 않는다.

## 채널과 반복 수

각 dataset/seed/K/조건별16 channel draws, draw당8 independent noise pairs. utility는 고정 첫2개 noise pair를 전체 test10,000개에 평가한다(조건/method당32개 모델). 모든8개는 model/statistic 감사에 쓴다. 단일 CUDA worker로 순차 실행한다.

조건: unit, iid Rayleigh, 삭제 A만 gain .01, 잔존 client1만 gain .01, Rayleigh+5% log-amplitude CSI 오차. unit/weak 조건의 source는 unit, 삭제 channel만 명시 조건을 적용한다. Rayleigh에서는 source/deletion 채널을 독립 draw한다. CSI 조건은 source 및 deletion 추정오차도 독립이다. weak/CSI는 스트레스 검사이며 현실 채널 분포 추정이 아니다. dropouts/coherence timeout을 생략했으므로 실제 무선 배포 실험은 아니다.

수신 thermal noise variance nu^2=.01, per-client payload vector의 1회 block-energy cap P=1. 이것을 per-symbol20dB SNR이라고 부르지 않는다. 목표 noise에 맞는 scaling과 cap에 맞는 최소 정수 반복 수를 사용한다. deep fade를 제거하거나 clip하지 않는다.

CSI 오차에서 실제 gain ratio g0=h0/h0_hat, g1=h1/h1_hat. final A 계수는 beta(g0_A-g1_A), retain 계수는 beta*g0_i+(1-beta)*g1_i이다. nominal DP를 자동 적용하지 않고, evaluator가 실제 계수로 rho_i=rho0*(g0_i^2+c_i^2*g1_i^2/(1-beta^2))를 계산한다. 실제 source/change 계수 및 conditional Gaussian covariance가 알려졌다는 모형하의 privacy 감사다. 서버가 이 evaluator 정보를 정책 선택에 쓰지는 않는다.

## 비용을 어떻게 셀 것인가

AirComp shared payload q*r real uses와 source 준비를 포함한다. phase마다 활성 client당 UL pilot8 real, 공통 DL pilot8 real, reliable control480bit(요청/지시+FP64 beta/scale/variance+정수 반복), client별 CSI feedback64bit, 최종 FP32 head broadcast16*10*32bit를 포함한다. control은 별도 안정된 링크에서 rate=.5log2(101)bit/real, CP factor1.125라고 가정한다. 실제 control outage/retry는 모델링하지 않는다. channel-adaptive method는 A의 pilot/feedback도 부담한다.

payload 송신 energy는 실제 clipped 통계와 inversion 신호에서 계산하며 public bound도 비교한다. 별도 normalized accounted TX energy는 payload+UL pilots+DL pilot/control/CSI/head 전송 에너지를 센다: pilot/DL energy per real use=1, CP 포함. **실제 RF joule이 아니며 circuit/PA efficiency/수신 energy는 미측정**이다. pilot energy per real use0/.001/.01/1 민감도와 payload 절감이 허용하는 추가 overhead energy(손익분기)를 기록한다. 이미 측정된 CSI 재사용을 공짜 이득으로 주 비교에 넣지 않는다. 전송량과 에너지를 서로 바꾸어 부르지 않는다.

초기 encoder 규칙은 사전 설치된 공개 seed이고 encoder weight 방송비를 생략한다. source 준비, 삭제, lifecycle은 분리한다. 통계 계산 sample 수/대략적 dense MAC, client/server cache byte와 source/deletion head solve 수를 공개한다. 실제 저장/전력/지연 측정은 아니다.

## 사전 판정 및 한계

Primary feasibility는 K1000, Rayleigh 정확 CSI, 두 dataset. fixed 및 power 방법에 대해 (1) exact-CSI mean KL<=1e-18, coupled model 상대오차<=1e-9, rho<=.5+1e-10; (2) 3seed 평균 test accuracy가 같은 clipped clean retained head보다5%p 이내이고 public-only보다10%p 이상 높음; (3) Independent 대비 counted normalized TX energy5% 이상 절감, 총 삭제 real uses가5% 넘게 증가하지 않는지 본다. 세 조건이 함께 통과해야 이번 설정의 실용 후보로 채택한다. payload-only 개선이나 같은 noisy distribution만으로 채택하지 않는다.

K10/100, unit/weak/CSI를 모두 공개한다. CSI 실패는 정확 CSI 정리 실패와 구분하며 강건한 privacy/unlearning을 주장하지 않는다. accuracy는 seed평균 및 seed SD, 분포 내 noise variation을 보고하며 seed3개만으로 광범위 일반화를 하지 않는다. zero-head보다 좋아도 높은 성능의 task model이라는 뜻은 아니다. Gaussian oracle binary attack AUC(A의 실제 통계 대 빈 통계, 서버에 유리한 알려진 두 가설)를 계산해 DP가 완전 은닉이 아님도 명시한다.

원시 데이터, client 통계, 모델 checkpoint는 저장소에 업로드하지 않는다. 공개할 것은 코드, 집계 지표, data/source/protocol hash, 실행 정보, 비용과 실패 기록이다. 실제 데이터 결과를 보고 추가 encoder/epsilon/lambda/K를 탐색하지 않는다.
