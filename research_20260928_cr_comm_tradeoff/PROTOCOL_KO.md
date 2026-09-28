# CR-Air: 통신 중심의 프라이버시 완화 실험 — 실행 전 고정

2026-09-28. 사용자가 통신 효과를 주로 평가하고 client-level 프라이버시 강도를 완화하도록 허용한 후속 실험이다. 이전 엄격한 설정과 결과를 덮어쓰지 않는다. 이번 판단의 주 지표는 **같은 privacy/final noise law에서 삭제 전송량 및 환산 지연**이다. 모든 에너지 항목의5% 절감을 의무화했던 이전 채택 기준을 적용하지 않는다. 에너지 결과도 별도로 공개한다.

## 고정 설정과 프라이버시 범위

이전 real-data 실험의 공개16차원 random-ReLU feature head, local sample-mean Gram/moment296차원, client L2 bound B=1, ridge lambda=.01, PSD projection/고정 초기K 정규화/FP32 head 방송을 유지한다. MNIST/FashionMNIST 각 private10,000/test10,000, label-2 clients K10/100/1000, client0 삭제. 새 seed202609293/294/295. K에 따라 삭제 sample은1000/100/10개로 달라진다. 특징 추출기와 classifier를 결과를 보고 바꾸지 않는다.

전체 학습+삭제 관측의 client-level replacement zCDP cap rho_bar=[.5,1,2,4,8], delta=1e-5. 공통 초기/최종 aggregate noise variance sigma^2=4/rho_bar=[8,4,2,1,.5]. epsilon 상한은 rho_bar+2sqrt(rho_bar log(1/delta)), 약[5.30,7.79,11.60,17.57,27.19]다. 큰 epsilon은 상당히 약한 보호이며 전체 구간을 '조금 완화' 또는 강한 privacy로 부르지 않는다. record-level privacy로 바꾸어 수치를 좋게 보이지 않는다. 실제 데이터는 공개 benchmark이고 이번 실험이 사용자의 private 데이터 보호 설정을 변경하지 않는다.

서버가 실제 A 통계/빈 통계의 두 가설을 알고 있다는 Gaussian oracle binary AUC를 같이 계산한다. 이는 실제 membership-inference 학습 공격이나 DP의 대체 증명이 아니다. privacy 상한과 출력분포에 대한 기존 가정(honest-but-curious 단일 sum receiver, 공개 고정 삭제대상1명, 데이터 독립 control, noise floor 신뢰)을 유지한다.

## 비교 방법

- Independent: beta=0인 독립 noisy retained 재집계. 각 privacy 단계에서 variance도 똑같이 변경한다.
- CR-Fixed: beta=.5.
- CR-Power: 공개 h/bound에 대해 payload energy 상한을 최소화하는 beta, [0,sqrt(.5)] 제약.
- CR-Uses: 공개 h/bound에 대해 필요한 반복 수를 최소화하는 beta=clip(a_R/(a_A+a_R),0,sqrt(.5)). 기존 CR 대수 안의 통신 목적함수 변형이며 별개의 새 unlearning 알고리즘이라는 주장이 아니다. 정수 반복 수가 같은 beta가 여러 개일 때 전력까지 최적화하지는 않는다.
- A-Only: trade-off 주장에 대한 강한 단순 비교군. 같은 source Y0에 A만 -s_A+Z1을 보내 더한다. source와 추가 noise variance는 각각sigma^2여서 final variance=2*sigma^2. 전체 rho_A=4/sigma^2=rho_bar, rho_R=2/sigma^2로 같은 privacy cap이지만 다른 final noise law다. 정확 CSI에서 A의 평균 영향은 제거되지만 본 CR의 원래 noise variance reference와 분포가 같지 않다. 이 성능 차이를 숨겨 matched-output 비교군으로 부르지 않는다. 여러 client의 공중 합산도 사용하지 않는 단순 competitor다. 특정 논문 코드의 재현이라는 주장이 아니다.

모든 CR은 A의 음의 통계와 retained 보충 통계를 같은 자원에서 동시에 전송한다. channel-adaptive 계수 선택을 위해 A의 pilot/CSI feedback을 반드시 센다. source noisy no-op 정확도 및 clean retained 정확도를 참고로 기록한다. 정확 CSI에서 최종 law가 같은 방법들의 작은 정확도 차이는 finite-noise 차이로 해석한다.

## 채널·GPU 평가

unit, iid Rayleigh, iid Rayleigh+5% log-amplitude CSI error. source/deletion 채널과 CSI 오차는 서로 독립. 각 dataset/seed/K/조건별16 channel draws, draw당8 noise pairs; 첫2개를 전체 test set으로 평가. privacy 단계 사이에서 표준 Gaussian noise/channel을 공유해 비교 분산을 줄인다. 각 단계의 데이터/분할은 같다. source와 deletion을 실제 weighted-channel 합산으로 구성하고 반복 평균의 Gaussian 등가 noise로 GPU head를 평가한다. source noise도 각 단계의 sigma^2에 맞게 바꾼다.

exact CSI에서는 ideal retained reference mean KL/coupled head/A-empty counterfactual/전체 rho cap을 검산한다. CSI error에서는 실제 gain ratio를 사용한 조건부 rho 초과, A 잔여 영향, ideal 및 fresh-radio retained reference KL을 따로 기록한다. nominal epsilon을 CSI error까지 보장한다고 주장하지 않는다.

## 실제로 세는 통신 항목

수신 real noise variance nu^2=.01, client payload vector1회 energy cap=1. 이 수치를 per-symbol20dB SNR이라고 부르지 않는다. target noise 분산과 전력 cap으로 최소 정수 반복을 결정한다. weak channel을 버리거나 client를 제거하지 않는다.

가변 길이의 실제 전송 자원 수로 정수화한다. payload block296 real + cyclic prefix37 real을 r번 전송한다. client별 UL pilot8+prefix1, 공통 DL pilot8+prefix1, control480bit + active-client별 CSI feedback64bit + FP32 head5120bit를 포함한다. control은 별도 안정된 링크의 effective rate=.5log2(101)bit/real이다. digital real symbols=ceil(total_bits/rate), prefix=ceil(digital_symbols/8). source 준비와 삭제, lifecycle을 모두 센다. encoder는 공개 seed로 설치돼 있다.

전송 지연은 effective real-symbol rate1M real/s로 환산한다. 100k/10M real/s 민감도도 기록한다. GPU compute time은 GPU timing으로 따로 기록한다. **SDR/무선 장비 실측 지연·joule이 아니다.** 별도 control 링크의 오류/재전송과 payload repetition의 block-static channel 가정이 있다. 고정50ms 지연 기준의 초과 fraction도 보고하되 이를 coherence 모델 검증으로 부르지 않는다.

동일 signal을 실제 반복 배치하고 Gaussian noise를 각각 생성하는 real baseband sample 검증을 추가한다. synthetic channel/statistics의 K1000, rho=[.5,2,8], unit/Rayleigh,4방법에 대해 raw r회 수신 평균 noise, payload sample 수와 정수 frame count를 검산한다. 큰 tensor는 반복256회 단위로 처리해 GPU memory를 제한한다. 해당 표본 검증은 무선 hardware test가 아니다.

위 raw baseband 검산의4방법은 Independent, CR-Fixed, CR-Power, CR-Uses다. A-Only는 main GPU 평가와 joint covariance/privacy 수식 감사에 포함한다.

## 에너지와 진단

실제 clipped statistic의 transmission tensor에서 payload TX energy를 계산한다. pilot/DL energy1 per real 및 prefix를 센 normalized accounted TX energy를 별도로 공개한다. 통신 지연 감소와 TX energy 감소를 혼동하지 않는다.

prefix 에너지는 길이 비례 근사와 함께 실제 송신 vector의 마지막37개 성분을 복사한 에너지를 계산하며, 주 비교는 실제 복사 성분 값을 사용한다. 이는 단일 flat channel의 real baseband frame 모형이며 OFDM 주파수 선택성/IFFT 구현을 검증하는 것이 아니다.

RF circuit/수신 energy는 실측하지 않는다. 대신 식으로 드러나는 추가 비용을 진단한다. per-client 한 active payload real 동안의 circuit energy를 kappa=[0,1e-6,1e-4,.01]로 가정해 TX+payload-circuit 민감도를 계산하고, payload circuit을 포함했을 때 Independent보다 유리해지는 손익분기 kappa도 계산한다. downlink RX circuit 등 누락 항목이 있어 이 sensitivity를 전체 실제 RF energy라고 하지 않는다.

## 판정 및 공개

각 privacy 단계/두 dataset에서 K1000 정확 CSI Rayleigh의 정확도 손실이 clean retained보다<=5%p이고 Independent 대비 총 삭제 real symbols가>=10% 줄면 '통신·utility 기준 통과'다. 조건은3seed 평균으로 판단하고 seed SD를 기록한다. 사용 가능한 가장 강한 단계(가장 작은 rho)를 보고하되, 필요 epsilon/AUC를 숨기지 않는다. 개인정보 보호의 실무 적정성을 이 기준만으로 판정하지 않는다. nominal CSI 보장과 실제 추정오차 결과도 분리한다.

별도로 같은 rho cap에서 A-Only와의 정확도·통신 좌표를 비교한다. A-Only까지 포함해 모두를 지배한다고 주장하려면 accuracy/원래 reference 일치 여부까지 일치시켜야 한다. trade-off 측정, Pareto 선택지, 새 trade-off 법칙의 최초 발견, 논문 신규성 확정은 서로 구분한다.

K10/100 결과와 unit에서 개선이 없거나 악화되는 조건을 함께 공개한다. 결과를 본 뒤 privacy 범위·lambda·encoder를 추가 탐색하지 않는다. 원시 데이터/client statistics/checkpoint는 업로드하지 않는다. 실행 코드, 모든 집계행, frame/energy ledger, 검산/hash, 보고서와 그림을 같은 GitHub에 올린다.
