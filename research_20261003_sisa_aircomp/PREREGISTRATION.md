# SISA-AirComp: 고정 shard와 집계 전송 비용 개선, 1차 유한 실험

작성: 2026-10-03, 실행 전. 사용자가 구체적인 실험 설계와 개선을 요청했다. 기존 연구 결과를 덮어쓰지 않는다. 새 GPU worker는 하나만 실행한다.

## 질문과 개선 후보

client별 데이터를 하나의 고정 shard에만 배정한다. shard별 독립 CNN을 FedAvg 형태로 학습하고 예측 확률을 평균한다. client 전체 삭제 시 영향받은 shard만 초기 checkpoint부터 정확히 replay한다. 전원이 처음부터 참여하므로 중간 slice checkpoint의 추가 이득을 주장하지 않는다. 이 실험에서는 재학습 라운드 수를 줄이지 않는다.

1. random_fixed: 무작위 균등 shard, 공개 clipping bound로 AirComp gain 설정, MSE 조건을 맞추는 반복 전송.
2. channel_fixed: 장기 채널 gain이 비슷한 client를 같은 크기의 shard로 묶음. 공개 bound 사용.
3. random_norm: 무작위 균등 shard + 현재 clipped update의 norm scalar로 gain 조절. 강한 비교군.
4. channel_norm: 채널 기반 고정 shard + 동일한 norm 조절. 추가 shard 설계의 후보.
5. K=1 random_norm: 단일 모델 AirComp full retraining 비교. 동일 shard 알고리즘의 exactness reference와 구분한다.

채널 grouping과 norm 조절은 기존 통신 기법의 조합을 평가하는 후보이며 문헌 최초성을 주장하지 않는다. channel_norm 대 random_norm이 추가 설계의 주 비교다. channel_norm 대 random_fixed만으로 신규 기여를 주장하지 않는다.

## 정보 조건과 isolation

서버 인터페이스는 shard 집계, client ID, CSI, 그리고 norm 방식일 때 현재 update당 32-bit norm scalar만 사용한다. 개별 gradient 벡터, 과거 client update 이력, 삭제 client의 협조는 요구하지 않는다. norm scalar 노출을 기록한다. 시뮬레이터가 client별 벡터를 일시적으로 다루는 것은 물리 채널 모사용이며 서버 학습 코드에는 집계만 전달한다. 연구 evaluator의 원자료 접근은 프로토콜과 구분한다.

모델/optimizer 상태의 shard 간 공유는 없다. 초기화·minibatch·잡음 난수는 shard, client, round로 분리한다. 고정 routing policy에 조건부인 같은 알고리즘의 A-free 재학습이 reference다. grouping은 학습 전에 장기 채널 정보만으로 결정하며 삭제 시 재배정하지 않는다. 데이터 삭제가 장기 CSI/ID 등 고정 routing metadata의 삭제까지 요구하는 문제는 범위 밖이다.

shard 사이 전송은 직교 자원이고 내부는 analog sum이다. TDMA full-band, simultaneous fixed FDMA, work-conserving elastic FDMA의 자원 ledger를 모두 계산한다. 부호화되지 않은 여러 합을 한 scalar 관측에서 분리할 수 있다고 가정하지 않는다. 이상적인 직교성과 CSI를 가정한 baseband simulation이며 실제 RF 검증/암호학적 프라이버시/DP 인증은 아니다. 최소 삭제 후 집계 인원 3을 조건으로 한다. 이것은 프라이버시 보장이 아니라 노출 조건이다.

## 데이터·학습

- 기존 core.py의 raw FashionMNIST CNN(약 38k 파라미터), fresh initialization. 공개 backbone 없음.
- 20 clients, Dirichlet alpha=0.5. train 12000 / dev 2000 / official test 10000. 원 인덱스 저장.
- K=1,2,4,5. client-balanced loss: active client update를 균등 평균. sample-weighted FedAvg가 아님.
- 120 rounds, local SGD 2 minibatches x64, lr .05, momentum/BN/dropout 없음. 매 local 호출 optimizer reset.
- L2 clipping C=1. client 0 전체 삭제. 모든 client는 t=0부터 참여. 각 source에서 독립 단일 삭제만 실시.
- source와 A-free all-shard fresh reference를 각각 실제 실행하고 affected-shard replay도 별도로 실제 실행. source와 reference의 무영향 shard에는 동일 random tape를 사용하고 affected suffix에는 새 noise key를 사용한다. 이는 구현 검증 coupling이며 과거 RF 잡음 재현을 운영 조건으로 요구하지 않는다.
- model checkpoint t=0,60과 final 모델 저장. full per-client history는 저장하지 않는다.

## 무선 모델과 ledger

장기 amplitude gain h_i=10^(U[-20,0]/20), channel seed는 데이터와 독립. 이 값에 client/round별 독립 bounded fading amplitude U[.85,1.15]를 곱한다. 실제 fast fading과 채널 추정 오차는 별도 후속 문제다. 수신 잡음 variance sigma2=.01, per-real-symbol energy cap P=1(정규화), 집계 update 전체 벡터의 기대 squared noise norm 목표 epsilon=1e-4. 코드 내 모든 변수는 이 단위로 저장한다.

update dimension D, active count m, weight w=1/m. fixed 방식 beta=max_i w*C/(h_i*sqrt(D)); norm 방식 beta=max_i w*||clip(delta_i)||/(h_i*sqrt(D)). client waveform x_i=w*delta_i/(h_i*beta), 따라서 ||x_i||^2/D<=1. r=max(1,ceil(D*beta^2*sigma2/epsilon))번 같은 update를 송신/평균하면 집계 잡음은 beta*sqrt(sigma2/r)이다. r번 실제 파형을 모두 메모리에 만들지 않고 동등 Gaussian 평균을 샘플링한다. r을 더 작게 보이게 cap하지 않는다. 반복마다 local gradient를 다시 계산하지 않는다.

- UL analog RE: r*D. pilot: client당 round당 8 RE. control: client당 64 bits와 shard당 64 bits, norm 방식 추가 client당 32 bits. 신뢰적 digital control의 gross 환산 2 bits/RE를 가정한다.
- DL: shard model당 32*D bits/round, 2 bits/RE 가정. 오류 없는 디지털 DL을 모델링한다. 높은 DL efficiency(6 bits/RE) 민감도를 별도로 계산한다.
- normalized UL signal energy: r*sum_i ||x_i||^2. pilot/control의 unit-symbol energy도 별도 기록. 실제 RF Joule로 표현하지 않는다. BS DL energy/회로전력 미포함.
- all-shard source/ref의 한 라운드: full-band TDMA와 이상적 elastic FDMA의 communication delay =sum_k load_k / total_resource_rate. 고정 equal FDMA =K*max_k load_k / total_resource_rate. 삭제 시 영향을 받은 shard만 동작하며 고정 FDMA는 기존 1/K bandwidth를 유지한다. empty spectrum 비용과 useful RE를 구분한다.
- normalized channel-use time만 보고하며 장비 ms/종단간 지연이라고 부르지 않는다. CPU/GPU simulation wall time은 별도다. 공정한 핵심 비교는 channel_norm 대 random_norm의 work-conserving RE다.
- 초기 학습, 삭제, lifecycle(initial + 1/10/100회 같은 비용의 삭제를 가정한 projection)을 구분. lifecycle projection은 실제 순차 삭제 실험이 아니다.

## 유한 실행 단계

0. CPU/NumPy 파형·직교성·noise variance audit와 GPU 4-round K=2 preflight(별도 seed 10600). 정확성·속도만 확인, 성능 tuning 금지.
1. Screening: seed10601, K=1 random_norm 1개 + K=2/4/5 x 4방법=12개, 총 13 source 조건. dev만 K 선택에 사용.
2. K 선택: channel_norm이 random_norm 대비 dev loss<=2pp, K1 대비<=5pp이며 correctness를 만족하는 K 중 J=.5*(delete local_calls/K1 local_calls)+.5*(delete total RE/K1 total RE)가 최소인 K. 동률이면 작은 K. eligible이 없으면 K2를 탐색적 확인으로 실행하고 eligibility failure 기록.
3. Confirmation: 새 seeds10611/10612/10613, 고정 선택 K의 random_norm/channel_norm과 K1 random_norm, 총9 source 조건. test로 재선택하지 않는다.
4. Stress: seed10621, 선택 K 두 방법과 K1, 총3조건. evaluator가 client dominant label 순으로 장기 channel gain을 정렬해 채널과 데이터 분포의 상관을 만든다. grouping algorithm은 CSI만 본다. 이 조건은 일반화 스트레스이며 primary confirmation에 합치지 않는다.

## 사전 성공·기각 기준

- Correctness: affected replay와 새 A-free all-shard reference의 전체 parameters bitwise 동일, maxabs<=1e-6도 보고. unaffected shard source 대비 동일, 삭제 client 호출 0, 타 shard replay 호출 0. 하나라도 실패하면 구현 오류로 다룬다.
- Radio audit: 송신 cap 위반 없음; predicted aggregate MSE<=epsilon; Monte Carlo measured MSE relative error<5%; noiseless waveform reconstruction maxabs<1e-10; disjoint-frequency leakage 0.
- 추가 개선 확인: confirmation에서 channel_norm의 평균 total RE가 random_norm보다 >=10% 작고, 평균 test accuracy 감소<=2pp, seeds3개 중2개 이상 RE 개선. 에너지 증가를 숨기지 않는다. 이 조건 불충족은 해당 주장 미확인/기각이며 실험 실패가 아님.
- UL-only 이득은 별도 보고하며 total 통신 절감으로 바꾸어 쓰지 않는다. grouping 정확도 손실>2pp이면 비용-정확도 교환관계로 남기고 동일 품질 개선으로 채택하지 않는다.
- Norm 개선과 grouping 개선을 분리한 ablation, K별 accuracy/local compute/UL/DL/energy/storage Pareto 표 작성. 작은 표본이며 통계적 일반성/실제 프라이버시 보장을 주장하지 않는다.

## 비용·보존·문헌

RTX3080 utilization0%,12MiB, compute inventory에는 ChatGPT display process만 있었음. CIM 전체 프로세스 조회는 권한 제한으로 실패했으므로 실행 직전 nvidia-smi compute inventory를 다시 확인. 1 worker, 유료 외부 서비스 없음. Power meter는 GPU board draw 5초 sampling, host 전체/전기요금은 미측정. preflight/본실험/실패 실행 포함 총 wall time·board Wh·local calls를 보고한다.

참고: SISA https://arxiv.org/abs/1912.03817 ; IJCAI2024 isolated/coded sharding https://www.ijcai.org/proceedings/2024/0503.pdf ; multi-model orthogonal AirComp https://www.comm.toronto.edu/~liang/publications/ICASSP2024.pdf . 기존 코드는 모델·데이터 loader만 재사용. 결과/분할/모델/확률/round ledger/코드 hash/선택 이유/완료 marker와 독립 audit를 새 폴더에 저장한다.
