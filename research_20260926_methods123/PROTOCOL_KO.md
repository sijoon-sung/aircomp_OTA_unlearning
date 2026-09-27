**1·2·3 순차 GPU 검증 — 실행 전 고정, 2026-09-26**

목적은 공식 공개 코드의 핵심 연산을 가져와 축소된 client-unlearning 문제에서 성립 여부를 확인하는 것이다. 논문 전체 benchmark 재현이나 새로운 방법의 우월성 확증은 아니다. SFL은 이번 실행에서 제외한다. 삭제 후 별도의 recovery는 하지 않는다.

**출처와 구현 구분**

- ① CuReNU/CuReNUS ICLR2026 공식 Supplemental code를 확보했다. GitHub 검색에서는 찾지 못했지만 학회 공개 zip의 `helper/sto_cubic_func.py`의 cubic subsolver와 `settings/models/convnet.py`의 ConvNet2를 사용한다. Local gradient/HVP를 aggregate oracle로 바꾸는 부분은 우리가 작성한다. Sophia 공식 GitHub도 비교용 자료로 보존하지만 이번 실행을 Sophia 재현이라고 하지 않는다.
- ② FedOrtho 공식 GitHub 저장소는 논문 링크·GitHub repository 검색에서 확인되지 않았다. Orthogonal Convolutional Neural Networks 공식 GitHub의 `orth_dist`를 사용해 논문의 kernel 정규화를 계산한다. Activation 차이·rank soft pruning·2단계 FL은 논문 수식 기반 재구현이다. 공식 FedOrtho 재현이라고 표기하지 않는다.
- ③ FedQUIT 공식 GitHub는 TensorFlow/Linux 환경이다. 설치 스크립트를 실행하지 않고 virtual-teacher logit 변환과 KL을 검토하여 PyTorch에 이식한다. CPU NumPy 식 및 gradient 항등식으로 검증한다. 독립 teacher 제외 + fresh student는 우리의 구조 변경이며 FedQUIT 원형이 아니다.
- Downloaded 코드는 원본 그대로 보존하고 commit/archive SHA256을 기록한다. 선택한 모델·순수 연산 함수만 읽어 로드한다. 전체 설치·훈련 shell script는 실행하지 않는다.

**수학 검사 — 학습보다 먼저 통과해야 함**

1. 같은 θ와 minibatch에서 weighted local gradient/HVP 합이 concatenated empirical objective의 gradient/HVP와 같다. Tiny smooth model의 dense Hessian·finite difference와 비교한다.
2. Ridge 이차문제는 W−H_R⁻¹g_R=W_R*가 성립한다. DNN에는 동일성을 일반화하지 않는다. Cubic model M‖δ‖³/6의 gradient는 M‖δ‖δ/2임을 autograd로 검증한다. 공식 코드도 이 계수를 쓴다. 논문 pseudo-code의 계수와 차이를 기록한다.
3. Kernel orthogonality loss와 공개 `orth_dist²`를 비교한다. Orthogonal kernel≠독립 feature 반례, 불균등 데이터 수의 activation aggregate 식, soft-pruning 순위/동률을 검사한다.
4. FedQUIT teacher 변환 및 KL gradient를 비교한다. 초기 student=old teacher이면 retain KD gradient=0이어서 target gradient만 합에 남는 privacy 퇴화도 검사한다. 실험의 mixed-KD는 잔존 CE gradient를 함께 보내 이 퇴화를 피하지만 정식 privacy 보장은 아니다.
5. 독립 teacher와 초기값의 target 의존성이 없는 조건에서 target-free KD objective가 동일함을 증명한다. Student를 old global teacher로부터 시작하는 경우는 동일성 보장이 없음을 분리한다.
6. Coherent real-valued MAC 합산을 실제 송신 정규화·채널·수신 복원 식으로 검증한다. 이 단계는 baseband 수치 검사이고 RF/CSI/CFO 실증은 아니다.

**공통 데이터와 모델**

기존 FashionMNIST raw 파일을 읽기 전용으로 사용한다. 각 class에서 private 1,200개, public query 200개, validation 200개를 무작위로 분리한다. Test는 공식 10,000개 전부다. 총 private 12,000개, public 2,000개, validation 2,000개이며 겹침을 검사한다. Public label은 KD 학습에 사용하지 않는다.

10 clients, 각1,200개. 각 class의 600개는 client마다60개씩 배분하고 나머지600개는 해당 class와 직전 class의 client에300개씩 배분한다. Client별 50% IID +50% two-label preference인 moderate non-IID다. Client0의1,200개 전체를 삭제한다. 잔존9 client는 같은 class를 일부 보유하므로 forget 정확도0이 목표가 아니다.

CuReNU의 ConvNet2(28×28, in_channels1, n_kernels8, hidden32, classes10)를 사용한다. 두 conv의 output channel 수는8,16이어서 row orthogonality의 차원 조건을 만족한다. BN/dropout은 없고 FP32 학습, TF32 off, CUDA deterministic 설정을 쓴다. Seeds271/272/273; source와 paired retrain은 같은 초기값·client별 sampling 난수를 사용한다.

**Source 학습 및 calibration**

무잡음 OTA 집계 극한의 FedAvg로 삭제 연산 자체를 먼저 분리한다. Learning rate .03, local batch64, local SGD4 steps/round, momentum0, weight decay0. Validation만 보는 별도 calibration seed270에서100 rounds를 학습한다. Val accuracy≥75%이면100 rounds를 고정하고, 아니면150 rounds를 사용한다.150에서도75% 미만이면 quality gate 실패로 기록하고 큰 sweep으로 확대하지 않는다. Test·삭제 재학습 모델로 선택하지 않는다.

②는 각 round의 local2 steps→aggregate→global reference를 두고 local2 fusion steps→aggregate하는 구조다. Ortho 계수 .001, alignment 계수 .0001. Paired retrain도 같은 구조다. Source가 test 정확도를 보고 선택되지는 않는다. 실험 중 설정을 바꾸면 새 protocol revision을 먼저 남기고 기존 결과를 보존한다.

**① HVP unlearning**

공식 FMNIST instance config의 M5, outer10, inner5, inner LR.01, decay.1, perturbation norm20을 사용한다. 각 retained client에서 gradient16/Hessian8 samples를 독립 추출하므로 전체144/72개다. Full-parameter HVP를 사용한다. Hessian/basis 행렬을 전송하지 않는다.

비교: no-op, 같은 초기값의 target-free retrain, 공식 subsolver+무잡음 aggregate, OTA20dB repeats1, OTA20dB repeats4, OTA30dB repeats1, retained-gradient SGD70 steps LR.01. 마지막은 recovery가 아니라 first-order 삭제 기준선이다. 10 gradient+60 HVP 관측과70 gradient 관측의 통신 횟수를 비교하지만 HVP의 추가 local compute는 별도로 센다. Solver와 batch schedule은 통신 비교 안에서 동일하다.

**② Orthogonal preparation + pruning**

각 layer의 local spatial-max activation sum을 forget/retain count로 정규화한 signed aggregate를 얻는다. 삭제 요청 client의 참여가 필요하다. Layer별 top25%를 선택하고 strength=max(.5,1−rank/k)로 kernel weight와 bias를 함께 감쇠한다. Bias 감쇠는 affine channel 전체를 줄이기 위한 구현 선택으로 기록한다. Recovery는0이다.

비교: no-op, paired ortho retrain, noiseless pruning, OTA uniform8, boundary(initial4 + 경계 주변25% 좌표16추가), uniform total-cost matched, random mask 대조. Non-orthogonal source의 noiseless pruning도 preparation ablation으로 평가한다. Boundary는 수신 점수만 보고 선택한다. SNR10/20dB, seed당 paired channel noise4개. Control/DL 때문에 UL-only 동예산을 총비용 개선으로 부르지 않는다. Total-matched uniform은 상대 방식의 비용 모형 안에서 가능한 추가 반복을 배분한다.

**③ KD unlearning**

A. FedQUIT-logit-min teacher 변환을 적용한 target KD gradient와 retained CE gradients를 .5/.5 가중 합산한다. Source에서100 steps, LR.01, batch32/client, temperature1. Noiseless와 OTA20dB repeats4 비교. 이 adaptation은 원래 on-device 전체 모델 교체와 다르며 추가 retained 참여가 필요하다.

B. 독립 teacher 구조: 동일 공개 random 초기값에서 각 client를 독립500 SGD steps(LR.03, batch64) 학습한다. Client별 sampling seed가 ID에만 의존하고 global feedback은 없다. Public queries에서 temperature2 확률을 집계한다. Student는 공개 random 초기값으로부터600 KD steps(LR.05, batch64) 학습한다. Original ensemble student, A제외 fresh student(noiseless/20dB repeats1/20dB repeats4), 원래 global teacher를 복사하는 fresh student 대조를 비교한다. Probability는 noisy 수신 뒤 simplex로 투영한다. Teacher와 student 훈련비용을 준비비용에 기록한다.

Target-free reference는 같은 독립-teacher 알고리즘이다. Seed271에서는 잔존 teacher들을 처음부터 다시 학습하여 checkpoint equality를 확인하고, target-free student도 다시 학습해 일치 여부를 검증한다. 다른 seed는 독립성 조건하에서 보존된 retained teacher를 사용한다. 일반 FedAvg retrain과의 예측 차이도 별도 기록하지만 동일 기준이라고 합치지 않는다.

**언러닝 평가와 비용**

Primary: forget/private sample의 paired target-free retrain 대비 JS divergence, no-op JS로 정규화한 비율. Test/retain prediction JS, forget/test accuracy 차이, class-matched nonmember에 대한 loss-threshold MIA AUC를 보조 기록한다. MIA AUC는 한 공격의 결과이고 인증이 아니다. Seed271의 추가 독립 retrain으로 reference 자체 변동도 검사한다. Reference와 test는 resource choice·early stop에 쓰지 않는다.

좋은 결과의 탐색 기준은 mean normalized forget JS<.8이고 세 seed에서 모두1미만인 경우다. No-op 자체의 JS가 reference 변동보다 작으면 삭제 신호가 약하다고 표시한다. Retain 성능은 관측하되 회복 튜닝은 하지 않는다. ② mask 오류만 감소하고 FU JS가 개선되지 않으면 실제 삭제 개선이라고 하지 않는다. ③ 자기 알고리즘 reference와의 일치만으로 FedAvg 삭제 성공이라고 하지 않는다.

Analog 전송은 weighted local vector를 채널 inversion·공통 최대 RMS scale로 정규화하고 independent Gaussian 잡음을 받는 모형이다. Client별 channel amplitude는 [.7,1,.8,1.2,.9,1.1,.75,1.05,.85,1.15]로 고정한다. 정보는 aggregate와 scalar norm/count, 요청 좌표다. 개별 vector는 simulator의 client oracle 내부에서만 사용한다. 이 구현의 process 격리가 보안 경계라는 뜻은 아니다.

비용 단위는 real-symbol equivalent다. Digital goodput=.5log2(1+SNR) bits/real use, float metadata32bit, pilot8 real uses/client/message, CP계수1.125. UL payload·DL model/vector/query ID/mask·norm/count·feedback를 별도 기록한다. 초기 학습·orthogonal preparation·teacher/student 계산량·unlearning walltime도 기록한다. 같은 품질이 아니면 통신 절감 대신 trade-off로 보고한다. CSI estimation error·CFO·RF hardware latency는 이번 범위 밖이다.

GPU worker는 항상 하나만 실행하고 기존 사용자 작업을 중단하지 않는다. 결과·checkpoint·실패 로그를 이 폴더에 보존한다.
