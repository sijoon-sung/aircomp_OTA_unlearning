**직교성 필요 여부 — 실행 전 고정, 2026-09-27**

질문: 동일 삭제 목적함수·step 크기에서 완전 직교화가 필요하며, 이를 완화하면 실제 삭제 품질과 AirComp 잡음 민감도가 어떻게 달라지는가? 새로운 알고리즘의 신규성을 주장하지 않는다. Recovery, SFL, KD는 확장하지 않는다.

**수학과 구현**

잔존 client들의 정규화 CE gradient를 열로 갖는 G, A의 정규화 UCE gradient를 u라 둔다. c=G^+u, v_beta=u-beta Gc. beta=0은 투영 없음, .5는 부분 투영, 1은 원래 완전 투영이다. 업데이트는 theta <- theta-.01 * ||g_A,UCE|| * v_beta/||v_beta||. 1차 지표 G^T v_beta=(1-beta)G^T u (정확한 연산)를 검사한다. 이 연구는 모델 업데이트의 직교성이지 무선 사용자 코드의 직교성이 아니다.

전체 CE 정상점에서 alpha g_A+(1-alpha)g_R=0이면 g_A를 g_R에 직교 투영한 결과는 0일 수 있다. 2차 손실의 exact deletion과 비교하고, 실제 CNN의 CE 및 UCE/full-local-data 기하를 별도 진단한다. UCE에 같은 소멸 결론을 가정하지 않는다.

FedOSD형 target-norm 정규화가 잡음을 증폭할 수 있다. 따라서 exact/noiseless beta0/.5/1은 정규화 없이 v_beta에 원래 target norm만 곱하는 raw-scale ablation도 함께 실시한다. 수학 검사는 FP64, 모델과 실행 gradient는 기존 FP32다.

**정보·통신 시나리오**

1. Exact oracle: full local vectors로 projection. 계산상 비교군이며 서버 정보 제약 위반.
2. Sketch/noiseless: 각 client는 공개된 하나의 고정 SRHT2048(16bit) sketch를 전달. 서버가 계수만 계산·방송. 각 client가 [-beta*c_i,1]로 자신의 unit gradient를 가중해 동시 송신. 새 공개 기저를 반복해서 묻지 않는다.
3. Sketch/AirComp20,40dB: 같은 방식에 실제 channel inversion과 AWGN, 평균 송신전력<=1, peak amplitude^2<=4, 반복1. 같은 seed/step/noise Z를 beta끼리 사용하되 peak 제약으로 물리 noise scale은 달라질 수 있다.
4. beta0은 잔존 계수가 모두0이라 A-only 송신이 된다. 계산 진단으로만 사용하며 privacy-compatible AirComp 해법으로 채택하지 않는다. .5도 sketch·norm·보정 aggregate 공개가 있으며 DP/전체 transcript 비복원은 증명하지 않는다.
5. 서버가 정확한 retained aggregate를 A에게 넘긴 뒤 A가 이를 상쇄하는 방식은 y-r로 A의 송신을 분리하므로 본 실행 후보에서 제외. HVP query, 추가 mixed observation, training preparation은 이번 질문과 solver를 바꾸므로 실시하지 않는다.

가까운 FU는 [FedOSD](https://arxiv.org/html/2412.20200v1): UCE 및 retained-gradient span projection을 사용한다. 공개 MIT 코드에 대응하는 기존 relation_protocol.py를 재사용한다. 통신은 [COTAF](https://arxiv.org/abs/2009.12787)의 client precoding/합산 구조와 기존 Legacy-Sketch-V1에 연결된다. beta 완화 자체는 알려진 선형 조합이며 신규성 근거가 아니다.

**고정 실험**

- CUDA RTX3080 한 worker. 새로운 학습/data seeds381/382/383. FashionMNIST private12000/public2000/validation2000/test10000,10clients,client0 전체 삭제. 기존 common.py 분할(50% IID+50% two-class 선호).
- ConvNet2 63562 parameters, plain FedAvg150rounds,local4,batch64,lr.03. 원래 FedOSD가 사전 kernel orthogonality를 요구하지 않으므로 이를 추가하지 않는다. 이전 ortho-source 표와 숫자를 직접 비교하지 않는다.
- 각 seed에 같은 초기값·동일 retained data로 reference2개: sampling_offset0 및900000. Reference/test로 beta·lr·전력·stopping을 선택하지 않는다. 기존 checkpoint를 덮어쓰지 않는다.
- 삭제: UCE target, retained CE,client당 minibatch256,step10,lr.01 고정. beta0/.5/1. Exact oracle, sketch-noiseless, OTA20/40dB(각4noise) 각각 실행. Step1/10에서 평가. Step1은 A의 마지막1회 참여 시나리오; step10은 A가10번 참여하는 별도 확장이다.
- Exact/raw-scale ablation beta0/.5/1을 추가해 step norm 혼동을 진단. 학습률 탐색과 결과 기반 조건 추가는 하지 않는다.
- 고정 입력 첫 step에서는 beta별 이상적 방향, cosine, 잔존 span 성분, 신호/잡음 norm, 실제 전력과 float32 MAC 오차를 기록. Exact,sketch,noise 효과를 분리한다.
- Primary: 각 seed의 reference0 대비 forget JS / no-op JS. Absolute JS,reference1 동일 지표,parameter distance ratio,test/forget accuracy,loss-MIA도 기록한다. 두 reference를 독립 학습 seed로 세지 않는다. Source MIA 자체가 약하면 삭제 증거로 사용하지 않는다.
- 사전 탐색 성공: 평균 정규화 JS<.8이며3seed 평균 모두<1. 부분 투영 우위는 같은 channel/step에서 beta1 대비 paired 개선과 reference1 방향 일치를 확인. 단순히 beta1보다 덜 나쁜 것과 삭제 성공을 구분한다. 모든 결과를 기록하며 최선 숫자만 선택하지 않는다.

**비용**

모든 step의 model DL,최종model DL,sketch UL,공개기저seed,계수DL,target-norm/송신scale metadata,pilot,CP1.125를 포함. Noiseless/oracle도 비용 단위의 비교를 위해20dB goodput를 사용. OTA40은 실제40dB goodput를 사용하되20dB 고정환산 비용도 함께 제공. Digital sketches/제어는 error-free Shannon goodput 모델이며 RF coding 검증이 아니다. 초기 source/reference 학습비용과 준비 시간을 별도 기록. Public/sketch basis 생성 계산은 포함된 simulator walltime이며 단말 실측 latency가 아니다.

출력: protocol/code hashes, math_checks.json, 각 seed 원자료,summary.json, RESULTS_KO.md, 비교 그림. 처음 기록한 조건을 결과를 보고 바꾸지 않는다.
