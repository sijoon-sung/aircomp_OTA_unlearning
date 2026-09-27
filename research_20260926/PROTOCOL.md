# 범위를 줄인 AirComp FU / SFL 판별 실험

2026-09-26. 기존 ota_ful 결과는 읽기 전용으로 사용한다. 새 결과는 이 폴더에만 저장한다.
이것은 국내 학회 주제 선정을 위한 탐색 실험이며 새 DNN unlearning 성공의 확증 실험이 아니다.

## 사전 고정 범위

1. 수학 반례: 두 quadratic client의 순차 SGD에서 과거 update 차감 후 잔여 오차,
   한 시점의 gradient 직교와 다음 시점 영향의 차이, activation 합의 목적함수 변경을 수치 확인한다.
2. V12 축소 진단: 학습되지 않은 공개 random ReLU 64차원+bias, Fashion-MNIST ridge head.
   seed 202609260, 202609261, 202609262. 매 class 1,000 training samples, 10 client 각각 두 labels.
   client 0 삭제, retained 9명 재참여, alpha=0.01, 정확한 source/retained optimum.
   H를 32회 noisy AirComp로 관측; eigenvalue floor=alpha. 65개 eigenmode를 13개씩 5그룹.
   gradient 통신 반복 총량 40: uniform / gradient-MSE / inverse-curvature-MSE.
   수신 Hessian 및 로컬 power 통계만으로 배정. SNR 10/20/30 dB, seed당 16 paired noise.
   exact-H는 진단용 oracle로 분리. 공식 test는 배정·선택에 사용하지 않음.
   primary: 초기 삭제 격차 대비 parameter squared error. 보조: objective gap, test accuracy.
   H acquisition, eigenbasis DL, gradient UL 및 model DL 비용을 포함한다.
   성공 조건: noise 감소뿐 아니라 total error도 uniform보다 개선; no-op보다 가까워야 함.
3. 이기종 SFL: 784→64→256→10 MLP, cut after first ReLU, widths 16×5/32×3/64×2.
   Fashion-MNIST 각 class 첫 1,000개의 seed shuffle training samples, 각 client 두 labels.
   위와 같은 3 seeds, 80 rounds, round/client마다 SGD 2 batches×32, lr=.03, momentum 없음.
   client front와 server suffix를 로컬에서 각자 update 후 masked average(SplitFedV1 연산 모사).
   individual activation/gradient 교환은 FP16 디지털; front parameter synchronization만 AirComp.
   비교: exact sync(oracle), digital8, AirComp uniform(그룹당4회), coverage/power-aware(총16회).
   width 그룹은 각 16 neurons인 4블록으로 통일, active counts=10/5/2/2.
   uniform 및 adaptive는 총16회: 같은 블록 크기·동일 analog resource.
   SNR=10/20 dB; 알려진 정렬 채널 |h|=1, 평균 송신전력 제한. client norm metadata 포함.
   extra basis나 solver 추가 없음. 동일 초기값·batch schedule·noise innovations 사용.
   성공 조건: same-resource adaptive가 uniform보다 parameter aggregation MSE를 낮추고
   accuracy 저하를 줄이는지. 정확도는 3 seed 전부와 평균±표본표준편차, 유의성 주장 없음.
   전체 비용은 activation UL, cut-gradient DL, front UL/DL, labels, norms, pilots/CP 포함.
   계산량은 dense MAC 산술 추정. 에너지·하드웨어 latency·RF waveform 측정으로 표현하지 않음.

## 공통 제한

- 하나의 CUDA worker만 실행; 기존 사용자 GPU 작업을 중단하지 않음.
- 전송 모델은 coherent, equalized, 평균전력 AWGN 추상화. CFO/추정 CSI/clipping/packet 오류 미검증.
- 디지털 payload rate는 1/2 log2(1+SNR) bit/complex use라는 명시적 비교 가정; 코딩 구현 아님.
- pilot 8 uses/client/block, metadata 32bit/client/block, CP 배율1.125.
- exact optimum/oracle gradient는 평가·진단에만 사용. adaptive 자원 선택에 제공하지 않음.
- 노출되는 norm, width, curvature, activation과 individual digital updates를 기록한다.
- 새 알고리즘 또는 통신 방식의 최초성은 주장하지 않는다. 기존식의 작은 적용 개선 후보다.

## 실행 전 보충

SFL의 총 반복수 문구는 코드 작성 전 최종적으로 **4개 동일 크기 그룹에 총16회**로 확정했다.
모든 local pool은 disjoint. client i의 label은 i 및 (i+1) mod10이며 각 label의 1,000개를
두 소유 client에 500개씩 배분한다. 축소 데이터 범위이며 기존 V1–V21 수치와 직접 합치지 않는다.
곡률 진단에는 같은 noisy H와 원래 좌표의 합산 C=XᵀY/N로 바로 ridge를 재학습하는
stats_retrain 비교도 포함한다(gradient와 같은 좌표당 8회). 이는 source를 쓰지 않으며
eigenbasis 배포도 요구하지 않는 강한 기준선이다. 낮은 차원의 sufficient-statistic 방식에 한정한다.
