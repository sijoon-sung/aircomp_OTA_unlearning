# 데이터–채널 연결 강도 통제 실험: 실행 전 기록

2026-10-03. 기존 결과를 덮어쓰지 않는 신규 탐색 실험. 현실의 데이터–채널 상관 유무를 검증하는 실측 실험이 아니다. 현재 로컬 FashionMNIST에는 대응하는 무선 관측이 없다.

## 조건과 가설

- 독립 seed 10701/10702/10703, FashionMNIST train12000/dev2000/test10000, client20, Dirichlet alpha .5, K4 (5명/shard), 240 rounds, local SGD 2 steps x64, lr .05. 기존 CNN/동일 초기값/동일 client별 minibatch seed 사용.
- 비교: random_norm, channel_norm (채널 세기순으로 5명씩). 신규 배정 알고리즘을 제안하는 단계가 아니다.
- 데이터 분포 p_i의 centered histogram 첫 principal component를 score로 사용한다. PC 부호는 최대 절댓값 loading이 양수가 되도록 고정한다. label 번호를 수치로 상관 계산하지 않는다.
- score와 직교하도록 만든 무작위 벡터 e에 대해 z=lambda*standardize(score)+sqrt(1-lambda^2)*e. lambda=0/.5/1. 이 lambda는 합성 latent 연결 계수이지 관측된 JS-distance 상관이 아니다. lambda0 역시 유한 표본의 직교화 통제이며 모든 비선형 독립을 보장하지 않는다.
- seed마다 같은 20개 Uniform[-20,0] dB 채널 값 집합을 z 순서로 할당한다. 채널 주변분포·데이터·초기값을 보존하고 연결만 바꾼다. 라운드 fading .85–1.15, 기존 norm 제어와 목표 집계 MSE 1e-4 유지. 채널은 주로 통신비용에 영향을 주는 조건이다.
- 3 seeds x3 levels x2 methods =18 cases. source, client0 삭제 전체 SISA reference, affected shard replay를 각각 실행한다. 단일 삭제 client0로 삭제 대상 일반성을 주장하지 않는다. 동일 SISA reference와 비교하며 단일 전역모델보다 절약했다는 주장은 하지 않는다.

## 분석·판정

1. 각 seed/level의 client histogram JS distance와 채널 dB 절대거리 사이 Spearman rho, client channel label permutation1999회 양측 p. 190개 pair를 독립 표본으로 취급하지 않는다. 9개 p에 BH 보정. 설정으로 만든 관계이므로 현실 상관의 증거가 아니다.
2. shard 균등 client 평균분포와 전체 균등 client 평균분포의 JS divergence: 평균/최대, 삭제 후 해당 shard JS. source/deleted test accuracy, macro-F1, 클래스별 recall 및 최저 recall.
3. paired channel-random: 삭제 accuracy 차이(pp), 최저 recall 차이(pp), UL와 total RE 상대 절감, 초기 학습 총 RE; 그룹당 fading-min gain과 대표성. 초기·삭제 비용을 따로 보고한다.
4. 탐색적 실용 기준(각 level): 평균 삭제 total RE 절감>=10%, 3 seeds 중>=2에서 절감, 평균 정확도 손실<=2pp, 모든 exact replay 통과. 통과해도 통계적 확증/신규성 증명은 아니다.
5. 연결강도 증가에 따라 채널 배정의 대표성·정확도 손실이 늘어나는지는 paired 차이를 기술하고, 3 seeds로 일반적인 유의성 주장은 하지 않는다. 반대/무차이도 보고한다.

## 비용·범위

단일 GPU worker, 시작 전 compute process 확인, 실행 시간과 GPU board energy(시뮬레이션 RF 에너지와 구분), local calls, 정상/실패 모두 기록. 18 cases로 종료한다. 원자료 correlation_control_v1/, 코드 correlation_control.py. 이전 결과 변경 없음.

grouping 서버는 CSI를 사용한다. 데이터 histogram과 PC는 평가자/합성 환경 생성에만 사용하며 운영 알고리즘에 전달하지 않는다. norm 제어는 현재 update norm scalar를 제공한다. DP나 별도 privacy 보장은 검증하지 않는다. 이상적 직교 shard 자원 분리로 간섭은 모델링하지 않는다. 고정 MSE 통신량 비교이며 고정 radio budget에서의 정확도 실험은 아니다.
