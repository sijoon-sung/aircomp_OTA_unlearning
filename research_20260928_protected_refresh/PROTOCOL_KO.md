# CR-Air 후보: 실행 전 범위 고정

2026-09-28. 세 조건(삭제, 서버 전체 관측 프라이버시, AirComp 기여)을 함께 다루는 **새 학습 구조 후보의 수학 검산**이다. 기존 DS-Air의 성능 개선 실험이 아니다. 공개 특징 추출기, client 단위 L2-clipped additive statistics, 최초 학습부터 noisy aggregate, 마지막 삭제 라운드에 A 및 잔존 client 참여를 가정한다.

한 번의 공개된 삭제 요청, 정확한 CSI, 단일 합산 관측 수신기, 수신기 Gaussian 열잡음의 신뢰 가능한 분산, 프로토콜을 준수하는 honest-but-curious 서버가 범위다. client 신원은 숨기지 않는다. client 데이터 전체 replacement adjacency를 쓴다. 반복 전송 횟수, scaling, clipping bound는 공개 데이터 독립 정책이다. 원래 private norm을 별도로 전송하지 않는다.

실행은 단일 CUDA 프로세스 FP64. seed=202609289. beta=0,.25,.5,sqrt(.5),.9,.99의 평균 상쇄, Gaussian covariance, ridge decoder 결과 및 A counterfactual 불변성, joint Gaussian precision으로 구한 zCDP, beta=.5 minmax, 물리 송신 합산과 block-energy bound, 에너지 최적 beta의 grid 교차 검산을 한다. 독립 refresh beta=0 및 위험한 같은 잡음 재사용 beta=1을 포함한다.

Monte Carlo 100000회는 Gaussian covariance에만 사용한다. ridge는 synthetic statistics이며 실제 MNIST/FashionMNIST 분류 실험이 아니다. 분포 일치 증명은 수식으로 하고 Monte Carlo/공유 잡음 coupling은 구현 검산으로만 사용한다. 기존 deterministic retrain에 대한 exact unlearning을 주장하지 않는다.

cost 예시는 K=10, d=16, C=10, q=296, block-energy cap=1, physical per-real noise variance=.01, client statistic bound B=1, target statistic noise variance=8, delta=1e-5, joint rho cap=.5. unit 및 약한 A/잔존 채널(h=.01)을 포함한다. payload TX energy와 real channel uses는 분리한다. 전력증폭기, RF circuit, downlink energy는 미측정. source 준비, pilot(활성 client당8 real), control384bit, FP32 최종 head broadcast를 단순 비용 모형에 포함한다. rate=.5log2(1+100), CP=1.125. 프로토콜/실험 결과 hash를 저장한다.

판정: 수식이 맞으면 '가정 아래 수학적으로 일관된 후보'다. real-data utility, realistic CSI, 반복 삭제, 동일 privacy/utility의 강한 비교군, 선행연구 차별성까지 검증해야 실용적인 세 조건 충족 및 논문 기여를 주장할 수 있다. 결과를 보고 이 판정을 완화하지 않는다.
