"""실험 전체에서 고정하는 값."""

N_CLIENTS = 20        # client 수
DIRICHLET = 0.5       # client 데이터 비균질도 (class 비율 Dirichlet 농도)
MIN_SAMPLES = 60      # client 최소 표본 수
LR = 0.05             # 로컬 SGD 학습률
LOCAL_STEPS = 2       # 라운드당 로컬 step
BATCH = 64
CLIP = 1.0            # client update L2 상한 C (전력 정렬의 공개 상한과 같다)
P_MAX = 1.0           # 심볼(칩)당 평균 송신 전력 상한
EPS = 10.0            # 허용 집계 오차 (noise 실험의 측정값 eps* = 10)
ROUNDS = 160
SEED0 = 81001
