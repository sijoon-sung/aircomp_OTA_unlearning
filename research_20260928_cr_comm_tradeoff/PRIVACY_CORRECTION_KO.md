# 복사한 cyclic prefix를 포함하는 프라이버시 정정

2026-09-28. 본 실행의 정확도·통신량 결과는 유지한다. 실행 후 전체 관측을 점검하면서 prefix 관측을 누락한 프라이버시 계산을 발견해 정정했다. 동결한 `PROTOCOL_KO.md`, `run_experiment.py`, 원시 집계 JSON의 `rho_cap`, `epsilon_cap`, `rho_all`, `oracle_AUC`는 **prefix를 버리는 수신기 기준의 nominal 값**이다. 서버는 그 prefix도 저장할 수 있으므로 이를 전체 관측 보장으로 인용하면 안 된다. 이전 대화의 nominal ε≈11.60도 이 실행의 최종 전체 관측 상한으로 사용할 수 없다.

## 추가 관측의 수학

이번 frame은 q=296개의 실수와 마지막 p=37개를 복사한 prefix를 전송한다. 각 실수 수신에 독립 Gaussian 잡음이 붙는다. 선형 복사 연산자를 T라고 두면

\[
Ts=(s_{q-p+1:q},s),\qquad
T^TT=\operatorname{diag}(1,\ldots,1,2,\ldots,2).
\]

따라서 마지막 37개 성분은 두 번 관측된다. 서버가 두 복사본을 평균하면 해당 성분의 noise variance는 절반이 된다. r회 반복의 residual은 평균과 독립이어도, **복사 prefix는 평균 payload와 독립인 데이터 없는 residual이 아니다.**

L2 bound B인 client 통계의 replacement 차이를 Δs라고 하면 ||Δs||≤2B이다. 한 관측의 Gaussian divergence에 들어가는 quadratic form은

\[
\|T\Delta s\|^2=\|\Delta s\|^2+\|\Delta s_{CP}\|^2
\le 2\|\Delta s\|^2\le8B^2.
\]

source와 deletion 모두 같은 prefix를 사용하므로, 기존 nominal 식에 대한 전체 관측의 보수적 zCDP 상한은

\[
\rho_{A,\mathrm{frame}}=2\rho_{A,\mathrm{nom}},\quad
\rho_{R,\mathrm{frame}}=2\rho_{R,\mathrm{nom}},\quad
\bar\rho_{\mathrm{frame}}=2\bar\rho_{\mathrm{nom}}.
\]

클리핑된 모든 벡터를 허용하는 범위에서의 상한이다. 실제 Gram/moment가 가질 수 있는 더 작은 domain을 이용한 최적 상한이라는 주장은 하지 않는다. CSI error 조건의 coefficient/gain을 포함한 nominal 식도 같은 배수로 보수적으로 보정한다. nominal cap을 넘었던 조건이 이 보정으로 해결되지는 않는다.

δ=10⁻⁵, ε=ρ+2√(ρ log(1/δ))를 적용하면 다음과 같다.

| Frozen nominal ρ | Full-frame ρ 상한 | Full-frame ε 상한 |
|---:|---:|---:|
| 0.5 | 1 | 7.7861 |
| 1 | 2 | 11.5971 |
| 2 | 4 | 17.5723 |
| 4 | 8 | 27.1941 |
| 8 | 16 | 43.1446 |

Oracle AUC는 무조건 2배로 대체하지 않는다. 실제 A 통계와 빈 통계의 알려진 두 가설에 대해, D² = ρ_A,nom (||s_A||²+||s_A,CP||²)/2를 계산한 뒤 AUC=Φ(√(D²/2))를 적용한다. 이는 membership inference 학습 공격이 아니라, side information을 가진 두 가설 Gaussian 최적 검정의 진단이다.

## 구현과 검증

`audit_full_frame.py`는 CUDA에서 TᵀT 및 explicit frame quadratic form을 검산하고, 같은 데이터 hash/분할/특징으로 A 통계를 다시 계산해 450개 조건의 수정 상한과 oracle AUC를 저장한다. 데이터나 통계 자체는 저장하지 않는다. `results/full_frame_accounting.json`이 보정의 근거이고, 재생성한 결과표와 그림은 이 값을 사용한다. 메인 실험 코드는 바꾸거나 재실행하지 않았다.

최종 모델 decoder는 실험처럼 prefix를 버린 payload 평균만 사용하므로 출력 분포와 정확도, frame 수, 에너지는 변하지 않는다. 모델이 사용하는 데이터보다 공격자인 서버가 이용할 수 있는 관측이 많다는 차이를 계산한 것이다.

향후 zero guard 또는 전체 frame combining을 사용하면 다른 프로토콜이 된다. 이를 이번 결과에 소급 적용해 더 좋은 프라이버시나 정확도를 주장하지 않는다.
