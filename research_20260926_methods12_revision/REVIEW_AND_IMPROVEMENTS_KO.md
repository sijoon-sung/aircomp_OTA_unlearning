**①은 원래 삭제신호 합산 구조로 다시 봐야 한다. ②는 클래스 구성 차이와 삭제 신호를 먼저 구별해야 한다.**

2026-09-26. 사용자 지적 후 원래 코드, 최근 실행 코드, 선행논문을 다시 대조했다. 이번에는 새로운 학습을 하지 않고 수학 검사와 기존 조건·checkpoint를 이용한 작은 CUDA 진단만 순차 수행했다. 이전 결과를 보고 정한 사후 탐색이며 독립 seed 검증은 아니다.

**먼저 바로잡을 부분**

| 구분 | 서버가 받는 것과 수행하는 일 | 실제 확인한 범위 |
|---|---|---|
| 원래 V1 | 작은 개별 sketch로 계수 계산 → client가 full vector에 계수 적용 → weighted AirComp로 삭제 방향 수신 | 방향 합성 연산과 백도어 조건의 효용이 이전에 확인됨. 일반 client 삭제 보장은 별도 |
| 처음의 ① convex-head | 집계 Hessian과 잔존 local residual을 받고 서버에서 공통 inverse 적용 | 고정 특징 ridge 목적에서는 무잡음 보정이 재학습 최적점과 일치. 20dB 오차 .0501→.0202 확인 |
| 최근의 ① CNN CuReNUS | 잔존 gradient와 HVP를 반복 수신해 cubic subproblem을 풂 | HVP 합산은 맞지만, 해당 solver 설정에서 target-free reference 접근 실패 |

최근 CuReNUS 실패를 위의 두 구조가 불가능하다는 결론으로 확대하면 안 된다. 특히 “잔존 client만 참여했으므로 틀렸다”는 해석도 아니다. Convex-head 역시 잔존 client만으로 정확한 보정이 가능하다. 실제 차이는 모델·목적함수·출발점·solver와 그 전제가 바뀐 것이다.

원래 V1의 sketch는 삭제 update를 복원하는 압축본으로 쓰지 않는다. 정규화 target 삭제 gradient를 u, retained gradient들을 G의 열로 두면

    c_hat = argmin_c ||S u − S G c||²
    target 전송: u,    retained i 전송: −c_hat_i g_i
    서버 수신: p_hat = u − G c_hat + e

이다. 서버는 합산된 삭제 방향을 모델에 적용한다. `relation_protocol.py`의 `relation_coefficients`, `projected_direction`, `aircomp_transmit`가 이 구조다. Sketch 획득과 계수 방송까지 포함하면 전체 프로토콜이 한 번의 통신인 것은 아니다. 원래 방식은 본질적으로 기존 FedOSD 방향을 구현하며 새 직교투영 알고리즘 자체가 아니다.

최근 CuReNUS의 다른 진단도 있다. 첫 gradient norm은 seeds271/272/273에서 .972/1.328/1.483인데, 가져온 공식 config의 gradient perturbation norm은20이었다. 비율은20.57/15.06/13.49배다. 모델 축소 후 해당 scale을 보정하지 않았다. 이는 확인된 설정 불균형이며, 실패의 지배 원인인지는 별도 ablation 없이 확정할 수 없다. 지금은 이 solver 튜닝으로 연구 범위를 확장하지 않는다.

**① 개선안: 공통 곡률 보정을 송신 전에 적용해 삭제 보정량을 직접 합산**

성공했던 고정-feature ridge head를 그대로 둔다. 잔존 client의 local residual을 b_i=H_iW₀−C_i, 데이터 비중을 p_i라 하고 공통 보정 행렬을 P=H_R⁻¹라 하자. 모든 client가 **같은 P**를 적용하면

    Δ_i = −P b_i
    Σ_i p_i Δ_i = −P Σ_i p_i b_i = W_R* − W₀.

따라서 서버는 수신한 삭제 보정량을 W₀에 더하면 된다. 실제 통신에서는 noisy H_hat_R의 공통 기저 Q와 고유값 λ를 쓰므로 마지막 등식은 근사다. Client들은 Qᵀb_i를 λ로 나눠 보낸다. 서버는 합계에 Q만 적용한다. 수신한 gradient에 서버가 뒤늦게 역곡률을 곱하던 이전 방식과 무잡음 연산은 같다. 이번 수치 검사는 이 등식도 확인했다.

각 client가 자기 H_i⁻¹를 따로 쓰면 위 등식은 깨진다. 예를 들어 H₁=1,H₂=4,b₁=b₂=1이면 local inverse 적용 후 평균은.625, global Hessian inverse 적용 결과는.4다.

차이는 통신 잡음의 위치다. 단일 블록에서 p_i를 local payload에 포함하고, 수신 noise covariance가 σ²I일 때

    합산 후 보정:  δ_hat = P(b + a_raw z),    noise MSE = σ² a_raw² ||P||_F²
    송신 전 보정:  δ_hat = P b + a_pre z,    noise MSE = σ² a_pre² d

이다. a_raw와 a_pre는 각 방식의 실제 payload 및 channel에 맞춘 전력 정규화 계수다. 사전 보정의 이득 조건은 d a_pre² < a_raw²||P||_F²다. 잡음 위치만 바꾸고 동일한 a를 가정하면 불공정하다. `math_checks.json`에는 동일 전력에서 사전 보정이 좋아지는 예와 약1.98배 나빠지는 반례를 모두 기록했다. 여러 출력열이면 해당 차원 수를 포함한다.

다중 block에서는 `Σ_block variance_block × residual_inverse_Frobenius² / repetition_block`을 최소화한다. 기존 방식과 후보 모두 같은 정수 최적 반복 배정 코드를 쓴다. 이 비교에서 알고리즘의 삭제 목적함수나 재학습 reference는 바꾸지 않았다.

**① 작은 CUDA 진단 결과**

이전과 동일한 FashionMNIST, 공통 무작위 고정 encoder, 65×10 ridge head, λ=.01,3 seeds×16 통신 잡음,20dB다. 오차는 `||W_hat−W_R*||² / ||W₀−W_R*||²`이며 no-op=1이다.

| 방식 | 평균 정규화 제곱오차 ↓ | 총 complex channel uses |
|---|---:|---:|
| 기존 uniform,40회 | .050145 | 95,611 |
| 기존 gradient-MSE,40회 | .021867 | 95,611 |
| 기존 deletion-MSE,40회 | .020228 | 95,611 |
| 기존 deletion-MSE, 추가 비용 허용48회 | .017838 | 96,196 |
| **송신 전 공통 보정,40회** | **.007300** | **96,236** |

후보의 추가 비용은 고유값65개 방송 약625 uses다. 기존 방식에도 이만큼 예산을 주면8회 반복을 추가할 수 있다. 마지막 두 방식의 비용 차이는 약40 uses,0.042%다. 후보는 원래 강한 비교군 대비63.9%, 추가 예산 비교군 대비59.1% 낮은 오차를 보였다. 초기 noise seeds를 재사용한 결과이며 독립 재현 성공률이 아니다.

후보3개 seed 평균은.007245/.007426/.007229였다. 이 조건부 Gaussian 모형의 분석적 기대오차는.007316으로 실제 평균과 가깝다. 기존 방식에 모든 추가 비용을 fractional repetition으로 사용할 수 있게 한 낙관적인 하한의 기대오차도.017991이었다. 따라서 이 모형에서 보인 차이를 단순 추가 송신 예산만으로 설명하기는 어렵다. 이 하한은 **고정된5개 block과 같은 H·기저** 안에서의 하한이며 모든 통신 설계에 대한 하한이 아니다.

이전 평균전력-only 모형은 peak를 제한하지 않았다. 후보만 유리하게 다루지 않도록 두 방식 모두 최대 복소 심볼 전력을 허용 평균전력의4배로 제한하는 scale도 검사했다.

| Peak≤4인 추가 진단 | 평균 정규화 제곱오차 ↓ |
|---|---:|
| 기존 deletion-MSE,48회 | .110861 |
| 송신 전 공통 보정,40회 | .010236 |

송신 평균전력≤1,peak≤4를 실제 계산한 payload로 확인했다. 전송 좌표의 크기 분포가 달라지므로 peak 제한의 영향도 달랐다. 이는 RF 측정이나 CSI/CFO 내성 검증이 아니다. 현재 자료는 이상적인 coherent, unit-gain, AWGN 모형이다.

**아직 해결하지 않은 비용과 신규성**

Hessian 수집·기저 전달의 큰 준비비용은 그대로다. 앞선82.9% 병목을 해결한 것처럼 쓰지 않는다. 수정의 실제 이점은 비슷한 총비용에서 더 작은 삭제 보정 오차다. 작은 convex-head에서 확인한 결과를 전체 DNN의 역사적 영향 제거로 확대할 수 없다. 다음 최소 비교는 이 same-head 설정에서 같은 목표오차에 필요한 총비용, noisy 충분통계 재학습, diagonal/block 대체를 함께 보는 것이다. 일단 다른 비선형 Newton solver로 갈아타지 않는다.

[OTA Fed-Sophia §III-A/B](https://arxiv.org/html/2410.07662v1)는 local gradient·diagonal curvature의 집계를 받아 서버가 나누고 clipping한다. 이번 후보는 공통 곡률을 먼저 공유하고 residual을 client에서 보정한다는 차이가 있다. 그러나 **송신 전 선형 가공 자체는 일반 OTA learning에도 쓰이는 설계**이며, [COTAF](https://arxiv.org/abs/2009.12787) 등 precoding 선행연구가 있다. 따라서 “공중에서 Newton/삭제 방향을 합했다”만으로 새롭다고 주장하지 않는다.

논문 기여 후보는 **삭제 시점의 작은 보정량을 어느 쪽에서 가공해야 최종 삭제오차를 적은 총비용으로 맞추는가**, 그리고 같은 common curvature를 사용해 local-inverse 편향 없이 구현하는 조건이다. 전력·peak·준비비용을 포함한 비교가 핵심이다. 이 문제 설정과 방법이 기존 communication-aware Newton 연구와 충분히 다른지는 추가 원문 대조가 필요하다. 현재 수치로 국내학회 신규성까지 확정하지 않는다.

**①의 정보 공개 조건에서 확인한 구체적 한계**

“각 client의 전체 gradient를 직접 보내지 않는다”와 “서버가 그것을 추론할 수 없다”는 다르다. 이번 정확한 quadratic 출발점에서는 삭제 client 비중을 α라 하면

    0 = g_all(W₀) = α g_A(W₀) + (1−α)g_R(W₀).

서버가 무잡음 H_R와 완전한 보정량 Δ=−H_R⁻¹g_R를 알면

    g_A(W₀) = ((1−α)/α) H_R Δ

로 target의 gradient를 역산할 수 있다. 이는 단순한 추상적 우려가 아니라 이 이상적 실험 조건의 반례이며 `check_stationarity_disclosure.py`에서 확인했다. 처음의 서버 후처리 방식에도 똑같이 적용된다. Noisy H·수신 잡음이 있으면 정확도가 낮아지지만, 잡음이 있다는 이유로 비공개나 DP가 증명되지는 않는다. 따라서 이번 .00730은 **삭제 보정의 전송 성능 결과**이며 사용자가 요구한 full-transcript 정보 제한까지 충족한 결과로 발표하지 않는다.

추후 동일 방향을 유지하면서 그 제한을 만족시키려면 공개되는 보정의 범위나 보장 조건을 명시적으로 바꿔야 한다. 예를 들어 고정된 r<d 차원 공통 부분공간에만 보정을 허용하면 이 특정 역산식으로 얻는 정보도 제한되지만, 일부 삭제 성분을 포기하므로 approximation error가 추가된다. 동일 지점에서 다른 기저를 반복 조회하면 다시 전체 정보가 드러날 수 있다. 이 변경은 아직 이번 실험에 넣지 않았으며 새로운 privacy 보장으로 주장하지 않는다. 정확한 최적점 가정, 출력으로부터의 추론, 보조정보를 명확히 하지 않은 채 단지 AirComp라는 이유로 안전하다고 쓰면 안 된다.

이번 곡률 기저·고유값은32bit 하향 비용만 반영하고 실제 적용은 FP64로 수행했다. 실제 양자화 오차와 인증된 privacy는 다음 구현 범위다.

**② 개선안: 클래스 구성 차이를 제거한 점수와 작은 연속 보정부터**

기존2번은 kernel 정규화와 activation 차이로 channel을 선택한 다음 선택 채널을50–75% 줄였다. 사용자 client0은 특정2개 class를 더 많이 가진다. 단순 평균 차이는 그 구성 차이를 client 고유 기여처럼 취급할 수 있다.

    s_raw = Σ_c π_A,c μ_A,c − Σ_c π_R,c μ_R,c
          = Σ_c π_A,c(μ_A,c−μ_R,c) + Σ_c(π_A,c−π_R,c)μ_R,c.

여기서 첫 항 s_matched는 같은 class 안에서 A와 잔존 데이터를 비교하고, 두 번째 항은 class 비율 차이만으로 생긴다. 이번 실제 checkpoint에서는 다음처럼 구성 차이 항이 컸다. 벡터들이 상쇄할 수 있으므로 norm을 영향 비율로 해석하지 않는다.

| Seed | 원래 점수 norm | Class 구성 차이 항 norm | 같은 class 비교 항 norm |
|---|---:|---:|---:|
| 271 | .2502 | .2519 | .0329 |
| 272 | .1808 | .1958 | .0268 |
| 273 | .2368 | .2460 | .0285 |

양쪽에 class c가 존재하면 각 client가 π_A,c와 global retained class count로 local 통계를 가중해 **24차원 벡터 하나**로 합친 뒤 OTA 전송할 수 있다. 서버는 client별 full gradient를 받지 않는다. Class count를 추가 공개·방송하는 비용은 이번 단순 구현에서 약1,190 real uses다. 통신 이득이라고 하기에는 기존 통계 전송보다 큰 추가비용이므로 공개 가능한 사전 count가 있는지 등 조건을 먼저 정해야 한다. ①의 complex-use 단위와 직접 섞어 비교하지 않는다.

또한 kernel Gram 오차가 줄어도 실제 feature correlation은 기존3개 seed 모두 조금 증가했다. `WWᵀ≈I`만으로 사전 얽힘 방지가 성공했다고 볼 수 없다. 다음 학습 변형을 한다면 feature의 class-conditional covariance를 관찰하고, kernel 정규화만 강화하기 전에 실제 분리가 생겼는지 확인해야 한다. Feature decorrelation 역시 client 영향의 독립성을 증명하지는 않는다.

큰 pruning 강도를 원인에서 분리하기 위해 같은6개 channel에 고정10% 감쇠를 적용했다. 같은 checkpoint를 사용하는 사후 진단이며 강도를 test/reference로 선택하지 않았다.

| 방식 | 삭제 데이터의 재학습-reference JS / no-op JS ↓ |
|---|---:|
| No-op | 1.000 |
| 기존 점수·50–75% 감쇠 | 24.625 |
| 기존 점수·10% 감쇠 | 1.149 |
| 같은 class 비교 점수·10% 감쇠 | 1.131 |
| Random 채널·10% 감쇠 | 1.081 |

과도한 개입의 문제는 확인됐지만, **같은 class 점수로 바꾼 것만으로 삭제 성공은 확인되지 않았다.** Random보다 평균적으로 좋은 것도 아니다. 따라서 다음 개선을 “pruning 양을 줄여서 성공”으로 표현하지 않는다. 핵심은 원래 순위가 무엇을 나타내는지 고치고, 확실한 신호가 없는데 항상25%를 선택하는 규칙을 재검토하는 것이다. 이후에는 같은-class 점수의 신뢰도와 AirComp noise를 이용한 무개입/작은 gate 보정을 후보로 검토할 수 있지만, 이것 역시 아직 구현·성공한 FU 알고리즘은 아니다.

[FedOrtho 본문 §3, 식4](https://arxiv.org/html/2506.19891v2)는 forget sample의 true-label probability를0에 가깝게 하는 조건도 사용한다. 원 논문은 class/client/sample 실험을 모두 제시하지만, 이 조건을 **같은 class를 잔존 client도 가진 clean-client 삭제**의 충분조건으로 가져올 수는 없다. 재학습 모델도 해당 sample을 맞힐 수 있기 때문이다. 원문 전체를 실패했다고 평가하지 않고 우리 삭제 단위·점수·reference의 적합성을 검토한다.

**현재 우선순위**

①은 “삭제신호를 만들어 공중 합산한다”는 의도를 유지한 작은 수정에서 수학적 동치와 실제 수치 개선이 확인됐다. 성공했던 convex-head를 유지하고 total-cost·공통곡률 준비 문제를 좁혀 보는 것이 합리적이다. ②는 class 구성 혼동과 너무 큰 intervention이 확인됐지만 새로운 삭제 규칙의 우월성은 아직 없다. Retain recovery나 SFL,③으로 이 판단을 대신하지 않는다.

실행 원자료: `math_checks.json`, `curvature_rows.json`, `class_score_rows.json`, `cost_peak_rows.json`, `summary.json`, `cost_peak_summary.json`. 코드: `check_signal_design.py`, `check_cost_and_peak.py`. 실행 범위와 사후 추가 비교는 `PROTOCOL_KO.md`, `COST_PEAK_ADDENDUM_KO.md`에 보존했다.
