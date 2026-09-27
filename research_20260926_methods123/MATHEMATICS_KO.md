**실험 이전에 확인한 성립 조건과 반례**

2026-09-26. 이 문서는 알고리즘의 계산 가능성과 삭제 보장을 구분한다. 수학 검사 원자료는 `results/math_validation.json`, 실제 CNN 연결 검사는 `results/integration_checks.json`이다. CPU/GPU의 작은 수치 오차는 정리의 일반 증명을 대체하지 않는다.

**① 분산 HVP는 계산할 수 있다. 이것만으로 재학습과 같아지지는 않는다.**

삭제 후 경험위험을 L_R(θ)=Σ_i p_i L_i(θ), Σp_i=1로 정의한다. 모든 client가 같은 θ와 같은 query v를 사용하면 미분의 선형성에 의해

    g_R(θ)=Σ_i p_i g_i(θ),      H_R(θ)v=Σ_i p_i H_i(θ)v

가 성립한다. 데이터 수가 다르면 p_i를 맞춰야 한다. 각 local에서 여러 SGD step을 먼저 진행한 뒤 계산한 H_i(θ_i)v는 H_R(θ)v와 같지 않다. 이번 unlearning은 매 outer step 같은 θ를 방송하고, 고정한 local Hessian minibatch에서 여러 v를 조회한다.

Smooth tiny model의 dense Hessian, local HVP 합, 중심차분을 비교한 오차는 각각 약1.1×10⁻¹⁶,1.8×10⁻¹⁰이다. 실제 학습된 CNN에서는 저자 코드의 global HVP와 우리가 만든 local HVP 합의 상대오차가2.45×10⁻⁷이었다. 이 정도 차이는 FP32 연산 순서의 차이다.

고정 특징의 ridge head라면 g_R=H_R(W₀−W_R*)이므로 W₀−H_R⁻¹g_R=W_R*가 정확하다. 반면 일반 CNN은 H가 θ에 따라 변하고 훈련 알고리즘도 유한 step이므로, Newton step이 특정 SGD 재학습 모델과 동일하다는 등식은 없다. 동일하게 낮은 retained loss를 갖는 다른 모델도 가능하다.

CuReNU의 cubic surrogate를 q(δ)=gᵀδ+δᵀHδ/2+M‖δ‖³/6로 두면

    ∇q(δ)=g+Hδ+(M/2)‖δ‖δ

이다. 저자 공개 구현은 M/2를 사용한다. 논문 Algorithm2에는 L 계수가 등장하고 본문의 한 식에는 cubic gradient가 빠져 있어, 이식에서는 **공식 코드의 M/2와 Cauchy 초기화·inner LR 감소**를 기준으로 삼았다. 수식의 이름만 보고 임의 구현하지 않았다. M이 실제 Hessian Lipschitz 상수의 상계라는 것은 이번 CNN에서 인증하지 않았으므로 전역 수렴 보장을 가져오지 않는다.

통신 오차가 e_g,e_H(v)라면 inner direction 오차에는 둘 다 들어간다. 작은 고유값/negative curvature뿐 아니라 출발점의 비정류성, 유한 minibatch, perturbation, 제한된 inner step, 재학습 경로 차이가 결과를 바꾼다. 무잡음 solver부터 실패한다면 transmission 반복을 늘려도 그 원인이 해결되지 않을 수 있다.

**② Kernel 직교화와 삭제 mask 계산은 성립하지만, clean-client 삭제는 별도 검증해야 한다.**

Kernel matrix W가 output channel을 행으로 가지면 R(W)=‖WWᵀ−I‖²_F이다. 행 수가 열 수보다 많으면 WWᵀ=I는 rank 때문에 불가능하다. 이번 첫 conv는8×9, 두 번째는16×72여서 가능한 차원이다. 공개 `orth_dist`가 이 모양에서 반환하는 norm을 제곱해 사용했고 직접 계산과 일치했다.

선형 입력 공분산 Σ_x에 대해 Σ_h=WΣ_xWᵀ다. WWᵀ=I라도 Σ_x=I가 아니면 off-diagonal이 남을 수 있다. 실제로 W=I₂, Σ_x의 off-diagonal=.8이면 출력 공분산도.8이다. ReLU·max pooling·후속 학습이 더해지면 단순 kernel 조건만으로 client별 기능 독립성을 얻었다고 할 수 없다.

Activation 통계의 올바른 global 식은

    s_j = Σ_i A^sum_{F,i,j}/N_F − Σ_i A^sum_{R,i,j}/N_R.

Client별 local 평균을 다시 균등 평균하는 것은 local count가 다르면 다르다. 우리의 clean-client 삭제에서는 F는 client0 전체, R은 나머지9명이다. Client0은 mean_F를, 나머지는 −mean_R,i/9를 같은 좌표에 보내 합산한다. 서버가 추가로 보는 정보는 channel별 작은 통계, count와 norm이다. Training update 자체를 복원하지 않는다.

Pruning 경계의 점수 차이가 Δ=s_j−s_k>0이고 수신오차가 Gaussian이면 순위 역전 확률은 Φ(−Δ/√Var(e_j−e_k))다. 따라서 경계 부근 재전송은 mask 오류를 줄일 수 있다. 하지만 mask와 삭제의 관계는 별도 문제다. 특히 **class에 민감한 channel**이 **한 client의 고유 기여**와 같은 것은 아니다. 다른 client가 같은 class를 가지고 있는 이번 문제에서 이 차이가 드러날 수 있다.

추가 전송의 좌표집합은 수신값으로만 정하고, 같은 local 통계·같은 가중치를 반복한다. Subset별 RMS가 달라지므로 초기4회와 추가16회를 무조건4:16으로 평균하지 않고 수신 잡음 분산의 역수로 가중 평균한다. Norm 재전달과 선택 index의 하향 비용도 센다. 반복 측정 자체는 새로운 projection 알고리즘이 아니다.

**③ KD의 계산과 target 독립성은 다르다. 초기 KD의 privacy 퇴화도 존재한다.**

FedQUIT-logit-min은 teacher의 true-class logit을 그 sample의 최소 logit으로 바꾼 후 softmax를 취한다. 이 변환과 KL loss의 student-logit 미분 (p_student−p_teacher)/B를 NumPy 식으로 별도 검사했다. 오차는 각각1.67×10⁻¹⁶,9.32×10⁻¹⁸이다. 이는 TensorFlow 프로젝트 전체를 실행했다는 뜻이 아니다.

Student=original teacher인 초기 시점에는 retained KD gradient가0이다. 따라서

    y = α g_forget + Σ_i β_i g_KD,retain,i = α g_forget

가 되어 aggregate가 target의 전체 gradient를 그대로 드러낼 수 있다. “모두 동시에 전송했으니 개별 update가 감춰진다”는 주장에 대한 반례다. 이번 mixed-KD에서는 retained CE gradient를 포함한다. 이것은 위의 결정적 영벡터 반례를 피하는 것이며 transcript privacy/DP의 증명은 아니다.

다른 구조로 각 client teacher T_i=A(D_i;u_i)를 독립적으로 학습하고, u_i 및 공개 초기값이 target과 독립이라고 하자. Global student의 feedback이 teacher에 돌아가지 않아야 한다. 그러면

    q_R(x)=Σ_{i≠A} p_i T_i(x)

는 D_A에 의존하지 않는다. Student를 target-independent 초기값에서 같은 KD 절차로 학습하면 해당 **독립-teacher 알고리즘**을 처음부터 A 없이 실행한 출력과 같은 분포를 갖는다. 채널 law도 같아야 하고 surviving weights를 재정규화해야 한다. 고정한 같은 난수로 coupling하면 수치적으로 같은 출력까지 검사할 수 있다.

기존 global teacher를 사용하거나 기존 student parameter에서 시작하면 위의 독립성 전제가 깨진다. 또한 이 정리는 monolithic FedAvg 재학습과 같다는 뜻이 아니다. 이번에 이 두 reference를 별도 기록하는 이유다.

**AirComp 송수신과 정보 조건**

Client의 가중 vector를 t_i=p_i u_i, channel을 h_i, scale을 a=max_i RMS(t_i)/|h_i|로 놓는다. Client는 x_i=t_i/(h_i a)를 송신하고 server는

    y=a(Σ_i h_i x_i+z)=Σ_i t_i+a z

를 받는다. Repetition r의 평균 잡음 분산은 a²/(SNR·r)이다. 무잡음 복원의 최대오차는4.44×10⁻¹⁶이었다. Norm을 알고 있는 조건이며 peak-power/clipping을 보장하는 모형은 아니다. 채널 역수 보상이 가능하고 위상 정렬이 완벽한 real MAC 모델이다.

모든 client 데이터·model이 한 연구 프로세스에 존재하는 것은 FL simulation을 위한 구현 편의다. 서버 역할에는 aggregate와 명시한 partial metadata만 전달한다. 저장된 local teacher checkpoint는 simulator의 client-owned reference이지 통신 protocol의 서버 수신물로 세지 않는다. 서버가 개별 vector들을 독립 안테나 관측이나 반복 혼합식으로 복원하지 못하는지는 별도의 full-transcript 조건이며, 현재 실험으로 cryptographic 보안을 인증하지 않는다.
