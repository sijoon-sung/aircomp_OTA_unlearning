**② 직교화 계열 — 보류한 OG-Air와 현재 후보인 Sketch-Air 부분 투영형**

현재 설명할 후보는 **Legacy-Sketch-V1에서 투영 강도를 .5로 완화하고 A가 마지막 한 번 참여하는 방식**이다. 설명용으로 ‘②-B Sketch-Air 부분 투영형’이라 부른다. 처음 제안한 사전 직교화가 성공했다고 이름을 바꾸는 것이 아니다.

**②-A: 처음 제안과 보류 이유**

처음 제안은 학습 중 CNN kernel에 ||WW^T-I||_F² 정규화를 걸어 얽힘을 줄이는 것이었다. OG-Air Original은 A와 잔존 데이터의 채널 활성도 차이로 채널을 골라50~75% 감쇠했다. OG-Air V1은 같은 학습 모델에서24차원 gate gradient의 alpha(g_A-g_R)를 공중 합산해 작은 gate 보정을 했다.

Original의20dB 정규화 삭제JS는12.4667, 무잡음도32.1267이었다. V1은.999739로 미삭제1과 거의 같았다. Kernel 직교성이 client별 기능 분리를 보장하지 않고, 활성도 차이는 class 구성 차이를 반영할 수 있으며, gate 보정은 작고 과거 학습을 되돌리는 식이 아니었다. [FedOrtho](https://arxiv.org/html/2506.19891v2)의 전체 방법을 반박하는 결과가 아니라, 우리 축소 구현과 적용 조건의 결과다.

**②-B: 삭제 시점의 관계를 이용하는 방법**

기존 글로벌 CNN theta_0를 유지한다. A는 삭제용 UCE gradient를, 나머지는 CE gradient를 계산한다. UCE는 L_U=-mean log(1-p_y/2)이며 이 손실을 gradient descent로 줄인다. 정답확률을 억제하는 목표가 target-free 재학습과 자동으로 같아지는 것은 아니다.

g_A^U=grad L_U, u=g_A^U/||g_A^U||, u_i=g_i^CE/||g_i^CE||, G=[u_i]_{i≠A}로 둔다. 정확한 관계계수 c=G^+u에 대해

`v_beta = u-beta Gc = (I-beta P_G)u`, `P_G=GG^+`.

beta=1은 완전 투영, .5는 부분 투영,0은 투영 없음이다. 현재 주 후보는 .5다. Gram–Schmidt·QR·SVD는 projection을 계산하는 방법이며, 다른 client의 정보를 얻는 통신 방법은 아니다. 구현은 작은 sketch 행렬의 SVD를 사용한다.

**누가 무엇을 계산하고 보내는가**

1. 서버가 현재 theta, 고정 공개 SRHT seed, sketch 크기2048, beta=.5, 학습률.01, 참여 집합을 방송한다.
2. 각 client는 로컬 gradient와 norm을 계산한다. A는 UCE, 나머지는 CE를 사용한다. 16bit로 양자화한 sketch s_A=S u, s_i=S u_i와 허용한 norm/scale metadata를 보낸다. 서버는 전체 gradient를 받지 않는다.
3. 서버는 sketch만으로 c_hat=argmin_c ||s_A-[s_i]c||²를 계산한다. 계수 w_A=1, w_i=-beta*c_hat_i를 방송한다.
4. A는 자신의 u를, 나머지는 -beta*c_hat_i*u_i를 같은 자원에서 동시 송신한다. 물리 채널 보상과 전력 정규화를 적용한다.
5. 서버는 y≈u-beta Gc_hat를 수신하고 d=||g_A^U|| y/||y||로 정규화한다. y가 사실상0이면 보정을0으로 둔다. theta_U=theta_0-.01d로 한 번 수정한다.
6. 결과 모델을 배포하고 A는 더 이상 참여하지 않는다. 이번 후보에는 이후 recovery가 없다.

```text
서버 → 현재모델·고정sketch설정 → A와 잔존clients
각client → 작은sketch·norm → 서버
서버 → 관계계수 → A와 잔존clients
A의 u + 잔존clients의 (-.5*c_i*u_i) → 동시 OTA 합산
서버 → 수신방향으로 최종1회 보정
```

**현재 구현에서는 A가 다른 client의 합을 받아 혼자 완성된 투영벡터를 계산하지 않는다.** A는 자기 gradient를 정규화해 보내고, projection의 보정 항은 다른 client의 가중 송신에 실린다. 그래서 서버가 완전한 retained aggregate를 A에게 알려줄 필요가 없다. ‘서버→A만 지시→A만 보정’과 동일한 구조로 설명하면 안 된다.

전체 gradient payload는 사용자별 직교 자원으로 분리하지 않는다. 보조 sketch와 계수는 디지털 메시지다. 따라서 모든 통신이 단 한 번의 analog 송신으로 끝나는 방법도 아니다. 여기서 ‘마지막1회 참여’는 sketch·계수방송·최종OTA를 포함하는 하나의 삭제 보정 절차다.

**직교성을 완화하는 수학적 의미**

정확한 projection에서는 G^T v_beta=(1-beta)G^T u다. beta=.5는 잔존 방향과 겹치는 성분을0으로 만들지 않는다. 대신 완전 projection이 제거하던 일부 방향을 유지한다. 이것이 삭제에 도움이 되는지는 재학습 reference로 평가해야 한다.

P_G u와(I-P_G)u가 직교이므로 ||v_beta||²=(1-beta)²||P_Gu||²+||(I-P_G)u||²다. 완전 투영은 residual을 작게 만들 수 있다. 같은 peak 전력 제약 아래 수신잡음이 상대적으로 커지고, 서버의 norm 재정규화가 잡음 방향을 확대할 수 있다. 다만 실제 noise scale도 송신계수와 payload에 따라 바뀌므로 단순한 norm 식만으로 최종 이득을 증명하지 않는다.

완전 직교성은 일반적인 삭제 필요조건이 아니다. 특히 CE 정상점에서 A와 retained 평균 gradient가 반대 방향이면 그 평균에 직교화한 A gradient가0이 되는 반례가 있다. 그러나 실제CNN은 정상점에 충분히 가깝지 않았고 UCE를 사용하므로, 이 반례를 실제 실패의 직접 원인으로 단정하지 않는다.

**정보 공개 조건**

서버는 개별 sketch·norm·전송scale, 관계계수, 수신보정, source/결과모델을 안다. beta=0에서는 나머지 계수가0이 되어 A-only 방향이 노출되므로 배포 후보에서 제외한다. beta=.5도 privacy 변환이 아니다. 서버가 정확한 P_G와 정규화 전 v를 안다면

`u = [I + beta/(1-beta) P_G]v` (beta<1)

로 되돌릴 수 있다. 실제 sketch protocol이 P_G 전체를 공개하지 않는다는 점과, 전체 transcript로 다른 역산이 불가능하다는 증명은 구분한다. 이번에는 전자를 구현했으며 후자는 미해결이다. 같은 gradient에 여러 beta를 요청하거나 독립 sketch를 추가해 정보가 누적되는 서비스로 바꾸면 안 된다. 실험의 여러 비교군은 별개의 protocol 실행이다.

**실험: 마지막1회 참여**

FashionMNIST private12000,10clients,client0 삭제. Plain FedAvg150rounds,ConvNet2 63,562parameters,새seed381/382/383. 각seed에 retained reference를2개 준비했다. 삭제 gradient는 client당256 samples,lr=.01,주 비교에서 첫 step norm을 동일하게 맞췄다. Noise는20/40dB 각4회이며 첫 reference 대비 normalized forget JS를 사용한다.

| 조건 | beta=1 | beta=.5 | 상대 오차 감소 |
|---|---:|---:|---:|
|정확한 full-vector oracle|.8672|.7035|18.9%|
|Sketch/무잡음|.8835|.7053|20.2%|
|AirComp20dB|.9812|.8670|11.6%|
|AirComp40dB|.8947|.6784|24.2%|

20dB에서 beta별 비용은1,557,477 real uses로 같다.40dB의 비용은 높은 goodput로 환산해816,134이며,20dB goodput 고정 환산으로는 동일하다. 다른SNR을 같은채널 비용절감으로 주장하지 않는다.

20dB의 수신/이상적 방향 cosine은 완전.0896→부분.1741,noise/signal norm은11.25→5.73이었다.40dB cosine은.6675→.8669였다. 무잡음에서도 부분 투영의 이득이 있어 단순한잡음효과로만 설명할 수 없다. 하지만20dB의 방향왜곡 자체는 여전히 크다.

대체 reference에서는20dB 부분/완전 평균.9594/.9959,40dB .9177/.9879였다. 평균 경향은 유지됐지만40dB의 한seed에서는 순서가 바뀌었다. 따라서 일반적인 통계적 우월성이나 완전 삭제로 표현하지 않는다.

**채택하지 않는 확장**

같은 보정을10번 반복하면 무잡음 완전7.05/부분105.79,40dB 완전3.73/부분75.25로 악화됐다. 부분 투영은 계속 UCE를 줄이면서 공유된 분류 지식까지 크게 바꿨다. 정확한 원인별 기여를 분리한 것은 아니지만, 현재10회 반복을 채택할 근거는 없다. Step 크기 정규화를 없앤 진단에서도 반복 삭제가 성공하지 않았다.

**선행연구와 현재 판정**

[FedOSD](https://arxiv.org/html/2412.20200v1)의 UCE/retained-gradient projection과 기존 Legacy-Sketch-V1이 직접적인 출발점이다. 원 논문의 recovery까지 재현한 실험은 아니다. 투영 강도 완화만으로 신규 알고리즘을 주장하지 않는다. AirComp 관점에서 검토할 지점은 작은 residual의 전력·잡음·삭제오차 관계다.

**②-A는 보류.②-B는 마지막1회 참여·부분 투영의 제한된 후보로 유지한다. Full-transcript 정보 조건은 아직 인증하지 못했다.**

[99개trajectory 실험보고서](../research_20260927_orthogonality/RESULTS_KO.md) · [실험코드](../research_20260927_orthogonality/run_experiment.py) · [원자료](../research_20260927_orthogonality/results/summary.json)
