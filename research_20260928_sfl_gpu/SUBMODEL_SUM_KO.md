# Submodel을 공중에서 합산할 수 있는가?

**가능하다.** 앞의 보고서에서 ‘이번 보정 후보를 새 논문 기여로 채택하기 어렵다’는 판단은 공중 submodel 결합 자체가 불가능하다는 뜻이 아니다. 실제 학습 실험에서도 branch 출력의 OTA 합산이 동작했다.

## 가장 직접적인 channel 분할

원래 convolution의 입력 채널을 장치별로 나눈다. 장치 k는 자신이 맡은 입력 채널 h_k와 그에 대응하는 kernel W_k로, **모든 장치에 공통된 출력 채널 좌표**의 기여 u_k를 계산한다.

$$u_k=W_k*h_k,\qquad z=\sum_k u_k+b,\qquad y=\phi(z).$$

A가 맡은 입력 채널에서 오는 기여, B가 맡은 입력 채널에서 오는 기여를 무선에서 합친다. 적절한 channel inversion과 정렬을 하면 수신값은 z에 수신 잡음이 더해진 형태다. 잡음이 없으면 원래 convolution과 정확히 같다. 이 식에는 **sketch, 저차원 근사, Gram–Schmidt가 필요 없다.** Bias는 합산 후 한 번만 더한다.

서버가 δ=∂ℓ/∂z를 broadcast하면 각 장치는 자신의 W_k,h_k에 대한 convolution backward를 계산할 수 있다. 다른 장치의 원본 feature나 weight를 모두 받을 필요는 없다. 단, 공통 입력의 channel 분할, sample/공간 위치 정렬, 송수신 조건은 갖춰야 한다.

`verify_channel_sum.py`는 8 input channels를 4장치에 나눠 12 output channels의 3×3 convolution을 만드는 경우를 GPU float64로 검산한다. Full convolution과 공중 합산 전의 수학적 partial sum의 forward/backward가 일치하는지 확인한다. 무선 잡음 하의 exact equality를 주장하는 검산은 아니다.

실제 검산 결과 forward 최대오차 **1.07e-14**, backward 최대오차 **0**으로 통과했다. [수치 원장](channel_sum_validation.json)

## 독립 submodel 전체를 끝에서 합치는 것도 가능한가?

같은 샘플 x에 대해 각자 작은 네트워크 f_k를 돌리고, 합산형 모델을 처음부터

$$F(x)=g\left(\sum_k f_k(x)\right)$$

로 정의하면 가능하다. 각 f_k의 출력 차원/의미를 공통으로 설계하고 합산 결과의 loss로 공동 학습하면 된다. 이번 `width20` 실험은 이 계열의 channel branch 모델이다. 이 모델은 하나의 큰 dense CNN을 아무 곳이나 잘라 만든 것과 자동으로 같지는 않다.

큰 dense CNN과 **층마다 같은 계산**을 유지하려면 각 층의 cross-channel 결합을 구현해야 한다. 별도의 작은 CNN 여러 개를 끝까지 실행한 뒤 한 번만 더하면 보통 ensemble/branch 구조라는 다른 모델이 된다. 어느 쪽도 불가능한 것이 아니라, 보존하려는 모델이 다르다.

## 무엇을 바로 더하면 안 되나?

- 역할이 다른 raw feature h_A,h_B를 대응 가중치 적용 없이 더하면 channel 정보가 사라질 수 있다. W_A=1,W_B=2일 때 입력(1,1)과(2,0)은 raw sum이 둘 다2이지만 필요한 weighted output은3과2다.
- 비선형 연산의 위치를 임의로 바꿀 수 없다. ReLU(1−1)=0이지만 ReLU(1)+ReLU(−1)=1이다.
- 서로 다른 샘플의 출력을 합하면 원래 샘플별 loss가 보존되지 않는다. 위 식은 같은 샘플/사건에 대한 공동 계산이다.

따라서 현재 결론은 **‘submodel 결합은 가능하며, 이를 기본 구조로 사용할 수 있다’**다. 별개로 따져야 하는 것이 그 구조 위에서 무엇을 새로 개선할지, 실제 전체 통신비와 장치 부하가 얼마나 줄어드는지다. 선행 중 MIMO OTA-SL은 선형층/conv의 무선 구현을 이미 다뤘고, FedDCT는 폭 분할 공동 학습을 다뤘으므로, 결합 가능성만으로 최초성을 주장하지 않는다.
