# 이번 AirComp 실험의 구조와 해석 범위

## 서버가 수신하는 값

하나의 SISA shard 안에서는 동시 아날로그 전송으로 update 평균을 받는다. shard 사이에는 직교 슬롯/주파수를 사용한다. shard 모델은 서로 공유하지 않고 예측만 평균한다. 삭제 시 해당 shard만 초기 상태부터 재학습한다. 여러 shard를 같은 시간·같은 주파수에서 임의로 섞어 놓고 분리가 된다고 가정하지 않는다. 다중 안테나를 이용한 공간 분리는 이번에 구현하지 않았다.

삭제 후 exact 비교의 reference는 최초 shard 배정을 고정하고 삭제 client만 제외하여 처음부터 학습한다. 공개된 histogram/routing 자체를 삭제하거나, 삭제 후 배정을 새로 계산하는 전체 절차와 동일하다고 주장하지 않는다.

각 client update u_i는 L2 norm C=.1로 clip한다. m명, 모델 차원D, 채널 진폭h_i, per-symbol 평균 전력 상한P=1일 때

\[
\beta_{\rm phys}=\frac{C}{m\sqrt{DP}\min_i|h_i|},\qquad x_i=\frac{u_i}{m h_i\beta}.
\]

위상은 정확히 보상한다고 가정한다. R회 반복된 동일 update를 평균하고 rescale하면

\[
\widehat u=\frac{1}{m}\sum_i u_i+z,\qquad
z\sim\mathcal N(0,\tau^2 I),\quad\tau^2=\frac{\beta^2\sigma^2}{R}.
\]

따라서 고정 전력 한도에서 β=β_phys를 쓰면 집계 MSE는 C²σ²/(R m² P min|h|²)다. 잡음을 줄이기 위한 반복 전송은 비용을 늘린다. 명목 SNR은P/σ²이며, 실제 update 평균의 에너지로 계산한 집계 SNR과는 다르다. 두 값을 별도로 저장한다.

## 이번 joint 배정

랜덤 배정과 채널순 배정에서 시작하여 shard 간 client를 한 명씩 교환하는 local search다. 목표는 정규화한 AirComp source/deletion MSE 지표와 shard label 분포의 JS divergence를 같은 비중으로 줄이는 것이다. 가능한 삭제 client20명을 모두 평균한 radio 지표를 이용하여 client0의 채널만 유리하게 최적화하지 않는다. 그러나 실제 학습/삭제 성능의 주 실험은client0이고 일부 조건만client7/14를 추가 검증한다.

목표함수 계수 .5/.5는 사전에 고정했고 SNR별로 재조정하지 않았다. 라운드별 fading에 따라 grouping을 변경하지도 않는다. 그러므로 최적 또는 SNR-adaptive grouping이라고 부르지 않는다. 배정은 초기 slow CSI와 제공받은 label histogram만 사용하고 dev/test 정확도로 선택하지 않는다. hist 공개와 conditional privacy 범위를 모든 방법에서 동일하게 맞춘다.

## 조건부 privacy와 shard 크기

공개된 label histogram/count/CSI/routing을 유지하면서 client 하나의 feature 데이터 전체를 교체하는 인접관계에서 한 번 공개하는 평균의 민감도는 Δ<=2C/m이다. 정규분포 잡음을 갖는 집계는 rho=Δ²/(2τ²)의 zCDP 상한을 갖는다. 전체 source+한 번의 삭제 재학습 transcript에 대해 client별 참여 round의 rho를 더하고 가장 큰 값을 사용한다. 삭제 client는 원래 공개된 학습 transcript의 privacy 비용이 사라지지 않는다.

자연 잡음과 최대 허용 alignment를 사용하면

\[
\rho_t=\frac{2DP R\min_i|h_i|^2}{\sigma^2}.
\]

이 이상적인 모델에서는 m과C가 식에서 상쇄된다. 따라서 같은 bottleneck channel을 유지한 채 shard 크기만 바꾸면 이 보수적 per-round privacy 상한은 같다. 실제 grouping에서는 bottleneck channel, 학습 과정, 삭제 시 참여 구성과 round 수가 달라져 전체 trade-off가 바뀐다. R을 늘리면 MSE는 줄지만 rho는 늘어난다. SNR을 높여 잡음을 줄여도 같은 방향의 trade-off가 있다.

큰 epsilon 상한을 얻었다는 것은 해당 분석으로 강한 보호를 입증하지 못했다는 뜻이다. 실제 공격에 반드시 취약하다고 증명하거나, 정확한 최적 epsilon을 계산한 것은 아니다. 소규모20 clients, client 전체 feature 교체, 모든 client가 매 round 참여, 고차원 모델에 대한 보수적 상한이라는 조건을 함께 읽어야 한다.

epsilon8/100 제한 조건에서는β를 더 크게 잡아 송신 전력을 낮추고, rescale 후 잡음을 키워 동일 source+한 삭제의 한도를 맞춘다. 이때 정확도가 크게 낮아질 수 있으며 그대로 보고한다. 충분히 좋은 결과가 나올 때까지 조건을 바꾸지 않는다.

Gaussian zCDP와 합성 근거: https://arxiv.org/abs/1605.02065 . 무선 잡음/전력 제어를 통한 privacy 선행연구: https://arxiv.org/abs/2006.05459 . 이 문서는 해당 가정 아래 구현을 설명하며, 모든 신호·메타데이터와 공격자를 포괄하는 새 privacy 정리를 주장하지 않는다.
