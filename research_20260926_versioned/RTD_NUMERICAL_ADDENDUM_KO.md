**③ 수치 정밀도 수정 — 최초 결과 보존 후 동일 조건 재비교**

최초 RTD 실험에서 무잡음 V1의 target 확률은 reference와 최대1.79e-7 달랐지만,600step KD 후 모델 차이가 커졌다. Teacher의FP32 softmax확률합이정확히1이아니고,FP32평균·9차원변환·simplex복원순서도다르다. 따라서 수학적으로 같은연산을실험하면서reference의부동소수연산을일치시키지못했다. 이 차이를AirComp잡음이나삭제성능의효과로계산하지않도록양쪽을동시에수정한다.

1. 동일한저장teacher가계산한확률을FP64로변환하고client에서sum=1로정규화한다.
2. Original평균과V1 Helmert변환/수신/복원을모두FP64로계산한다. Helmert행렬도FP64다. 송신전력은수정한실제payload로다시계산한다.
3. Reference는이정규화된확률의FP64평균을마지막에FP32로변환한target으로학습한다. A포함source도같은규칙이다. Student는기존과동일한FP32 CNN,lr.05,600steps,같은초기값과minibatch난수다.
4. 무잡음Original/V1 target의FP32값을reference와비교한다. 작은roundoff를숨기기위해reference모델을사후선택하거나noise조건별로바꾸지않는다.
5. 처음의결과는`RTD_seed*.json`에그대로보존하고수정결과는`RTDfp64_seed*.json`에쓴다. 기존teacher와plainFedAvg외부비교군도보존하여재사용한다. 새source/referenceKD만필요하다.

이는통신·정규화의수치수정이며더좋은test결과를선택하는hyperparameter탐색이아니다. FP64를사용해도noisyKD가항상안정적이라는보장은없다. 최종보고에서는최초문제와수정후무잡음오차를명시한다. 학술적으로는FP64전처리의client계산비용도고려해야하며32bit RF성능을실측했다고하지않는다. 전송되는파형좌표수와모델배포bit수는변하지않는다.
