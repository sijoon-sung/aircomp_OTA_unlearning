# SFL: 별도 비교 트랙

2026-09-28. 일반 OTA-FL 후보 선정과 합치지 않는다. 사용자가 제안한 channel/width 방향을 검토하되, 서로 독립인 client 샘플을 합쳐도 원래 SFL과 같다는 가정을 하지 않는다.

|선행|공중에서 무엇을 합하는가|조건 및 후속 문제|
|---|---|---|
|[Communication-Efficient Split Learning Based on Analog Communication and Over the Air Aggregation, 2021](https://arxiv.org/html/2106.00999v1)|각 view의 선형 projection을 합쳐 다음 activation으로 전달|같은 샘플의 서로 다른 view; projection 연산과 출력 차원도 비용에 포함해야 함|
|[Over-the-Air Split Learning with MIMO-Based Neural Network and Constellation-Based Activation, 2022](https://arxiv.org/html/2210.03914)|MIMO 채널과 precoder/combiner를 NN 선형층으로 사용|준정적 채널, reciprocity를 이용한 역전파; 정확한 reciprocity 밖의 동작은 별도 검증 필요|
|[Latency Minimization in AirComp-Assisted Split Federated Learning with Layer-Wise Aggregation, ICC Workshops 2026](https://ieeexplore.ieee.org/document/11586362)|client-side 모델의 layer별 parameter 집계|Activation/label 업로드는 FDMA. Width 방향 activation 합산과 다름|
|[SplitFC](https://arxiv.org/html/2307.10805v2)|디지털 feature compression|feature dropout/quantization을 사용하는 강한 통신 비교군|

2021/2022 논문의 구조와 가정은 이번에 본문을 다시 확인했다. ICC Workshops와 SplitFC의 상세 내용은 앞선 `research_20260926_followup/SFL_CHANNEL_REVIEW_KO.md`의 본문 검토 기록을 사용했다. ICC 논문의 cut-gradient/broadcast downlink 지연 생략 가정을 총비용 비교에서는 그대로 복사하지 않는다.

가능한 구조는 동일 샘플/정렬된 multi-view 입력에 대해

$$
h=[h_1;\ldots;h_K],\quad W=[W_1,\ldots,W_K],\quad
a=\sum_k W_kh_k+b.
$$

Activation은 합한 뒤 적용한다. 공통 backward signal을 $\delta$라 하면 $\nabla_{W_k}L=\delta h_k^T$, $\nabla_{h_k}L=W_k^T\delta$다. **공통 $\delta$가 client parameter gradient까지 같다는 뜻은 아니다.** 이 선형 대수 자체와 forward 합산은 신규성이 아니다.

서로 다른 이미지/label의 $h_k$를 단순 합치면 horizontal SFL의 원래 손실이 바뀐다. 이 구조를 택하려면 aligned multi-view/model-parallel task를 명시하거나, 별도로 목적함수의 타당성을 증명해야 한다.

별도 후속 후보는 **width/cut/projection dimension을 줄일 때 절약하는 local MAC·메모리와 추가되는 projection·forward/backward 통신의 총비용**을 비교하는 작은 연구다. 유효 채널 rank와 reciprocity 오차까지 조건에 넣어야 한다. Width를 줄여 얻은 계산 절감을 AirComp의 단독 효과로 보고하지 않는다.

현재 판정: 구조적으로 가능한 조건과 가까운 선행은 확인됐다. 새 알고리즘 우월성은 미검증이다. 일반 OTA-FL 후보가 실패했다는 이유로 이 트랙을 자동 채택하지 않는다.
