# 데이터–채널 상관과 SISA/AirComp 그룹 배정: 완료 결과와 판단

2026-10-03. 18/18 실험 완료. 실제 무선 관측이 없는 FashionMNIST에서 만든 합성 연결 민감도 실험이다. **현실에서 client 데이터 분포와 무선 채널 사이에 상관이 있다는 것을 검증한 결과는 아니다.**

조건: 20 clients, Dirichlet .5, K4(5명/shard), 240 rounds, 3 독립 seeds, 무작위 배정 vs 채널 세기순 배정, client0 삭제. 초기 학습 / 같은 SISA 전체 재학습 / 해당 shard 삭제 재학습을 별도로 실행했다. 학습 데이터12000, dev2000, test10000. 목표 집계 MSE 1e-4, 전체 FP32 downlink, shard는 이상적인 직교 자원으로 분리한다.

|합성 연결 계수|측정된 거리 상관 rho 범위|무작위 정확도|채널순 정확도|정확도 차이|삭제 총 통신자원 절감|비용 개선 seed|사전 실용 기준|
|---|---|---|---|---|---|---|---|
|0|−0.052~−0.010|72.98%|72.68%|−0.30pp|5.63%|2/3|미달|
|0.5|0.060~0.072|73.01%|73.06%|+0.05pp|19.13%|3/3|통과|
|1|0.329~0.438|73.03%|72.83%|−0.20pp|20.88%|2/3|통과|

통신자원은 UL+DL+pilot+control을 기존 단위 RE로 환산한 값이다. 각 seed에서 상대 절감을 구한 뒤 평균했다. 무작위와 채널순은 같은 seed/연결 계수에서 데이터·채널·초기값·client minibatch 난수를 공유한다. 사전 기준은 평균 삭제 통신 절감>=10%, 2/3 이상 seed 개선, 평균 정확도 손실<=2pp, exact replay 전부 통과다. 통과는 통계적 확증이나 신규성 증명이 아니다.

## 확인한 것

- 강한 연결 조건에서는 3 seeds 모두 채널순 shard의 데이터 대표성이 나빠졌다. 전체 client 평균분포와 shard 분포의 JS divergence가 무작위 대비 각각 +0.0161/+0.0271/+0.0224, 평균 +0.0219 증가했다.
- 그러나 정확도 변화는 −1.09/+1.23/−0.73pp로 방향이 일치하지 않았다. **데이터 대표성 악화가 이번 설정에서 일관된 전체 정확도 저하로 이어졌다고 결론내릴 수 없다.**
- 강한 연결에서도 통신 절감은 +23.28/+51.53/−12.17%로 한 seed에서 비용이 늘었다. 단일 삭제 client0를 사용했으므로 예상 삭제 비용 전반의 개선을 주장하지 않는다.
- 중간 조건은 전체 정확도 평균이 거의 같았지만 최저 클래스 recall은 평균3.10pp 하락, 한 seed에서는7.90pp 하락했다. 전체 정확도만을 제약으로 삼으면 일부 클래스 손실을 놓친다.
- 초기 학습 총 통신 절감 평균은 연결계수0/0.5/1에서 각각9.48/15.59/20.38%. 삭제 시 절감과 구분해야 한다.
- 18/18에서 reference와 replay 모델이 bitwise 일치하고, 저장된 예측도 배열 단위 동일했다. 고정 shard/초기화/난수의 같은 SISA 알고리즘에 대한 검증이며, 전역 단일 모델과의 동일성 또는 DP 보장이 아니다.

## 다음 판단

채널순 배정은 비교 기준선으로 유지할 가치가 있다. 다만 보편적 최적안으로 채택하지 않는다. 다음 방법을 검토할 때는 **무선 비용을 줄이면서 shard의 클래스별 표본 확보를 보완하는 배정**을 후보로 삼을 수 있다. 먼저 복수 삭제 client와 고정 무선 예산 조건에서 비교해 단일 삭제 대상·고정 MSE의 영향을 분리해야 한다. 이 후속 실험과 신규 방법은 이번에 실행하지 않았다. 실제 상관관계의 존재 여부는 데이터와 채널이 같은 client/시간/장소에 연결된 실측 자료가 있어야 검증할 수 있다.

## 총비용·검증·원자료

- 정상 실험 wall time696.280초(11.60분), GPU board energy31.076Wh, client local training 호출185,760회(각2 SGD steps). GPU는 단일 worker 사용. 외부 유료 서비스0; 전기요금은 산정하지 않았다.
- 첫 실행은 scipy import 오류로 학습 전 종료(약1.8초, GPU 학습0). NumPy 순위 구현으로 수정 후18개 모두 완료했다. 별도 CPU audit/plot 명령1.74초. 기타 문서 작성/도구 대기 시간은 실험 wall time에 포함하지 않았다.
- 무선 전력/MSE 검사 통과. 18 cases의 동일 partition, 채널 주변분포, paired 채널, 고정 random routing, 학습 호출 수 및 저장 예측 일치 검사 모두 통과(audit.json).
- 메타분석은 3 seeds라 일반적 통계 확증이 아니다. 고정K/단일 dataset/240 rounds이며 수렴을 입증하지 않았다. 실제 radio fading trace/간섭/오차 있는 CSI를 검증하지 않았다. 현재 update norm scalar 제공을 허용했으며 histogram은 합성환경/평가자만 사용한다.
- 원자료: 같은 디렉터리 각 coupling*_s*_K4_* 폴더의 config.json, partition.npz, 모델/예측/ledger/result/extra_metrics. 전체 results.json, comparisons.json, association_summary.json, cost_total.json, audit.json. 순열 null·histogram·채널은 association_s*.json.
- 재현 코드 ../correlation_control.py, ../correlation_audit_plot.py; 실행 당시 코드/사전기록 해시는 environment.json. 기존 run_v1/ 결과는 변경하지 않았다.

![결과 그래프](C:/Users/DISLAB/Desktop/성시준_논문/논문/research_20261003_sisa_aircomp/correlation_control_v1/control_results.png)
