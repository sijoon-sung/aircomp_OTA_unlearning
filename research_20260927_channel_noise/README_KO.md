**공중 잡음 활용: 수식은 성립하지만 현재 곡률 오차가 삭제 품질을 제한한다**

2026-09-27. 새 FashionMNIST 분할3개에서 GPU 검증을 완료했다. 이번 결과는 ① convex-head의 후속 실험이며 ② CNN 전체/③ 증류/SFL 결과로 확대하지 않는다. NM-Air와 JS-Air는 연구용 이름이고 확정된 신규 알고리즘 명칭이 아니다.

- [알고리즘·수식·선행연구·한계](METHOD_KO.md)
- [실행 전 고정한 프로토콜](PROTOCOL_KO.md)
- [모든 설정의 결과표](RESULT_TABLES_KO.md)
- [원자료와 seed별 요약](results/summary.json), [검증 결과](results/verification.json)
- [잡음 보정·모델 분포 구분 진단](results/diagnosis.json)

**두 가지 실험 결론**

| 후보 | 활용 방식 | 확인한 이점 | 남은 문제 |
|---|---|---|---|
|NM-Air|목표 출력분산을 채널 잡음으로 만들도록 전력·반복 수 설정|같은 출력분포 비교군보다 보정 payload75.1%, 총 삭제비3.19% 감소|Hessian 추정 오차로 retained 모델과 평균이 다름. 현재 분포 일치 실패|
|JS-Air|알려진 수신 잡음 분산으로 correction을 shrink|20dB,반복5에서 동일 비용 MSE12.0% 감소;반복40에서1.8% 감소|기존 추정 기법의 적용. Noisy-H에서 일반 보장 없음. 10dB는 no-op보다 나쁨|

공중 잡음을 이용한 삭제 법칙은 명확해졌지만 **현재 설정에서 통신비와 엄격한 분포 언러닝을 동시에 성공시켰다고 결론내릴 수 없다.** JS-Air는 작은 point-error 개선안으로 남길 수 있다.

**NM-Air 주 비교:20dB,tau=.003**

| 항목 | Aware-fixed40 | NM-Air |
|---|---:|---:|
|보정 단계 analog payload, CP 전|5,200|1,294.58|
|전체 삭제비, source cache|137,775.97|133,382.37|
|초기 준비+삭제1회|148,432.04|144,038.45|
|목표 분산과의 covariance KL|0|0|
|retained target과의 조건부 KL 평균|1,186.17|1,186.17|
|조건부 TV 상계 평균|1.0|1.0|
|test accuracy|69.324%|69.331%|

통신 단위는 real-use equivalent다. 두 방식은 같은 조건부 평균·covariance를 만들므로 삭제 분포상 동등하다. Accuracy의 작은 차이는 서로 다른 noise 조합의 유한 표본 결과이며 NM-Air 성능 우위가 아니다. Randomized retrain accuracy는69.433%다.

기존 DS-fixed40의 결정론적 reference MSE 비율은.201098이고 NM-Air는.244412다. Randomized target의 분산을 맞추느라 point MSE가 약21.5% 증가했다. 목표가 다르다는 사실을 숨기지 않는다. 삭제 보정 없이 noise만 추가한 경우는1.052709로, no-op 기준1보다 나쁘다.

Hessian 수집과 Q/Λ 방송만 약123,610.84 real uses로 NM-Air 총비용의92.7%다. 그래서 correction payload를75.1% 줄여도 전체 이득은3.19%다. 균등 반복을 사용하는 같은 분포 비교군 대비 총비용 이득은 약.865%다. 비용에는 source 준비, H32, 기저/eigenvalue, scalar metadata, pilot, 제어, 마지막 모델 배포가 들어 있다. 반복 동안 채널이 유지되고 정확한 noise variance를 안다는 조건이다.

**왜 언러닝 판정은 실패했는가**

목표 variance를 맞춘 후에도 μ−W_R=(I−PH_R)(W₀−W_R)가 남는다. 이번 평균 squared bias가 주 조건의 Gaussian KL1,186.17을 만든다. Noise로는 이 평균 오차를 없앨 수 없다.

상계가 느슨하기 때문인지도 추가 검사했다. 새 Hessian 잡음128개×payload16개=2,048개 모델/seed와 독립 randomized retrain2,048개/seed를 생성했다. Evaluator가 exact retained 모델을 알고 있을 때 거리 ||W−W_R||²만으로 생성 절차를 구분하는 AUC는 다음과 같다.

| 조건 | 3seed AUC |
|---|---|
|NM-Air, noisy Hessian|1.000 /1.000 /1.000|
|NM-Air, exact Hessian oracle|.491 /.484 /.499|
|삭제 없이 noise만 추가|1.000 /1.000 /1.000|

이것은 **unlearned sampler와 retrained sampler의 구분 진단**이며 데이터 membership attack이 아니다. Oracle에서는20dB,tau=.003의 조건부 KL이6.35e-10으로 떨어져 수식의 의도는 확인된다. 하지만 무잡음 곡률을 실제 채널에서 같은 비용으로 얻었다는 결과는 아니다. 사후 진단은 주 실험 설정을 바꾸거나 성공 기준을 완화하는 데 사용하지 않았다.

**JS-Air: 같은 비용에서 얻은 작은 개선**

20dB, deterministic retained optimum을 reference로 사용했다. 표의 오차는 retained 모델과의 parameter squared distance를 초기 삭제 격차로 나눈 값이다.

| 전체 block 반복 수 | Raw | JS-Air | 오차 감소 | 동일 총 삭제비 |
|---|---:|---:|---:|---:|
|5|.276772|.243447|12.04%|132,657.22|
|10|.233176|.217391|6.77%|133,388.47|
|20|.212598|.204998|3.58%|134,850.97|
|40|.202782|.199103|1.81%|137,775.97|

3seed 모두 같은 비용의 raw보다 개선했다. 다만20반복 JS-Air(.204998)는40반복 raw(.202782)보다 오차가1.09% 높아, 그 쌍에 대해서 “비용과 오차를 동시에 낮췄다”고 표현할 수 없다.30dB의 개선도 작다.10dB에서 큰 상대 개선을 보이지만 최선의 표 안에서도 no-op보다 나쁘므로 성공으로 세지 않는다.

**Noise variance 추정은 공짜가 아니다**

Main 결과는 정확한 σ²를 안다고 가정했다. 무신호 N개로 잡음 분산을 새로 추정하는 진단에서는 다음 추가 비용과 covariance-only conditional KL 평균이 나왔다.

| N | 추가 real uses | covariance KL 평균 |
|---|---:|---:|
|64|72|5.6212|
|256|288|1.2343|
|1,024|1,152|.3330|
|4,096|4,608|.0782|
|16,384|18,432|.0184|

두 방식에 같은 calibration이 필요하면 양쪽에 더해야 한다. NM-Air에만 새4,096개 측정이 필요하다면 기본 절감량4,393.59보다 비용이 크다.100개 삭제 요청 동안 noise floor를 재사용할 수 있다면4,608/100=46.08로 나눌 수 있지만, 실제 장비의 시간 안정성은 확인하지 않았다. 이 calibration 자체도 수신 잡음만의 진단이며 Hessian bias를 고치지 않는다.

**연산·메모리·검증 범위**

주 GPU 실행은 RTX3080/PyTorch2.11.0+cu128에서 약5.88초, 후속 분포/분산 진단 약4.71초였다. 이 시간은 모든 seed·조건의 CUDA 시뮬레이터 실행시간이며 RF latency나 실제 client의 end-to-end 시간과 다르다. 주 실행에서 관측한 최대 PyTorch CUDA allocated memory는458,523,136bytes다. Local H/C 준비, 데이터 처리와 평가를 포함하며 실제 다중 단말의 메모리 합계가 아니다.

이전65×10 DS 실험과 같은 local statistics/선형 보정 계산을 사용한다. NM-Air 추가 resource 선택은5개 block의 max/ceil/sqrt다. JS-Air는650개 수신 좌표의 norm 및 scale 계산을 추가하며 extra uplink가 없다. 반복마다 local gradient를 다시 계산하지 않고 같은 payload를 재송신한다. Tensor 저장량은 기존 DS의 H/C·기저·head와 같고 새 controller scalar는 block당 몇 개다. Radio 송신 symbol energy를 정규화 단위로 기록했지만 실제 전력W나 Joule로 변환하지 않았다.

1458개 Gaussian 비교 행+648개 point 비교 행을 기록했고, 전력 제한·총비용·동일 분포 비교군243쌍·Gaussian KL 수식을 독립 재계산했다. 코드를 실행 전 고정한 프로토콜 해시와 연결했다. GPU 수학 검사에서는 quadratic 항등식 오차1.16e-15, MAC 합산 오차2.22e-16,20,000번 반복평균의 covariance 최대 상대오차1.65%를 확인했다. 이는 통계적 검증이며 전 분포에 대한 증명은 수식의 명시된 조건에 한정된다.

**실행**

외부 vendor 코드 없이 PyTorch·NumPy와 FashionMNIST raw IDX 데이터가 필요하다. GPU에서 하나씩 순차 실행한다.

```bash
python research_20260927_channel_noise/run_experiment.py --stage math
python research_20260927_channel_noise/run_experiment.py --stage run --data /path/to/FashionMNIST/raw
python research_20260927_channel_noise/diagnose.py --data /path/to/FashionMNIST/raw
python research_20260927_channel_noise/summarize.py
```

`summarize.py`는 표준 라이브러리만으로 보관 결과를 검증한다. 그림은 Matplotlib 설치 후 `--plots`를 추가한다. 새 결과는 실험 폴더를 복사해 실행하는 것을 권장한다. 공개한 폴더를 그대로 재실행하면 해당 폴더의 결과 파일을 다시 만든다.

이번 범위에서 다음 판단은 명확하다. NM-Air를 완성된 프라이버시 언러닝으로 채택할 근거는 없으며, 공통 곡률 오차와 준비비를 먼저 줄여야 한다. JS-Air는 비용을 늘리지 않는 작은 개선안으로 유지할 수 있지만, 선행 통계 기법과 AirComp noise matching의 중복을 고려하면 이것만으로 신규성 주장은 약하다. 전체 server transcript의 프라이버시 조건도 해결되지 않았다.
