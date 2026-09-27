**AirComp 언러닝 방법론 정리 — ① DS-Air / ② 직교화 계열과 부분 투영 / ③ RTD-Air**

2026-09-27 기준. 이 문서는 현재 구현·수학·실험을 연결한 방법론 명세다. 방법론을 정리했다는 것과 학술적 신규성·삭제 인증·프라이버시 증명을 완료했다는 것은 구분한다. 기존 원자료와 실패 기록은 변경하지 않았다.

**현재 사용할 세 가지 설명**

| 번호 | 현재 설명할 방법 | 무엇을 공중에서 합산하는가 | 삭제 시 A 참여 | 현재 판단 |
|---|---|---|---|---|
|① DS-Air|공통 곡률로 송신 전에 보정하는 삭제 residual 합산|잔존 client의 공통 inverse가 적용된 residual 좌표|검증한 구현에서는 불필요|통신 오차 개선 확인. Full 방식의 정보 제약 미충족;48차원 변형은 별도 후보|
|②-B Sketch-Air 부분 투영형|Sketch로 관계 계수를 구한 뒤 완전 직교성을 완화한 삭제 방향 합산|A의 UCE gradient와 잔존 CE gradient의 가중 합|필요. 현재 후보는 마지막1회|완전 투영 대비1회 보정 개선. 반복10회는 채택하지 않음|
|③ RTD-Air|독립 teacher에서 A를 제외하고 공중 예측 평균으로 fresh student 학습|같은 공개 query에 대한 잔존 teacher 확률|불필요|Original 구조 유지.9좌표 전송V1은 우월성 미확인|

②의 원래 제안은 **학습 중 사전 직교화**였다. 그 구현인 **②-A OG-Air Original/V1**은 각각 pruning 손상과 거의 no-op라는 문제로 보류한다. 최신②-B는 삭제 시점의 gradient projection 계열로,②-A가 개선되어 성공한 것처럼 합치지 않는다. Legacy-Sketch-V1의 부분 투영 변형이며 새로운 알고리즘 이름으로 신규성을 주장하지 않는다.

| 버전 표기 | 의미 |
|---|---|
|Legacy-Sketch-V1|기존 V1–V21 중 sketch→관계계수→가중 OTA로 projection을 계산하는 방법|
|DS-Air Original / V1|공통 inverse의 적용 위치: 수신 후 / 송신 전|
|OG-Air Original / V1|activation pruning /24차원 gate-gradient 보정. 둘 다 현재 보류|
|Sketch-Air β=1 / β=.5|동일 UCE·sketch·통신 구조에서 완전 / 부분 투영|
|RTD-Air Original / V1|10개 확률좌표 /9개 contrast좌표 전송|

**모든 방법에 적용하는 문제 정의**

클라이언트 A의 전체 데이터를 삭제한다. 미삭제 모델의 A 데이터 정답률을0으로 만드는 것이 목표가 아니다. A를 제외한 동일 학습 규칙의 결과에 가까워지는 것을 평가한다. Exact unlearning은 일반적으로 target-free 학습 알고리즘의 출력 분포 동등성과 연결되며, 한 난수 실행에서의 weight equality와 구분한다.

①의 reference는 고정된 공개 encoder 위 ridge의 retained 최적해다.②는 같은 CNN 학습 규칙의 retained FedAvg 모델이다.③은 독립 teacher ensemble에서 A를 뺀 KD 절차다. **세 숫자를 나란히 놓고 어느 방법이 더 잘 삭제한다고 순위를 매기면 안 된다.**

**공통 무선 모델과 공개 정보**

코드에서 각 client의 실제 합산할 payload를 z_i라 한다. 채널 h_i에 대해 x_i=z_i/(a h_i)를 동시에 송신하면 서버는 y=a(sum_i h_i x_i+n)=sum_i z_i+a n을 받는다. a는 실제 payload의 RMS와 peak 및 h_i를 고려해 공통으로 정한다. Mean transmit power<=1,peak amplitude^2<=4를 적용한다. Negative coefficient는 동기화된 부호/위상 전처리로 구현하는 모델이다.

이 실험은 coherent real MAC,정확한CSI,AWGN 모형이다. 표의20/40dB는 기준 전력과 잡음의 설정이며,peak 제약 후 작은 residual의 실제 수신SNR이 그 값이라는 뜻은 아니다. 실제RF,추가 MIMO 관측,CSI/CFO 오차,도청자 채널,collusion은 검증하지 않았다.

서버가 삭제 요청자가 A임을 아는 것과 A의 전체 update를 복원하는 것은 다르다. 후자가 사용자의 핵심 제한이다. Sketch·norm·count·곡률·teacher 확률 등 부분 정보도 공개량으로 기록한다. 합산했다는 사실이나 차원 감소만으로 DP를 주장하지 않는다. 정확한 나머지 합 r을 서버가 알고 y=x_A+r을 받으면 x_A=y-r이라는 점을 모든 분기에서 점검한다. [관련 반복 관측 공격](https://proceedings.mlr.press/v139/lam21b.html).

**현재 근거를 대표하는 수치**

| 비교 | 정규화 오차의 변화 | 비용·조건 | 해석 |
|---|---|---|---|
|① Original→V1|.369047→.210444|20dB,peak제약;143,940→144,643 real uses|43.0%감소. 비용을 더 준Original 대비도40.5%감소|
|②-B β1→β.5|.981156→.867013|20dB,마지막1회;동일1,557,477 real uses|11.6%감소. 사전 삭제품질 기준은미통과|
|②-B β1→β.5|.894661→.678367|40dB,마지막1회;beta간동일비용|24.2%감소.주reference 탐색기준통과,대체reference 일부seed역전|
|③ Original→V1|.668330→.755425|20dB,1회;731,755→729,505 real uses|9좌표V1 채택근거부족.Original 유지|

①은 parameter squared error,②③은 예측JS다. 각각 미삭제 상태를1로 정규화한다. 평균은 noise를 먼저 seed 안에서 평균한 뒤3seed로 계산했다. 위 표의 비용은 준비·학습비용을 숨겨0으로 만드는 것이 아니라 삭제 단계 통신 ledger이며 초기 학습과 공개 query 준비는 별도 문서에 기록했다.

**방법론별 명세**

- [① DS-Air: 수식·송수신·정보 조건·실험](METHOD_1_DS_AIR_KO.md)
- [②-A 실패 기록 및②-B 부분 투영: 수식·송수신·실험](METHOD_2_SKETCH_AIR_KO.md)
- [③ RTD-Air: 독립 teacher 조건·공중 증류·실험](METHOD_3_RTD_AIR_KO.md)
- [이번 추가 검증](VALIDATION_KO.md)

**검증과 현재 결정**

원자료780개 평가 행을 감사하고 주요 평균을 재계산했다. GPU에서①의 quadratic 항등식·정보복원 반례,②의 기존checkpoint 첫step3개,③의 teacher checkpoint 기반 무잡음target6개를 추가 검증했다. 새 모델을 재학습하지 않았다. 대규모 실험을 반복해야 할 불일치는 발견되지 않았다.

①은 작은 convex-head 설정의 통신 개선 결과로 유지한다. 사용자 정보 제한을 엄격히 만족하는 full-vector 배포안으로 쓰지 않는다.②는 마지막1회 참여의 부분 투영 후보만 유지한다.③은 독립 teacher 기반Original을 유지하고,AirComp 고유 기여는 추가로 확립해야 한다. 세 방법 모두 ‘서버가 개별 데이터를 전혀 추론할 수 없는 언러닝’으로 표현하지 않는다.

SFL/이기종 모델 조사와 retain recovery는 이 방법론의 실험 범위에 추가하지 않았다. 각 방법의 미해결 문제를 한 문장으로 좁히면①은 **곡률 준비비용과 전체 관측의 정보 누출**,②는 **마지막1회 삭제 방향의 품질과 전력 제한 아래 합산 오차**,③은 **독립 teacher 구조에서 최종 student 오차에 실제로 도움이 되는 통신 설계**다.
