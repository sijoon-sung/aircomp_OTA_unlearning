**방법론 통합 검증 — 완료, 2026-09-27**

문서 설명이 구현과 맞는지 확인하는 검증을 수행했다. 기존 GPU 실험 전체를 반복하거나 새 학습률을 탐색하지 않았다. RTX3080의 단일 worker로 작은 수학 검사와 저장 checkpoint 재계산을 수행했고, 새 training run은0개다.

| 확인한 내용 | 결과 | 의미 |
|---|---:|---|
|기존 원자료 완료 상태·유한 수치·주요 평균|780개 평가 행 감사|DS441,OG54,수정RTD87,최신부분투영198|
|기존 protocol·실험 코드 hash|일치|조건/코드 변경 없이 기존 결과 사용|
|① quadratic retained 삭제 항등식|최대오차2.43e-17|고정된 이차 목적함수의 정확한 보정식 확인|
|① 공통 inverse와 합산의 교환|최대오차1.48e-17|무잡음 Original/V1 계산 동치|
|① local inverse 평균과의 차이|norm .6673|client별 역행렬 평균은 같은 알고리즘이 아님|
|① 정상점·full 보정에서 A gradient 복원|최대오차2.36e-15|full 방법의 정보 제한 문제 재확인|
|① 제한된 부분공간의 비식별 예제|source/관측 차이<7e-16,A gradient 차이1.7321|같은 투영 관측으로 full gradient가 달라질 수 있는 예제. DP 증명 아님|
|② 알려진 retained span에서 부분 투영 가역성|역산오차2.23e-16|부분 투영 자체가 privacy 변환은 아님|
|② 기존 checkpoint3개로 첫 step 재계산|normalized JS 차이<2.41e-8|부분 투영·무잡음 결과 재현|
|③ 기존 teacher3seed×Original/V1 수신 target|6개 모두 FP32 max error=0|target-free KD 입력과 정확히 같은 값|
|③ 기존600step KD equality 기록|6개 모두 parameter max error=0|기존 검증 감사. 이번에 KD를 다시 학습하지 않음|
|③ 삭제 전후 정확한 평균에서 A 예측 복원|최대오차3.34e-15미만|서버의 이전 aggregate 이력도 정보 감사에 포함해야 함|

추가 GPU 계산 구간은 약2.03초였다. 이는 simulator 함수의 실행시간이며 무선 전송 latency나 단말 성능 수치가 아니다. 프로그램 시작·라이브러리 로딩까지 포함한 명령 walltime과도 구분한다.

**해석을 제한해야 하는 부분**

이번 검증은 ‘방법론의 식과 코드가 일치하는가’를 확인했다. 작은 선형 예제의 정보 누출이나 비식별성은 해당 가정에 대한 계산이며, 실제 전체 학습 기록·모든 부가정보·collusion·MIMO 수신을 포함한 privacy theorem이 아니다. 세 방법 모두 formal DP 또는 certified unlearning 검증을 통과했다고 표현하지 않는다.

②의 개선은 마지막1회 보정의 제한된 결과이고,10회 반복은 실패 기록을 유지한다.③의 무잡음 equality는 독립 teacher와 같은 KD 절차에 대한 결과이며,일반 FedAvg 재학습과의 equality가 아니다.①의 정확한 항등식은 고정feature ridge에 대한 것이며,private-data-trained CNN encoder까지 삭제하는 정리가 아니다.

**재현 파일**

- [검증 계획](VALIDATION_PLAN_KO.md)
- [검증 코드](validate.py)
- [검증 결과·원자료 재집계](validation/verification.json)
- [기존①②-A③ 상세 결과](../research_20260926_versioned/RESULTS_KO.md)
- [최신②-B 상세 결과](../research_20260927_orthogonality/RESULTS_KO.md)
