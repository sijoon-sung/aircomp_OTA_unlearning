**방법론 통합에 필요한 추가 검증 — 2026-09-27, 실행 전 기록**

목표는①②③의 수식·실제 코드·결과·공개 정보가 일치하는지 확인하는 것이다. 대규모 재학습이나 새 hyperparameter 탐색을 하지 않는다. 기존 기록을 덮어쓰지 않고 새 폴더에 검증 결과를 쓴다. GPU worker는1개다.

1. 기존 source/protocol/code hash와 raw JSON의 완료 상태·개수·유한 수치를 확인한다. 결과 평균을 raw에서 seed별로 먼저 다시 계산한다.
2. ① 작은 FP64 quadratic 문제에서 retained Newton 삭제 항등식, 송신 전 공통 inverse와 수신 후 inverse의 무잡음 동치, 서로 다른 local inverse를 평균하면 일반적으로 다르다는 반례를 확인한다. 전체 source 정상점 및 full correction에서 target gradient의 복원이 가능한지 직접 계산한다. 고정 부분공간에서 관측되지 않는 방향의 통계 변경은 원래 gradient를 바꾸면서 같은 source·같은 projected correction을 만드는 예를 확인한다. 이는 특정 복원 경로의 한계 확인이며 DP 증명이 아니다.
3. ② 기존 새 seed381/382/383의 checkpoint로 부분 투영·무잡음·첫 step을 재계산한다. Raw 결과의 JS·이상적 residual norm과 대조한다. 알려진 retained span에서 beta=.5 연산이 가역이고 beta=1 연산은 rank deficient라는 점을 작은 FP64 예제로 확인한다. 부분 투영을 privacy 기법으로 부르지 않는다.
4. ③ 기존 teacher checkpoint3seed에서 공개 query2000개를 다시 평가한다. Original/V1 무잡음 수신 target이 target-free KD target과 정확히 같은 FP32 값인지 확인한다. 이미 확인된600step KD 모델 equality6개는 원자료를 감사하며 학습을 반복하지 않는다. 삭제 전후 정확한 aggregate로 A의 teacher prediction을 복원하는 차분식을 검증한다.

공통 기준: FP64 선형 항등식 최대오차<1e-10,모델 first-step normalized JS 재계산 차이<1e-5,RTD target FP32값 equality,max power 평균1/peak4(float 허용오차). 불일치가 나오면 문서를 먼저 고치고, 검증되지 않은 것을 통과로 기록하지 않는다.
