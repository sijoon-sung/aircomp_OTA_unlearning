**1·2·3 코드 확보 → 수학 검사 → 순차 GPU 실험 완료**

**①·② 후속 정정과 개선안:** [삭제신호 공중 합산의 의도 복원 및 제한된 재검증](../research_20260926_methods12_revision/REVIEW_AND_IMPROVEMENTS_KO.md). 최근① CuReNUS 결과로 V1/convex-head를 기각하지 않는다. 이전 raw 결과는 그대로 보존했다.

가장 먼저 볼 문서는 [실험 결과와 판단](RESULTS_KO.md)이다. 원본 논문의 전체 benchmark 재현과 우리 조건에 맞춘 adaptation을 구분했다. 기존 연구 폴더는 수정하지 않았다.

| 자료 | 내용 |
|---|---|
| [결과 보고서](RESULTS_KO.md) | 방법별 결과·원인 진단·비용·privacy 조건·현재 판단 |
| [전체 결과표](RESULT_TABLES_KO.md) | 모든 비교군, seed 평균±표본표준편차 |
| [실행 전 프로토콜](PROTOCOL_KO.md) | 데이터 분할·reference·hyperparameter·순서·성공 기준 |
| [수학 검증](MATHEMATICS_KO.md) | HVP 합산, cubic 계수, 직교화 반례, KD privacy 퇴화, 독립 teacher 조건 |
| [코드 출처](CODE_PROVENANCE_KO.md) | 실제 GitHub commit·학회 코드·이식/재구현 범위 |
| [추가 reference 감사](REFERENCE_AUDIT_PROTOCOL_KO.md) | 재학습 변동 때문에 추가한 제한된 검사와 비용 장부 보완 |
| [수학 원결과](results/math_validation.json) | 학습 전에 통과한 numerical assertions |
| [전체 CNN 연결 검사](results/integration_checks.json) | 저자 HVP와 local aggregate 비교, budget·simplex 검사 |
| [모든 집계 결과](results/summary.json) | 방법별·seed별 수치 |
| [Reference 감사 원결과](results/reference_audit.json) | 다른 sampling stream의 retained retrain과 비교 |
| [비용 감사](results/cost_audit.json) | 준비비용, public images, local teacher/student 연산 |
| [재실행 명령](run_all.ps1) | 기존 결과를 보존하며 미완료 단계만 순차 실행 |

실행 코드는 [수학 검사](math_validation.py), [공통 학습·AirComp](common.py), [공개 코드 adapter](vendor_adapters.py), [실험 queue](run_experiments.py), [reference 감사](audit_references.py)다. Vendor source는 다운로드한 상태로 보존했다. 결과를 다시 만들 때 기존 JSON을 덮어쓰기보다 이 폴더를 별도 복사해 새로운 실행으로 관리하는 편이 안전하다.

이번에는① HVP의 연산 정확성을 확인했지만 삭제 개선은 확인하지 못했고,②의 사전 직교화·boundary 통계전송 이점도 실제 CNN에서는 확인되지 않았다.③은 **독립 teacher를 제외하고 fresh student를 만드는 변경된 학습 방식**에서 수학적 조건과 GPU 재실행이 일치했다. 일반 FedAvg의 exact unlearning이나 AirComp 전용 신규성 확보를 뜻하지 않는다.

추가 sweep·recovery·SFL 실험은 실행하지 않았다. 이 폴더의9개 method-seed 결과와6개 reference audit가 완료되었고 현재 실행 중인 연구 GPU worker는 없다.
