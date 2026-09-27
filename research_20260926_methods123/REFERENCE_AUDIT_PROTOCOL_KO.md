**후속 검증 범위 고정 — 원래 결과를 바꾸지 않는 reference 감사**

최초1·2·3 실험을 완료한 뒤, plain seed271에서 두 retained 재학습의 차이가 no-op와 paired reference의 차이보다 큰 것을 확인했다(정규화 비율1.371). 이 때문에 단일 reference 거리만으로 방법의 실패·성공을 단정할 수 없는 문제가 남았다.

추가 실행은 **각 seed와 plain/ortho 학습 구조에 대해 sampling stream만 다른 두 번째 retained reference 한 개**로 제한한다. Plain seed271은 이미 있으므로 재사용한다. 동일 private partition·초기값·150 rounds·고정 학습률을 사용하며 offset500000이다. 추가 reference를 본 뒤 어떤 model/hyperparameter도 선택하거나 재학습하지 않는다. 원래 결과 JSON과 표는 보존한다.

Source, 무잡음 HVP, OTA HVP20dB, 무잡음 pruning이 두 reference 각각과 얼마나 다른지 보고, reference 간 JS도 함께 기록한다. 또한 기존 source/HVP/reference의 retained CE와 gradient norm을 측정하여 출발점 비정류성·최적화 목표 차이를 진단한다. 이 CE 평가는 recovery 학습이 아니다. 두 모델만으로 full distribution unlearning을 인증하지 않는다.

비용 정리에서 independent-teacher 방식의 최초 student 학습 계산과 최초 global student 배포비용이 raw preparation row에 누락된 것을 발견했다. 삭제 실행 ledger는 보존하고, derived preparation ledger에600×64 student samples와32bit×parameter model DL을 명시적으로 추가한다. Public2,000개 image는 사전 배포 가정과 미배포 시12,544,000bit 추가비용을 각각 적는다. 이는 비용 장부 수정이며 결과·전송 알고리즘 변경이 아니다.
