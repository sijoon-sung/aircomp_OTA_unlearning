# Clipping 진단 추가: 사후 관측에 따른 유한 확장

2026-10-03. 주 실험의 기본45조건을 관측한 뒤, R4/privacy-cap 실행이 끝나기 전에 기록한다. 기본 natural 조건의 source clipping 비율이91.28~95.50%이고 dev 최고68.75%로 사전 목표70%에 도달하지 못했다. 따라서 작은 SNR 정확도 차이가 학습 제약에 가려진 것인지 확인할 필요가 있다. 주57조건·성공기준·결과는 변경하지 않는다. 추가 결과를 주 실험의 독립 확인 결과처럼 합치지 않는다.

추가8개 source/reference/replay 진단:
1. seed10801, K4, natural R1, rounds160, C1.0, SNR0/20 x random/channel/joint =6. 이미 실행한 같은seed/K/SNR/method의C0.1과 대응 비교한다. C1.0은 이전 experiment.py의 고정 clipping 기준이며 결과에 맞춘 탐색으로 선택한 값이 아니다.
2. 같은seed/K4/random/natural R1/rounds160, 명목SNR120dB에서 C0.1/C1.0 =2. 120dB는 현실적 무선 배치 주장용이 아니라 잡음을 거의 없앤 수치 기준이다.

주 분석: clipping율, dev/test 정확도, 목표70% 도달 여부/체크포인트 자원량. C를 바꾸면 최대 허용 alignment에 따른 집계 잡음 크기도 함께 바뀌므로 0/20dB 비교만으로 순수한 clipping 인과효과라 주장하지 않는다. 120dB 기준에서C1이C0.1보다 dev/test 각각2pp 이상 개선되면 강한 clipping이 학습을 제약했다는 진단 근거로 삼는다(한seed 진단, 일반화/유의성 주장 아님). 반대·무차이도 보고한다. 같은 C에서SNR을 바꾼 변화도 별도 보고한다. 자연잡음 privacy 식에서C가 상쇄되는지 원자료에서 확인한다.

주 worker 종료를 확인한 뒤 단일 추가 worker로 실행한다. snr_clip_diagnostic_v1/에 분리 저장한다. 주 실험과 추가 진단의 시간·GPU board Wh·호출 수를 각각 보고하고 합산한다. 추가 확장은 이8개로 종료한다.
