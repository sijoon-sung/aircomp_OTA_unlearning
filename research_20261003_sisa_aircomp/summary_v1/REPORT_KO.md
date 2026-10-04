# SISA-AirComp 1차 실험 결과

2026-10-03. 실제 신규 학습·전체 재학습·부분 replay를 실행했다. 기존 결과는 수정하지 않았다.

판단: **사전 기준에서 추가 grouping 개선 미확인**. 선택 K=4, dev 선택 적격=True. 문헌 최초성은 검증하지 않았다.

## 조건과 개선 기전

FashionMNIST train12000/dev2000/test10000, 20 clients, Dirichlet .5, raw CNN, 120 rounds, local SGD 2x64, client 균등 평균. 전원 t=0 참여; client0 삭제. 해당 shard 초기 checkpoint부터 같은 120 rounds를 다시 수행한다. 이번 client 전체 삭제에서는 중간 slice checkpoint 절감이 없다.

무작위/채널 기반 고정 배정 × 공개 clipping bound/현재 norm scalar gain 조절을 교차 비교했다. long-term channel amplitude -20~0dB, bounded per-round fading, 완전 CSI, 이상적 직교 shard 전송, aggregate squared noise norm 목표1e-4. 반복 전송 횟수는 이 목표와 전력 상한으로 정했다. 실제 무선 장비가 아닌 baseband simulation이다.

random_norm은 channel_norm과 동일한 현재 norm 정보 및 power control을 사용하는 강한 비교군이다. 새로운 grouping의 성능은 이 둘로 판단하며, fixed-bound 대비 큰 차이를 신규 grouping 기여로 바꾸어 쓰지 않는다.

## 실행 전 기준과 검증

- 전체 25조건에서 source, A-free 모든 shard 재학습, 해당 shard replay를 각각 실행. CPU audit 25/25 통과.
- 모든 shard parameter와 reference/replay 예측 bitwise 동일. 무영향 shard 불변. 삭제 client 및 타 shard client의 replay 호출0.
- 파형 reconstruction, 직교 bin leakage0, Monte Carlo noise variance, 모든 송신 power cap, ledger 및 accuracy 독립 재계산 통과.
- 신규 개선 채택 기준: confirmation 평균 total RE 10% 이상 절감, 평균 정확도 감소2pp 이내, 3 seeds 중2개 이상 RE 감소. 테스트로 K를 재선택하지 않았다.

## Screening: seed10601, dev 선택에만 사용

| K | 방식 | Dev % | Test % (선택 미사용) | 삭제 local calls | UL RE (million) | 전체 RE (million) | 초기 전체 RE (million) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | random_norm | 67.80 | 65.99 | 2280 | 5.168 | 78.801 | 78.655 |
| 2 | random_fixed | 69.25 | 67.52 | 1080 | 167.943 | 241.492 | 598.183 |
| 2 | channel_fixed | 67.90 | 66.39 | 1080 | 57.078 | 130.627 | 511.054 |
| 2 | random_norm | 69.15 | 67.51 | 1080 | 9.149 | 82.715 | 167.702 |
| 2 | channel_norm | 67.95 | 66.40 | 1080 | 4.709 | 78.274 | 164.448 |
| 4 | random_fixed | 71.30 | 69.13 | 480 | 278.502 | 352.026 | 2990.357 |
| 4 | channel_fixed | 69.70 | 67.80 | 480 | 279.688 | 353.213 | 2390.057 |
| 4 | random_norm | 71.25 | 69.06 | 480 | 16.921 | 90.453 | 397.823 |
| 4 | channel_norm | 69.75 | 67.83 | 480 | 17.227 | 90.759 | 378.299 |
| 5 | random_fixed | 71.75 | 69.50 | 360 | 335.733 | 409.253 | 5231.657 |
| 5 | channel_fixed | 70.85 | 67.91 | 360 | 769.813 | 843.332 | 3997.215 |
| 5 | random_norm | 71.75 | 69.60 | 360 | 19.830 | 93.356 | 551.108 |
| 5 | channel_norm | 70.85 | 67.96 | 360 | 35.411 | 108.936 | 509.955 |

선택 점수: J=.5×(삭제 local calls/K1)+.5×(삭제 전체 RE/K1). channel_norm이 random_norm보다 dev2pp, K1보다5pp를 넘게 낮아지지 않는 K에서 최소 점수를 선택. K=4를 confirmation 전에 고정했다. 상세값: ../run_v1/selection.json.

## 새 seed 3개: 추가 grouping 개선

| Seed | Random 정확도 % | Channel 정확도 % | 차이 pp | UL 절감 % | 전체 RE 절감 % | 정규화 UL 에너지 변화 % |
|---:|---:|---:|---:|---:|---:|---:|
| 10611 | 68.91 | 68.28 | -0.63 | -91.43 | -27.00 | +158.17 |
| 10612 | 69.12 | 65.77 | -3.35 | 90.26 | 43.36 | -87.42 |
| 10613 | 67.61 | 69.64 | +2.03 | -15.37 | -4.90 | +23.05 |

주 판단의 분모는 random_norm. 평균 자원량 비율로 UL 절감 20.95%, 전체 RE 절감 7.89%, 평균 정확도 차이 -0.65pp. RE 개선 seed 1/3. DL 효율을2에서6 bits/RE로 바꾼 ledger 민감도에서 전체 RE 절감 13.50%. 이는 채널 모델을 다시 실행한 결과가 아니다.

## 단일 모델 재학습과 동일 SISA 재학습의 분모 구분

| Seed | K1 정확도 % | 제안 정확도 % | 삭제 계산 / K1 | 삭제 RE / K1 | 삭제 RE / 동일 SISA 전체 | 초기 RE / K1 | 최소 삭제 후 인원 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10611 | 68.98 | 68.28 | 0.211 | 1.674 | 0.336 | 4.828 | 4 |
| 10612 | 64.02 | 65.77 | 0.211 | 0.945 | 0.157 | 6.079 | 4 |
| 10613 | 66.78 | 69.64 | 0.211 | 1.389 | 0.266 | 5.188 | 4 |

K1은 다른 학습 알고리즘이므로 정확 언러닝 기준이 아니다. 동일 SISA 전체 재학습의 reference와 bitwise 같다는 결과를 단일 전역 모델 재학습과 같다고 해석하지 않는다.

## 초기 비용을 포함한 lifecycle projection

| 가정한 삭제 횟수 | Channel / Random 전체 RE | Channel / K1 전체 RE |
|---:|---:|---:|
| 1 | 0.897 | 3.349 |
| 10 | 0.912 | 1.695 |
| 100 | 0.920 | 1.367 |

위 표는 각 삭제가 이번 측정과 같은 비용이라는 대입 계산이다. 실제 누적·순차 삭제나 shard 고갈을 실행하지 않았다.

## 채널-데이터 상관 스트레스 (별도 seed)

| Seed | Random 정확도 % | Channel 정확도 % | 차이 pp | 전체 RE 절감 % |
|---:|---:|---:|---:|---:|
| 10621 | 64.59 | 65.21 | +0.62 | 46.98 |

dominant label과 채널 순위를 연결해 채널 grouping이 데이터 분포까지 몰리게 했다. 이는 알고리즘에 label 정보를 준 것이 아니라 evaluator의 환경 생성이다. 주 confirmation과 평균을 섞지 않는다.

## 정보·전송·비용 해석

- norm 방식은 client당 round마다 현재 clipped update norm32bits를 추가로 노출한다. CSI/ID도 사용한다. 개별 gradient/history를 서버에 저장하지 않는다. 최소집계 인원은 보장된 프라이버시 지표가 아니며 MIA/gradient inversion 실험은 이번 범위에 없다.
- 고정 FDMA와 elastic FDMA 차이에는 비어 있는 대역폭을 회수하는 효과가 있다. 그러나 work-conserving TDMA와 elastic FDMA의 유효 RE와 이상적 통신 시간은 이 모델에서 동일하다. 동시 전송 자체의 spectral gain을 주장하지 않는다.
- 모든 RE는 전체 공유 시간-주파수 자원을 센 값. 2 bits/RE의 control/DL 가정, pilots 포함. 본문 time은 GPU simulation wall time과 통신 ledger를 구분한다. 실제 ms 지연이나 RF Joule은 측정하지 않았다.
- normalized UL energy에는 analog waveform와 단위 symbol energy의 control/pilot를 포함한다. 수신기/회로/BS DL 에너지 제외. 전송 횟수 절감이 같은 비율의 에너지 절감은 아니다.
- exactness 검증은 같은 고정 routing, RNG coupling, 이상적 채널 simulator의 같은 알고리즘 비교다. 실제 RF에서의 분포 인증, metadata 자체 삭제, 공격 방어 보장이 아니다.

## 총 실행 비용

- preflight와 본실험 포함 측정 wall time 537.11s (8.95min). GPU board energy 23.5616Wh. 유료 외부 서비스 사용0. host 전체 소비전력/전기요금 미측정.
- 결과 폴더 파일 크기(보고서 생성 전) 121.20MiB. 각 실행의 power samples는 cost_total.json에 보존. preflight는5초 미만이라 에너지 적분 표본이 부족할 수 있다.
- 단일 신규 GPU worker로 직렬 실행. CPU audit/report 생성 시간은 위 GPU 실험비에 포함하지 않았다.

## 다음 판단

현재 channel_norm을 전체 통신 비용의 검증된 개선으로 채택하지 않는다. UL-only 이득, 정확도 손실, 에너지 변화를 분리해서 다음 설계를 정한다. 단순 대역폭 재배정을 독립적인 신규 개선으로 주장하지 않는다.

## 원자료

- ../PREREGISTRATION.md: 실행 전 조건/기준. ../experiment.py: 실행 코드.
- ../run_v1/: 각 source/reference/replay 모델, 확률, partition, labels, event ledger, 코드 hash, selection, completion.
- all_metrics.csv / paired_results.json / decisions.json: 수치와 결정.
- audit.json: CPU 독립 검증. experiment_cost.json: 총비용.
- frontier.png / confirmation.png: 그림. scientific pilot이며 일반적인 우월성을 확정하지 않는다.
