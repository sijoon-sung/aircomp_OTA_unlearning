# 가설별 묶음: aircomp_OTA_unlearning (2026-10-10)

이 문서는 같은 내용을 **"무엇을 확인하려 했나"** 기준으로 다시 묶은 것이다.
수치는 각 폴더의 결과 문서에서 옮겼고, 마지막 절은 결과표를 직접 대조하며 찾은 문제다.

## 뿌리 질문

**Q0. AirComp가 전역 모델에 디지털 FL과 다른 영향을 주어, 언러닝을 더 싸게·정확하게·안전하게 만드는가?**

아래 H1~H8은 전부 Q0의 하위 가설이다. H9는 Q0를 내려놓은 뒤의 후보다.

| 가설 | 한 줄 | 판정 | AirComp 고유? |
|---|---|---|---|
| H1 잡음 활용 | 수신 잡음이 언러닝에 필요한 무작위성을 공짜로 준다 | 기각 | 아님. 디지털 서버가 같은 잡음을 더하면 같은 분포 |
| H2 전처리 위치 | 공통 inverse를 송신 전에 걸면 삭제 오차가 준다 | 조건부 성립 | 아님. 통신 오차 일반 |
| H3 직교·투영 | 삭제 방향을 부분 직교화하면 1회 삭제가 좋아진다 | 부분 성립 | 아님 |
| H4 teacher 증류 | A 뺀 teacher 확률의 공중 평균으로 재학습을 근사 | 구조만 동작 | 아님 |
| H5 SISA 배정 | 채널·데이터 기반 배정이 학습·삭제 비용을 교환시킨다 | 성립 | 데이터 쪽 이득은 아님 |
| H6 SISA 격리 | shard 간 물리 결합이 삭제 흔적을 전파한다 | 성립 | 자명. 공유 매체 일반 |
| H7 SISA 노출 | 작은 shard·이탈·차분이 개인 update를 드러낸다 | 성립 | 아님. SecAgg도 동일 |
| H8 비용 구조 | SISA를 얹으면 학습 K배, 삭제 통신 불변 | 성립 | 성립하나 식에서 바로 나옴 |
| H9 OTA-FL 일반 | 분할 집계 편향(SegOTA), AirComp-SFL | 후보/보류 | 분할 동시 전송의 교차 누설은 고유 |

---

## H1. 잡음 활용 — "수신 잡음은 언러닝의 무작위성을 공짜로 제공한다"

| 하위 가설 | 폴더 (브랜치) | 핵심 결과 |
|---|---|---|
| 잡음으로 출력분포를 맞추면 반복 전송이 준다 (NM-Air) | `research_20260927_channel_noise` (main) | 총 삭제비 3.19% 절감. noisy Hessian 조건 분포 검증 실패 |
| 알려진 잡음 분산으로 shrinkage (JS-Air) | 같은 폴더 | 같은 비용 point MSE 1.8–12.0% 감소. 고전 James–Stein 적용 |
| 잡음 충분통계 갱신 + 전체 관측 DP (CR-Air) | `research_20260928_protected_refresh`, `_cr_realdata`, `_cr_comm_tradeoff` (main) | 정확 CSI에서 분포 일치·DP 증명. K1000 FMNIST 45.16% (clean 56.37%). payload 에너지 82.8%↓, 전체 TX 에너지 개선 없음. CSI 5% 오차에서 실패 |
| 배정 간 정확도 차이를 잡음이 설명하는가 | `aircomp_noise_control_20261003` (SISA 브랜치) | 잡음 0/.01/1 비교. 잡음 외 효과 남음 |

**판정:** 기각. 같은 잡음 법칙은 디지털 서버가 재현한다. 선행: Koda 2020, Liu & Simeone(WFLMC), CAFU(LWC 2026).

## H2. 송신 전처리 위치 — "공통 곡률 inverse를 송신 전에 적용하면 OTA 잡음 증폭이 줄어든다"

| 하위 가설 | 폴더 | 핵심 결과 |
|---|---|---|
| Original vs V1 (inverse 위치) | `research_20260926_methods123`, `_versioned`, `research_20260927_methodology` (main) | unit 채널 0.369 → 0.210 (43%↓) |
| fading에서도 성립 + Switch/Subspace | `research_20260928_ds_revision` (main) | FMNIST 20dB Rayleigh: Original 3.34, V1 2.91, Subspace 0.84 (미삭제=1). MNIST Subspace 2.92 실패 |
| deep fade를 주파수 다양성으로 보완 | `research_20260928_cohort_diversity` (main) | FD8 0.97 → 0.49. MNIST 4/96 세션은 여전히 실패 |
| 비용 원장 | `research_20260927_costs` (main) | 삭제 1회 137,765 real uses (V1) |

**판정:** unit 채널에서만 깨끗이 성립. full V1은 A의 gradient가 역산된다(오차 2.36e-15). AirComp 고유 아님.

## H3. 직교·투영 — "삭제 방향을 잔존 gradient와 직교화하면 1회 삭제 품질이 오른다"

| 하위 가설 | 폴더 | 핵심 결과 |
|---|---|---|
| 학습 중 사전 직교화 (OG-Air) | `research_20260926_methods123`, `_versioned` | pruning 손상 / 거의 no-op. 보류 |
| 삭제 시 부분 투영 (Sketch-Air β) | `research_20260927_orthogonality`, `support/` | β 1 → .5: 20dB 0.981 → 0.867, 40dB 0.895 → 0.678. 반복 10회 불채택 |

**판정:** 마지막 1회만 부분 성립. gradient projection 일반 기법.

## H4. teacher 증류 — "A를 뺀 독립 teacher의 확률을 공중 평균하면 student가 재학습에 가까워진다"

| 폴더 | 핵심 결과 |
|---|---|
| `research_20260926_methods123`, `_versioned`, `research_20260927_methodology/METHOD_3_RTD_AIR_KO.md` | Original 0.668 vs V1 0.755. V1 개선 없음 |

**판정:** 구조는 동작. AirComp가 하는 일은 평균뿐.

## H5. SISA 배정 — "채널·데이터 기반 shard 배정이 학습 비용과 삭제 비용을 교환시킨다"

| 하위 가설 | 폴더 | 핵심 결과 |
|---|---|---|
| 초기 최적 배정 ≠ 삭제 최적 배정 (조합 계산) | `aircomp_shard_objective_test_20261003` | 5775개 분할 전수. 신경망 아님 |
| 채널/데이터/결합 배정 (학습 포함) | `research_20261003_sisa_aircomp`, `aircomp_deletion_partition_20261003` | JS+MSE 가중합 최적화. 실제 삭제비 최소화와 다름 |
| 데이터–채널 상관이 있을 때만 이득인가 | `research_20261003_sisa_aircomp/correlation_control_v1` | 연결 0: 삭제 RE 5.6% 절감(기준 미달). 0.5/1: 19–21% 절감 |
| 실측 삭제비로 배정 선택 | `aircomp_mechanism_20261004` | **18조건 중 10조건에서 중단.** 첫 seed data 배정 UL 8.97%↓, 준비비 4.1e8 |
| 참여 시점 지연 | `aircomp_staged_entry_20261003` | 완료 |
| 배정 9종 파레토 | `aircomp_sim_20261007` P1, `aircomp_sisa/asisa/exp/assignment.py` | 돌려 담기 삭제 에너지 −13.8% (최대 전력) |
| 배정 안정성·누적 비용·공정성 | `aircomp_sisa/asisa/exp/stability.py`, `lifecycle.py`, `fairness.py`, `placement.py` | 순위 규칙은 삭제 1회에 3.4–3.9 shard 재학습. 해시는 1개지만 최소 인원 2.8, 노출 41% |

**판정:** trade-off는 있다. 10/5 문서의 판정 그대로, 데이터 중심 이득은 AirComp 기여가 아니다.

## H6. SISA 격리 — "shard 사이 물리 결합이 삭제 흔적을 다른 shard로 전파한다"

| 결합 경로 | 폴더 | 핵심 결과 (미삭제 shard 파라미터 흔적) |
|---|---|---|
| 합성 선형 혼합 수신 | `aircomp_isolation_test_20261003` | 혼합 ρ>0에서 전파, ZF로 제거 |
| 코드 분할 간섭·near-far | `aircomp_sim_20261007` P0-A·P2, `aircomp_sisa/asisa/exp/interference.py`, `codes.py`, `placement.py` | 20dB 칩오차 .3 채널정렬: 0.321 (난수 0.235). ZCZ 구간 안 0 |
| 전체 전력 정렬·스케줄링 | `aircomp_sisa/asisa/exp/control.py` | −10dB 전체 정렬 0.476 ± 0.091, shard 안 0 |
| 재학습 전송 자체 | `aircomp_sisa/asisa/exp/retrain.py` | PN16 0.124, TDMA 0.0013 |
| 부반송파 주파수 오차 | `aircomp_sisa/asisa/exp/subcarrier.py` | OFDMA 채널 인식 0.203 ± 0.090 |
| 전체 검증 조기 종료 | `aircomp_sisa/asisa/exp/earlystop.py` | 2번 정체 규칙 0.124. shard 안 규칙 0 |
| 표준 AirComp 조건 (Rayleigh·절단·RF 손상) | `aircomp_sisa/asisa/exp/standard.py` | **코드만 있음. 미실행** |
| 보조 노트 | `무선통신과정에서 SISA를 지키려면/` (HTML 22개) | 문서 |

**판정:** 측정은 정교하지만 결론은 "공유 매체를 공유하면 결합된다, 자기 정보만 쓰고 직교 자원을 써라"로 자명하다.

## H7. SISA 노출 — "작은 shard·이탈·삭제 전후 차분이 개인 update를 드러낸다"

| 하위 가설 | 폴더 | 핵심 결과 |
|---|---|---|
| shard 크기와 노출 | `aircomp_sisa/asisa/exp/resources.py` (c) | 20dB 주 label 적중: n=1 84%, n=5 27.5%, n=20 9.9% |
| 이탈로 실제 인원 감소 | `dropout.py` | n_t=1 라운드 노출 34.5%. 3명 미만 비송신 규칙 |
| 삭제 전후 차분 | `differencing.py` | n=2 20dB 같은 초기값 90%, 새 초기값 80% (아래 문제 2) |
| SNR과 조건부 privacy | `research_20261003_sisa_aircomp/SNR_PRIVACY_*` | 큰 ε 상한 |

**판정:** 노출은 확인. SecAgg에서도 같은 구조라 AirComp 고유 아님.

## H8. 비용 구조 — "SISA를 AirComp에 얹으면 학습 자원은 K배, 삭제 통신은 그대로"

| 폴더 | 핵심 결과 |
|---|---|
| `aircomp_sim_20261007` P0(b), `aircomp_sisa/asisa/exp/resources.py`, `lifecycle.py` | 학습 UL K=1 6.1e6 → K=10 6.1e7. 삭제 UL 6.125e6 고정. 누적 통신을 줄이는 건 slicing뿐 (전체 재학습 대비 해시+병합 122.6% → slicing 추가 91.7%) |

**판정:** 성립. "AirComp 시간은 참가자 수와 무관"에서 한 줄로 나온다.

## H9. Q0 밖 — OTA-FL 일반 문제

| 후보 | 폴더 | 상태 |
|---|---|---|
| 분할 동시 전송(SegOTA)의 그룹 가중치 편향·교차 누설 | `research_20260928_ota_problem_review` | 산술 검산만. 단순 보정은 전체 모델 LMMSE를 못 이김. CNN 미실행 |
| AirComp-SFL | `research_20260928_sfl_gpu`, `research_20260926` | OTA-Stein 65.94% < 정규화 71.34%, digital4 71.21%. 보류 |

---

## 결과표를 대조하며 찾은 문제

1. **모든 SISA 정확도 비교가 학습 부족 영역에서 나왔다.** 로컬 step 2에서 K=1 정확도 70.2%, step 10에서 79.3%다(`sharding`). 이 영역에서는 잡음이 정확도를 올린다. eps=10에서 +3pp, 이 PC 축소 실행에서는 2명 shard가 +14pp였다. 그래서 eps*=10 선정, "공통 수신 크기가 정확도 +3.5pp" 같은 결론은 잡음이 regularizer로 작동한 결과일 수 있다. step 10에서 다시 재야 한다.
2. **차분 실험의 해석이 자기 표와 맞지 않는다.** 보고서는 "새 초기값 재학습이 차분을 막는다"고 쓰지만, n=2·20dB에서 cos 0.148 → 0.141, 적중 90% → 80%로 거의 그대로다. n=5에서 75% → 50%로 떨어지지만 배치만 바꿔도 45%다. 새 초기값의 효과는 배치 변경 수준이다.
3. **정렬 방식 사이의 흔적 비교 단위가 다르다.** 공통 수신 크기의 파라미터 난수 변동은 1.39, 최대 전력은 0.22다. "흔적 ÷ 난수"가 작아 보이는 것은 잡음이 기준선을 부풀린 탓이다. `placement` 보고서에 주의 문구는 있지만 `interference` 판정에는 반영되지 않았다.
4. **디지털 비교군이 32bit × D의 Shannon 전송이다.** P0 표의 225배 차이는 이 기준선 때문이다. 같은 저장소의 SFL 실험에서는 4bit 디지털이 OTA와 거의 같았다.
5. **미완:** `aircomp_mechanism_20261004` 10/18, `standard` 미실행, `sisa_arch_20261006` 결과 없음.
6. **재현 경로:** B세대 10/3 실험은 `research_20260929_pdf_ota_unlearning/core.py`의 DISLAB PC 절대경로에 묶여 있다.

## 묶음을 실제 폴더로 옮길 때

폴더를 물리적으로 옮기면 깨지는 곳이 두 군데 있다. `aircomp_sim_20261007`이 `sisa_arch_20261006`을 import하고, B세대 실험들이 `research_20260929_pdf_ota_unlearning/core.py`를 상대경로로 찾는다. 그래서 첫 단계는 이동이 아니라 최상위에 `HYPOTHESES.md` 하나(이 문서)를 두고 각 가설에서 기존 폴더로 링크하는 것이다. 이동은 A·B세대를 `archive/`로 내릴 때 한 번에 한다.
