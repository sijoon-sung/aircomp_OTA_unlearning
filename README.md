# AirComp OTA Unlearning — 실험 및 비용 기록

2026-09-28 기준, 세 가지 AirComp 언러닝 방법의 수학적 조건, Original/V1 비교, 실패 기록과 비용 산정을 모은 연구 저장소다. 이전 날짜 문서는 당시의 탐색 결과이며, 이후 수정 사항을 덮어쓰지 않았다.

**최신 deep-fade 보완: Cohort Diversity** — [GPU 결과](research_20260928_cohort_diversity/README_KO.md) · [알고리즘·수식·비용](research_20260928_cohort_diversity/METHOD_KO.md). 잔존9명을 모두 유지하면서 공유 주파수8개 중 자원을 고르는 FD8-Subspace를 검증했다. 새 seed3개, 조건당32개 channel/H draw로10,368개 평가 기록을 추가했다. 20dB 독립 Rayleigh에서 같은 예산 Subspace 대비 오차가 FashionMNIST `.9695 → .4873`, MNIST `1.7897 → .6360`으로 감소했다(미삭제=1). 탐색비를 포함한 사전 주 기준은 통과했지만, MNIST4/96개 session은 여전히 E>=1이다. 완전히 정적인 후보/큰 경로손실의 실패, 대기 방식의 timeout·지연과 gradient 복원 위험도 공개했다. 주파수 availability/coherence 가정이 있는 결과이며 프라이버시 보장이나 신규성 확정은 아니다.

**앞선 DS-Air 보완 검증** — [GPU 결과](research_20260928_ds_revision/README_KO.md) · [수식·선행연구·한계](research_20260928_ds_revision/METHOD_KO.md). 두 데이터셋/3개 모델 폭/3개 채널/SNR 조건에서 5,568개 평가 기록을 추가했다. 추가 제어비를 포함한 Switch는 채택 기준에 실패했다. FashionMNIST 20dB Rayleigh 주 조건의 오차는 Original 3.3407, V1 2.9135, 같은 예산 Subspace 0.8382다(미삭제=1). 저비용 Subspace는 비용을39.8% 줄였지만 오차1.0942로 미삭제보다 나쁘다. MNIST 20dB Rayleigh에서는 같은 예산 Subspace도2.9224로 실패했다. 아래 과거 unit-channel 개선을 fading 전반의 삭제 성공으로 확대하지 않는다. Full transcript 프라이버시도 미해결이다.

**추가 GPU 검증: 공중 잡음 활용** — [NM-Air / JS-Air 결과](research_20260927_channel_noise/README_KO.md) · [알고리즘과 수식](research_20260927_channel_noise/METHOD_KO.md). NM-Air는 같은 출력분포 비교군 대비 총 삭제비를3.19% 줄였지만, noisy Hessian 조건의 분포 언러닝 검증은 통과하지 못했다. JS-Air는20dB에서 같은 비용의 point MSE를1.8–12.0% 줄였다. 잡음만으로 삭제가 완성된다는 결론은 아니다. 새 데이터 seed3개, Gaussian/point 비교2106행과 별도 분포 구분·잡음 추정비용 진단을 공개했다.

- [①②③ 최신 방법론](research_20260927_methodology/METHODS_KO.md)
- [통신·계산·저장·초기 준비 총비용](research_20260927_costs/COST_REPORT_KO.md)
- [비용 가정과 미측정 항목](research_20260927_costs/ASSUMPTIONS_KO.md)
- [수학 및 기존 GPU 결과 검증](research_20260927_methodology/VALIDATION_KO.md)
- [재현 안내](REPRODUCIBILITY_KO.md), [외부 코드 출처](THIRD_PARTY_NOTICES.md)

**현재 방법과 판단**

| 방법 | 공중 합산 대상 | 현재 판단 |
|---|---|---|
|① DS-Air Original / V1|잔존 client residual / 공통 곡률로 송신 전에 보정한 residual|Unit 채널 V1 개선. FD8-Subspace가 두 dataset의20dB 독립 fading에서 추가 개선. 정적 채널/큰 경로손실/전체 관측 프라이버시는 미해결|
|②-A OG-Air Original / V1|사전 직교화 모델의 activation 통계 / gate gradient|삭제 품질 문제로 둘 다 보류|
|②-B Sketch-Air β=1 / β=.5|A의 UCE gradient와 잔존 CE gradient의 가중 합|완전 직교성을 완화한 마지막 1회 참여가 비교군보다 개선. 반복 10회는 채택하지 않음|
|③ RTD-Air Original / V1|독립 teacher의 10차원 확률 / 9차원 contrast|독립 teacher 기반 Original 유지. V1의 최종 student 개선 근거 부족|

②-B는 기존 Legacy-Sketch-V1의 부분 투영 변형이다. 원래 제안한 사전 직교화 ②-A와 구분한다. 공중 합산이나 차원 감소만으로 개인정보 보호, DP 또는 정확한 삭제를 보장한다고 주장하지 않는다.

**보완된 대표 통신비**

20dB, 단위는 **real channel-use equivalent**다. 모델이 이미 배포된 상태의 삭제와 초기 준비부터 시작하는 비용을 구분했다. 초기 준비와 삭제 사이의 source 모델 중복 전송은 차감했다.

| 방식 | 모델 cache 후 삭제 1회 | 초기 준비 + 삭제 1회 | 공개 query 최초 배포도 필요한 경우 |
|---|---:|---:|---:|
|① DS-Air V1|137,765|148,410|148,410|
|②-B Sketch-Air β=.5|870,223|115,416,238|115,416,238|
|③ RTD-Air Original, 반복 1회|731,842|1,463,682|5,702,658|

세 방법은 모델과 재학습 기준, 삭제 품질이 달라 이 표를 효율 순위로 해석할 수 없다. RTD의 query 최초 배포는 2,000장의 28×28 uint8 이미지, 총 12,544,000bit를 가정했다. 공개 데이터가 이미 설치된 경우에는 이 항목이 추가되지 않는다.

총비용 보고서에는 13개 설정의 Original/V1·부분공간·반복 수 비교, 제어·scale·pilot·CP, 계산 sample 수와 MAC, tensor 저장량, 가정한 전송률과 재전송 민감도가 있다. 실제 RF 지연·전력·금액을 측정한 값은 아니다.

**대표 성능 결과**

| 비교 | 정규화 오차 | 조건 |
|---|---|---|
|① Original → V1|.369047 → .210444|20dB, peak 제약, parameter 제곱오차|
|②-B β=1 → β=.5|.981156 → .867013|20dB, 마지막 1회, 예측 JS|
|②-B β=1 → β=.5|.894661 → .678367|40dB, 마지막 1회, 예측 JS|
|③ Original → V1|.668330 → .755425|20dB, 반복 1회, 예측 JS; V1 개선 없음|

각 오차는 해당 방법의 미삭제 상태를 1로 정규화했다. 세 family 사이에는 reference와 지표가 다르다. 비용 보완 때문에 모델을 다시 학습한 것은 아니며, 위 품질 결과는 기존 프로토콜의 결과다. 초기 source는 이상적 집계로 생성했으므로 noisy end-to-end 학습까지 검증한 것으로 확대하지 않는다.

**저장소 구성과 실행**

- `research_20260928_cohort_diversity/`: 전원 참여를 유지하는 주파수 선택/인과적 대기, 탐색비·지연·timeout 포함 GPU 검증.
- `research_20260928_ds_revision/`: DS 보완 Switch/부분공간, 두 데이터셋 GPU 결과, Newton 대조군, 전력·전체 비용·복원 공격 감사.
- `research_20260927_methodology/`: 최신 수식, 송수신 절차, 정보 공개 조건, 780개 평가 행 감사와 GPU 검증 기록.
- `research_20260927_channel_noise/`: 후속 NM-Air/JS-Air 수식, 새 GPU 실험, 잡음 보정 비용과 실패 원인.
- `research_20260927_costs/`: 추가 비용 계산 코드, 가정, JSON 원장과 보고서.
- `research_20260927_orthogonality/`: 완전/부분 투영 비교, 마지막 1회 및 반복 실패 기록.
- `research_20260926_versioned/`: DS/OG/RTD Original·V1 실험 및 RTD 수치 정밀도 보정.
- `research_20260926_methods123/`: 초기 구현, 비교군, 선행 코드 출처와 당시 실패 결과.
- `support/`: 기존 sketch/weighted AirComp 계산 모듈의 보관본.

Python 표준 라이브러리만으로 비용표와 배포 파일을 검증할 수 있다.

```bash
python research_20260927_costs/calculate_costs.py
python verify_bundle.py
python research_20260927_channel_noise/summarize.py
python research_20260928_ds_revision/summarize.py
python research_20260928_cohort_diversity/summarize.py
```

이전 GPU 학습 코드는 당시 파일을 보존했다. 데이터셋, checkpoint, Python/CUDA 환경과 전체 vendor 소스는 포함하지 않았다. 이전 GPU 실험 재실행에는 [재현 안내](REPRODUCIBILITY_KO.md)의 외부 의존성과 경로 설정이 필요하다. 새로운 `research_20260927_channel_noise`는 vendor 없이 PyTorch·NumPy와 `--data` 경로로 실행할 수 있다. SFL 후속 연구와 retain recovery를 이 실험 묶음의 성과에 합치지 않았다.
