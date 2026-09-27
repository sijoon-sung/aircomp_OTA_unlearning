**Original / V1 및 비교군 — 전체 수치표**

각 값은 noise를 seed 내부에서 평균한 뒤 구한3 seed 평균±표본표준편차다. 모든 비용은 real channel-use equivalent다.

**① DS-Air: fixed-feature ridge reference**

| 방법 / 버전 / 조건 | 삭제 오차 ↓ | Test accuracy % | Forget accuracy % | 전체 real uses |
|---|---:|---:|---:|---:|
| No-op / baseline | 1.000000 ± 0.000000 | 71.02 ± 0.20 | — | 0 |
| Exact retrain / reference | 0.000000 ± 0.000000 | 70.02 ± 0.24 | — | oracle |
| DS-Air / Original / peak | 0.369047 ± 0.007061 | 69.83 ± 0.19 | — | 143,940 |
| DS-Air / V1 / peak | 0.210444 ± 0.003549 | 69.86 ± 0.23 | — | 144,643 |
| Original-cost-matched / baseline / peak | 0.353435 ± 0.006766 | 69.83 ± 0.19 | — | 144,525 |
| gradient_MSE / baseline / peak | 0.383222 ± 0.008061 | 69.83 ± 0.18 | — | 143,940 |
| Noisy stats retrain / baseline / peak | 31.054967 ± 0.259219 | 69.06 ± 0.11 | — | 90,456 |
| Diagonal Newton / baseline / peak | 10.867346 ± 0.065014 | 52.01 ± 0.46 | — | 22,604 |
| DS-Air-subspace48 / V1 / peak | 0.367611 ± 0.006037 | 70.24 ± 0.22 | — | 88,173 |
| DS-Air / Original / average-only | 0.020109 ± 0.000497 | 70.00 ± 0.23 | — | 143,940 |
| DS-Air / V1 / average-only | 0.006992 ± 0.000082 | 70.01 ± 0.23 | — | 144,643 |
| Retained GD40 noiseless / baseline | 0.657331 ± 0.004508 | 71.12 ± 0.21 | — | 324,568 |

**② OG-Air: paired orthogonal FedAvg reference**

| 방법 / 버전 / 조건 | 삭제 오차 ↓ | Test accuracy % | Forget accuracy % | 전체 real uses |
|---|---:|---:|---:|---:|
| No-op / baseline | 1.000000 ± 0.000000 | 75.38 ± 1.75 | 81.61 ± 1.58 | 0 |
| Target-free ortho retrain / reference | 0.000000 ± 0.000000 | 75.19 ± 1.69 | 80.36 ± 1.23 | 227,708,074 |
| OG-Air-noiseless / Original | 32.126656 ± 21.561317 | 75.15 ± 1.16 | 81.14 ± 1.19 | 1,375,097 |
| OG-Air-noiseless / V1 | 0.999745 ± 0.000403 | 75.38 ± 1.75 | 81.61 ± 1.58 | 1,375,097 |
| OG-Air / Original | 12.466650 ± 10.792941 | 73.83 ± 1.36 | 80.67 ± 1.66 | 1,375,097 |
| OG-Air / V1 | 0.999739 ± 0.000410 | 75.38 ± 1.75 | 81.61 ± 1.58 | 1,375,097 |
| Original-mild / ablation | 1.159949 ± 0.172819 | 75.39 ± 1.59 | 81.47 ± 1.48 | 1,374,908 |
| Class-matched-mild / ablation | 1.817556 ± 0.577859 | 75.55 ± 1.30 | 81.78 ± 1.15 | 1,376,205 |
| Retained gate step / baseline | 1.015401 ± 0.008774 | 75.39 ± 1.76 | 81.58 ± 1.61 | 1,374,888 |
| Retained SGD10 / baseline | 2.334506 ± 1.439860 | 75.98 ± 1.09 | 81.17 ± 1.66 | 8,277,610 |
| FedOSD-core10 / baseline | 7.789275 ± 3.382082 | 74.50 ± 1.70 | 77.83 ± 1.83 | 76,295,789 |
| Legacy-Sketch-V1 / baseline | 0.643417 ± 0.128312 | 75.37 ± 1.73 | 81.42 ± 1.45 | 9,388,485 |

**③ RTD-Air: independent-teacher KD reference**

| 방법 / 버전 / 조건 | 삭제 오차 ↓ | Test accuracy % | Forget accuracy % | 전체 real uses |
|---|---:|---:|---:|---:|
| No-op / baseline | 1.000000 ± 0.000000 | 72.13 ± 1.74 | 79.19 ± 1.34 | 0 |
| Target-free KD / reference | 0.000000 ± 0.000000 | 72.02 ± 0.67 | 77.86 ± 1.29 | 731,658 |
| RTD-Air-noiseless / Original | 0.000000 ± 0.000000 | 72.02 ± 0.67 | 77.86 ± 1.29 | 731,755 |
| RTD-Air-noiseless / V1 | 0.000000 ± 0.000000 | 72.02 ± 0.67 | 77.86 ± 1.29 | 729,505 |
| RTD-Air / Original / r1 | 0.668330 ± 0.315429 | 71.97 ± 1.49 | 77.88 ± 1.36 | 731,755 |
| RTD-Air / Original / r4 | 0.595757 ± 0.343364 | 72.10 ± 1.42 | 78.09 ± 1.38 | 799,255 |
| RTD-Air / V1 / r1 | 0.755425 ± 0.410669 | 72.03 ± 1.33 | 77.91 ± 1.29 | 729,505 |
| RTD-Air / V1 / r4 | 0.877398 ± 0.207775 | 71.98 ± 1.71 | 78.02 ± 1.37 | 790,255 |
| Centered-full10 / baseline / r1 | 0.772382 ± 0.256878 | 71.86 ± 1.71 | 77.97 ± 1.48 | 731,755 |
| RTD-Air-query1000 / V1 / r1 | 0.852840 ± 0.282789 | 71.59 ± 1.01 | 77.56 ± 0.94 | 708,567 |
| Digital8 / baseline / r1 | 0.782174 ± 0.131396 | 71.82 ± 1.65 | 78.17 ± 1.80 | 1,195,775 |

**외부 비교: plain FedAvg reference**

| 방법 / 버전 / 조건 | 삭제 오차 ↓ | Test accuracy % | Forget accuracy % | 전체 real uses |
|---|---:|---:|---:|---:|
| No-op | 1.000000 ± 0.000000 | 75.45 ± 1.80 | 81.31 ± 1.65 | 0 |
| Target-free FedAvg | 0.000000 ± 0.000000 | 75.29 ± 1.97 | 80.06 ± 1.04 | 113,854,037 |
| FedQUIT-logit-min + CE adaptation | 152.811879 ± 18.876689 | 62.06 ± 2.18 | 35.44 ± 3.46 | 76,592,014 |
| Old-global-teacher KD | 2.413818 ± 0.400150 | 73.60 ± 1.80 | 79.81 ± 1.42 | 687,341 |

DS와 CNN의 오차 정의 및 reference가 다르므로 표 사이의 숫자로 전체 우열을 정하지 않는다.
Original은 우리 기존 구현이다. FedOSD/FedQUIT 표기는 본문에 명시한 adaptation이며 원 논문 전체 benchmark 재현이 아니다.
