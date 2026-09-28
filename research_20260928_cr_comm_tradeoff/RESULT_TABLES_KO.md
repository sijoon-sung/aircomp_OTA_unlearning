# 통신·프라이버시 완화 결과표

지연은 시뮬레이션의 frame real symbol 수를 1M real/s로 환산한 값이다. SDR 실측이 아니다. 3 seed, K1000, 정확 CSI Rayleigh 결과다. A-Only의 final variance는 다른 방법의 2배이고 원래 noisy-retrain reference와 일치하지 않는다.

**프라이버시 보정:** 복사한 cyclic prefix도 서버 관측이다. 아래 rho/epsilon은 prefix를 포함한 보수적 전체 관측 상한이다. 동결 raw JSON/PROTOCOL의 nominal rho는 이 값의 절반이며 epsilon도 그대로 쓰면 안 된다. 근거는 README와 audit_full_frame.py를 참조한다.

| Dataset | Full-frame rho cap | Full-frame epsilon cap | Independent acc / ms | CR-Power acc / ms | A-Only acc / ms | CR 통신 절감 |
|---|---:|---:|---|---|---|---:|
| FashionMNIST | 1.0 | 7.79 | 45.40% / 35.281 | 45.38% / 33.203 | 39.19% / 2.266 | 5.89% |
| FashionMNIST | 2.0 | 11.60 | 49.97% / 37.945 | 50.02% / 33.619 | 45.38% / 2.266 | 11.40% |
| FashionMNIST | 4.0 | 17.57 | 52.96% / 43.204 | 53.03% / 34.514 | 50.02% / 2.273 | 20.11% |
| FashionMNIST | 8.0 | 27.19 | 54.68% / 53.728 | 54.75% / 36.318 | 53.03% / 2.294 | 32.40% |
| FashionMNIST | 16.0 | 43.14 | 55.53% / 74.783 | 55.58% / 39.925 | 54.75% / 2.335 | 46.61% |
| MNIST | 1.0 | 7.79 | 29.76% / 34.088 | 29.97% / 32.995 | 24.95% / 2.266 | 3.21% |
| MNIST | 2.0 | 11.60 | 34.60% / 35.538 | 34.74% / 33.196 | 29.97% / 2.266 | 6.59% |
| MNIST | 4.0 | 17.57 | 38.32% / 38.417 | 38.45% / 33.696 | 34.74% / 2.266 | 12.29% |
| MNIST | 8.0 | 27.19 | 40.72% / 44.147 | 40.79% / 34.653 | 38.45% / 2.266 | 21.51% |
| MNIST | 16.0 | 43.14 | 42.01% / 55.636 | 42.11% / 36.623 | 40.79% / 2.266 | 34.17% |

## Full-frame rho=4 (nominal rho=2) 세부 비교

| Dataset | 방법 | Accuracy ± seed SD | 지연 mean / p50 / p95(ms) | >50ms | 원래 reference KL | oracle AUC |
|---|---|---:|---:|---:|---:|---:|
| FashionMNIST | Independent | 52.962 ± 0.323 | 43.204 / 35.330 / 73.658 | 0.104 | 3.25e-27 | 0.7055 |
| FashionMNIST | CR-Fixed | 52.987 ± 0.318 | 36.207 / 33.529 / 46.433 | 0.042 | 2.841e-27 | 0.7336 |
| FashionMNIST | CR-Power | 53.026 ± 0.294 | 34.514 / 33.196 / 39.706 | 0.021 | 3.459e-27 | 0.7776 |
| FashionMNIST | CR-Uses | 53.024 ± 0.294 | 34.514 / 33.196 / 39.706 | 0.021 | 3.414e-27 | 0.7775 |
| FashionMNIST | A-Only | 50.015 ± 0.241 | 2.273 / 2.266 / 2.266 | 0.000 | 45.41 | 0.7776 |
| MNIST | Independent | 38.324 ± 0.324 | 38.417 / 35.163 / 48.466 | 0.042 | 3.359e-27 | 0.6978 |
| MNIST | CR-Fixed | 38.381 ± 0.327 | 34.625 / 33.529 / 38.074 | 0.021 | 2.239e-27 | 0.7251 |
| MNIST | CR-Power | 38.447 ± 0.332 | 33.696 / 33.196 / 35.410 | 0.000 | 2.955e-27 | 0.7681 |
| MNIST | CR-Uses | 38.447 ± 0.332 | 33.696 / 33.196 / 35.410 | 0.000 | 2.955e-27 | 0.7681 |
| MNIST | A-Only | 34.738 ± 0.311 | 2.266 / 2.266 / 2.266 | 0.000 | 45.41 | 0.7681 |

## 모든 privacy 단계의 채택 기준

| Dataset | 방법 | Full-frame rho | Clean gap(%p) | 통신 절감(%) | 평균 기준 | 모든 seed utility 기준 |
|---|---|---:|---:|---:|---|---|
| FashionMNIST | CR-Fixed | 1.0 | 10.909 | 4.789 | False | False |
| FashionMNIST | CR-Power | 1.0 | 10.921 | 5.890 | False | False |
| FashionMNIST | CR-Uses | 1.0 | 10.920 | 5.890 | False | False |
| FashionMNIST | CR-Fixed | 2.0 | 6.287 | 9.188 | False | False |
| FashionMNIST | CR-Power | 2.0 | 6.281 | 11.400 | False | False |
| FashionMNIST | CR-Uses | 2.0 | 6.282 | 11.400 | False | False |
| FashionMNIST | CR-Fixed | 4.0 | 3.310 | 16.195 | True | True |
| FashionMNIST | CR-Power | 4.0 | 3.271 | 20.113 | True | True |
| FashionMNIST | CR-Uses | 4.0 | 3.272 | 20.113 | True | True |
| FashionMNIST | CR-Fixed | 8.0 | 1.573 | 26.090 | True | True |
| FashionMNIST | CR-Power | 8.0 | 1.549 | 32.404 | True | True |
| FashionMNIST | CR-Uses | 8.0 | 1.550 | 32.404 | True | True |
| FashionMNIST | CR-Fixed | 16.0 | 0.721 | 37.502 | True | True |
| FashionMNIST | CR-Power | 16.0 | 0.713 | 46.612 | True | True |
| FashionMNIST | CR-Uses | 16.0 | 0.712 | 46.612 | True | True |
| MNIST | CR-Fixed | 1.0 | 13.418 | 2.636 | False | False |
| MNIST | CR-Power | 1.0 | 13.295 | 3.206 | False | False |
| MNIST | CR-Uses | 1.0 | 13.295 | 3.206 | False | False |
| MNIST | CR-Fixed | 2.0 | 8.556 | 5.223 | False | False |
| MNIST | CR-Power | 2.0 | 8.532 | 6.589 | False | False |
| MNIST | CR-Uses | 2.0 | 8.532 | 6.589 | False | False |
| MNIST | CR-Fixed | 4.0 | 4.889 | 9.870 | False | False |
| MNIST | CR-Power | 4.0 | 4.823 | 12.289 | True | True |
| MNIST | CR-Uses | 4.0 | 4.823 | 12.289 | True | True |
| MNIST | CR-Fixed | 8.0 | 2.530 | 17.231 | True | True |
| MNIST | CR-Power | 8.0 | 2.476 | 21.506 | True | True |
| MNIST | CR-Uses | 8.0 | 2.476 | 21.506 | True | True |
| MNIST | CR-Fixed | 16.0 | 1.213 | 27.452 | True | True |
| MNIST | CR-Power | 16.0 | 1.162 | 34.173 | True | True |
| MNIST | CR-Uses | 16.0 | 1.162 | 34.173 | True | True |

## 전체 조건

| Dataset | K | Channel | Full-frame rho cap | 방법 | acc(%) | ms | TX energy | Full-frame rho bound max | cap초과 fraction |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| FashionMNIST | 10 | csi5 | 1.0 | A-Only | 10.098 | 2.266 | 1933 | 1.250 | 0.500 |
| FashionMNIST | 10 | csi5 | 1.0 | CR-Fixed | 10.387 | 2.542 | 2209 | 0.869 | 0.000 |
| FashionMNIST | 10 | csi5 | 1.0 | CR-Power | 10.192 | 2.542 | 2209 | 1.250 | 0.354 |
| FashionMNIST | 10 | csi5 | 1.0 | CR-Uses | 10.227 | 2.542 | 2209 | 1.250 | 0.292 |
| FashionMNIST | 10 | csi5 | 1.0 | Independent | 10.530 | 2.511 | 2178 | 1.289 | 1.000 |
| FashionMNIST | 10 | csi5 | 2.0 | A-Only | 10.214 | 2.266 | 1933 | 2.500 | 0.500 |
| FashionMNIST | 10 | csi5 | 2.0 | CR-Fixed | 10.590 | 2.542 | 2209 | 1.738 | 0.000 |
| FashionMNIST | 10 | csi5 | 2.0 | CR-Power | 10.381 | 2.542 | 2209 | 2.500 | 0.354 |
| FashionMNIST | 10 | csi5 | 2.0 | CR-Uses | 10.429 | 2.542 | 2209 | 2.500 | 0.292 |
| FashionMNIST | 10 | csi5 | 2.0 | Independent | 10.741 | 2.511 | 2178 | 2.577 | 1.000 |
| FashionMNIST | 10 | csi5 | 4.0 | A-Only | 10.456 | 2.273 | 1933 | 5.000 | 0.500 |
| FashionMNIST | 10 | csi5 | 4.0 | CR-Fixed | 10.793 | 2.542 | 2209 | 3.475 | 0.000 |
| FashionMNIST | 10 | csi5 | 4.0 | CR-Power | 10.690 | 2.542 | 2209 | 5.000 | 0.354 |
| FashionMNIST | 10 | csi5 | 4.0 | CR-Uses | 10.716 | 2.542 | 2209 | 5.000 | 0.292 |
| FashionMNIST | 10 | csi5 | 4.0 | Independent | 10.853 | 2.532 | 2178 | 5.155 | 1.000 |
| FashionMNIST | 10 | csi5 | 8.0 | A-Only | 10.713 | 2.287 | 1933 | 10.000 | 0.500 |
| FashionMNIST | 10 | csi5 | 8.0 | CR-Fixed | 11.122 | 2.556 | 2209 | 6.950 | 0.000 |
| FashionMNIST | 10 | csi5 | 8.0 | CR-Power | 11.072 | 2.542 | 2209 | 10.000 | 0.354 |
| FashionMNIST | 10 | csi5 | 8.0 | CR-Uses | 11.089 | 2.542 | 2209 | 10.000 | 0.292 |
| FashionMNIST | 10 | csi5 | 8.0 | Independent | 11.015 | 2.567 | 2179 | 10.309 | 1.000 |
| FashionMNIST | 10 | csi5 | 16.0 | A-Only | 11.105 | 2.328 | 1933 | 19.999 | 0.500 |
| FashionMNIST | 10 | csi5 | 16.0 | CR-Fixed | 11.678 | 2.584 | 2209 | 13.901 | 0.000 |
| FashionMNIST | 10 | csi5 | 16.0 | CR-Power | 11.574 | 2.556 | 2209 | 19.999 | 0.354 |
| FashionMNIST | 10 | csi5 | 16.0 | CR-Uses | 11.586 | 2.556 | 2209 | 19.999 | 0.292 |
| FashionMNIST | 10 | csi5 | 16.0 | Independent | 11.448 | 2.657 | 2179 | 20.619 | 1.000 |
| FashionMNIST | 10 | rayleigh | 1.0 | A-Only | 10.092 | 2.266 | 1933 | 1.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 1.0 | CR-Fixed | 10.388 | 2.542 | 2209 | 0.667 | 0.000 |
| FashionMNIST | 10 | rayleigh | 1.0 | CR-Power | 10.245 | 2.542 | 2209 | 1.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 1.0 | CR-Uses | 10.292 | 2.542 | 2209 | 1.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 1.0 | Independent | 10.531 | 2.511 | 2178 | 1.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 2.0 | A-Only | 10.209 | 2.266 | 1933 | 2.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 2.0 | CR-Fixed | 10.597 | 2.542 | 2209 | 1.333 | 0.000 |
| FashionMNIST | 10 | rayleigh | 2.0 | CR-Power | 10.449 | 2.542 | 2209 | 2.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 2.0 | CR-Uses | 10.503 | 2.542 | 2209 | 2.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 2.0 | Independent | 10.738 | 2.518 | 2178 | 2.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 4.0 | A-Only | 10.461 | 2.266 | 1933 | 4.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 4.0 | CR-Fixed | 10.786 | 2.549 | 2209 | 2.667 | 0.000 |
| FashionMNIST | 10 | rayleigh | 4.0 | CR-Power | 10.663 | 2.542 | 2209 | 4.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 4.0 | CR-Uses | 10.727 | 2.542 | 2209 | 4.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 4.0 | Independent | 10.852 | 2.532 | 2178 | 4.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 8.0 | A-Only | 10.708 | 2.266 | 1933 | 8.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 8.0 | CR-Fixed | 11.120 | 2.556 | 2209 | 5.333 | 0.000 |
| FashionMNIST | 10 | rayleigh | 8.0 | CR-Power | 11.058 | 2.549 | 2209 | 8.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 8.0 | CR-Uses | 11.071 | 2.549 | 2209 | 8.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 8.0 | Independent | 11.008 | 2.560 | 2179 | 8.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 16.0 | A-Only | 11.106 | 2.266 | 1933 | 16.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 16.0 | CR-Fixed | 11.681 | 2.570 | 2209 | 10.667 | 0.000 |
| FashionMNIST | 10 | rayleigh | 16.0 | CR-Power | 11.601 | 2.556 | 2209 | 16.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 16.0 | CR-Uses | 11.581 | 2.556 | 2209 | 16.000 | 0.000 |
| FashionMNIST | 10 | rayleigh | 16.0 | Independent | 11.444 | 2.629 | 2179 | 16.000 | 0.000 |
| FashionMNIST | 10 | unit | 1.0 | A-Only | 10.092 | 2.266 | 1933 | 1.000 | 0.000 |
| FashionMNIST | 10 | unit | 1.0 | CR-Fixed | 10.388 | 2.542 | 2209 | 0.667 | 0.000 |
| FashionMNIST | 10 | unit | 1.0 | CR-Power | 10.209 | 2.542 | 2209 | 1.000 | 0.000 |
| FashionMNIST | 10 | unit | 1.0 | CR-Uses | 10.388 | 2.542 | 2209 | 0.667 | 0.000 |
| FashionMNIST | 10 | unit | 1.0 | Independent | 10.531 | 2.511 | 2178 | 1.000 | 0.000 |
| FashionMNIST | 10 | unit | 2.0 | A-Only | 10.209 | 2.266 | 1933 | 2.000 | 0.000 |
| FashionMNIST | 10 | unit | 2.0 | CR-Fixed | 10.597 | 2.542 | 2209 | 1.333 | 0.000 |
| FashionMNIST | 10 | unit | 2.0 | CR-Power | 10.461 | 2.542 | 2209 | 2.000 | 0.000 |
| FashionMNIST | 10 | unit | 2.0 | CR-Uses | 10.597 | 2.542 | 2209 | 1.333 | 0.000 |
| FashionMNIST | 10 | unit | 2.0 | Independent | 10.738 | 2.511 | 2178 | 2.000 | 0.000 |
| FashionMNIST | 10 | unit | 4.0 | A-Only | 10.461 | 2.266 | 1933 | 4.000 | 0.000 |
| FashionMNIST | 10 | unit | 4.0 | CR-Fixed | 10.786 | 2.542 | 2209 | 2.667 | 0.000 |
| FashionMNIST | 10 | unit | 4.0 | CR-Power | 10.708 | 2.542 | 2209 | 4.000 | 0.000 |
| FashionMNIST | 10 | unit | 4.0 | CR-Uses | 10.786 | 2.542 | 2209 | 2.667 | 0.000 |
| FashionMNIST | 10 | unit | 4.0 | Independent | 10.852 | 2.511 | 2178 | 4.000 | 0.000 |
| FashionMNIST | 10 | unit | 8.0 | A-Only | 10.708 | 2.266 | 1933 | 8.000 | 0.000 |
| FashionMNIST | 10 | unit | 8.0 | CR-Fixed | 11.120 | 2.542 | 2209 | 5.333 | 0.000 |
| FashionMNIST | 10 | unit | 8.0 | CR-Power | 11.106 | 2.542 | 2209 | 8.000 | 0.000 |
| FashionMNIST | 10 | unit | 8.0 | CR-Uses | 11.120 | 2.542 | 2209 | 5.333 | 0.000 |
| FashionMNIST | 10 | unit | 8.0 | Independent | 11.008 | 2.511 | 2178 | 8.000 | 0.000 |
| FashionMNIST | 10 | unit | 16.0 | A-Only | 11.106 | 2.266 | 1933 | 16.000 | 0.000 |
| FashionMNIST | 10 | unit | 16.0 | CR-Fixed | 11.681 | 2.542 | 2209 | 10.667 | 0.000 |
| FashionMNIST | 10 | unit | 16.0 | CR-Power | 11.651 | 2.542 | 2209 | 16.000 | 0.000 |
| FashionMNIST | 10 | unit | 16.0 | CR-Uses | 11.681 | 2.542 | 2209 | 10.667 | 0.000 |
| FashionMNIST | 10 | unit | 16.0 | Independent | 11.444 | 2.511 | 2178 | 16.000 | 0.000 |
| FashionMNIST | 100 | csi5 | 1.0 | A-Only | 13.224 | 2.266 | 1933 | 1.145 | 0.438 |
| FashionMNIST | 100 | csi5 | 1.0 | CR-Fixed | 14.842 | 5.305 | 4965 | 0.881 | 0.000 |
| FashionMNIST | 100 | csi5 | 1.0 | CR-Power | 14.775 | 5.298 | 4965 | 1.145 | 0.438 |
| FashionMNIST | 100 | csi5 | 1.0 | CR-Uses | 14.755 | 5.298 | 4965 | 1.145 | 0.417 |
| FashionMNIST | 100 | csi5 | 1.0 | Independent | 15.166 | 5.310 | 4936 | 1.266 | 1.000 |
| FashionMNIST | 100 | csi5 | 2.0 | A-Only | 14.775 | 2.266 | 1933 | 2.290 | 0.438 |
| FashionMNIST | 100 | csi5 | 2.0 | CR-Fixed | 16.901 | 5.319 | 4966 | 1.762 | 0.000 |
| FashionMNIST | 100 | csi5 | 2.0 | CR-Power | 16.820 | 5.305 | 4965 | 2.290 | 0.438 |
| FashionMNIST | 100 | csi5 | 2.0 | CR-Uses | 16.800 | 5.305 | 4965 | 2.290 | 0.417 |
| FashionMNIST | 100 | csi5 | 2.0 | Independent | 17.251 | 5.386 | 4937 | 2.533 | 1.000 |
| FashionMNIST | 100 | csi5 | 4.0 | A-Only | 16.827 | 2.273 | 1933 | 4.580 | 0.438 |
| FashionMNIST | 100 | csi5 | 4.0 | CR-Fixed | 19.715 | 5.360 | 4966 | 3.525 | 0.000 |
| FashionMNIST | 100 | csi5 | 4.0 | CR-Power | 19.512 | 5.319 | 4966 | 4.580 | 0.438 |
| FashionMNIST | 100 | csi5 | 4.0 | CR-Uses | 19.492 | 5.319 | 4966 | 4.580 | 0.417 |
| FashionMNIST | 100 | csi5 | 4.0 | Independent | 20.088 | 5.601 | 4939 | 5.065 | 1.000 |
| FashionMNIST | 100 | csi5 | 8.0 | A-Only | 19.507 | 2.287 | 1933 | 9.161 | 0.438 |
| FashionMNIST | 100 | csi5 | 8.0 | CR-Fixed | 23.455 | 5.499 | 4968 | 7.049 | 0.000 |
| FashionMNIST | 100 | csi5 | 8.0 | CR-Power | 23.157 | 5.374 | 4967 | 9.161 | 0.438 |
| FashionMNIST | 100 | csi5 | 8.0 | CR-Uses | 23.161 | 5.367 | 4967 | 9.161 | 0.417 |
| FashionMNIST | 100 | csi5 | 8.0 | Independent | 23.864 | 6.080 | 4943 | 10.130 | 1.000 |
| FashionMNIST | 100 | csi5 | 16.0 | A-Only | 23.151 | 2.321 | 1933 | 18.322 | 0.438 |
| FashionMNIST | 100 | csi5 | 16.0 | CR-Fixed | 28.444 | 5.791 | 4971 | 14.099 | 0.000 |
| FashionMNIST | 100 | csi5 | 16.0 | CR-Power | 28.184 | 5.520 | 4968 | 18.322 | 0.438 |
| FashionMNIST | 100 | csi5 | 16.0 | CR-Uses | 28.210 | 5.513 | 4968 | 18.322 | 0.417 |
| FashionMNIST | 100 | csi5 | 16.0 | Independent | 28.821 | 7.044 | 4952 | 20.260 | 1.000 |
| FashionMNIST | 100 | rayleigh | 1.0 | A-Only | 13.219 | 2.266 | 1933 | 1.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 1.0 | CR-Fixed | 14.838 | 5.305 | 4965 | 0.667 | 0.000 |
| FashionMNIST | 100 | rayleigh | 1.0 | CR-Power | 14.735 | 5.298 | 4965 | 1.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 1.0 | CR-Uses | 14.719 | 5.298 | 4965 | 1.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 1.0 | Independent | 15.169 | 5.330 | 4936 | 1.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 2.0 | A-Only | 14.770 | 2.273 | 1933 | 2.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 2.0 | CR-Fixed | 16.888 | 5.326 | 4966 | 1.333 | 0.000 |
| FashionMNIST | 100 | rayleigh | 2.0 | CR-Power | 16.772 | 5.305 | 4965 | 2.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 2.0 | CR-Uses | 16.763 | 5.305 | 4965 | 2.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 2.0 | Independent | 17.243 | 5.441 | 4937 | 2.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 4.0 | A-Only | 16.814 | 2.280 | 1933 | 4.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 4.0 | CR-Fixed | 19.696 | 5.388 | 4966 | 2.667 | 0.000 |
| FashionMNIST | 100 | rayleigh | 4.0 | CR-Power | 19.479 | 5.333 | 4966 | 4.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 4.0 | CR-Uses | 19.468 | 5.333 | 4966 | 4.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 4.0 | Independent | 20.086 | 5.691 | 4939 | 4.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 8.0 | A-Only | 19.494 | 2.294 | 1933 | 8.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 8.0 | CR-Fixed | 23.431 | 5.548 | 4968 | 5.333 | 0.000 |
| FashionMNIST | 100 | rayleigh | 8.0 | CR-Power | 23.116 | 5.402 | 4967 | 8.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 8.0 | CR-Uses | 23.098 | 5.402 | 4967 | 8.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 8.0 | Independent | 23.856 | 6.232 | 4944 | 8.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 16.0 | A-Only | 23.126 | 2.335 | 1933 | 16.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 16.0 | CR-Fixed | 28.427 | 5.915 | 4971 | 10.667 | 0.000 |
| FashionMNIST | 100 | rayleigh | 16.0 | CR-Power | 28.139 | 5.575 | 4968 | 16.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 16.0 | CR-Uses | 28.141 | 5.569 | 4968 | 16.000 | 0.000 |
| FashionMNIST | 100 | rayleigh | 16.0 | Independent | 28.813 | 7.391 | 4953 | 16.000 | 0.000 |
| FashionMNIST | 100 | unit | 1.0 | A-Only | 13.219 | 2.266 | 1933 | 1.000 | 0.000 |
| FashionMNIST | 100 | unit | 1.0 | CR-Fixed | 14.838 | 5.298 | 4965 | 0.667 | 0.000 |
| FashionMNIST | 100 | unit | 1.0 | CR-Power | 14.770 | 5.298 | 4965 | 1.000 | 0.000 |
| FashionMNIST | 100 | unit | 1.0 | CR-Uses | 14.838 | 5.298 | 4965 | 0.667 | 0.000 |
| FashionMNIST | 100 | unit | 1.0 | Independent | 15.169 | 5.268 | 4935 | 1.000 | 0.000 |
| FashionMNIST | 100 | unit | 2.0 | A-Only | 14.770 | 2.266 | 1933 | 2.000 | 0.000 |
| FashionMNIST | 100 | unit | 2.0 | CR-Fixed | 16.888 | 5.298 | 4965 | 1.333 | 0.000 |
| FashionMNIST | 100 | unit | 2.0 | CR-Power | 16.814 | 5.298 | 4965 | 2.000 | 0.000 |
| FashionMNIST | 100 | unit | 2.0 | CR-Uses | 16.888 | 5.298 | 4965 | 1.333 | 0.000 |
| FashionMNIST | 100 | unit | 2.0 | Independent | 17.243 | 5.268 | 4935 | 2.000 | 0.000 |
| FashionMNIST | 100 | unit | 4.0 | A-Only | 16.814 | 2.266 | 1933 | 4.000 | 0.000 |
| FashionMNIST | 100 | unit | 4.0 | CR-Fixed | 19.696 | 5.298 | 4965 | 2.667 | 0.000 |
| FashionMNIST | 100 | unit | 4.0 | CR-Power | 19.494 | 5.298 | 4965 | 4.000 | 0.000 |
| FashionMNIST | 100 | unit | 4.0 | CR-Uses | 19.696 | 5.298 | 4965 | 2.667 | 0.000 |
| FashionMNIST | 100 | unit | 4.0 | Independent | 20.086 | 5.268 | 4936 | 4.000 | 0.000 |
| FashionMNIST | 100 | unit | 8.0 | A-Only | 19.494 | 2.266 | 1933 | 8.000 | 0.000 |
| FashionMNIST | 100 | unit | 8.0 | CR-Fixed | 23.431 | 5.298 | 4965 | 5.333 | 0.000 |
| FashionMNIST | 100 | unit | 8.0 | CR-Power | 23.126 | 5.298 | 4965 | 8.000 | 0.000 |
| FashionMNIST | 100 | unit | 8.0 | CR-Uses | 23.431 | 5.298 | 4965 | 5.333 | 0.000 |
| FashionMNIST | 100 | unit | 8.0 | Independent | 23.856 | 5.268 | 4936 | 8.000 | 0.000 |
| FashionMNIST | 100 | unit | 16.0 | A-Only | 23.126 | 2.266 | 1933 | 16.000 | 0.000 |
| FashionMNIST | 100 | unit | 16.0 | CR-Fixed | 28.427 | 5.298 | 4966 | 10.667 | 0.000 |
| FashionMNIST | 100 | unit | 16.0 | CR-Power | 28.149 | 5.298 | 4965 | 16.000 | 0.000 |
| FashionMNIST | 100 | unit | 16.0 | CR-Uses | 28.427 | 5.298 | 4966 | 10.667 | 0.000 |
| FashionMNIST | 100 | unit | 16.0 | Independent | 28.813 | 5.268 | 4937 | 16.000 | 0.000 |
| FashionMNIST | 1000 | csi5 | 1.0 | A-Only | 39.204 | 2.266 | 1933 | 1.101 | 0.562 |
| FashionMNIST | 1000 | csi5 | 1.0 | CR-Fixed | 45.399 | 33.210 | 3.253e+04 | 0.925 | 0.000 |
| FashionMNIST | 1000 | csi5 | 1.0 | CR-Power | 45.377 | 33.016 | 3.253e+04 | 1.101 | 0.562 |
| FashionMNIST | 1000 | csi5 | 1.0 | CR-Uses | 45.384 | 33.016 | 3.253e+04 | 1.101 | 0.542 |
| FashionMNIST | 1000 | csi5 | 1.0 | Independent | 45.400 | 34.115 | 3.251e+04 | 1.388 | 1.000 |
| FashionMNIST | 1000 | csi5 | 2.0 | A-Only | 45.378 | 2.273 | 1933 | 2.201 | 0.562 |
| FashionMNIST | 1000 | csi5 | 2.0 | CR-Fixed | 50.011 | 33.654 | 3.254e+04 | 1.849 | 0.000 |
| FashionMNIST | 1000 | csi5 | 2.0 | CR-Power | 50.023 | 33.217 | 3.254e+04 | 2.201 | 0.562 |
| FashionMNIST | 1000 | csi5 | 2.0 | CR-Uses | 50.027 | 33.210 | 3.254e+04 | 2.201 | 0.542 |
| FashionMNIST | 1000 | csi5 | 2.0 | Independent | 49.977 | 35.579 | 3.253e+04 | 2.776 | 1.000 |
| FashionMNIST | 1000 | csi5 | 4.0 | A-Only | 50.017 | 2.287 | 1933 | 4.403 | 0.562 |
| FashionMNIST | 1000 | csi5 | 4.0 | CR-Fixed | 52.995 | 34.639 | 3.255e+04 | 3.699 | 0.000 |
| FashionMNIST | 1000 | csi5 | 4.0 | CR-Power | 53.031 | 33.702 | 3.254e+04 | 4.403 | 0.562 |
| FashionMNIST | 1000 | csi5 | 4.0 | CR-Uses | 53.034 | 33.696 | 3.254e+04 | 4.403 | 0.542 |
| FashionMNIST | 1000 | csi5 | 4.0 | Independent | 52.953 | 38.528 | 3.256e+04 | 5.551 | 1.000 |
| FashionMNIST | 1000 | csi5 | 8.0 | A-Only | 53.023 | 2.315 | 1933 | 8.805 | 0.562 |
| FashionMNIST | 1000 | csi5 | 8.0 | CR-Fixed | 54.721 | 36.602 | 3.257e+04 | 7.398 | 0.000 |
| FashionMNIST | 1000 | csi5 | 8.0 | CR-Power | 54.749 | 34.722 | 3.255e+04 | 8.805 | 0.562 |
| FashionMNIST | 1000 | csi5 | 8.0 | CR-Uses | 54.750 | 34.708 | 3.255e+04 | 8.805 | 0.542 |
| FashionMNIST | 1000 | csi5 | 8.0 | Independent | 54.680 | 44.341 | 3.262e+04 | 11.103 | 1.000 |
| FashionMNIST | 1000 | csi5 | 16.0 | A-Only | 54.746 | 2.377 | 1933 | 17.611 | 0.562 |
| FashionMNIST | 1000 | csi5 | 16.0 | CR-Fixed | 55.580 | 40.501 | 3.261e+04 | 14.795 | 0.000 |
| FashionMNIST | 1000 | csi5 | 16.0 | CR-Power | 55.593 | 36.748 | 3.257e+04 | 17.611 | 0.562 |
| FashionMNIST | 1000 | csi5 | 16.0 | CR-Uses | 55.596 | 36.720 | 3.257e+04 | 17.611 | 0.542 |
| FashionMNIST | 1000 | csi5 | 16.0 | Independent | 55.535 | 56.059 | 3.273e+04 | 22.206 | 1.000 |
| FashionMNIST | 1000 | rayleigh | 1.0 | A-Only | 39.188 | 2.266 | 1933 | 1.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 1.0 | CR-Fixed | 45.388 | 33.591 | 3.254e+04 | 0.667 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 1.0 | CR-Power | 45.376 | 33.203 | 3.253e+04 | 1.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 1.0 | CR-Uses | 45.377 | 33.203 | 3.253e+04 | 1.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 1.0 | Independent | 45.397 | 35.281 | 3.252e+04 | 1.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 2.0 | A-Only | 45.376 | 2.266 | 1933 | 2.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 2.0 | CR-Fixed | 50.010 | 34.459 | 3.254e+04 | 1.333 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 2.0 | CR-Power | 50.015 | 33.619 | 3.254e+04 | 2.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 2.0 | CR-Uses | 50.015 | 33.619 | 3.254e+04 | 2.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 2.0 | Independent | 49.975 | 37.945 | 3.253e+04 | 2.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 4.0 | A-Only | 50.015 | 2.273 | 1933 | 4.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 4.0 | CR-Fixed | 52.987 | 36.207 | 3.255e+04 | 2.667 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 4.0 | CR-Power | 53.026 | 34.514 | 3.254e+04 | 4.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 4.0 | CR-Uses | 53.024 | 34.514 | 3.254e+04 | 4.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 4.0 | Independent | 52.962 | 43.204 | 3.257e+04 | 4.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 8.0 | A-Only | 53.026 | 2.294 | 1933 | 8.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 8.0 | CR-Fixed | 54.723 | 39.710 | 3.258e+04 | 5.333 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 8.0 | CR-Power | 54.747 | 36.318 | 3.255e+04 | 8.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 8.0 | CR-Uses | 54.747 | 36.318 | 3.255e+04 | 8.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 8.0 | Independent | 54.676 | 53.728 | 3.264e+04 | 8.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 16.0 | A-Only | 54.747 | 2.335 | 1933 | 16.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 16.0 | CR-Fixed | 55.576 | 46.738 | 3.263e+04 | 10.667 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 16.0 | CR-Power | 55.584 | 39.925 | 3.258e+04 | 16.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 16.0 | CR-Uses | 55.585 | 39.925 | 3.258e+04 | 16.000 | 0.000 |
| FashionMNIST | 1000 | rayleigh | 16.0 | Independent | 55.527 | 74.783 | 3.278e+04 | 16.000 | 0.000 |
| FashionMNIST | 1000 | unit | 1.0 | A-Only | 39.188 | 2.266 | 1933 | 1.000 | 0.000 |
| FashionMNIST | 1000 | unit | 1.0 | CR-Fixed | 45.388 | 32.863 | 3.253e+04 | 0.667 | 0.000 |
| FashionMNIST | 1000 | unit | 1.0 | CR-Power | 45.376 | 32.863 | 3.253e+04 | 1.000 | 0.000 |
| FashionMNIST | 1000 | unit | 1.0 | CR-Uses | 45.388 | 32.863 | 3.253e+04 | 0.667 | 0.000 |
| FashionMNIST | 1000 | unit | 1.0 | Independent | 45.397 | 32.832 | 3.25e+04 | 1.000 | 0.000 |
| FashionMNIST | 1000 | unit | 2.0 | A-Only | 45.376 | 2.266 | 1933 | 2.000 | 0.000 |
| FashionMNIST | 1000 | unit | 2.0 | CR-Fixed | 50.010 | 32.863 | 3.253e+04 | 1.333 | 0.000 |
| FashionMNIST | 1000 | unit | 2.0 | CR-Power | 50.015 | 32.863 | 3.253e+04 | 2.000 | 0.000 |
| FashionMNIST | 1000 | unit | 2.0 | CR-Uses | 50.010 | 32.863 | 3.253e+04 | 1.333 | 0.000 |
| FashionMNIST | 1000 | unit | 2.0 | Independent | 49.975 | 32.832 | 3.25e+04 | 2.000 | 0.000 |
| FashionMNIST | 1000 | unit | 4.0 | A-Only | 50.015 | 2.266 | 1933 | 4.000 | 0.000 |
| FashionMNIST | 1000 | unit | 4.0 | CR-Fixed | 52.987 | 32.863 | 3.253e+04 | 2.667 | 0.000 |
| FashionMNIST | 1000 | unit | 4.0 | CR-Power | 53.026 | 32.863 | 3.253e+04 | 4.000 | 0.000 |
| FashionMNIST | 1000 | unit | 4.0 | CR-Uses | 52.987 | 32.863 | 3.253e+04 | 2.667 | 0.000 |
| FashionMNIST | 1000 | unit | 4.0 | Independent | 52.962 | 32.832 | 3.25e+04 | 4.000 | 0.000 |
| FashionMNIST | 1000 | unit | 8.0 | A-Only | 53.026 | 2.266 | 1933 | 8.000 | 0.000 |
| FashionMNIST | 1000 | unit | 8.0 | CR-Fixed | 54.723 | 32.863 | 3.253e+04 | 5.333 | 0.000 |
| FashionMNIST | 1000 | unit | 8.0 | CR-Power | 54.747 | 32.863 | 3.253e+04 | 8.000 | 0.000 |
| FashionMNIST | 1000 | unit | 8.0 | CR-Uses | 54.723 | 32.863 | 3.253e+04 | 5.333 | 0.000 |
| FashionMNIST | 1000 | unit | 8.0 | Independent | 54.676 | 32.832 | 3.251e+04 | 8.000 | 0.000 |
| FashionMNIST | 1000 | unit | 16.0 | A-Only | 54.747 | 2.266 | 1933 | 16.000 | 0.000 |
| FashionMNIST | 1000 | unit | 16.0 | CR-Fixed | 55.576 | 32.863 | 3.254e+04 | 10.667 | 0.000 |
| FashionMNIST | 1000 | unit | 16.0 | CR-Power | 55.584 | 32.863 | 3.253e+04 | 16.000 | 0.000 |
| FashionMNIST | 1000 | unit | 16.0 | CR-Uses | 55.576 | 32.863 | 3.254e+04 | 10.667 | 0.000 |
| FashionMNIST | 1000 | unit | 16.0 | Independent | 55.527 | 32.832 | 3.252e+04 | 16.000 | 0.000 |
| MNIST | 10 | csi5 | 1.0 | A-Only | 10.212 | 2.266 | 1933 | 1.194 | 0.562 |
| MNIST | 10 | csi5 | 1.0 | CR-Fixed | 10.195 | 2.542 | 2209 | 0.853 | 0.000 |
| MNIST | 10 | csi5 | 1.0 | CR-Power | 10.150 | 2.542 | 2209 | 1.194 | 0.438 |
| MNIST | 10 | csi5 | 1.0 | CR-Uses | 10.134 | 2.542 | 2209 | 1.194 | 0.375 |
| MNIST | 10 | csi5 | 1.0 | Independent | 9.920 | 2.511 | 2178 | 1.223 | 1.000 |
| MNIST | 10 | csi5 | 2.0 | A-Only | 10.213 | 2.266 | 1933 | 2.388 | 0.562 |
| MNIST | 10 | csi5 | 2.0 | CR-Fixed | 10.179 | 2.542 | 2209 | 1.706 | 0.000 |
| MNIST | 10 | csi5 | 2.0 | CR-Power | 10.157 | 2.542 | 2209 | 2.388 | 0.438 |
| MNIST | 10 | csi5 | 2.0 | CR-Uses | 10.169 | 2.542 | 2209 | 2.388 | 0.375 |
| MNIST | 10 | csi5 | 2.0 | Independent | 10.004 | 2.511 | 2178 | 2.446 | 1.000 |
| MNIST | 10 | csi5 | 4.0 | A-Only | 10.239 | 2.266 | 1933 | 4.776 | 0.562 |
| MNIST | 10 | csi5 | 4.0 | CR-Fixed | 10.180 | 2.542 | 2209 | 3.412 | 0.000 |
| MNIST | 10 | csi5 | 4.0 | CR-Power | 10.279 | 2.542 | 2209 | 4.776 | 0.438 |
| MNIST | 10 | csi5 | 4.0 | CR-Uses | 10.292 | 2.542 | 2209 | 4.776 | 0.375 |
| MNIST | 10 | csi5 | 4.0 | Independent | 10.079 | 2.532 | 2178 | 4.892 | 1.000 |
| MNIST | 10 | csi5 | 8.0 | A-Only | 10.344 | 2.266 | 1933 | 9.553 | 0.562 |
| MNIST | 10 | csi5 | 8.0 | CR-Fixed | 10.287 | 2.542 | 2209 | 6.825 | 0.000 |
| MNIST | 10 | csi5 | 8.0 | CR-Power | 10.431 | 2.542 | 2209 | 9.553 | 0.438 |
| MNIST | 10 | csi5 | 8.0 | CR-Uses | 10.434 | 2.542 | 2209 | 9.553 | 0.375 |
| MNIST | 10 | csi5 | 8.0 | Independent | 10.172 | 2.587 | 2179 | 9.783 | 1.000 |
| MNIST | 10 | csi5 | 16.0 | A-Only | 10.480 | 2.266 | 1933 | 19.105 | 0.562 |
| MNIST | 10 | csi5 | 16.0 | CR-Fixed | 10.457 | 2.570 | 2209 | 13.649 | 0.000 |
| MNIST | 10 | csi5 | 16.0 | CR-Power | 10.572 | 2.542 | 2209 | 19.105 | 0.438 |
| MNIST | 10 | csi5 | 16.0 | CR-Uses | 10.573 | 2.542 | 2209 | 19.105 | 0.375 |
| MNIST | 10 | csi5 | 16.0 | Independent | 10.252 | 2.691 | 2179 | 19.567 | 1.000 |
| MNIST | 10 | rayleigh | 1.0 | A-Only | 10.208 | 2.273 | 1933 | 1.000 | 0.000 |
| MNIST | 10 | rayleigh | 1.0 | CR-Fixed | 10.191 | 2.556 | 2209 | 0.667 | 0.000 |
| MNIST | 10 | rayleigh | 1.0 | CR-Power | 10.271 | 2.549 | 2209 | 1.000 | 0.000 |
| MNIST | 10 | rayleigh | 1.0 | CR-Uses | 10.294 | 2.549 | 2209 | 1.000 | 0.000 |
| MNIST | 10 | rayleigh | 1.0 | Independent | 9.920 | 2.560 | 2178 | 1.000 | 0.000 |
| MNIST | 10 | rayleigh | 2.0 | A-Only | 10.218 | 2.287 | 1933 | 2.000 | 0.000 |
| MNIST | 10 | rayleigh | 2.0 | CR-Fixed | 10.177 | 2.584 | 2209 | 1.333 | 0.000 |
| MNIST | 10 | rayleigh | 2.0 | CR-Power | 10.248 | 2.556 | 2209 | 2.000 | 0.000 |
| MNIST | 10 | rayleigh | 2.0 | CR-Uses | 10.281 | 2.556 | 2209 | 2.000 | 0.000 |
| MNIST | 10 | rayleigh | 2.0 | Independent | 10.001 | 2.622 | 2179 | 2.000 | 0.000 |
| MNIST | 10 | rayleigh | 4.0 | A-Only | 10.234 | 2.308 | 1933 | 4.000 | 0.000 |
| MNIST | 10 | rayleigh | 4.0 | CR-Fixed | 10.181 | 2.625 | 2209 | 2.667 | 0.000 |
| MNIST | 10 | rayleigh | 4.0 | CR-Power | 10.321 | 2.577 | 2209 | 4.000 | 0.000 |
| MNIST | 10 | rayleigh | 4.0 | CR-Uses | 10.349 | 2.577 | 2209 | 4.000 | 0.000 |
| MNIST | 10 | rayleigh | 4.0 | Independent | 10.077 | 2.754 | 2179 | 4.000 | 0.000 |
| MNIST | 10 | rayleigh | 8.0 | A-Only | 10.338 | 2.349 | 1933 | 8.000 | 0.000 |
| MNIST | 10 | rayleigh | 8.0 | CR-Fixed | 10.285 | 2.715 | 2210 | 5.333 | 0.000 |
| MNIST | 10 | rayleigh | 8.0 | CR-Power | 10.454 | 2.611 | 2209 | 8.000 | 0.000 |
| MNIST | 10 | rayleigh | 8.0 | CR-Uses | 10.478 | 2.611 | 2209 | 8.000 | 0.000 |
| MNIST | 10 | rayleigh | 8.0 | Independent | 10.166 | 2.997 | 2180 | 8.000 | 0.000 |
| MNIST | 10 | rayleigh | 16.0 | A-Only | 10.475 | 2.433 | 1934 | 16.000 | 0.000 |
| MNIST | 10 | rayleigh | 16.0 | CR-Fixed | 10.454 | 2.917 | 2211 | 10.667 | 0.000 |
| MNIST | 10 | rayleigh | 16.0 | CR-Power | 10.597 | 2.702 | 2210 | 16.000 | 0.000 |
| MNIST | 10 | rayleigh | 16.0 | CR-Uses | 10.613 | 2.702 | 2210 | 16.000 | 0.000 |
| MNIST | 10 | rayleigh | 16.0 | Independent | 10.247 | 3.538 | 2182 | 16.000 | 0.000 |
| MNIST | 10 | unit | 1.0 | A-Only | 10.208 | 2.266 | 1933 | 1.000 | 0.000 |
| MNIST | 10 | unit | 1.0 | CR-Fixed | 10.191 | 2.542 | 2209 | 0.667 | 0.000 |
| MNIST | 10 | unit | 1.0 | CR-Power | 10.218 | 2.542 | 2209 | 1.000 | 0.000 |
| MNIST | 10 | unit | 1.0 | CR-Uses | 10.191 | 2.542 | 2209 | 0.667 | 0.000 |
| MNIST | 10 | unit | 1.0 | Independent | 9.920 | 2.511 | 2178 | 1.000 | 0.000 |
| MNIST | 10 | unit | 2.0 | A-Only | 10.218 | 2.266 | 1933 | 2.000 | 0.000 |
| MNIST | 10 | unit | 2.0 | CR-Fixed | 10.177 | 2.542 | 2209 | 1.333 | 0.000 |
| MNIST | 10 | unit | 2.0 | CR-Power | 10.234 | 2.542 | 2209 | 2.000 | 0.000 |
| MNIST | 10 | unit | 2.0 | CR-Uses | 10.177 | 2.542 | 2209 | 1.333 | 0.000 |
| MNIST | 10 | unit | 2.0 | Independent | 10.001 | 2.511 | 2178 | 2.000 | 0.000 |
| MNIST | 10 | unit | 4.0 | A-Only | 10.234 | 2.266 | 1933 | 4.000 | 0.000 |
| MNIST | 10 | unit | 4.0 | CR-Fixed | 10.181 | 2.542 | 2209 | 2.667 | 0.000 |
| MNIST | 10 | unit | 4.0 | CR-Power | 10.338 | 2.542 | 2209 | 4.000 | 0.000 |
| MNIST | 10 | unit | 4.0 | CR-Uses | 10.181 | 2.542 | 2209 | 2.667 | 0.000 |
| MNIST | 10 | unit | 4.0 | Independent | 10.077 | 2.511 | 2178 | 4.000 | 0.000 |
| MNIST | 10 | unit | 8.0 | A-Only | 10.338 | 2.266 | 1933 | 8.000 | 0.000 |
| MNIST | 10 | unit | 8.0 | CR-Fixed | 10.285 | 2.542 | 2209 | 5.333 | 0.000 |
| MNIST | 10 | unit | 8.0 | CR-Power | 10.475 | 2.542 | 2209 | 8.000 | 0.000 |
| MNIST | 10 | unit | 8.0 | CR-Uses | 10.285 | 2.542 | 2209 | 5.333 | 0.000 |
| MNIST | 10 | unit | 8.0 | Independent | 10.166 | 2.511 | 2178 | 8.000 | 0.000 |
| MNIST | 10 | unit | 16.0 | A-Only | 10.475 | 2.266 | 1933 | 16.000 | 0.000 |
| MNIST | 10 | unit | 16.0 | CR-Fixed | 10.454 | 2.542 | 2209 | 10.667 | 0.000 |
| MNIST | 10 | unit | 16.0 | CR-Power | 10.618 | 2.542 | 2209 | 16.000 | 0.000 |
| MNIST | 10 | unit | 16.0 | CR-Uses | 10.454 | 2.542 | 2209 | 10.667 | 0.000 |
| MNIST | 10 | unit | 16.0 | Independent | 10.247 | 2.511 | 2178 | 16.000 | 0.000 |
| MNIST | 100 | csi5 | 1.0 | A-Only | 10.903 | 2.266 | 1933 | 1.151 | 0.458 |
| MNIST | 100 | csi5 | 1.0 | CR-Fixed | 11.648 | 5.340 | 4965 | 0.889 | 0.000 |
| MNIST | 100 | csi5 | 1.0 | CR-Power | 11.515 | 5.312 | 4965 | 1.151 | 0.458 |
| MNIST | 100 | csi5 | 1.0 | CR-Uses | 11.521 | 5.312 | 4965 | 1.151 | 0.438 |
| MNIST | 100 | csi5 | 1.0 | Independent | 11.666 | 5.414 | 4936 | 1.288 | 1.000 |
| MNIST | 100 | csi5 | 2.0 | A-Only | 11.513 | 2.266 | 1933 | 2.303 | 0.458 |
| MNIST | 100 | csi5 | 2.0 | CR-Fixed | 12.483 | 5.388 | 4966 | 1.777 | 0.000 |
| MNIST | 100 | csi5 | 2.0 | CR-Power | 12.341 | 5.340 | 4966 | 2.303 | 0.458 |
| MNIST | 100 | csi5 | 2.0 | CR-Uses | 12.349 | 5.340 | 4966 | 2.303 | 0.438 |
| MNIST | 100 | csi5 | 2.0 | Independent | 12.506 | 5.656 | 4938 | 2.576 | 1.000 |
| MNIST | 100 | csi5 | 4.0 | A-Only | 12.341 | 2.266 | 1933 | 4.606 | 0.458 |
| MNIST | 100 | csi5 | 4.0 | CR-Fixed | 13.646 | 5.534 | 4967 | 3.554 | 0.000 |
| MNIST | 100 | csi5 | 4.0 | CR-Power | 13.554 | 5.388 | 4966 | 4.606 | 0.458 |
| MNIST | 100 | csi5 | 4.0 | CR-Uses | 13.569 | 5.388 | 4966 | 4.606 | 0.438 |
| MNIST | 100 | csi5 | 4.0 | Independent | 13.749 | 6.156 | 4941 | 5.152 | 1.000 |
| MNIST | 100 | csi5 | 8.0 | A-Only | 13.558 | 2.266 | 1933 | 9.211 | 0.458 |
| MNIST | 100 | csi5 | 8.0 | CR-Fixed | 15.458 | 5.860 | 4969 | 7.109 | 0.000 |
| MNIST | 100 | csi5 | 8.0 | CR-Power | 15.367 | 5.541 | 4967 | 9.211 | 0.458 |
| MNIST | 100 | csi5 | 8.0 | CR-Uses | 15.377 | 5.541 | 4967 | 9.211 | 0.438 |
| MNIST | 100 | csi5 | 8.0 | Independent | 15.546 | 7.197 | 4947 | 10.304 | 1.000 |
| MNIST | 100 | csi5 | 16.0 | A-Only | 15.368 | 2.273 | 1933 | 18.423 | 0.458 |
| MNIST | 100 | csi5 | 16.0 | CR-Fixed | 18.082 | 6.526 | 4973 | 14.218 | 0.000 |
| MNIST | 100 | csi5 | 16.0 | CR-Power | 17.991 | 5.881 | 4969 | 18.423 | 0.458 |
| MNIST | 100 | csi5 | 16.0 | CR-Uses | 18.001 | 5.874 | 4969 | 18.423 | 0.438 |
| MNIST | 100 | csi5 | 16.0 | Independent | 18.174 | 9.299 | 4959 | 20.608 | 1.000 |
| MNIST | 100 | rayleigh | 1.0 | A-Only | 10.904 | 2.266 | 1933 | 1.000 | 0.000 |
| MNIST | 100 | rayleigh | 1.0 | CR-Fixed | 11.645 | 5.319 | 4965 | 0.667 | 0.000 |
| MNIST | 100 | rayleigh | 1.0 | CR-Power | 11.509 | 5.305 | 4965 | 1.000 | 0.000 |
| MNIST | 100 | rayleigh | 1.0 | CR-Uses | 11.513 | 5.305 | 4965 | 1.000 | 0.000 |
| MNIST | 100 | rayleigh | 1.0 | Independent | 11.666 | 5.386 | 4936 | 1.000 | 0.000 |
| MNIST | 100 | rayleigh | 2.0 | A-Only | 11.509 | 2.266 | 1933 | 2.000 | 0.000 |
| MNIST | 100 | rayleigh | 2.0 | CR-Fixed | 12.475 | 5.360 | 4966 | 1.333 | 0.000 |
| MNIST | 100 | rayleigh | 2.0 | CR-Power | 12.336 | 5.326 | 4965 | 2.000 | 0.000 |
| MNIST | 100 | rayleigh | 2.0 | CR-Uses | 12.340 | 5.326 | 4965 | 2.000 | 0.000 |
| MNIST | 100 | rayleigh | 2.0 | Independent | 12.510 | 5.552 | 4938 | 2.000 | 0.000 |
| MNIST | 100 | rayleigh | 4.0 | A-Only | 12.336 | 2.266 | 1933 | 4.000 | 0.000 |
| MNIST | 100 | rayleigh | 4.0 | CR-Fixed | 13.643 | 5.478 | 4967 | 2.667 | 0.000 |
| MNIST | 100 | rayleigh | 4.0 | CR-Power | 13.550 | 5.367 | 4966 | 4.000 | 0.000 |
| MNIST | 100 | rayleigh | 4.0 | CR-Uses | 13.554 | 5.367 | 4966 | 4.000 | 0.000 |
| MNIST | 100 | rayleigh | 4.0 | Independent | 13.742 | 5.955 | 4940 | 4.000 | 0.000 |
| MNIST | 100 | rayleigh | 8.0 | A-Only | 13.550 | 2.266 | 1933 | 8.000 | 0.000 |
| MNIST | 100 | rayleigh | 8.0 | CR-Fixed | 15.446 | 5.749 | 4969 | 5.333 | 0.000 |
| MNIST | 100 | rayleigh | 8.0 | CR-Power | 15.353 | 5.485 | 4967 | 8.000 | 0.000 |
| MNIST | 100 | rayleigh | 8.0 | CR-Uses | 15.358 | 5.485 | 4967 | 8.000 | 0.000 |
| MNIST | 100 | rayleigh | 8.0 | Independent | 15.541 | 6.815 | 4946 | 8.000 | 0.000 |
| MNIST | 100 | rayleigh | 16.0 | A-Only | 15.353 | 2.273 | 1933 | 16.000 | 0.000 |
| MNIST | 100 | rayleigh | 16.0 | CR-Fixed | 18.062 | 6.290 | 4972 | 10.667 | 0.000 |
| MNIST | 100 | rayleigh | 16.0 | CR-Power | 17.982 | 5.763 | 4969 | 16.000 | 0.000 |
| MNIST | 100 | rayleigh | 16.0 | CR-Uses | 17.986 | 5.763 | 4969 | 16.000 | 0.000 |
| MNIST | 100 | rayleigh | 16.0 | Independent | 18.161 | 8.543 | 4957 | 16.000 | 0.000 |
| MNIST | 100 | unit | 1.0 | A-Only | 10.904 | 2.266 | 1933 | 1.000 | 0.000 |
| MNIST | 100 | unit | 1.0 | CR-Fixed | 11.645 | 5.298 | 4965 | 0.667 | 0.000 |
| MNIST | 100 | unit | 1.0 | CR-Power | 11.509 | 5.298 | 4965 | 1.000 | 0.000 |
| MNIST | 100 | unit | 1.0 | CR-Uses | 11.645 | 5.298 | 4965 | 0.667 | 0.000 |
| MNIST | 100 | unit | 1.0 | Independent | 11.666 | 5.268 | 4935 | 1.000 | 0.000 |
| MNIST | 100 | unit | 2.0 | A-Only | 11.509 | 2.266 | 1933 | 2.000 | 0.000 |
| MNIST | 100 | unit | 2.0 | CR-Fixed | 12.475 | 5.298 | 4965 | 1.333 | 0.000 |
| MNIST | 100 | unit | 2.0 | CR-Power | 12.336 | 5.298 | 4965 | 2.000 | 0.000 |
| MNIST | 100 | unit | 2.0 | CR-Uses | 12.475 | 5.298 | 4965 | 1.333 | 0.000 |
| MNIST | 100 | unit | 2.0 | Independent | 12.510 | 5.268 | 4935 | 2.000 | 0.000 |
| MNIST | 100 | unit | 4.0 | A-Only | 12.336 | 2.266 | 1933 | 4.000 | 0.000 |
| MNIST | 100 | unit | 4.0 | CR-Fixed | 13.643 | 5.298 | 4965 | 2.667 | 0.000 |
| MNIST | 100 | unit | 4.0 | CR-Power | 13.550 | 5.298 | 4965 | 4.000 | 0.000 |
| MNIST | 100 | unit | 4.0 | CR-Uses | 13.643 | 5.298 | 4965 | 2.667 | 0.000 |
| MNIST | 100 | unit | 4.0 | Independent | 13.742 | 5.268 | 4936 | 4.000 | 0.000 |
| MNIST | 100 | unit | 8.0 | A-Only | 13.550 | 2.266 | 1933 | 8.000 | 0.000 |
| MNIST | 100 | unit | 8.0 | CR-Fixed | 15.446 | 5.298 | 4965 | 5.333 | 0.000 |
| MNIST | 100 | unit | 8.0 | CR-Power | 15.353 | 5.298 | 4965 | 8.000 | 0.000 |
| MNIST | 100 | unit | 8.0 | CR-Uses | 15.446 | 5.298 | 4965 | 5.333 | 0.000 |
| MNIST | 100 | unit | 8.0 | Independent | 15.541 | 5.268 | 4936 | 8.000 | 0.000 |
| MNIST | 100 | unit | 16.0 | A-Only | 15.353 | 2.266 | 1933 | 16.000 | 0.000 |
| MNIST | 100 | unit | 16.0 | CR-Fixed | 18.062 | 5.298 | 4966 | 10.667 | 0.000 |
| MNIST | 100 | unit | 16.0 | CR-Power | 17.982 | 5.298 | 4965 | 16.000 | 0.000 |
| MNIST | 100 | unit | 16.0 | CR-Uses | 18.062 | 5.298 | 4966 | 10.667 | 0.000 |
| MNIST | 100 | unit | 16.0 | Independent | 18.161 | 5.268 | 4937 | 16.000 | 0.000 |
| MNIST | 1000 | csi5 | 1.0 | A-Only | 24.954 | 2.266 | 1933 | 1.231 | 0.500 |
| MNIST | 1000 | csi5 | 1.0 | CR-Fixed | 29.864 | 33.571 | 3.254e+04 | 0.954 | 0.000 |
| MNIST | 1000 | csi5 | 1.0 | CR-Power | 29.992 | 33.182 | 3.253e+04 | 1.231 | 0.500 |
| MNIST | 1000 | csi5 | 1.0 | CR-Uses | 29.992 | 33.182 | 3.253e+04 | 1.231 | 0.500 |
| MNIST | 1000 | csi5 | 1.0 | Independent | 29.784 | 35.288 | 3.252e+04 | 1.321 | 1.000 |
| MNIST | 1000 | csi5 | 2.0 | A-Only | 29.983 | 2.266 | 1933 | 2.462 | 0.500 |
| MNIST | 1000 | csi5 | 2.0 | CR-Fixed | 34.726 | 34.445 | 3.254e+04 | 1.908 | 0.000 |
| MNIST | 1000 | csi5 | 2.0 | CR-Power | 34.756 | 33.598 | 3.254e+04 | 2.462 | 0.500 |
| MNIST | 1000 | csi5 | 2.0 | CR-Uses | 34.756 | 33.598 | 3.254e+04 | 2.462 | 0.500 |
| MNIST | 1000 | csi5 | 2.0 | Independent | 34.615 | 37.917 | 3.254e+04 | 2.642 | 1.000 |
| MNIST | 1000 | csi5 | 4.0 | A-Only | 34.751 | 2.266 | 1933 | 4.925 | 0.500 |
| MNIST | 1000 | csi5 | 4.0 | CR-Fixed | 38.390 | 36.214 | 3.255e+04 | 3.816 | 0.000 |
| MNIST | 1000 | csi5 | 4.0 | CR-Power | 38.454 | 34.500 | 3.254e+04 | 4.925 | 0.500 |
| MNIST | 1000 | csi5 | 4.0 | CR-Uses | 38.454 | 34.500 | 3.254e+04 | 4.925 | 0.500 |
| MNIST | 1000 | csi5 | 4.0 | Independent | 38.331 | 43.190 | 3.257e+04 | 5.284 | 1.000 |
| MNIST | 1000 | csi5 | 8.0 | A-Only | 38.451 | 2.266 | 1933 | 9.850 | 0.500 |
| MNIST | 1000 | csi5 | 8.0 | CR-Fixed | 40.746 | 39.710 | 3.258e+04 | 7.632 | 0.000 |
| MNIST | 1000 | csi5 | 8.0 | CR-Power | 40.799 | 36.304 | 3.255e+04 | 9.850 | 0.500 |
| MNIST | 1000 | csi5 | 8.0 | CR-Uses | 40.799 | 36.304 | 3.255e+04 | 9.850 | 0.500 |
| MNIST | 1000 | csi5 | 8.0 | Independent | 40.719 | 53.679 | 3.264e+04 | 10.567 | 1.000 |
| MNIST | 1000 | csi5 | 16.0 | A-Only | 40.799 | 2.266 | 1933 | 19.699 | 0.500 |
| MNIST | 1000 | csi5 | 16.0 | CR-Fixed | 42.058 | 46.710 | 3.263e+04 | 15.263 | 0.000 |
| MNIST | 1000 | csi5 | 16.0 | CR-Power | 42.109 | 39.912 | 3.258e+04 | 19.699 | 0.500 |
| MNIST | 1000 | csi5 | 16.0 | CR-Uses | 42.109 | 39.912 | 3.258e+04 | 19.699 | 0.500 |
| MNIST | 1000 | csi5 | 16.0 | Independent | 42.019 | 74.672 | 3.279e+04 | 21.134 | 1.000 |
| MNIST | 1000 | rayleigh | 1.0 | A-Only | 24.951 | 2.266 | 1933 | 1.000 | 0.000 |
| MNIST | 1000 | rayleigh | 1.0 | CR-Fixed | 29.852 | 33.189 | 3.253e+04 | 0.667 | 0.000 |
| MNIST | 1000 | rayleigh | 1.0 | CR-Power | 29.975 | 32.995 | 3.253e+04 | 1.000 | 0.000 |
| MNIST | 1000 | rayleigh | 1.0 | CR-Uses | 29.975 | 32.995 | 3.253e+04 | 1.000 | 0.000 |
| MNIST | 1000 | rayleigh | 1.0 | Independent | 29.759 | 34.088 | 3.251e+04 | 1.000 | 0.000 |
| MNIST | 1000 | rayleigh | 2.0 | A-Only | 29.975 | 2.266 | 1933 | 2.000 | 0.000 |
| MNIST | 1000 | rayleigh | 2.0 | CR-Fixed | 34.714 | 33.682 | 3.254e+04 | 1.333 | 0.000 |
| MNIST | 1000 | rayleigh | 2.0 | CR-Power | 34.738 | 33.196 | 3.253e+04 | 2.000 | 0.000 |
| MNIST | 1000 | rayleigh | 2.0 | CR-Uses | 34.738 | 33.196 | 3.253e+04 | 2.000 | 0.000 |
| MNIST | 1000 | rayleigh | 2.0 | Independent | 34.600 | 35.538 | 3.253e+04 | 2.000 | 0.000 |
| MNIST | 1000 | rayleigh | 4.0 | A-Only | 34.738 | 2.266 | 1933 | 4.000 | 0.000 |
| MNIST | 1000 | rayleigh | 4.0 | CR-Fixed | 38.381 | 34.625 | 3.255e+04 | 2.667 | 0.000 |
| MNIST | 1000 | rayleigh | 4.0 | CR-Power | 38.447 | 33.696 | 3.254e+04 | 4.000 | 0.000 |
| MNIST | 1000 | rayleigh | 4.0 | CR-Uses | 38.447 | 33.696 | 3.254e+04 | 4.000 | 0.000 |
| MNIST | 1000 | rayleigh | 4.0 | Independent | 38.324 | 38.417 | 3.256e+04 | 4.000 | 0.000 |
| MNIST | 1000 | rayleigh | 8.0 | A-Only | 38.447 | 2.266 | 1933 | 8.000 | 0.000 |
| MNIST | 1000 | rayleigh | 8.0 | CR-Fixed | 40.740 | 36.540 | 3.257e+04 | 5.333 | 0.000 |
| MNIST | 1000 | rayleigh | 8.0 | CR-Power | 40.794 | 34.653 | 3.255e+04 | 8.000 | 0.000 |
| MNIST | 1000 | rayleigh | 8.0 | CR-Uses | 40.794 | 34.653 | 3.255e+04 | 8.000 | 0.000 |
| MNIST | 1000 | rayleigh | 8.0 | Independent | 40.716 | 44.147 | 3.261e+04 | 8.000 | 0.000 |
| MNIST | 1000 | rayleigh | 16.0 | A-Only | 40.794 | 2.266 | 1933 | 16.000 | 0.000 |
| MNIST | 1000 | rayleigh | 16.0 | CR-Fixed | 42.057 | 40.362 | 3.261e+04 | 10.667 | 0.000 |
| MNIST | 1000 | rayleigh | 16.0 | CR-Power | 42.108 | 36.623 | 3.257e+04 | 16.000 | 0.000 |
| MNIST | 1000 | rayleigh | 16.0 | CR-Uses | 42.108 | 36.623 | 3.257e+04 | 16.000 | 0.000 |
| MNIST | 1000 | rayleigh | 16.0 | Independent | 42.008 | 55.636 | 3.273e+04 | 16.000 | 0.000 |
| MNIST | 1000 | unit | 1.0 | A-Only | 24.951 | 2.266 | 1933 | 1.000 | 0.000 |
| MNIST | 1000 | unit | 1.0 | CR-Fixed | 29.852 | 32.863 | 3.253e+04 | 0.667 | 0.000 |
| MNIST | 1000 | unit | 1.0 | CR-Power | 29.975 | 32.863 | 3.253e+04 | 1.000 | 0.000 |
| MNIST | 1000 | unit | 1.0 | CR-Uses | 29.852 | 32.863 | 3.253e+04 | 0.667 | 0.000 |
| MNIST | 1000 | unit | 1.0 | Independent | 29.759 | 32.832 | 3.25e+04 | 1.000 | 0.000 |
| MNIST | 1000 | unit | 2.0 | A-Only | 29.975 | 2.266 | 1933 | 2.000 | 0.000 |
| MNIST | 1000 | unit | 2.0 | CR-Fixed | 34.714 | 32.863 | 3.253e+04 | 1.333 | 0.000 |
| MNIST | 1000 | unit | 2.0 | CR-Power | 34.738 | 32.863 | 3.253e+04 | 2.000 | 0.000 |
| MNIST | 1000 | unit | 2.0 | CR-Uses | 34.714 | 32.863 | 3.253e+04 | 1.333 | 0.000 |
| MNIST | 1000 | unit | 2.0 | Independent | 34.600 | 32.832 | 3.25e+04 | 2.000 | 0.000 |
| MNIST | 1000 | unit | 4.0 | A-Only | 34.738 | 2.266 | 1933 | 4.000 | 0.000 |
| MNIST | 1000 | unit | 4.0 | CR-Fixed | 38.381 | 32.863 | 3.253e+04 | 2.667 | 0.000 |
| MNIST | 1000 | unit | 4.0 | CR-Power | 38.447 | 32.863 | 3.253e+04 | 4.000 | 0.000 |
| MNIST | 1000 | unit | 4.0 | CR-Uses | 38.381 | 32.863 | 3.253e+04 | 2.667 | 0.000 |
| MNIST | 1000 | unit | 4.0 | Independent | 38.324 | 32.832 | 3.25e+04 | 4.000 | 0.000 |
| MNIST | 1000 | unit | 8.0 | A-Only | 38.447 | 2.266 | 1933 | 8.000 | 0.000 |
| MNIST | 1000 | unit | 8.0 | CR-Fixed | 40.740 | 32.863 | 3.253e+04 | 5.333 | 0.000 |
| MNIST | 1000 | unit | 8.0 | CR-Power | 40.794 | 32.863 | 3.253e+04 | 8.000 | 0.000 |
| MNIST | 1000 | unit | 8.0 | CR-Uses | 40.740 | 32.863 | 3.253e+04 | 5.333 | 0.000 |
| MNIST | 1000 | unit | 8.0 | Independent | 40.716 | 32.832 | 3.251e+04 | 8.000 | 0.000 |
| MNIST | 1000 | unit | 16.0 | A-Only | 40.794 | 2.266 | 1933 | 16.000 | 0.000 |
| MNIST | 1000 | unit | 16.0 | CR-Fixed | 42.057 | 32.863 | 3.254e+04 | 10.667 | 0.000 |
| MNIST | 1000 | unit | 16.0 | CR-Power | 42.108 | 32.863 | 3.253e+04 | 16.000 | 0.000 |
| MNIST | 1000 | unit | 16.0 | CR-Uses | 42.057 | 32.863 | 3.254e+04 | 10.667 | 0.000 |
| MNIST | 1000 | unit | 16.0 | Independent | 42.008 | 32.832 | 3.252e+04 | 16.000 | 0.000 |
