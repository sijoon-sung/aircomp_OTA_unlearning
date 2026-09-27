**GPU 순차 실험의 원자료 요약**

Channel noise 반복은 먼저 seed 내부에서 평균하고, 아래 mean±sd는 model/data seed 간 표본표준편차다. 단일 reference-variation 행은 seed271만이다. Ratio=forget JS / no-op forget JS; 낮을수록 paired reference에 가깝다.

**방법 1**

| 방식 | Reference | Forget JS 비율 | Test accuracy % | Forget accuracy % | MIA AUC | 삭제 통신 M real uses |
|---|---|---:|---:|---:|---:|---:|
| no_op | plain FL | 1.000 ± 0.000 | 75.53 ± 0.32 | 80.89 ± 1.42 | 0.517 ± 0.003 | — |
| target_free_retrain | plain FL | 0.000 ± 0.000 | 75.40 ± 0.11 | 79.75 ± 1.28 | 0.515 ± 0.003 | 113.854037 ± 0.000000 |
| independent_sampling_retrain | plain FL | 1.371 | 75.13 | 81.83 | 0.512 | 113.854037 |
| curenus_exact | plain FL | 7.654 ± 7.182 | 75.10 ± 0.54 | 80.61 ± 1.63 | 0.514 ± 0.001 | 53.819225 ± 0.000000 |
| curenus_ota20_r1 | plain FL | 7.662 ± 7.262 | 75.04 ± 0.53 | 80.67 ± 1.88 | 0.514 ± 0.001 | 53.819225 ± 0.000000 |
| curenus_ota20_r4 | plain FL | 7.661 ± 7.193 | 75.09 ± 0.53 | 80.64 ± 1.68 | 0.514 ± 0.001 | 68.835748 ± 0.000000 |
| curenus_ota30_r1 | plain FL | 7.647 ± 7.171 | 75.10 ± 0.53 | 80.64 ± 1.68 | 0.514 ± 0.001 | 37.615464 ± 0.000000 |
| retained_sgd70 | plain FL | 3.386 ± 1.652 | 75.95 ± 0.37 | 80.36 ± 2.24 | 0.515 ± 0.001 | 53.819225 ± 0.000000 |

**방법 2**

| 방식 | Reference | Forget JS 비율 | Test accuracy % | Forget accuracy % | MIA AUC | 삭제 통신 M real uses |
|---|---|---:|---:|---:|---:|---:|
| no_op | ortho FL | 1.000 ± 0.000 | 75.58 ± 0.32 | 81.06 ± 1.02 | 0.517 ± 0.003 | — |
| ortho_noiseless_prune | ortho FL | 24.625 ± 19.988 | 73.12 ± 0.75 | 77.06 ± 1.81 | 0.518 ± 0.004 | 0.000408 ± 0.000000 |
| target_free_ortho_retrain | ortho FL | 0.000 ± 0.000 | 75.53 ± 0.17 | 79.78 ± 0.70 | 0.515 ± 0.004 | 227.708074 ± 0.000000 |
| random_mask | ortho FL | 11.835 ± 8.277 | 73.37 ± 2.67 | 79.33 ± 2.90 | 0.518 ± 0.002 | — |
| no_ortho_noiseless_prune | plain FL | 22.972 ± 16.755 | 73.34 ± 0.34 | 77.47 ± 2.07 | 0.518 ± 0.003 | — |
| ota10_uniform | ortho FL | 13.310 ± 3.363 | 73.09 ± 2.64 | 79.67 ± 0.32 | 0.518 ± 0.003 | 0.000867 ± 0.000000 |
| ota10_boundary | ortho FL | 16.328 ± 1.458 | 72.85 ± 2.37 | 79.50 ± 0.20 | 0.518 ± 0.004 | 0.001184 ± 0.000000 |
| ota10_matched | ortho FL | 16.013 ± 3.561 | 73.14 ± 2.54 | 79.51 ± 0.24 | 0.518 ± 0.003 | 0.001164 ± 0.000000 |
| ota20_uniform | ortho FL | 17.052 ± 5.926 | 74.39 ± 0.60 | 79.83 ± 0.79 | 0.518 ± 0.003 | 0.000597 ± 0.000000 |
| ota20_boundary | ortho FL | 18.225 ± 6.384 | 74.37 ± 0.72 | 79.90 ± 1.04 | 0.518 ± 0.003 | 0.000806 ± 0.000000 |
| ota20_matched | ortho FL | 17.121 ± 5.817 | 74.29 ± 0.62 | 79.74 ± 0.88 | 0.518 ± 0.003 | 0.000786 ± 0.000000 |

**방법 3**

| 방식 | Reference | Forget JS 비율 | Test accuracy % | Forget accuracy % | MIA AUC | 삭제 통신 M real uses |
|---|---|---:|---:|---:|---:|---:|
| no_op | plain FL | 1.000 ± 0.000 | 75.53 ± 0.32 | 80.89 ± 1.42 | 0.517 ± 0.003 | — |
| fedquit_ce_exact | plain FL | 100.130 ± 26.061 | 61.17 ± 1.40 | 35.33 ± 2.05 | 0.515 ± 0.005 | 76.592014 ± 0.000000 |
| fedquit_ce_ota20_r4 | plain FL | 100.101 ± 26.133 | 61.22 ± 1.32 | 35.33 ± 2.05 | 0.515 ± 0.005 | 98.044189 ± 0.000000 |
| fresh_student_old_global_teacher | plain FL | 1.360 ± 0.567 | 73.95 ± 0.06 | 79.67 ± 0.74 | 0.516 ± 0.003 | — |
| no_op_ensemble_student | 독립 teacher KD | 1.000 ± 0.000 | 72.87 ± 0.24 | 78.47 ± 1.42 | 0.516 ± 0.004 | 0.731786 ± 0.000000 |
| exclude_fresh_noiseless | 독립 teacher KD | 0.000 ± 0.000 | 72.49 ± 0.69 | 77.64 ± 1.51 | 0.515 ± 0.004 | 0.731744 ± 0.000000 |
| exclude_fresh_ota20_r1 | 독립 teacher KD | 0.344 ± 0.331 | 72.63 ± 0.64 | 77.31 ± 1.64 | 0.515 ± 0.004 | 0.731744 ± 0.000000 |
| exclude_fresh_ota20_r4 | 독립 teacher KD | 0.208 ± 0.170 | 72.54 ± 0.61 | 77.42 ± 1.52 | 0.516 ± 0.004 | 0.799244 ± 0.000000 |

**방법2: 통신오류만의 비교**

| 방식 | 오선택 channel 수 | Score MSE | 무잡음 pruning 대비 forget JS |
|---|---:|---:|---:|
| ota10_uniform | 4.083 | 0.091136 | 0.024985 |
| ota10_boundary | 4.083 | 0.156637 | 0.025161 |
| ota10_matched | 3.833 | 0.038373 | 0.021485 |
| ota20_uniform | 3.500 | 0.009114 | 0.017587 |
| ota20_boundary | 3.417 | 0.015588 | 0.018457 |
| ota20_matched | 3.333 | 0.004861 | 0.017698 |

독립 teacher의 no-op ensemble student 행은 삭제비용이 아니라 최초 준비비용이며, cost_audit.json에서 보완한 model DL·초기 seed 비용을 사용했다. Local teacher/student 계산 및 public image 사전배포 비용은 별도 장부에 있다. Plain/ortho source 준비비용도 각 training JSON에 기록되어 있다.

