**Original / V1 비교 완료**

- [결과와 판단](RESULTS_KO.md)
- [전체 수치표](RESULT_TABLES_KO.md)
- [방법명·선행연구·코드 출처](METHODS_AND_PROVENANCE_KO.md)
- [실행 전 고정 protocol](PROTOCOL_KO.md)
- [③ 수치 정밀도 수정 기록](RTD_NUMERICAL_ADDENDUM_KO.md)
- [추가 진단 범위](DIAGNOSTIC_ADDENDUM_KO.md)
- [실행 코드](benchmark.py)
- [③ 수정 코드](rtd_fp64.py)
- [그림 PNG](comparison.png) · [PDF](comparison.pdf)

DS-Air는개선효과를확인했다. OG-Air V1은no-op수준이며,RTD-Air는Original의구조적성립을확인했지만통신V1의최종삭제오차개선은확인되지않았다. 모든결과를성공으로표현하지않는다. 최초③결과(`RTD_seed*.json`)와수정결과(`RTDfp64_seed*.json`)를모두보존했고,최종표는수정결과를사용한다.
