**가져온 코드와 실제 사용 범위**

2026-09-26. 다운로드 commit·archive SHA256은 `source_manifest.json`, 공식 학회 부록은 `supplement_manifest.json`에 기록했다. Git clone은 Windows TLS credential 오류로 실패했지만, 공개 GitHub API의 commit을 고정한 codeload archive 다운로드로 동일 source를 확보했다. 인증 우회나 외부 서비스 변경은 하지 않았다.

| 출처 | 고정 버전 | 실제 사용 |
|---|---|---|
| [CuReNU/CuReNUS 공식 ICLR Supplemental](https://proceedings.iclr.cc/paper_files/paper/2026/file/1fb4f62b8df5059a41df270a65e9b462-Supplemental-Conference.zip) | archive SHA256: supplement manifest | ConvNet2 class, cubic subsolver, HVP/reference vector helpers |
| [Orthogonal CNN 공식 GitHub](https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks) | aa7f56901c661a124e0cfe72eb2c9dc98045ce94 | `imagenet/utils.py::orth_dist`의 제곱 |
| [FedQUIT 공식 GitHub](https://github.com/alessiomora/FedQUIT) | e447fa0c75c6a44b1325be5fd561da98c91f4cc2 | TensorFlow virtual-teacher logit-min 변환을 읽고 PyTorch로 이식, KL gradient 검사 |
| [Sophia 공식 GitHub](https://github.com/Liuhong99/Sophia) | a7e157229b71d58cf995d32854f1be15c265b350 | source 검토·보존만 수행; Sophia 실험을 실행했다고 하지 않음 |

GitHub repository 검색에서 FedOrtho, CuReNU, CureNewton 이름의 결과는0건이었다. 검색결과0이 공개 코드의 부재를 증명하는 것은 아니다. CuReNU는 학회 supplemental로 해결했고, FedOrtho는 공식 code를 확인하지 못했으므로 **논문 수식 기반 구현**으로 명시했다.

`vendor_adapters.py`는 검토한 function/class의 AST만 불러온다. 전체 upstream import tree와 setup script를 실행하지 않는다. CuReNU의 cache clear와 CPU 변환 helper만 가벼운 동등 함수로 연결했다. 실제 cubic solver의 Cauchy 초기화, M/2 계수, perturbation, inner LR 0.1배 감소는 원본대로다.

주요 변경은 다음과 같다.

- CuReNU: 중앙 minibatch gradient/HVP 대신9 retained client의 weighted aggregate를 callback으로 제공한다. 모델은 저자 CNN 구조를 사용하되 kernels8/hidden32로 축소했다. 공식 FMNIST config의 M·outer/inner steps·LR·perturbation을 사용하며 class deletion 대신 moderate non-IID의 clean-client deletion으로 바꿨다.
- FedOrtho: 공개 orthogonal norm 연산과 논문의 두 단계 정규화·activation max·rank soft-pruning을 결합했다. 첫 conv의 rank 조건을 만족하도록8 channels를 사용한다. 같은 fraction의 bias도 감쇠한다. 통계 집계의 count 가중치는 clean-client 문제에 맞게 명시적으로 정의했다.
- FedQUIT: 원형은 target device에서 모델을 수정한다. 우리의 OTA adaptation은 target KD와 retained CE gradient를 같은 현재 모델에서 합산한다. TensorFlow 프로젝트의 full benchmark 재현이 아니다. 온전히 독립된 teacher+fresh student는 별도의 구조 변경이다.

출처마다 원래 datasets, 모델 크기, 학습 횟수, 삭제 단위가 달라 이번 숫자를 논문 표와 직접 대조하지 않는다. 결과가 부정적이어도 원 논문을 반박하는 재현 실패라고 하지 않으며, **우리 조건으로 옮긴 축소 구현의 결과**로 해석한다.

외부 source의 README·license는 vendor 안에 보존했다. 공개 archive를 읽고 로컬 연구에 사용한 것이며, 수정 코드를 외부에 배포하거나 GitHub에 push하지 않았다.
