**재현 범위와 실행 안내 — 2026-09-27**

비용 계산은 이 저장소의 원자료만으로 실행된다. 초기 배포의 GPU 실험은 원본 코드와 검증 기록을 보관했으며, 데이터와 외부 코드 준비가 추가로 필요하다. 초기 배포는 비용 재계산만 수행했고, 이후 `research_20260927_channel_noise`에서 새 GPU 실험을 실행했다. 후속 실험은 vendor 없이 실행 가능하며 [전용 안내](research_20260927_channel_noise/README_KO.md)의 `--data` 옵션을 사용한다.

**바로 재실행할 수 있는 작업**

저장소 루트에서 Python 3.10 이상으로 실행한다. PyTorch나 GPU가 필요하지 않다.

```bash
python research_20260927_costs/calculate_costs.py
python verify_bundle.py
```

첫 명령은 기존 JSON 원장을 읽어 `results/costs.json`과 `COST_REPORT_KO.md`를 다시 만든다. 둘째 명령은 원본/배포 파일 해시, 주요 실험 코드·프로토콜 해시, 780개 평가 행, 비용 계산의 기록된 코드 해시와 문서 링크를 확인한다. 이미 저장된 GPU 결과를 검사하는 작업이며 GPU 실험 재현 자체를 대신하지 않는다.

비용 계산으로 생성하는 두 출력 파일에 한해서 운영체제의 LF/CRLF 줄바꿈 차이를 허용한다. 수치나 문장이 달라지는 경우는 허용하지 않으며, 보관된 실험 코드와 프로토콜은 바이트 단위 해시를 그대로 검사한다.

**원자료와 배포본의 차이**

`EXPORT_MANIFEST.json`에는 보관 파일별 원본 SHA-256, 배포 SHA-256, 파일 크기가 있다. Markdown의 로컬 절대 경로 링크만 GitHub 상대 경로로 바꿨다. 실험 Python, 결과 JSON/CSV와 동결된 주요 프로토콜은 원본 바이트를 유지한다. 기존 `validation/document_audit.json`은 원본 문서의 검증 기록이므로 링크가 바뀐 배포 Markdown의 해시와 다를 수 있다. 배포본 검증은 `EXPORT_MANIFEST.json`과 `BUNDLE_VERIFICATION.json`을 사용한다.

이전 문서의 “아직 GitHub에 올리지 않았다” 등의 문장은 당시 작업 상태다. 최신 배포 범위는 루트 README를 기준으로 한다. 각 실험 폴더의 `results/environment.json`과 provenance manifest도 당시 기록으로 보존한다.

**전체 GPU 실험을 다시 실행할 때 필요한 준비**

검증에 사용한 환경은 NVIDIA GeForce RTX 3080, PyTorch 2.11.0+cu128, CUDA 12.8이다. NumPy와 결과 그림용 Matplotlib도 사용했다. 다른 GPU/라이브러리 버전의 bitwise equality를 보장하지 않는다. 각 실험의 `PROTOCOL_KO.md`에 seed, 데이터 분할과 순차 실행 조건이 있다.

1. FashionMNIST의 원본 IDX 데이터를 준비한다. `research_20260926_methods123/common.py::DATA`와 `research_20260926/run_pilots.py::DATA`가 당시 Windows 절대 경로를 사용한다. 실행용 복사본에서 데이터 경로를 설정해야 한다.
2. [외부 코드 출처](THIRD_PARTY_NOTICES.md)의 정확한 commit/공식 supplement를 내려받는다. `vendor_adapters.py`는 `vendor/CuReNU_supplement/code/`와 `vendor/Orthogonal-Convolutional-Neural-Networks/`에서 선택한 함수·클래스를 AST로 읽는다. 보관된 `fetch_references.py`는 실행 시점의 최신 branch를 조회하므로 과거 결과 재현에는 manifest의 고정 commit URL을 사용한다. 이 안내에서는 다운로드 스크립트를 자동 실행하지 않는다.
3. `support/relation_protocol.py`를 import할 수 있도록 Python 경로에 `support/`를 추가한다. `benchmark.py` 안의 당시 `ota_ful` 절대 경로는 보관된 설정이다. 새 환경에서 이 경로에 다른 모듈이 있다면 명시적으로 수정한 실행용 복사본을 사용한다.
4. 모델 checkpoint는 배포에 포함하지 않았다. 원본 학습 스크립트로 해당 checkpoint를 먼저 생성해야 `validate.py`의 checkpoint replay를 다시 실행할 수 있다. 학습 순서는 각 파일의 CLI와 프로토콜을 확인한다. 버전 비교 → RTD FP64 보정 → 부분 투영 실험 → 최신 검증 순서로, GPU worker는 하나만 실행한다.
5. 경로·환경을 바꾼 재실행은 별도 결과 폴더에서 수행하고 새 코드/환경 해시를 기록한다. 과거 결과나 검증 manifest를 새로운 실행으로 덮어쓰지 않는다. 경로를 수정한 코드에 원본 코드 해시 검사가 그대로 통과할 것으로 기대하면 안 된다.

**포함하지 않은 항목**

데이터셋, 학습 checkpoint, 전체 third-party 소스 트리, 로컬 Python/CUDA 런타임, 로그 캐시와 자격증명은 제외했다. 결과 그림 PDF는 포함하지만 논문 원문 PDF는 배포하지 않는다. 비용표의 tensor byte는 배포 ZIP 크기나 peak GPU 메모리와 다른 지표다.
