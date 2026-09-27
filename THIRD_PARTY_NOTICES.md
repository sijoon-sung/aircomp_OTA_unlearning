**외부 코드와 출처**

이 저장소의 방법 이름은 실험 구분을 위한 이름이다. 선행 알고리즘과 단순 결합 자체의 신규성을 주장하지 않는다. 전체 외부 저장소를 재배포하지 않으며, 실행에 필요한 출처와 당시 버전을 아래에 기록한다.

| 출처 | 당시 버전 | 사용 범위 |
|---|---|---|
|[CuReNU/CuReNUS 공식 ICLR 2026 supplement](https://proceedings.iclr.cc/paper_files/paper/2026/file/1fb4f62b8df5059a41df270a65e9b462-Supplemental-Conference.zip)|archive SHA-256은 [supplement manifest](research_20260926_methods123/supplement_manifest.json)|ConvNet2와 초기 cubic/HVP 실험 함수. 원본 전체는 포함하지 않음|
|[Orthogonal Convolutional Neural Networks](https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks/tree/aa7f56901c661a124e0cfe72eb2c9dc98045ce94)|`aa7f56901c661a124e0cfe72eb2c9dc98045ce94`|`orth_dist`; 사전 직교화 비교. [MIT notice](third_party_notices/Orthogonal_CNN_LICENSE)|
|[FedQUIT](https://github.com/alessiomora/FedQUIT/tree/e447fa0c75c6a44b1325be5fd561da98c91f4cc2)|`e447fa0c75c6a44b1325be5fd561da98c91f4cc2`|logit-min 규칙의 명시적 PyTorch 포트. 전체 TensorFlow 학습 pipeline 재현이 아님|
|[FedOSD](https://github.com/zibinpan/FedOSD/tree/41cc10635c6f2396795d0d9c6239ea181b7c9ee6)|`41cc10635c6f2396795d0d9c6239ea181b7c9ee6`|UCE 및 projection 정의를 참고한 sketch/weighted AirComp 구현. [MIT notice](third_party_notices/FedOSD_LICENSE)|
|[Sophia](https://github.com/Liuhong99/Sophia/tree/a7e157229b71d58cf995d32854f1be15c265b350)|`a7e157229b71d58cf995d32854f1be15c265b350`|초기 조사만 수행. 해당 optimizer 성능 실험으로 제시하지 않음|

세부 적용 범위와 최초 조회 기록은 [CODE_PROVENANCE_KO.md](research_20260926_methods123/CODE_PROVENANCE_KO.md), [source_manifest.json](research_20260926_methods123/source_manifest.json)에 보존했다. 외부 코드의 이용 조건은 해당 저작자의 조건을 따른다. 이 배포에서 연구 자료 전체에 새로운 포괄 라이선스를 부여하지 않았다.
