**①②③ 총비용 산정 범위 — 2026-09-27**

기존 결과의 수치·성능을 바꾸지 않고 비용을 보완한다. 비용 최적화 설정으로 모델을 다시 실험한 것처럼 표현하지 않는다. 단위는 real channel-use equivalent다. 기본 goodput R=.5log2(1+10^(SNR/10)) bit/real use, CP=1.125이며 디지털 구간은 error-free goodput 모델이다.

**구분할 비용**

- 삭제 시점의 통신: 원자료 ledger를 성분별로 재계산한다. 이미 포함된 곡률 수집·기저 방송·sketch·pilot·모델 배포는 다시 더하지 않는다.
- 명시적으로 빠진 제어: 삭제요청128bit UL, 참여/모델버전 등을 담는 공통지시128bit DL을 별도 가정으로 추가한다. 실제 표준 패킷 실측 크기가 아니다. Ethernet/IP/TLS 등의 protocol overhead까지 측정한 것으로 해석하지 않는다.
- 공통 수신 scale a의 방송: DS는 각 packet에32bit(H+보정block), OG는32bit를 추가한다. Sketch는 기존 계수DL의32bit에 포함돼 있어 중복하지 않는다. RTD는 기존 query-ID 이후32bit를 a로 명시한다. 온도·기저·가중치는 고정 공개 규칙이며 변하면 추가 metadata가 필요하다.
- 초기 source 준비: 기존 원자료 비용을 사용한다. FedAvg/OG source ledger에 빠진 마지막 완성 모델 DL을 추가하고, 각 학습 OTA packet의 receive-scale32bit 방송도 추가한다. DS source의 통계수집2packet scale64bit를 추가한다. RTD의 기존 source preparation에 빠진 scale metadata10×32bit와 초기 공개seed32bit를 추가한다. Source 학습 지시/시작 control128bit도 별도로 넣는다.
- 중복 방지: 준비 완료 모델을 client가 이미 cache한 시나리오의 삭제에서는 source 모델 재방송을 차감한다. Cold deletion은 기존처럼 source를 재방송하는 경우다. Lifecycle은 ‘준비 완료+cache된 상태의 삭제1회’로 계산한다.
- RTD query: 이미지2000×28×28×8=12,544,000bit의 최초 배포비를 uncached lifecycle에 포함한다. 기존 비용의 query ID32bit/sample과 구분한다. Public dataset의 적법한 설치·동일한query mapping은 전제이며 새로운 private 데이터를 업로드하지 않는다.
- Client 계산: 기존 gradient/forward sample 수, teacher/student 준비 및 기존 simulator walltime을 기록한다. DS에는 encoder/충분통계/보정의 MAC 산술량을 별도 계산한다. CNN forward MAC과 학습 sample 수는 제시하되 backward FLOPs를 임의 상수로 정밀 측정값처럼 쓰지 않는다.
- 저장: 모델, gradient, sketch, local statistics, teacher prediction cache의 tensor byte수를 산정한다. allocator·activation·optimizer·Python 메모리를 포함한 peak GPU/RAM 수치가 아니다.
- Reference 재학습 및 비교군 sweep은 연구 검증 비용으로 분리하며 실제 삭제 서비스 비용에 합산하지 않는다.

**환산·민감도**

가정한 처리율 1M/10M real uses/s로 통신시간을 환산한다. MHz 대역폭이나 실제 RF latency와 동일시하지 않는다. 전력 P watt가 주어지지 않았으므로 실측 J·원화비용은 산정하지 않는다. 단일 링크의 동작전력1W를 가정하면 해당 링크 시간초가 에너지J와 같다는 예시만 제공하며 전체 단말/서버 에너지가 아니다.

Packet error 0/1/5% 시 디지털 구간의 기대재송신비1/(1-p)만 별도 계산한다. Analog 재시도는 새 잡음과 결과를 바꾸므로 임의로 본 실험 비용에 넣지 않는다. 정확한CSI,기존pilot8real/client/packet,coherence가 반복들을 포괄한다는 기존 가정을 유지한다. ACK·재동기화·추가CSI비용은 구현 데이터가 없어 ‘미측정’이다.

계산 결과는 전체비용 순위가 아니라 각 family 내 Original/V1 비교와 초기비용/삭제비용의 분해에 사용한다. Family별 모델·학습 규칙·삭제품질이 다르다.
