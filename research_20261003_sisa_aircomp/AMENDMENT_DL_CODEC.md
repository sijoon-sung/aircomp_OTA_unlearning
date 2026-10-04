# 추가 1회: 정확한 replay의 무손실 downlink 압축

2026-10-03. run_v1이 실행 중인 상태에서 기록. screening과 첫 confirmation에서 grouping 개선이 일관되지 않고 DL32-bit 모델 비용이 크다는 것을 관측했다. 원래 조건·선택·성공 기준은 변경하지 않는다. 추가 codec 값은 아직 측정하지 않았다. 무제한 탐색 없이 아래 9개 기존 confirmation 조건의 삭제 replay만 한 번 더 수행한다.

## 알고리즘과 비교

각 client는 직전 수신 global shard model의 FP32 byte buffer 하나를 보관한다. local SGD 상태와 별개다. 서버가 다음 global model을 보내기 전에 아래 packet 중 길이가 가장 짧은 것을 선택한다.

- raw: full FP32 bytes.
- full_zlib: full raw bytes를 zlib level6로 압축; raw보다 길면 raw fallback. 강한 stateless compression baseline.
- xor_zlib: 현재 FP32 bit pattern과 직전 수신 모델 byte의 XOR를 zlib level6로 압축. full_zlib/raw 중 더 짧은 방식으로 fallback 가능. **float subtraction/quantization 아님**, 디코딩 후 모든 비트가 같아야 한다.

첫 모델은 이전 buffer가 없으므로 full/raw만 사용한다. 삭제 branch 시작 때 buffer reset해 삭제 전 모델을 초기 base로 재사용하지 않는다. 모든 packet에 128-bit codec/version/length/CRC header를 동일하게 센다. 압축 알고리즘 자체는 알려진 lossless delta coding이며 새 발명이라고 주장하지 않는다. 각 retained client에 D*4 byte 추가 캐시, 현재 모델 기반 압축 CPU 시간과 decoder 시간, DL 실제 byte 절감, 전체 RE 변화를 보고한다. 이는 전체 client 학습 이력을 수집하는 방식이 아니다.

## 실행 범위

run_v1 완료 뒤 동일한 하나의 GPU worker 정책을 적용한다. confirmation seeds10611/10612/10613 × K1 random_norm / 선택K random_norm / 선택K channel_norm =9개 삭제 replay. source/full reference를 다시 학습하지 않고 기존 초기 checkpoint와 동일 데이터·CSI·난수로 삭제 경로를 재계산한다. 저장된 reference 모델과 최종 비트 동일성을 확인한다. codec은 수신 모델 값을 바꾸지 않아 local 계산과 UL 값은 동일해야 한다.

각 라운드의 raw/full_zlib/xor_zlib 모두 실제 encode/decode하고 raw bytes와 정확히 비교한다. 평가 대상 모델이 실제 전달됐다고 가정하는 baseband digital packet 실험이며 실제 network latency나 RF Joule을 측정하지 않는다. packet 손실·cache eviction 시 full refresh 필요하지만 이번 loss-free 비교에서는 그 빈도를 추정하지 않는다.

## 추가안 사전 기준

- 매 round decoded bytes==raw, 최종모델==기존 full reference, 기존 replay ledger의 UL/local calls 동일.
- 주 비교: random_norm의 xor_zlib 대 같은 random_norm full_zlib, 3 seeds 평균 DL bytes>=10% 절감이면 후속 후보. 전체 RE 절감은 실제 UL·DL·control 합계로 별도 보고. 기준 미달이면 그대로 보고.
- stateful vs stateless encode/decode CPU 시간 및 추가 모델 cache 비용을 보고. 전송 속도와 실제 장비 CPU 특성에 따라 latency 이득이 달라지므로 packet byte 감소를 end-to-end latency 감소로 등치하지 않는다.
- 이번은 앞선 관측에 따른 추가 탐색으로, 새로운 held-out method confirmation이나 원래 grouping 성공으로 합산하지 않는다. 초기 학습 DL codec 비용·누적 삭제는 실행하지 않는다.
