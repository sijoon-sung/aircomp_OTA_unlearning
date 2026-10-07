# AirComp-SISA: AirComp 연합학습에서 SISA 구조로 정확 언러닝

AirComp(over-the-air computation) 연합학습에서 서버는 shard 의 합만 받는다. 정확 언러닝은 SISA 구조(shard 를 나눠 따로 학습하고, 삭제 요청이 오면 그 client 의 shard 만 처음부터 재학습)로 한다. 이 폴더는 그 구조에서 생기는 비용과 문제를 라운드 단위로 실제 송수신을 돌려 잰다.

이전 폴더 `sisa_arch_20261006/`, `aircomp_sim_20261007/` 의 코드를 하나의 패키지로 새로 정리한 것이다. 이전 폴더는 그대로 두었다. 데이터 분할·초기값·채널·배정 규칙·로컬 학습·송수신 식은 이전 구현과 같은 값을 낸다(부동소수점 반올림 수준, 1e-8). 잡음 난수 키는 정리하면서 바뀌었으므로, 이전 실행과 수치는 통계적으로만 같다.

## 1. 실행

```bash
pip install -r requirements.txt     # GPU 판 torch 는 https://pytorch.org 의 안내대로
python check.py                     # 물리 계층 식 검증 (CPU, 수십 초)
python run.py --quick               # 동작 확인 (수 분, 수치는 의미 없음)
python run.py                       # 전체 (FashionMNIST 자동 다운로드, seed 5, 160 라운드)
python run.py --only interference   # 일부만 (쉼표로 여러 개)
```

Windows 는 `run.bat` 더블클릭 (= check.py 후 run.py, 인자를 그대로 넘김).

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--only` | 전부 | `noise,resources,assignment,interference,codes,placement,control,stability,differencing,dropout,lifecycle,subcarrier,retrain,earlystop,sharding,fairness` 중 일부 |
| `--seeds` / `--rounds` | 5 / 160 | |
| `--eps` | 10 | 허용 집계 오차. noise 실험의 측정값 |
| `--chunk` | 64 | vmap 묶음 크기. GPU 메모리가 부족하면 32 |
| `--dataset` | fashionmnist | `mnist` 도 가능 |

결과: `runs/<시각>_<호스트>/<실험>/REPORT_KO.md`, `results.json`, 그리고 보고서를 모은 `SUMMARY_KO.md`.

## 2. 실험

| 실험 | 질문 | 이전 이름 |
|---|---|---|
| `noise` | 정확도를 해치지 않는 집계 잡음은 어디까지인가 (eps*) | G0 |
| `resources` | shard 수 K 에 따라 학습 자원, 삭제 비용(심볼·에너지·계산·시간)이 어떻게 변하는가. shard 가 작을수록 서버가 개인 정보를 얼마나 읽어 내는가 | P0 (b)(c) |
| `assignment` | shard 배정 규칙에 따라 학습 비용과 삭제 비용이 교환되는가 | P1 |
| `interference` | 같은 자원에 동시에 보내는 shard 사이 간섭이 삭제 대상의 흔적을 다른 shard 에 남기는가. 전력 정렬이 그 간섭을 약한 shard 로 몰아주는가 (near-far) | P0-A + P2 (도착 크기를 가장 약한 shard 에 맞추는 경우 추가) |
| `codes` | 시간 오차에 강한 코드(ZCZ: walsh 칩 사이에 0 칩을 넣고 창으로 모아 받음)로 shard 간 간섭을 구조적으로 0 으로 만들 수 있는가. 대가(칩 수, 잡음)는 | 새로 추가 |
| `control` (E1) | shard 를 직교 블록(시간 슬롯)으로 나눠도, 전력을 전체 최약 노드에 맞추거나 전체 기준으로 스케줄링하면 (기존 관행) 정확한 언러닝이 깨지는가. shard 안 결정은 비용이 드는가 | 새로 추가 |
| `stability` (E2) | 불안정한 배정에서 정확성에 필요한 재학습 shard 수, u 의 shard 만 재학습한 SISA 와 전체 재학습의 차이, 해시 + 국소 병합 (제안) | 새로 추가 |
| `differencing` (E3) | 재학습 때 서버가 삭제 전후 shard 합을 모두 보면 차분으로 u (와 첫 라운드 이탈자) 의 update 가 드러나는가. 새 초기값 재학습 (제안) 이 막는가 | 새로 추가 |
| `dropout` (E4) | 학습 중 이탈로 shard 합의 실제 인원이 줄 때 노출이 얼마나 커지는가. 라운드 최소 인원 규칙 (제안) 의 대가 | 새로 추가 |
| `lifecycle` (E2b) | 삭제·신규 참여가 이어질 때 배정·삭제 처리 방법별 누적 계산·통신·지연과 노출 (해시, 병합, 정지, 묶음, slicing, 부하 상한 해시, K 조정). slicing 정확성·정지의 정확도 손해 GPU 확인 | 새로 추가 |
| `subcarrier` (R1~R3) | TDMA 대신 FDMA·채널 인식 OFDMA·TDMA+FDMA 혼합·보호 대역으로 shard 를 배치할 때 라운드 시간, 잡음·반복, 주파수 오차로 인한 shard 간 섞임과 흔적 | 새로 추가 |
| `placement` | 같은 자원을 쓰는 shard 를 어떻게 배정해야 하는가. 배정이 near-far, 흔적, 삭제 비용, 최소 인원(노출), 삭제 안정성을 어떻게 바꾸는가 | 새로 추가 |
| `retrain` (B) | 학습 도중 삭제가 와서 같은 자원에서 재학습하면, 그 전송이 학습 중인 다른 shard 에 흔적을 남기는가 (직교 블록이면 0, 코드 분할이면 >0). 지금까지는 재학습을 따로 돌려 이 흔적을 안 쟀다 | 2026-10-08 추가 |
| `earlystop` (D) | 멈출 라운드를 앙상블 검증 정확도 (전체 결정) 로 정하면 u 와 무관한 shard 의 최종 모델이 달라지는가. shard 자기 검증으로 정하면 0 | 2026-10-08 추가 |
| `sharding` (A·E) | 로컬 step 을 늘려 제대로 배운 상태에서 K 에 따른 정확도 (조 모델 하나 vs 앙상블), 잡음이 정확도를 올리는 현상이 사라지는가, 기기 쪽 추론 내려받기 (K×32×D bit)·연산 (K 배) | 2026-10-08 추가 |
| `fairness` (C) | 삭제 재학습 비용을 사람별로 세면 누가 내는가. 요청자는 떠나고 조원이 낸다. 삭제가 한 묶음에 몰릴 때의 쏠림 (지니, 최대/평균) | 2026-10-08 추가 |

각 실험 파일 맨 위 설명에 설정·측정·판정 기준을 적었다. 보고서의 "읽는 법" 절에 가설이 맞을 때 보여야 하는 모습을 적었다.

## 3. 모델

- client 20명, FashionMNIST 를 class 마다 Dirichlet(0.5) 비율로 나눔, 작은 CNN (파라미터 38,282개). 로컬 SGD 2 step, update 는 L2 ≤ 1 로 자름.
- 채널 진폭: 장기 10^(U[-20,0]/20) × 라운드별 U[0.85, 1.15]. 위상은 송신 전에 보상.
- 한 shard 의 한 라운드 (`asisa/radio.py`):
  member i 가 s_i = x_i / (|h_i| β) 를 보내면 서버에 x_i/β 로 도착하고, 서버는 받은 합에 α = β/n 을 곱해 (1/n)Σx_i + α z 를 얻는다. 집계 오차 = D α² σ² / L.
  - 최대 전력: β = C / (min|h| √(PD)) (가장 약한 member 가 전력 상한에 닿음)
  - 공통 수신 크기: β = n_nom √(eps L / (D σ²)) (공개 상수). 못 맞추는 shard 만 최대 전력. n_nom 을 비우면 그 shard 의 인원 (= 집계 오차를 eps 에 맞춤)
  - 최대 전력의 집계 오차가 eps 를 넘으면 R = ⌈오차/eps⌉ 번 반복
- 다중화: `orth` (shard 마다 직교 블록, 간섭 없음) / `code` (같은 자원에 코드로 동시 전송, 칩 타이밍 오차가 있으면 shard 사이 간섭) / `ideal` (평균에 정해진 크기의 잡음만, noise 실험용)
- 코드: `walsh`, `pn`, `zcz` (walsh 칩 사이에 0 칩 gap 개, 수신기는 0..gap 칩 늦은 신호를 모두 모으는 창으로 역확산. 시간 오차 gap 칩 미만이면 간섭 0, 대가는 칩 수와 잡음 (gap+1) 배)
- 전력 정렬 `weakest` (코드 분할 전용): 같은 자원을 쓰는 shard 가 모두 가장 약한 shard 의 최대 크기로 도착
- OFDM (`mux='ofdm'`, `OfdmSystem`): 부반송파 64개 × 시간 슬롯 G 개. 주파수 선택적 채널 (4-tap), client 주파수 오차 (이웃 부반송파로 새는 간섭), 배치 방식 (연속 블록, 블록 위치 채널 인식, 섞어 배치, 부반송파 채널 인식), 보호 부반송파, 블록 순서. 기기 총전력 P 를 자기 부반송파에 나누므로 부반송파를 적게 쓰는 FDMA 는 부반송파당 전력이 크다.
- 시간 slicing (`Shard.entry`, `Shard.t0`): client 마다 참여 라운드를 두고, 체크포인트에서 이어 재학습할 수 있다.
- 이탈·스케줄링 (`asisa/fl.py`): client 는 라운드마다 확률 drop 으로 빠질 수 있고 (client·라운드로 정해져 모든 학습 경로에서 같음), `sched=('shard'|'global', q)` 면 채널 하위 q 는 송신하지 않는다. `align='global'` (직교 블록) 은 같은 system 전체의 최약 송신자 기준 전력 정렬. `min_present` 는 그 라운드 인원이 모자라면 보내지 않는 규칙. `noise_salt` 는 배치는 같고 잡음만 독립인 전송을 만든다.
- 흔적의 크기: 파라미터 차이 ||W(u 있음) − W(u 없이 처음부터)|| / ||W(u 없이)|| 와 그 난수 기준선. 예측 불일치는 작은 흔적에도 몇 % 로 포화되어 참고용 (P2 실측)
- 짝 비교: minibatch 와 잡음은 (seed, 라운드, client, salt) 로 정해진다. "u 있음" 과 "u 없이 처음부터" 는 같은 배치·잡음을 쓰므로 차이는 u 에서만 나온다. salt 를 바꾼 학습이 "학습 난수 변동" 기준선이다.

## 4. 폴더

```
aircomp_sisa/
  run.py          실행기
  run.bat         Windows 실행 (check.py -> run.py)
  check.py        물리 계층 식 검증
  asisa/
    config.py     고정값 (client 수, 학습률, eps 등)
    util.py       난수 키, 입출력, 보고서 표와 짝 비교 통계
    data.py       데이터 내려받기와 client 분할
    model.py      CNN, 초기값
    trainer.py    vmap 로컬 학습, 예측, 앙상블 평가
    channel.py    채널, shard 배정 규칙
    radio.py      AirComp 물리 계층 (전력 정렬, 반복, 직교 블록, 코드 분할, 장부)
    fl.py         라운드 루프 (로컬 학습 -> system 별 전송 -> 모델 갱신)
    exp/          noise, resources, assignment, interference, codes, placement, control, stability, differencing, dropout, lifecycle, subcarrier (+ deletion.py 공통 측정)
```
