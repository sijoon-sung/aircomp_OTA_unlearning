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
| `--only` | 전부 | `noise,resources,assignment,interference` 중 일부 |
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
| `interference` | 같은 자원에 동시에 보내는 shard 사이 간섭이 삭제 대상의 흔적을 다른 shard 에 남기는가. 전력 정렬이 그 간섭을 약한 shard 로 몰아주는가 (near-far) | P0-A + P2 |

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
    exp/          noise, resources, assignment, interference
```
