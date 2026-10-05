# SISA–AirComp 아키텍처 실험: 실행 안내

이 폴더는 "SISA 구조를 지키면서 언러닝에 유리한 AirComp 아키텍처"의 구성 요소가 실제로 효과가 있는지 하나씩 가려내는 실험 묶음입니다.
구성 요소는 채널 기준 shard 배정, 삭제해도 바뀌지 않는 배정 규칙(①), 채널 순서 참여 시점(②), 연속 삭제를 고려한 shard 크기(③), 삭제 중 대역 몰아주기(④), guard 대역(⑤)입니다.
실험마다 무엇을 바꾸고 무엇을 고정하며 어떤 기준으로 판정하는지는 [PLAN_KO.md](PLAN_KO.md) 에 실행 전에 적어 두었습니다.

## 1. 준비물

- NVIDIA GPU. 메모리 2GB 이상이면 기본 설정으로 돌아갑니다. GPU 가 없어도 CPU 로 돌지만 매우 느립니다.
- Python 3.9 이상
- 인터넷 연결: 첫 실행 때 FashionMNIST(약 30MB)를 `data/` 폴더에 자동으로 내려받습니다.

## 2. 설치 (처음 한 번)

```bash
git clone -b research/sisa-arch-experiments-20261006 https://github.com/sijoon-sung/aircomp_OTA_unlearning.git
cd aircomp_OTA_unlearning/sisa_arch_20261006
```

PyTorch 는 GPU(CUDA) 판으로 설치해야 합니다. https://pytorch.org/get-started/locally/ 에서 자기 환경에 맞는 명령을 고르면 됩니다. 예를 들어 CUDA 12.1 이면 다음과 같습니다.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy
```

설치가 됐는지는 아래 명령으로 확인합니다. 마지막 값이 `True` 이면 GPU 를 쓸 수 있는 상태입니다.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. 실행

**Windows:** `run_all.bat` 을 더블클릭합니다.
**Linux / macOS:** `bash run_all.sh`

처음에는 동작 확인부터 하는 것을 권장합니다. 2~5분이면 끝납니다. 동작 확인은 20라운드만 돌리므로 정확도 수치는 의미가 없고, 오류 없이 끝나는지만 봅니다.

```bash
python run_all.py --quick
```

동작 확인이 끝나면 전체 실행을 합니다.

```bash
python run_all.py
```

실험은 G0 → E1 → E2 → E3 → E4 → E5 순서로 자동 진행됩니다. G0 에서 정한 잡음 한계(eps)를 E1~E5 가 자동으로 이어받습니다.
한 실험이 실패해도 다음 실험은 계속 진행하고, 실패 원인은 `run_log.txt` 와 `status.json` 에 남습니다.

### 걸리는 시간

RTX 4060 노트북 GPU 에서 client 한 명의 로컬 학습(2 step) 한 번에 약 1.9ms 가 걸렸습니다. 이것을 기준으로 전체 실행은 **약 2~3시간**으로 추정합니다.
데스크톱 GPU 라면 더 빠릅니다. 진행 상황은 화면과 `run_log.txt` 에 라운드 단위로 찍힙니다.

### 중간에 끊겼을 때

같은 결과 폴더를 `--out` 으로 지정하고, 남은 실험만 `--only` 로 돌리면 됩니다. G0 결과(`eps_star.json`)가 그 폴더에 있으면 자동으로 이어받습니다.

```bash
python run_all.py --out runs/20261006_120000_MYPC --only e3,e4,e5
```

## 4. 결과 보는 법

결과는 `runs/<시각>_<컴퓨터이름>/` 에 쌓입니다.

| 파일 | 내용 |
|---|---|
| `SUMMARY_KO.md` | **여기부터 보면 됩니다.** 실험별 판정 요약과 모든 보고서를 한 파일에 모았습니다 |
| `g0_noise/REPORT_KO.md` 등 | 실험별 표와 판정 근거 |
| `*/results.json` | 원자료(조건별 수치) |
| `eps_star.json` | G0 가 정한 잡음 한계 |
| `environment.json` | 실행 환경(GPU, 버전, 인자) |
| `run_log.txt`, `status.json` | 진행 기록, 실험별 소요 시간·GPU 전력량·실패 원인 |

## 5. 결과를 돌려주는 법

결과 폴더는 수 MB 수준입니다. 아래처럼 같은 브랜치에 올리거나, 폴더를 압축해서 전달하면 됩니다.

```bash
git add runs/<결과폴더>
git commit -m "SISA-AirComp 아키텍처 실험 결과 (<컴퓨터이름>)"
git push
```

## 6. 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--quick` | 끔 | 작은 규모로 동작만 확인 |
| `--only` / `--skip` | 전체 | 예: `--only g0,e2`, `--skip e5` |
| `--out` | `runs/<시각>_<호스트>` | 결과 폴더 |
| `--device` | auto | `cuda`, `cuda:1`, `cpu` |
| `--chunk` | 64 | 한 번에 계산하는 client 수. GPU 메모리가 부족하면 32 |
| `--seeds` | 5 | GPU 학습 seed 수 |
| `--draws` | 1000 | CPU 해석의 채널 표본 수 (E1, E2, E4) |
| `--draws-e3` / `--seqs` | 300 / 20 | E3 의 채널 표본 수 / 삭제 순서 수 |
| `--rounds` | 160 | 학습 라운드 |
| `--eps` | G0 결과 | 잡음 한계를 직접 지정 |
| `--data` | `./data` | 데이터 폴더 |

## 7. 문제 해결

- **`CUDA out of memory`**: `--chunk 32` 로 다시 실행합니다. 끝난 실험은 `--only` 로 건너뛰면 됩니다.
- **`torch.cuda.is_available()` 가 False**: CPU 판 PyTorch 가 설치된 경우입니다. 2절의 GPU 판 설치 명령으로 다시 설치합니다.
- **데이터 다운로드 실패**: 인터넷이 막힌 컴퓨터라면 다른 컴퓨터에서 `data/FashionMNIST/raw/` 폴더(IDX 파일 4개)를 복사해 오면 됩니다.
- **화면의 한국어가 깨짐**: Windows 에서는 `run_all.bat` 으로 실행하면 UTF-8 로 맞춰집니다. 결과 파일은 항상 UTF-8 입니다.

## 8. 폴더 구성

```
sisa_arch_20261006/
  README_KO.md        이 문서
  PLAN_KO.md          실험별 가설·변수·통제·판정 기준 (실행 전 고정)
  run_all.py          원클릭 실행기
  run_all.bat / .sh   더블클릭 / 한 줄 실행
  requirements.txt
  sisa_arch/
    common.py         데이터·모델·채널·배정 규칙·참여 일정 (이전 실험과 같은 seed 에서 같은 분할·채널)
    costmodel.py      AirComp 라운드 비용·잡음 해석식
    trainer.py        배치(vmap) 로컬 학습, shard 별 AirComp 집계, 평가
    g0_noise.py, e1_stable.py, e2_entry.py, e3_lifetime.py, e4_bandwidth.py, e5_guard.py
```
