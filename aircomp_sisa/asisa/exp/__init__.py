"""실험 모음. 각 모듈은 run(cfg, log) 하나를 내놓고, cfg['out']/<이름>/ 에 REPORT_KO.md 와 results.json 을 쓴다.

  noise        허용 집계 잡음 eps* (다른 실험의 eps 기준)
  resources    shard 수 K 에 따른 자원·삭제 비용, shard 크기에 따른 개인 노출
  assignment   shard 배정 규칙별 학습 비용과 삭제 비용
  interference 같은 자원에 동시에 보내는 shard 사이의 간섭과 near-far: 도착 크기를 어떻게 맞추면 흔적이 줄어드는가
  codes        시간 오차에 강한 코드(ZCZ)로 간섭을 구조적으로 0 으로 만들 때의 대가
  placement    같은 자원을 쓰는 shard 의 배정이 바꾸는 것들 (near-far, 흔적, 삭제 비용, 노출, 안정성)
  control      E1 직교 블록에서도 전력·스케줄링을 shard 밖 정보로 정하면 정확성이 깨지는가
  stability    E2 불안정한 배정의 재학습 범위, 해시 + 국소 병합
  differencing E3 재학습 때 삭제 전후 shard 합의 차분으로 개인 update 가 드러나는가, 새 초기값 재학습
  dropout      E4 학습 중 이탈이 shard 합의 노출과 잡음에 주는 영향, 최소 인원 규칙
  lifecycle    E2b 삭제·참여가 이어질 때 배정·삭제 처리 방법별 누적 비용과 노출
  subcarrier   R1~R3 FDMA, 채널 인식 OFDMA, TDMA+FDMA 혼합, 보호 대역
  retrain      B 학습 도중 삭제 → 같은 자원에서 재학습할 때 그 전송이 다른 shard 에 남기는 흔적
  earlystop    D 멈출 라운드를 전체 검증 정확도로 정하면 (전체 결정) 다른 shard 모델이 달라지는가
  sharding     A·E shard 수 K 에 따른 정확도 (제대로 배운 상태) 와 기기 쪽 추론·내려받기 비용
  fairness     C 삭제 재학습 비용을 사람별로 누가 내는가 (요청자는 떠나고 조원이 냄)
  (deletion.py 는 interference·codes·placement 가 함께 쓰는 학습 경로 구성과 측정)
"""
from types import SimpleNamespace
import math
from ..data import load_split
from ..channel import channels
from ..trainer import LocalTrainer, Evaluator
from ..model import init_vector

def seed_context(cfg, seed):
    """seed 하나의 데이터·채널·학습기·초기값·평가기."""
    data = load_split(cfg['raw'], seed, cfg['device'])
    base, h = channels(seed, cfg['T'])
    tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk'])
    return SimpleNamespace(seed=seed, data=data, base=base, h=h, tr=tr, w0=init_vector(seed, cfg['device']), ev=Evaluator(tr, data))

def db(sigma2):
    return f'{round(10 * math.log10(1 / sigma2))}dB'
