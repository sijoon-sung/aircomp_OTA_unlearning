"""실험 모음. 각 모듈은 run(cfg, log) 하나를 내놓고, cfg['out']/<이름>/ 에 REPORT_KO.md 와 results.json 을 쓴다.

  noise        허용 집계 잡음 eps* (다른 실험의 eps 기준)
  resources    shard 수 K 에 따른 자원·삭제 비용, shard 크기에 따른 개인 노출
  assignment   shard 배정 규칙별 학습 비용과 삭제 비용
  interference 같은 자원에 동시에 보내는 shard 사이의 간섭과 near-far: 도착 크기를 어떻게 맞추면 흔적이 줄어드는가
  codes        시간 오차에 강한 코드(ZCZ)로 간섭을 구조적으로 0 으로 만들 때의 대가
  placement    같은 자원을 쓰는 shard 의 배정이 바꾸는 것들 (near-far, 흔적, 삭제 비용, 노출, 안정성)
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
