"""보고서 작성 유틸: 마크다운 표, 수치 형식, 판정 문장."""
import math
import numpy as np

def fmt(v, nd=3):
    if v is None:
        return '-'
    if isinstance(v, bool):
        return '예' if v else '아니오'
    if isinstance(v, (int, np.integer)):
        return f'{int(v):,}'
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return '-'
        if v != 0 and (abs(v) >= 1e6 or abs(v) < 1e-3):
            return f'{v:.3e}'
        return f'{v:,.{nd}f}'
    return str(v)

def pct(v, nd=2):
    return '-' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{100 * v:+.{nd}f}%'

def pp(v, nd=2):
    return '-' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{100 * v:+.{nd}f}pp'

def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)

def verdict_line(name, ok, detail):
    mark = '지지' if ok is True else ('미지지' if ok is False else '판단 보류')
    return f'- **{name}: {mark}** — {detail}'

def mean_ci(x):
    """평균과 정규근사 95% 구간(표본 수가 작으면 참고용)."""
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0:
        return float('nan'), float('nan'), float('nan')
    m = float(x.mean())
    if len(x) < 2:
        return m, m, m
    h = 1.96 * float(x.std(ddof=1)) / math.sqrt(len(x))
    return m, m - h, m + h
