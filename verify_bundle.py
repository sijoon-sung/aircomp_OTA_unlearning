"""Stdlib-only archive integrity checks; this does not rerun GPU training."""
from pathlib import Path
import hashlib
import json
import math
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def finite(obj):
    if isinstance(obj, float):
        assert math.isfinite(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            finite(value)
    elif isinstance(obj, list):
        for value in obj:
            finite(value)


def verify():
    manifest = load('EXPORT_MANIFEST.json')
    generated = {'research_20260927_costs/results/costs.json',
                 'research_20260927_costs/COST_REPORT_KO.md'}
    newline_only = []
    for item in manifest['files']:
        payload = (ROOT / item['path']).read_bytes()
        if sha(item['path']) != item['packaged_sha256']:
            # Python write_text uses platform line endings. A rerun may produce
            # LF instead of the archived Windows CRLF, with identical contents.
            assert item['path'] in generated, item['path']
            windows = payload.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
            assert hashlib.sha256(windows).hexdigest() == item['packaged_sha256'], item['path']
            assert len(windows) == item['bytes'], item['path']
            newline_only.append(item['path'])
        else:
            assert len(payload) == item['bytes'], item['path']
        if item['path'].endswith('.py'):
            assert item['source_sha256'] == item['packaged_sha256'], item['path']
    old = 'research_20260926_versioned'
    part = 'research_20260927_orthogonality'
    for folder, code in [(old, 'benchmark.py'), (part, 'run_experiment.py')]:
        env = load(folder + '/results/environment.json')
        assert env['code_sha256'] == sha(folder + '/' + code)
        assert env['protocol_sha256'] == sha(folder + '/PROTOCOL_KO.md')
    counts = {}
    for family in ['DS', 'OG', 'RTDfp64']:
        records = [json.loads(p.read_text(encoding='utf-8'))
                   for p in sorted((ROOT / old / 'results').glob(f'{family}_seed*.json'))]
        assert len(records) == 3 and all(r['complete'] for r in records)
        finite(records)
        counts[family] = sum(len(r['rows']) for r in records)
        if family == 'RTDfp64':
            for r in records:
                assert r['code_sha256'] == sha(old + '/rtd_fp64.py')
                assert r['numerical_addendum_sha256'] == sha(old + '/RTD_NUMERICAL_ADDENDUM_KO.md')
    records = [load(f'{part}/results/seed{s}_complete.json') for s in [381, 382, 383]]
    assert all(r['complete'] for r in records)
    finite(records)
    counts['partial'] = sum(len(r['rows']) for r in records)
    assert counts == {'DS': 441, 'OG': 54, 'RTDfp64': 87, 'partial': 198}
    costroot = 'research_20260927_costs'
    costs = load(costroot + '/results/costs.json')
    finite(costs)
    assert costs['code_sha256'] == sha(costroot + '/calculate_costs.py')
    assert costs['assumptions_sha256'] == sha(costroot + '/ASSUMPTIONS_KO.md')
    assert costs['verification']['passed'] and len(costs['records']) == 13
    checked = 0
    for p in ROOT.rglob('*.md'):
        if '.git' in p.parts:
            continue
        for link in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            link = link.strip('<>')
            if link.startswith(('https://', 'http://', '#', 'mailto:')):
                continue
            assert not re.match(r'^[A-Za-z]:[/\\]', link), (p, link)
            target = (p.parent / unquote(link.split('#')[0])).resolve()
            assert target.is_relative_to(ROOT) and target.is_file(), (p, link)
            checked += 1
    result = dict(passed=True, archived_files=manifest['archived_file_count'],
                  archived_bytes=manifest['archived_bytes'], raw_rows=counts,
                  scope='Initial archive integrity and current Markdown links; new GPU experiment has its own results/verification.json',
                  cost_variants=13, markdown_local_links_checked=checked,
                  archived_code_and_frozen_protocol_hashes_preserved=True,
                  generated_outputs_with_only_platform_newline_changes=newline_only,
                  gpu_training_rerun_by_this_verifier=False)
    (ROOT / 'BUNDLE_VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
