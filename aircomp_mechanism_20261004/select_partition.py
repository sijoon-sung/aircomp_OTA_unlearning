"""Retrospective finite-candidate deletion-cost selector; no test data used.

This profiles every source and deletion candidate, so its setup cost is high.
It is an executable design reference, not a validated cheap surrogate optimizer.
"""
from pathlib import Path
import json,argparse

def select(records,seed,target='q65',noise='nominal20dB'):
    rows=[r for r in records if r['seed']==seed and r['noise']==noise]
    evaluations=[]
    for r in rows:
        ds=r['deletions'];successful=[d for d in ds if d['stops'][target] is not None]
        feasible=r['source_success'][target] and len(successful)==len(ds)
        evaluations.append(dict(method=r['method'],source_complete=r['source_success'][target],
            deletion_completed=len(successful),deletion_total=len(ds),feasible=feasible,
            mean_delete_ul=sum(d['costs'][target]['ul_reals'] for d in ds)/len(ds) if feasible else None,
            worst_delete_ul=max(d['costs'][target]['ul_reals'] for d in ds) if feasible else None,
            source_ul=r['source_cost'][target]['ul_reals'],groups=r['groups'],
            profile_consumed_ul=r['source_cost'][target]['ul_reals']+sum(d['costs'][target]['ul_reals'] for d in ds)))
    eligible=[e for e in evaluations if e['feasible']]
    winner=min(eligible,key=lambda e:(e['mean_delete_ul'],e['worst_delete_ul'],e['source_ul'],e['method'])) if eligible else None
    return dict(seed=seed,target=target,noise=noise,status='selected' if winner else 'no_feasible_candidate',
        winner=winner,candidates=evaluations,total_profile_consumed_ul=sum(e['profile_consumed_ul'] for e in evaluations),
        uses_test_metrics=False,scope='retrospective selection on measured candidate profiles; online transfer not validated')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--target',choices=['q60','q65'],default='q65')
    args=parser.parse_args();records=json.loads((Path(__file__).parent/'results.json').read_text(encoding='utf-8'))
    print(json.dumps(select(records,args.seed,args.target),ensure_ascii=False,indent=2))
