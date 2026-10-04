"""Eight post-hoc diagnostic cases, separate from preregistered main outcomes."""
import json, os, subprocess, time, traceback
import snr_privacy as S
import experiment as E

def main():
    out=S.ROOT/'snr_clip_diagnostic_v1'; out.mkdir(exist_ok=False)
    assert (S.ROOT/'snr_privacy_v1/complete.json').exists(), 'Wait for main worker completion'
    inv=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True)
    foreign=[s for s in inv.splitlines()[1:] if s.strip() and 'ChatGPT.exe' not in s and not s.strip().startswith(str(os.getpid())+',')]
    if foreign: E.dump(out/'blocked.json',dict(processes=foreign)); raise RuntimeError('GPU occupied')
    E.dump(out/'environment.json',dict(inventory=inv,driver_sha=E.sha(__file__),engine_sha=E.sha(S.ROOT/'snr_privacy.py'),amendment_sha=E.sha(S.ROOT/'SNR_CLIP_DIAGNOSTIC_AMENDMENT.md')))
    jobs=[(1.,snr,method) for snr in [0,20] for method in ['random','channel','joint']]+[(c,120,'random') for c in [.1,1.]]
    E.dump(out/'jobs.json',jobs)
    rows=[]; start=time.perf_counter(); error=None
    with E.Power() as meter:
        try:
            for C,snr,method in jobs:
                S.C=C
                folder=out/f'C{C}'; folder.mkdir(exist_ok=True)
                result=S.run_case(folder,10801,4,method,snr,1,None)
                for r in result: r['C']=C; rows.append(r)
            E.dump(out/'results.json',rows)
        except BaseException:
            error=traceback.format_exc(); E.dump(out/'failure.json',dict(error=error,completed=len(rows)))
        finally: S.torch.cuda.synchronize()
    ledgers=list(out.glob('C*/*/*ledger.json'))
    E.dump(out/'cost_total.json',dict(seconds=time.perf_counter()-start,**meter.report(),source_cases=len(rows),
        local_calls=sum(json.loads(p.read_text())['cost']['local_calls'] for p in ledgers),error=error))
    if error: raise RuntimeError(error)
    E.dump(out/'complete.json',dict(cases=len(rows),all_exact=all(r['replay_exact'] for r in rows)))
    E.log(event='diagnostic_complete',cases=len(rows),seconds=time.perf_counter()-start)

if __name__=='__main__': main()
