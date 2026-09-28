"""Post-run check: FP32 CSI feedback must not change recorded scheduling.

Clients retain local high-precision CSI for channel inversion. The server's
band/wait decisions see a 32-bit gain per client. The frozen simulator made
these decisions from the unrounded values; this checks observational equality,
including threshold crossings, for every recorded session. It does not rerun
or tune the unlearning algorithm.
"""
import json
from pathlib import Path
import run_experiment as experiment

if __name__=='__main__':
    compared=changed=0
    for ds_index,ds in enumerate(['FashionMNIST','MNIST']):
        for seed in experiment.SEEDS:
            records=json.loads((experiment.OUT/f'{ds}_seed{seed}.json').read_text())['rows']
            index={(r['condition'],r['draw'],r['method']):r for r in records}
            for config in experiment.CONDITIONS:
                for draw in range(experiment.DRAWS):
                    ns=seed*100000+ds_index*10000+draw*100
                    _,he=experiment.bank(ns,config['corr'],config['csi'],config['weak'])
                    rounded=he.float().double()
                    for method in experiment.METHODS:
                        a=experiment.policy(method,he)
                        b=experiment.policy(method,rounded)
                        row=index[(config['name'],draw,method)]
                        assert a==(row['selected_band'],row['attempts'],row['status']=='ok')
                        compared+=1
                        changed+=a!=b
    assert changed==0,'FP32 CSI changes a scheduling decision; a versioned rerun is required.'
    result=dict(passed=True,decisions_compared=compared,decisions_changed_by_FP32_feedback=changed,
                interpretation='Exact decision equivalence on recorded sessions, not an all-input theorem',
                audit_code_sha256=experiment.base.sha(__file__),**experiment.hashes())
    experiment.base.write(experiment.OUT/'feedback_verification.json',result)
    print(json.dumps(result))
