from common import *
import json

assert json.loads((RESULTS/'math_validation.json').read_text())['passed']
assert not (RESULTS/'calibration.json').exists(), 'Preserve calibration; do not overwrite.'
data=load_data(270)
model,cost,elapsed=train_fl(data,270,100)
val=accuracy(model,data['x'][data['val']],data['y'][data['val']])
rounds=100
if val<.75:
    model,cost2,elapsed2=train_fl(data,270,150,start=model,start_round=100)
    elapsed+=elapsed2;rounds=150
    val=accuracy(model,data['x'][data['val']],data['y'][data['val']])
else:cost2=None
record={'seed':270,'chosen_rounds':rounds,'validation_accuracy':val,'quality_gate_passed':val>=.75,
        'wall_seconds':elapsed,'cost_first100':cost,'cost_extension':cost2,'test_used':False}
save_model(RESULTS/'calibration_source.pt',model)
write(RESULTS/'calibration.json',record);log(**record)
