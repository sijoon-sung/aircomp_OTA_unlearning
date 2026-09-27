from pathlib import Path
import json,statistics,math
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
rate=.5*math.log2(101)
records=[]
for seed in [271,272,273]:
    one=json.loads((OUT/f'method1_seed{seed}.json').read_text())
    two=json.loads((OUT/f'method2_seed{seed}.json').read_text())
    three=json.loads((OUT/f'method3_seed{seed}.json').read_text())
    d=one['training']['parameters']
    original=next(r for r in three['rows'] if r['method']=='no_op_ensemble_student')
    preparation=dict(original['cost'])
    preparation['dl_bits']+=d*32+32 # original student deployment and public random initialization seed
    preparation['student_train_samples']=600*64
    preparation['local_teacher_gradient_samples']=sum(t['gradient_samples'] for t in three['teacher_preparation'])
    preparation['total_real_uses']=1.125*(preparation['ul_payload_real']+preparation['pilot_real']+(preparation['dl_bits']+preparation['metadata_bits'])/rate)
    setup_bits=2000*28*28*8
    records.append({'seed':seed,'parameters':d,'plain_source_preparation_real_uses':one['training']['cost']['total_real_uses'],
        'orthogonal_source_preparation_real_uses':two['training']['cost']['total_real_uses'],
        'independent_teacher_preparation_corrected':preparation,
        'public_image_provisioning_bits_if_not_cached':setup_bits,
        'public_image_provisioning_extra_real_uses_if_not_cached':1.125*setup_bits/rate,
        'privacy_transcript_note':'Original and target-free aggregates on the same public queries can reveal the removed teacher predictions by differencing. They are partial output information, not full gradient updates; no DP claim.'})
(OUT/'cost_audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records[0],ensure_ascii=False,indent=2))
summary=json.loads((OUT/'summary.json').read_text())
for r in summary:
    if r['stage']==2 and 'total_real_uses' in r['metrics']:
        print(r['method'],r['metrics']['total_real_uses']['mean'])
