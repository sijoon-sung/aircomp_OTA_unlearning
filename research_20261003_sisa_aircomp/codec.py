"""One preregistered follow-up: actual lossless FP32 downlink packets on replay."""
import argparse, zlib
from experiment import *

class CodecEngine(Engine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.previous={}
        self.codec_events=[]

    def physical_receive(self,w,ids,c,t,noise_branch):
        raw=w.detach().cpu().contiguous().numpy().astype('<f4',copy=False).tobytes()
        before=time.perf_counter()
        full=zlib.compress(raw,level=6)
        full_encode=time.perf_counter()-before
        full_kind='zlib' if len(full)<len(raw) else 'raw'
        full_packet=full if full_kind=='zlib' else raw
        before=time.perf_counter()
        decoded=zlib.decompress(full_packet) if full_kind=='zlib' else full_packet
        full_decode=time.perf_counter()-before
        assert decoded==raw
        previous=self.previous.get(c)
        before=time.perf_counter()
        if previous is not None:
            xor=np.bitwise_xor(np.frombuffer(raw,dtype=np.uint8),np.frombuffer(previous,dtype=np.uint8)).tobytes()
            delta=zlib.compress(xor,level=6)
        else:
            delta=None
        xor_extra_encode=time.perf_counter()-before
        delta_selected=delta is not None and len(delta)<len(full_packet)
        chosen=delta if delta_selected else full_packet
        before=time.perf_counter()
        if delta_selected:
            rebuilt=np.bitwise_xor(np.frombuffer(zlib.decompress(chosen),dtype=np.uint8),np.frombuffer(previous,dtype=np.uint8)).tobytes()
        else:
            rebuilt=zlib.decompress(chosen) if full_kind=='zlib' else chosen
        xor_decode=time.perf_counter()-before
        assert rebuilt==raw and zlib.crc32(rebuilt)==zlib.crc32(raw)
        self.previous[c]=rebuilt
        self.codec_events.append(dict(t=t,c=c,clients=len(ids),raw_bytes=len(raw)+16,full_bytes=len(full_packet)+16,
             xor_bytes=len(chosen)+16,kind='xor_zlib' if delta_selected else full_kind,
             full_encode_seconds=full_encode,full_decode_seconds=full_decode,
             xor_encode_seconds=full_encode+xor_extra_encode,xor_decode_seconds=xor_decode,
             decoded_bitwise_equal=True,base_buffer_bytes=len(raw)))
        return super().physical_receive(w,ids,c,t,noise_branch)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='codec_v1');args=ap.parse_args()
    run=ROOT/'run_v1'
    assert (run/'complete.json').exists()
    out=ROOT/args.out;out.mkdir(exist_ok=False)
    inventory=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True)
    foreign=[s for s in inventory.splitlines()[1:] if s.strip() and 'ChatGPT.exe' not in s and not s.strip().startswith(str(os.getpid())+',')]
    if foreign:raise RuntimeError(f'Other GPU worker: {foreign}')
    dump(out/'environment.json',dict(pid=os.getpid(),inventory=inventory,code_sha256=sha(__file__),
         amendment_sha256=sha(ROOT/'AMENDMENT_DL_CODEC.md'),experiment_sha256=sha(ROOT/'experiment.py')))
    selected=[r for r in json.loads((run/'results.json').read_text(encoding='utf-8')) if r['phase']=='confirm']
    began=time.perf_counter();rows=[];error=None
    with Power() as meter:
        try:
            for row in selected:
                folder=run/row['case']
                config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
                data=load_data('FashionMNIST',.5,row['seed'])
                engine=CodecEngine(data,row['seed'],row['K'],row['method'],row['rounds'])
                source=torch.load(folder/'source_models.pt',map_location='cpu',weights_only=True)
                checkpoints=torch.load(folder/'checkpoints.pt',map_location='cpu',weights_only=True)
                start=[v.to('cuda') for v in source]
                start[engine.affected]=checkpoints[f'{engine.affected}_0'].to('cuda')
                learned,trace=engine.fit(start,deleted=True,only=engine.affected,label='lossless_codec')
                ref=torch.load(folder/'reference_models.pt',map_location='cpu',weights_only=True)
                exact=all(torch.equal(a.cpu(),b) for a,b in zip(learned,ref))
                assert exact
                for field in ['local_calls','ul_reals','control_bits','pilot_reals']:
                    assert trace['cost'][field]==row['delete_cost'][field]
                events=engine.codec_events
                sizes={name:sum(e[f'{name}_bytes'] for e in events) for name in ['raw','full','xor']}
                sums={f'{name}_{kind}_seconds':sum(e[f'{name}_{kind}_seconds'] for e in events)
                      for name in ['full','xor'] for kind in ['encode','decode']}
                base_re=row['delete_cost']['total_re']-row['delete_cost']['dl_bits']/2
                packet= dict(case=row['case'],seed=row['seed'],K=row['K'],method=row['method'],
                    test_accuracy=row['test_accuracy'],model_exact=exact,all_packets_exact=all(e['decoded_bitwise_equal'] for e in events),
                    sizes=sizes,total_re={name:base_re+size*8/2 for name,size in sizes.items()},
                    cpu=sums,per_client_cache_bytes=engine.D*4,retained_clients=len(engine.groups[engine.affected])-1,
                    delta_packets=sum(e['kind']=='xor_zlib' for e in events),rounds=len(events),events=events,
                    original_delete_cost=row['delete_cost'],rerun_cost=trace['cost'])
                dump(out/f"{row['case']}.json",packet);rows.append(packet)
                log(event='codec_case_complete',case=row['case'],exact=exact,dl_saving_vs_full=1-sizes['xor']/sizes['full'],
                    total_saving_vs_full=1-packet['total_re']['xor']/packet['total_re']['full'])
                del engine,data,learned,start;torch.cuda.empty_cache()
        except BaseException:
            error=traceback.format_exc();dump(out/'failure.json',dict(error=error))
            dump(out/'cost_interrupted.json',dict(seconds=time.perf_counter()-began,**meter.report(),error=error,completed_cases=len(rows)))
            raise
    dump(out/'cost_total.json',dict(seconds=time.perf_counter()-began,**meter.report(),error=error,completed_cases=len(rows),
         local_calls=sum(r['rerun_cost']['local_calls'] for r in rows)))
    dump(out/'results.json',rows)
    dump(out/'complete.json',dict(cases=len(rows),all_exact=all(r['model_exact'] and r['all_packets_exact'] for r in rows)))
    log(event='codec_worker_complete',cases=len(rows),seconds=time.perf_counter()-began)

if __name__=='__main__':main()
