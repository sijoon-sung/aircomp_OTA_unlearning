"""Validate source-adapter and AirComp interfaces before the main GPU queue."""
from common import *
from vendor_adapters import official_hvp_func
from run_experiments import hvp_local,statistic_vectors,prune,pruning_indices,stat_observation,simplex
data=load_data(270);m=load_model(RESULTS/'calibration_source.pt')
g=rng(765);batches=[batch(data,i,g,8) for i in [1,2,3]]
v=torch.randn_like(flat(m));v=v/v.norm()
local=sum(hvp_local(m,*b,v) for b in batches)/3
x=torch.cat([b[0] for b in batches]);y=torch.cat([b[1] for b in batches])
cb=official_hvp_func(m,F.cross_entropy,[(x,y)],device=DEVICE)
official=flatten(cb(split_params(m,v))[1]).detach()
err=float((local-official).norm()/official.norm().clamp_min(1e-8))
assert err<1e-4,err
vectors,diag=statistic_vectors(m,data);truth=sum(vectors)
noisy,cost=stat_observation(vectors,999,20,'boundary')
matched,mcost=stat_observation(vectors,999,20,'matched',cost.total())
assert mcost.total()<=cost.total()+1e-5
pm=prune(m,truth);ids=pruning_indices(truth)
assert ids.numel()==6 and not torch.equal(flat(m),flat(pm))
p=torch.randn(20,10,device=DEVICE)
q=simplex(p);assert bool((q>=0).all()) and torch.allclose(q.sum(1),torch.ones(20,device=DEVICE),atol=1e-6)
write(RESULTS/'integration_checks.json',{'passed':True,'official_vs_local_dnn_hvp_relative_error':err,
    'boundary_total_real_uses':cost.total(),'uniform_matched_total_real_uses':mcost.total(),
    'pruned_channels':ids.cpu().tolist(),'simplex_max_row_sum_error':float((q.sum(1)-1).abs().max())})
log(status='integration_checks_passed',hvp_relative_error=err)
