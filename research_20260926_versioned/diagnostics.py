from benchmark import *


@torch.no_grad()
def rotation_check():
    x,y,xt,yt=pilot.load_data();proj=torch.randn((784,64),device=DEV,generator=rng(20260922))/math.sqrt(784)
    z=torch.cat([F.relu((x-.5)@proj),torch.ones(len(x),1,device=DEV)],1).double()
    zt=torch.cat([F.relu((xt-.5)@proj),torch.ones(len(xt),1,device=DEV)],1).double()
    yy=F.one_hot(y,10).double();eye=torch.eye(65,device=DEV,dtype=z.dtype);rows=[]
    u=torch.linalg.qr(torch.randn(65,48,device=DEV,dtype=z.dtype,generator=rng(90317)),mode='complete').Q
    u=fp32(u)
    for seed in [202609263,202609264,202609265]:
        pools=pilot.partition(y,seed)
        hs=torch.stack([z[p].T@z[p]/len(p)+.01*eye for p in pools]);cs=torch.stack([z[p].T@yy[p]/len(p) for p in pools])
        h,c=hs[1:].mean(0),cs[1:].mean(0);w0=fp32(torch.linalg.solve(hs.mean(0),cs.mean(0)));wr=torch.linalg.solve(h,c)
        b=hs[1:]@w0-cs[1:];rh=u.T@hs[1:]@u;rb=u.T@b
        err=float((u@torch.linalg.solve(rh.mean(0),rb.mean(0))-torch.linalg.solve(h,b.mean(0))).abs().max())
        assert err<1e-10
        for noise in range(16):
            ns=seed*100+noise*20;hc=Ledger();hh=head_received_h(rh,ns,hc)
            for version in ['Original','V1']:
                led=Ledger();led.v=hc.v.copy();led.dl(2*650*32+32)
                delta,diag=head_correct(hh,rb,ns+1,version,led)
                w=fp32(w0-u@delta)
                rows.append(dict(seed=seed,noise=noise,version=version,
                    metrics=head_metrics(w,w0,wr,zt,yt,z[pools[0]]),cost=led.report(),noiseless_rotation_error=err))
    save(OUT/'rotation_diagnostic.json',rows)


def cosine(a,b):return float(F.cosine_similarity(a[None],b[None]))


def geometry_check():
    rows=[]
    for seed in SEEDS:
        data=cm.load_data(seed);model=cm.load_model(CP/f'seed{seed}_ortho_all.pt')
        streams={i:rng(seed*8100+i) for i in range(10)}
        vs=[model_grad(model,*cm.batch(data,i,streams[i],64)) for i in range(1,10)]
        vs.append(model_grad(model,*cm.batch(data,0,streams[0],64),target=True))
        unit,norms=rel.normalized_rows(torch.stack(vs))
        exactw=rel.relation_coefficients(unit);exact=exactw@unit
        sketches=rel.quantize_rows(rel.srht(unit,2048,seed*100),16);sw=rel.relation_coefficients(sketches)
        ideal=sw@unit
        noisy,power=transmit(list(unit),sw.tolist(),list(range(1,10))+[0],seed*200)
        exd=rel.rescale_fedosd(exact,norms[-1]);idd=rel.rescale_fedosd(ideal,norms[-1]);nd=rel.rescale_fedosd(noisy,norms[-1])
        rows.append(dict(seed=seed,sketch_ideal_cosine=cosine(exd,idd),sketch_noisy_cosine=cosine(exd,nd),
            exact_residual_norm=float(exact.norm()),sketch_residual_norm=float(ideal.norm()),
            noise_rms_norm=math.sqrt(len(noisy)*power['variance']),
            ideal_relative_direction_error=float((exd-idd).norm()/exd.norm()),
            noisy_relative_direction_error=float((exd-nd).norm()/exd.norm()),power=power))
    save(OUT/'legacy_geometry.json',rows)


if __name__=='__main__':
    rotation_check();geometry_check();log(stage='diagnostics',status='complete')
