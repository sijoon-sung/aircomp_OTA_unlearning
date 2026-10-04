# AirComp shard / SNR / conditional privacy experiment

2026-10-03, before execution. Existing outputs remain untouched. Finite exploratory study, not a novelty claim or convergence certification.

## Matrix

- FashionMNIST train12000/dev2000/test10000, 20 clients, Dirichlet .5, existing CNN38282 parameters, local2 SGD steps x64, lr .05, clip L2 C=.1, 160 rounds, checkpoints40/80/120/160. Each shard independently trained, predictions averaged.
- Screen seed10801: K2/4/5 x nominal SNR0/10/20dB x random/channel/joint =27 configurations. K4 is fixed in advance, not selected from the screen.
- Confirmation seeds10802/10803: K4 x same SNR x three methods =18 configurations.
- Repeat control: confirmation seeds, K4/SNR10, three methods, R4 =6 configurations; primary R1. Per-symbol power cap1. Repetition consumes four times analog UL symbols and energy; it is not free.
- Privacy-limited control: screen seed, K4/SNR10/R1, three methods x conditional total epsilon cap8/100 at delta1e-5 =6 configurations. Total57 source training configurations. Each source is followed by independently executed full SISA reference and affected-shard replay for deletion0. Screen K4/SNR10/R1/natural also evaluates deletions7,14, giving63 deletion evaluations with cached source models. No further grid expansion.
- Channel amplitudes10^(Uniform[-20,0]/20), independent of client data; per-round fadingUniform[.85,1.15], known phases compensated perfectly. Nominal SNR=Pmax/sigma2 (0/10/20dB); actual aggregate SNR also logged. This is simulated SISO, perfect CSI, no dropout. Shards use orthogonal time/frequency resources; no MIMO spatial multiplexing or RF hardware claim.

## Grouping

- random: fixed seeded equal-size permutation; channel: sort slow gain and split into equal sizes.
- joint: pair-swap local search over equal-size groups, minimizing .5*normalized radio objective + .5*normalized mean JS divergence of shard client-average histograms from all-client average. Radio objective is .5*mean source 1/(m^2 min(h)^2) + .5*mean over all20 possible single-client deletions 1/((m-1)^2 min(retained h)^2). Both denominators are the random partition's scores, with floor1e-12. Start random and channel, at most30 improving best-swap passes; deterministic, dev/test not used.
- This is an explicit AirComp-MSE/data-coverage heuristic and not claimed new. It does not directly minimize privacy or exploit measured data–channel correlation.
- All methods receive the same class histogram/count side information for fair disclosure accounting. Histograms10 float32 + count32 per client are charged once. CSI and per-round common scaling communicated; current update norms are NOT sent. Channel-only methods ignore histogram values for assignment.

## Radio and privacy accounting

For m active clients, update norm <=C, dimension D, per-symbol power capP=1:
beta_phys=C/(m*sqrt(D*P)*min|h|), x_i=u_i/(m*h_i*beta), received mean=mean(u_i)+N(0,tau2 I), tau2=beta^2*sigma2/R.
The same update is repeated R times. Server may see all R observations; the mean is sufficient under independent Gaussian noise, so accounting uses variance/R. No automatic repetition to meet fixed MSE.

Adjacency: replace all private features of one fixed client while public client identity, counts, label histogram, channels and routing remain fixed. Thus this is **conditional client-feature privacy**, NOT protection of the disclosed histogram or client membership. Server is honest-but-curious and sees aggregate transcripts, not individual waveforms/updates. Noise is trusted receiver noise; malicious receiver/collusion/extra antennas not covered. Evaluation logs/models and true updates used by the simulator are offline evaluator data, not additional protocol releases. Experiment PRNG seeds are audit artifacts; a deployment would require private unpredictable noise.

Per-step sensitivity<=2C/m, rho_t=(2C/m)^2/(2*tau2), zCDP adaptive sequential composition; epsilon=rho+2sqrt(rho log(1/delta)). Independent noise is used for source and affected-shard replay. Disjoint shards compose in parallel, so max client rho, not sum over all clients. Retained clients in the deleted shard pay source+replay; deleted client pays source only, unaffected clients source only. Full reference is evaluator-only. Each deletion evaluation is a separate source+one-request scenario, not a sequence of three published requests. Histogram disclosure lies outside this conditional guarantee.

At fixed full-power alignment, sensitivity and noise both scale as1/m: shard size alone does not guarantee stronger DP. Gains also depend on bottleneck channels. Natural DP epsilon may be extremely large; report it without calling it a useful guarantee.

Privacy caps8/100: rho_budget=(sqrt(log(1/delta)+epsilon)-sqrt(log(1/delta)))^2. Set per-step rho<=rho_budget/(2*160) by beta>=sqrt((2C/m)^2*R/(2*rho_step*sigma2)), reducing transmit power. This limits source+one replay worst-client epsilon to cap under the stated adjacency. No fake privacy benefit from merely omitting metadata from the analysis.

Sources: Gaussian zCDP/composition https://arxiv.org/abs/1605.02065 ; channel-noise/power privacy mechanism https://arxiv.org/abs/2006.05459 .

## Metrics and criteria

- Source/deleted dev and test accuracy, macro-F1 and worst-class recall at40/80/120/160. Per-shard class divergence, actual update clipping rate, aggregate signal/noise ratio, MSE, power, normalized RF signal energy.
- UL analog symbols, DL bits, pilot/control/metadata and total equivalent real resources (digital2bits/RE), local calls, initial and deletion costs separately. Cross-K fixed rounds have different total radio budget; do not label them equal-total-budget. Separately compare latest checkpoint under the K2/R1/160-round source+deletion budget for each seed.
- Budget comparisons: within each seed and nominal SNR, find earliest checkpoint meeting dev accuracy>=.70 and conditional cumulative epsilon<=8/100/1e5/1e7; report minimum initial+deletion RE for random/channel/joint. If no candidate, report infeasible on this finite grid. Selection uses dev, selected test only reported afterwards. Checkpoint analysis postprocesses the already available models; no cost-free release of additional checkpoints or full reference assumed.
- Practical grouping screen/confirmation at K4/R1/natural: joint vs random and vs channel, average test loss<=2pp, and >=10% total initial+deletion RE reduction at the common dev target .70, achieved in >=2/3 seeds at a given SNR. Both methods must meet target; failures/missing target are reported separately and cannot be hidden. Natural privacy changes also reported; improvement of all objectives is not assumed.
- Fixed-round resource count can be identical across methods. Then report no communication-count improvement and compare errors/energy/rounds-to-target instead. Do not claim fewer symbols solely from better channels.
- Replay exactness checked at every stored checkpoint against separately executed same-SISA reference, source and replay unaffected models remain equal. Conditional DP is a conservative analytic bound, not an attack experiment. No MIA success rate invented.

## Execution and cost

One GPU worker only; inventory checked before run. Record code hashes, snapshots, partitions, configs, per-round ledgers and power samples in snr_privacy_v1/. Preliminary physical/DP invariant audit precedes training. Stop on failed exactness or numerical validity, report completed scope and partial cost. GPU board Wh is not RF Joules. No external paid compute. Initial success/failure reports and final independent CPU audit included.
