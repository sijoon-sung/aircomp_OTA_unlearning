**Original / V1 순차 비교 — 실행 전 고정, 2026-09-26**

이름은 이번 연구 기록용이며 학술적 신규성이나 기존 문헌에 없는 명칭을 주장하지 않는다. Original은 우리 기존 구현이다. 논문 원형 전체 재현을 뜻하지 않는다. 과거 sketch V1은 Legacy-Sketch-V1로 표기한다. 세 family의 reference가 다르므로 단일 전체 순위를 만들지 않는다. Recovery와 SFL은 확장하지 않는다.

| Family | Original | V1 |
|---|---|---|
| DS-Air: Deletion-Signal AirComp | 집계 residual에 서버가 공통 inverse 적용 | client에서 동일한 inverse 좌표 보정 후 합산 |
| OG-Air: Orthogonal-Gate AirComp | FedOrtho 수식 기반 activation 차이, top25%,50–75% 감쇠 | 같은 ortho source에서 client 제거에 따른24개 gate-gradient 차이로 작은 signed 보정 |
| RTD-Air: Retained-Teacher Distillation over AirComp | 독립 teacher 확률10좌표 OTA, target-free fresh student | 확률의 알려진 공통 평균을 제거하고9개 직교 contrast 좌표 OTA, 동일 fresh student |

새 model/data seeds371/372/373을 사용한다. DS-Air는202609263/264/265. 어떤 reference/test도 hyperparameter·resource·stopping 선택에 쓰지 않는다. DS는 seed당16 noise, OG/RTD는 seed당4 noise. Source 모델을 만들고 source/reference를 평가하는 것은 결과 기록용이다. 기존 data와 코드의 검토된 함수만 재사용하며 모든 신규 출력은 이 폴더에 저장한다. GPU worker 하나로1→2→3 순서 실행한다.

**공통 수학 검사**

DS: 공통 선형 보정과 합산의 교환, local inverse 평균의 반례, 실제 송신/수신 정규화, exact ridge 삭제 항등식, rank48 연산의 nullspace를 검사한다. 전체 inverse와 정확한 stationary source는 target gradient 역산 위험이 남는다. DS의 full V1은 성능 비교용이며 strict transcript 조건을 통과했다고 하지 않는다.

OG: γ=1의 gated forward가 source와 같다. Original의 kernel·bias 양의 scaling과 ReLU gate의 동치를 확인한다. L_all=αL_A+(1−α)L_R일 때 같은 γ에서 g_all−g_R=α(g_A−g_R). 같은 출발점에서 all-data/retained 한 gradient step의 차이는 ηα(g_A−g_R)다. V1은 이를 gate 공간에 적용한다. 이는 과거 전체 경로 삭제 등식이 아니다.24차원 정보는63,562차원 전체 model gradient와 구별한다.

RTD: Q∈R^(10×9),QᵀQ=I,Qᵀ1=0. 확률 q는 q=1/10+Q(Qᵀq)로 복원되고, weighted 합과 변환이 교환된다. 알려진 DC를 빼고 송신 RMS를 재계산한다. Drop-last-class 방식처럼 마지막 class에 noise가 누적되지 않는다. 동일 target-free teacher·동일 초기값·KD schedule에서 무잡음 함수는 같다. 정확한 parameter equality는 FP32 재구성 오차로 깨질 수 있어 확률/gradient 오차와 실제 출력 차이를 구별한다.

**1. DS-Air**

FashionMNIST10,000개,label-2 client10명,client0 삭제. 기존과 같은 공개 무작위 ReLU784→64+bias encoder, λ=.01 ridge65×10. Encoder는 private data로 학습하지 않는다. Reference는 같은 head의 exact retained optimum.20dB, H upper triangle32 repeats, residual5 blocks×13×10, 반복 총40. Main channel은 real MAC 평균전력≤1,peak real amplitude²≤4. 평균전력-only는 ablation. Source/basis/eigenvalues의 방송은 실제FP32 roundtrip을 적용하고 계산은FP64로 한다.

Original/V1 외 비교: no-op, exact retained optimum, gradient-MSE allocation, Original에 V1의 추가 eigenvalue DL 예산을 준 정수 반복 배정, noisy sufficient-statistics retraining(H32,C8), diagonal Newton(diagH32,b8), retained GD40(lr.1). H/basis 준비비용과 final model delivery를 모두 센다. Primary는 parameter squared error/initial deletion gap이고 test accuracy와 output JS도 기록한다.

V1의정보 제한 ablation: 공개 seed의 고정 orthonormal65×48 subspace U만 사용한다. Client는 UᵀH_iU와 Uᵀb_i를 전송하고, 공통 reduced inverse를 받은 뒤48차원 보정을 합산한다. Server는U로 lift한다. full H_R·다른 subspace query를 함께 수신하지 않는다. 이것은 특정 선형 복원 경로를 rank 제한하는 것일 뿐 DP가 아니다. 원래full optimum 대비 projection bias를 숨기지 않는다. Subspace는결과를 보고 선택하지 않는다.

**2. OG-Air**

기존 common.py와 동일 FashionMNIST private12,000/public2,000/val2,000/test10,000, moderate non-IID,10 clients,client0 삭제. ConvNet2(8/16 channels,hidden32),63,562 parameters. Ortho source/retrain은150 rounds,local4 steps,lr.03,batch64,2단계 aggregation,λortho.001/alignment.0001이다. 이번 V1은 training을 바꾸지 않아 Original과 동일 source/reference를 사용한다.

Original: 기존 activation-difference score를 OTA8 repeats로 받고 layer별top2/top4,50–75% 감쇠. V1: source에서 각 client의 전체 local data 평균 gate gradient24개를 계산한다. Client0은αg_A,잔존client는−αp_R,i g_i를 보내합산한다. γ=1+clip(.1×received_direction,−.05,.05). LR.1 및 cap.05는 고정한다. 모델 source를 추가 최적화하는 보정 loop나 recovery는 없다. Noisy20dB8 repeats,4noise,무잡음 비교. Target gradient와 retained gradient를 따로 full dimension으로 공개하지 않는다.

비교: no-op, paired target-free ortho retrain, Original-mild(같은ranking10%감쇠), class-matched-mild(앞서제안한같은class점수), retained-gate one-step(24좌표,noiseless), retained full-gradient SGD10(lr.01,batch64/client), FedOSD-core10(UCE+exact projection,lr.01,batch64/client), Legacy-Sketch-V1(동일10steps,SRHT2048,16bit sketch,weighted OTA20dB). FedOSD-core는 recovery제외 adaptation이며 전체 원논문 성능재현이라고 하지 않는다. 개별fullgradient를받는oracle비교군은우리정보제약위반으로표시한다.

Primary=forget predictions JS to paired retrain / no-op JS. 절대JS,test/forget accuracy,lossMIA AUC,통신비용,source/retain준비비용을함께기록한다. V1이무개입에가깝다는이유만으로삭제성공으로보지않는다. 평균<.8이고3seed모두<1이탐색통과기준이다. Source부터MIA≈.5면privacy평가감도가약하다고기록한다.

**3. RTD-Air**

동일 CNN/data seeds371/372/373.10개 local teacher를각자공개초기값에서500steps(lr.03,batch64)독립학습;global feedback없음.공개2000query,T=2. A제외확률을받아공개초기값에서600KDsteps(lr.05,batch64)student학습.Original/V1/대조군모두같은teacher와같은KD난수stream. Reference는같은독립teacher알고리즘의A제외freshstudent다. Source는A포함ensemble로같은절차로학습한student다.

비교: no-op source, target-free noiseless, Original full10probability20dB r1/r4, V1 9contrast20dB r1/r4, centered-full10 ablation20dB r1, ideal-digital8bit client probability(round(q×255)/255후sum정규화), fixed1000query V1 ablation20dB r1. Digital은실제8bit양자화하되error-freepayload/shannon-goodput모형이며codedRF실증이아니다.1000query는label을보지않는공개seed고정subsample이며reference는2000query유지한다. 근접exact0을얻으려고reference를바꾸지않는다.

FedQUIT-logit-min+retained CE100steps와기존globalteacher KD는별도FedAvg source/reference문맥의외부비교표에만놓는다. Method2에서준비한ortho source에강제로섞어RTD보다좋다/나쁘다순위를매기지않는다. 필요한plain source/retain을같은150round로따로준비해새seed로평가한다.

**통신 ledger와 판단**

이번모든표는real channel-use equivalent로통일한다. R=.5log2(1+SNR) bits/real use, CP factor1.125. 총비용=1.125×(ULreal+pilotreal+(DLbits+ULdigitalbits+metadata)/R). Norm32bit/client/packet,pilot8real/client/packet. block별수신분산이달라지는경우각packet을센다. Final fullmodel32bitDL포함.신호기저는공개seed로만드는경우seed32bit,학습된기저는실제행렬전송비용. Source준비/teacher훈련/KD/기존publicdata배포비용은별도기록한다. 기존95,611complex-use표와이번real-use표를직접비교하지않는다.

Simulatedchannel은coherentrealMAC,고정positiveh,평균/peakpower제약,AWGN이다. DSpilot은h=1,OG/RTD는기존h=[.7,1,.8,1.2,.9,1.1,.75,1.05,.85,1.15].CSI/CFO/실제RFwalltime을확인했다고하지않는다. 전송오차만의비교에서는같은수신목적함수·학습schedule을유지한다.3seed평균과seed간표준편차를보고하고noise반복은seed안에서평균한다.실험완료후성공기준을바꾸지않는다.
