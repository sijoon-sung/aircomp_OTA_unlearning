**두 가지 제한된 추가 진단 — 초기 세 방법의 설정과 결과 보존**

①의 rank48 변형은 차원뿐 아니라 공개 회전도 도입한다. 따라서 그 효과를 rank 감소만의 이점으로 해석하지 않도록 같은48차원 basis를65차원 orthonormal basis로 완성한 비교군을 추가한다. 회전된 full65차원에서 Original과V1을 각각 실행한다. 같은3 seeds×16noise,20dB,peak제약,H32,residual40,budget를 유지한다. 이 추가 진단은 초기 결과를 본 뒤 정한 사후 ablation이며, 처음부터 고정한 주 비교와 표를 분리한다. 새 설정으로 최초 결과를 덮어쓰지 않는다.

②의 Legacy-Sketch-V1이 exact FedOSD보다 낮은 reference 오차를 보였으므로, 첫 step에서 exact direction, sketch+무잡음 합산 direction, sketch+noisy 합산 direction의cosine과relative error를 검사한다. 이는 noisy결과가더정확한projection이라는주장을피하기위한기하진단이다. 현재모델·seed·batchschedule을유지하고정책을재선택하지않는다. 새로운recovery나전체학습을추가하지않는다.

이 작업은③수치수정실험이끝난뒤같은GPU worker정책으로순차실행한다. 원래모든raw결과를보존한다.
