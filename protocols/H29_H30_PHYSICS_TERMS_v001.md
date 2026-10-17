# H29 + H30 physics terms for blind-window cooling transients — protocol v001

Registered 2026-10-13 before running. Author's hypotheses: H29 (heat-transfer coefficient: main oil cools in proportion to its difference from the cooled inlet oil), H30 (high load or wind from an unfavourable direction adds heat and cancels the cooler). Coefficients are learned, no MM92 physical constants are assumed.

Data/protocol identical to `LOCAL_LATE_TRUTH_LONG_v001` Amendment 1 (Kelmarsh WT1–6, 2017, synchronous, lag 0, truth released 3 steps later, fleet HGB with turbine id, monthly retraining on released rows only, evaluation Feb–Dec 2017).

Feature sets (all from information available at decision time; oil only up to t-3):
- base: as before (9 helpers at t, 1/2/3-step changes, oil(t-3), oil(t-3)-oil(t-4)).
- h29: base + [oil(t-3) - mean(inlet t-2..t)] (driving temperature difference over the blind window) + [inlet(t) - inlet(t-3)].
- h30: h29 + [power(t) x (oil(t-3) - mean inlet)] + [rotor speed(t)^2] + [|vane position 1+2(t)|] (yaw misalignment) + [power(t) x (inlet(t)-inlet(t-3))].
Each with and without the local late-truth correction (k=20, memory 8640).
Metrics: MAE all, calm (below-median |y(t)-y(t-3)|) and transient (top 10%), per turbine, averaged; per month.
Decision: H29 supported if h29 beats base on transient MAE by >= 3% and does not worsen calm by > 1%. H30 supported if h30 beats h29 on transient by >= 2% under the same calm condition. All results kept.
