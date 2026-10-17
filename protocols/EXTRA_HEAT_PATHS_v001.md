# Extra heat-path channels on the monthly-retrain pipeline — protocol v001

Registered 2026-10-14 before running. Motivation: exploratory sync screen (2019 dev week) showed ~15% gain from unused gearbox-side temperature channels; physically these are other heat paths of the main gear oil (open hypothesis).

Identical setup to `H29_H30_PHYSICS_TERMS_v001` (Kelmarsh 2017, Feb–Dec evaluation, monthly retraining on released rows only, fleet HGB with turbine id, synchronous, lag 0, local correction k=20 memory 8640).
Extra channels (value at t and 1/2/3-step changes): Front bearing temperature, Rear bearing temperature, Rotor bearing temp, Gearbox speed. Columns that are all-NaN/constant in a training window are dropped for that window only (logged).
Variants: base_x (base + extra), h30_x (h30 + extra), each with and without local correction.
Decision: extra channels are "supported" if h30_x+local beats h30+local (0.2402) by >= 3% overall without worsening the transient subset. All results kept.

## Amendment 1 (2026-10-14, registered before any result of v001 was seen) — H31 convection terms
Author's hypothesis H31: bearings heat the oil by convection; air temperature and speed matter. Variant h31_x = h30_x + [front bearing − oil(t-3)], [rear bearing − oil(t-3)], [rotor bearing − oil(t-3)], [oil(t-3) − nacelle temperature], [gearbox speed^2]; with and without local correction.
Decision: H31 supported if h31_x beats h30_x on transient MAE by >= 2% without worsening calm by > 1%.
