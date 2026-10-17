# Value of freshness per sensor — protocol v001
Registered 2026-10-14 before running. Author's hypothesis H34: place the sensor at the closest installable point to the bearing. Data analogue: leave-one-sensor-stale.
Setting: Kelmarsh 2017, monthly retraining of fleet HGB (PARAMS of the long protocol), Feb–Dec evaluation, synchronous, lag 0, oil truth 3 steps late; features = h30_x set (h30 core + bearing temps + gearbox speed). No local correction (speed).
For each sensor s in {Power, Wind speed, Rotor speed, Generator RPM, Nacelle ambient, Nacelle, Grid current, Generator bearing rear, Gear oil inlet, Front bearing, Rear bearing, Rotor bearing, Gearbox speed}: replace s and its 1/2/3-step changes by values ending at t-3 (all derived physics terms using s also use the stale value); all other sensors fresh.
Metric: MAE increase relative to all-fresh (overall and transient). Prediction H34: the two axle-bearing temperatures rank in the top 2.
