# Which boundary message to send immediately — protocol v001
Registered 2026-10-14 before running (U13). Setting as FRESHNESS_VALUE_v001 (h30_x features, monthly retraining fleet HGB, Feb–Dec 2017, sync, lag 0, oil truth 3 steps late, no local correction). Front axle bearing treated like rear (sent together as "bearing message").
Budget: at every 10-min step exactly one of {bearing message, inlet message} is fresh (age 0); the other arrives 3 steps late (its last fresh copy may also be used if newer — the model receives each sensor's latest available value and its age in steps).
Policies: always_bearing; always_inlet; alternate (switch every 3 steps); event: send the sensor whose own value changed more in the last step, normalized by its training std (sender-side decision, uses only the sender's own measurements).
Metric: MAE all and transient. Decision: choose the policy with lowest transient MAE whose overall MAE is within 1% of the best overall.
