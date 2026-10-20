# Trial 1 Review: TIC-VLA / DynaNav 24 s Slow-Path Latency Injection

Date: 2026-10-14

## Summary

The injected delay worked as built. It slowed the slow path a lot, and the fast path kept running. The four-episode suite does not show more navigation failures or collisions overall. The one clear harm is episode 1, and the two episode-1 runs disagree with each other. The setup does not reproduce the paper's Yo-Yo threat model.

This is an evidence analysis of the existing exports. Nothing was changed, and no new runs were made. All numbers were recomputed in memory from the exported JSON files and logs. Every expected value in the analysis plan matched.

Path shorthand: `export/` means `ticvla_experiment_export/`.

### Comparison set

| Comparison | Baseline (0 s) | Injected (24 s) |
|---|---|---|
| Four-episode suite | `export/dod_baseline_0s/dod_pilot_results_20261013_223838.json`, `export/logs/dod_baseline_0s.log` | `export/dod_latency_24s/dod_pilot_results_20261014_042851.json`, `export/logs/dod_latency_24s.log` |
| Standalone episode 1 | `export/logs/episode_1_result.json`, `export/logs/container_episode_1.log` | `export/logs/dod_24s_episode1_result.json`, `export/logs/dod_24s_episode1.log` |

- **Pairing:** suite episodes are paired by `episode`. `scene`, `start`, `goal`, `instruction`, `timeout`, and `robot_type` match for all four pairs and for the standalone pair.
- **Latest aliases:** the `*_latest.json` files match their dated counterparts, ignoring `timestamp`. They are not counted as extra trials.
- **Excluded baseline:** `dod_pilot_results_20261013_211537.json` is excluded. All four of its records have `error: no_result_file`, so it is an infrastructure failure, not a navigation baseline.

## 1. Verdicts

| Question | Verdict |
|---|---|
| Were slow-path updates inhibited? | **Yes, but delayed, not blocked.** Usable updates fell sharply, and each one arrived about 29–40 s after its request. Late results were still accepted, so the slow model was not shut out. |
| Did the fast action expert continue? | **Yes.** Commands kept changing and the robot kept moving during every sleep window. The command count is a lower bound on model calls, not proof of an unchanged call rate. |
| Did navigation get worse? | **Not overall.** Suite success went 2/4 → 3/4, and episodes with a collision flag went 2/4 → 1/4. Only episode 1 got worse, and its two pairs disagree. |
| Is this a faithful Yo-Yo replication? | **No.** It is a fixed 24 s sleep inserted in the slow-path worker. There is no traffic burst from an ordinary user, no shared-resource contention, no autoscaling, and no deadline rejection. |

### Why "delayed, not blocked"

- **No deadline check:** `_poll_kv_cache_future` publishes any result that has a state, with no age check (`export/code/ticvla.py:962-966`).
- **Bootstrap exempt:** the first generation is never delayed (`ticvla.py:913-927`).
- **Old state kept:** until a new result arrives, the fast path keeps using the previous semantic state (`ticvla.py:1122`).
- **Planning doc:** `Denial_Plan.md:25` says the action expert should "never receive" the cache. The code does not do that.

### Why "the fast path continued"

- **Waits don't block the fast path:** the sleep runs in a background executor thread before VLM inference (`ticvla.py:820-864`, `929-937`).
- **Each fast call:** fresh current-frame image features plus the old cache go into `self.action_expert(image_embeds, state, kv_cache=kv_cache_to_use)` (`ticvla.py:1260-1292`).
- **What a `v_cmd` line means:** the local behavior copy prints `v_cmd` only after `predict_async` returns and the command is accepted (`DenialTest/TIC-VLA/DynaNav/behavior/nova_carter_test_ticvla.py:1347-1356, 1459-1466`).
- **Backup is separate:** no inference runs during backup recovery (`nova_carter_test_ticvla.py:790-792`).
- **Provenance limit:** this behavior file is not in the export, so I can't confirm it is the exact code that ran.

### Why it isn't Yo-Yo

- **Paper threat model (p. 3, §3.1):** the attacker is an ordinary user whose valid requests share network and compute with benign requests.
- **Paper Fig. 2 (p. 4):** contention from scaling and queueing makes results miss their freshness deadline, and the merger discards them as stale.
- **Paper Fig. 3 (p. 5):** Yo-Yo bursts push benign p99 latency past a 250 ms SLO.
- **TIC-VLA has none of these:** no burst, no contention, no autoscaling, and no deadline-based discard. Its merge rule is "use whatever cache is newest."

## 2. Measured mechanism

Counts below are ordinary completed generations "with model state", excluding bootstrap. Delivery is from request start to when a poll publishes the result, so it includes polling delay (`ticvla.py:1015-1022`).

| Run | Updates 0 s → 24 s | Mean delivery, wall s | Mean delivery, sim s (frames/30) | Max cache age used, sim s | Completed 24 s sleeps |
|---|---|---|---|---|---|
| Standalone ep 1 | 9 → 3 | 5.608 → 29.704 | 1.833 → 11.444 | 23.067 | 3 (+1 cut off at `dod_24s_episode1.log:4018`) |
| Suite ep 1 | 9 → 2 | 5.692 → 29.495 | 1.785 → 11.317 | 22.567 | 2 (+1 cut off) |
| Suite ep 27 | 38 → 3 | 5.852 → 29.225 | 1.563 → 9.133 | 24.767 | 4 (+1 cut off at `dod_latency_24s.log:7996`) |
| Suite ep 51 | 19 → 7 | 15.114 → 39.882 | 1.168 → 3.229 | 7.600 | 7 (+1 cut off) |
| Suite ep 65 | 42 → 9 | 7.577 → 30.757 | 2.548 → 10.489 | 21.700 | 10 (last never published) |

- **Sim time vs wall time:** the plan's assumption that 24 s equals 720 frames is wrong. Measured rates were 2.46–11.56 simulation frames per wall second (outdoor ep 51 ≈ 2.5). The 24 s sleep therefore lasted roughly 60–280 frames, depending on the scene.
- **Recorded sleep length:** each completed sleep logged `actual_s` of 24.000–24.010.
- **Behavior's delay input disagrees with real cache age:** the behavior-supplied `time_delay` differed by more than 0.1 s from the measured cache age in 98 of 661 calls in injected ep 27 (2–8 in the other runs). The largest gap was 11.6 s (e.g. `dod_24s_episode1.log:4016-4017`).

### Standalone 24 s timeline (`export/logs/dod_24s_episode1.log`)

| Sleep (lines) | Steps covered | `v_cmd` lines inside | Distinct (v, w) | Robot motion | Cache age range, sim s |
|---|---|---|---|---|---|
| 565–1469 | 9–279 | 128 | 63 | (−3.00, −1.20) → (2.78, −0.74), 5.79 m | 0.17–9.17 |
| 1691–2659 | 345–634 | 137 | 88 | (4.24, −0.54) → (10.75, 0.79), 6.65 m | 11.27–20.90 |
| 2862–3794 | 694–973 | 132 | 125 | stuck near (10.98, ~1.0), 0.27 m | 11.70–21.00 |
| 4018–end (cut off) | 1039–1200 | 76 | 57 | 0.09 m, w = −1.2 throughout | 11.57–16.93 |

Published updates in this run arrived at `:1687`, `:2858`, and `:4014`. Delivery was 29.962 / 29.514 / 29.637 wall seconds, or 336 / 349 / 345 frames.

### Side effects the injection added

- **Lost context images:** only injected runs logged "Failed to load image" (standalone 4, suite ep 1: 2, ep 27: 8, baseline: 0). Delayed jobs still point to frames the 4-frame image buffer has already deleted (`nova_carter_test_ticvla.py:770-774`). So the slow model also loses context images, not just time.
- **One update lost entirely:** in ep 27 all four input images were missing, and the generation completed "without model state" (`dod_latency_24s.log:7987-7992`). That one is input loss, not queue delay.

## 3. Navigation outcomes

Path and time in parentheses are absolute and percent change. Success radius is 1.5 m, except 4 m for ep 51 (`export/code/dod_pilot.yaml`). Timeouts are 40 / 60 / 100 / 110 simulated seconds.

| Suite episode | Success | Collision flag (physical streak) | Path, m | Duration, sim s | Final error, m | SPL |
|---|---|---|---|---|---|---|
| 1, hospital | ✓ → ✓ | 0 → 0 | 12.129 → 15.330 (+3.201, +26.39%) | 18.5 → 33.1 (+14.6, +78.92%) | 1.487 → 1.499 | 1.000 → 0.849 |
| 27, office | ✗ → ✓ | 1 → 1 (physical; 156 → 150) | 28.366 → 20.409 (−7.957, −28.05%) | 60.0 → 43.233 (−16.767, −27.94%) | 10.733 → 1.496 | 0 → 0.740 |
| 51, outdoor | ✓ → ✓ | 0 → 0 | 15.378 → 15.594 (+0.216, +1.41%) | 23.4 → 23.733 (+0.333, +1.42%) | 3.993 → 3.994 | 1 → 1 |
| 65, warehouse | ✗ → ✗ (timeout) | 1 (human) → 0 (20 → 40) | 61.953 → 68.015 (+6.062, +9.78%) | 110 → 110 | 6.260 → 8.432 | 0 → 0 |
| Total / mean | 2/4 → 3/4 (+25 pp) | 2/4 → 1/4 (−25 pp) | 29.456 → 29.837 | 52.975 → 52.517 | 5.618 → 3.855 | 0.500 → 0.647 |

The recomputed averages match `average_metrics` in both JSON files.

### How the benchmark counts things

- **Collision count:** `physical_collision_count` is the longest run of consecutive contact frames at ≥10 N, not a number of impacts (`export/code/benchmark.py:211/330, 1276-1295`). The physical flag only trips at 100 frames (`benchmark.py:130-131`).
- **Ep 65's 20 → 40:** a longer contact run that stayed under 100, so no physical flag. Its baseline flag came from human proximity (<0.2 m, `benchmark.py:698-701`). The log summaries agree: `Human Collision Rate: 25.0%` (`dod_baseline_0s.log:164076`) vs `0.0%` (`dod_latency_24s.log:164622`).
- **Success:** XY distance below the radius (`benchmark.py:636-647`). It does not check that the robot stopped at the object.
- **Path length:** the sum of per-frame XY motion (`benchmark.py:740-752`).
- **SPL:** uses the straight-line start-to-goal distance (`benchmark.py:715-738`). The run stops as soon as the robot is inside the radius, so path can be shorter than that distance. That is why SPL caps at 1 for baseline ep 1 (12.13 m vs 13.01 m) and ep 51 (≈15.4 m vs 18.42 m, ending at ≈3.99 m).
- **Planning doc:** `Denial_Plan.md:76` expects SPL to *rise* under latency. Higher SPL is better, so that expectation is backwards.

### Standalone episode 1

This pair is not pooled with the suite. It uses the same scene, start, goal, instruction, timeout, and robot.

| | Success | Duration, sim s | Path, m | Error, m | Collision | SPL |
|---|---|---|---|---|---|---|
| 0 s | ✓ | 18.767 | 12.313 | 1.493 | 0 | 1 |
| 24 s | ✗ timeout | 40.0 (+21.233, +113.14%) | 14.723 (+2.409, +19.57%) | 1.754 | 0 | 0 |

### Likely cause of the episode-1 harm (INFERRED from logs)

- **Baseline got turn-left plans early:** from step ~150 its updates included left-turn plans (`container_episode_1.log:1180-2077`). By x≈7.5 the robot was at y≈1.5–1.8.
- **Injected runs didn't:** they used the "go straight" bootstrap-era plan `(2.07,0,0)…` until step 343/328, so at x≈7.5 they were at y≈0.
- **Overshoot:** they passed about 1.8–1.9 m to the side of the goal and stopped near (10.98, 1.0).
- **Standalone outcome:** it never triggered recovery and timed out at 1.75 m.
- **Suite outcome:** backup started at frame 869 (`dod_latency_24s.log:3414-3415`), and the run ended a success at frame 993 with 1.499 m error. Last model position was 2.03 m from the goal, so backup recovery most likely carried it in. This is not clean evidence that the learned policy coped with the delay.

### Episode 27's improvement

- **Not a clean before/after:** the injected run hit a sustained collision near step 802 (`dod_latency_24s.log:7748`), then did backup from 851 to 1151 (`:7976-7977`, `:8667`).
- **How it finished:** 95 accepted model commands at steps 1153–1294 used a cache that was up to 24.77 s old. It entered the radius at frame 1297.
- **Baseline:** collided near step 1306, did backup 1308–1608 (`dod_baseline_0s.log:8366-8368`), and timed out.
- **Conclusion:** the paths differ too much to credit the result to the latency.

## 4. Causal limits and what's missing for stronger claims

- **Too few runs:** there is one run per episode and condition. Generation samples randomly (`do_sample=True`, `ticvla.py:1089`), and `seed: 666` does not make runs replay identically. The two episode-1 pairs disagree (suite succeeds, standalone fails). No significance test or failure-rate claim is supported.
- **Code provenance:** baseline logs have no DoD instrumentation, so I can't confirm both conditions ran the same build. The behavior file is a local copy, not exported.
- **Delivery includes stalls:** polling only happens inside `predict_async` (`ticvla.py:1081`), which doesn't run during backup. Baseline office has one 28.668 s / 321-frame delivery during backup (`dod_baseline_0s.log:8407`). That is a polling stall, not VLM compute time.
- **Log line order is unreliable:** Carb log lines are buffered out of order relative to stdout. Baseline `v_cmd` lines for steps 1302–1306 appear after the frame-1308 backup line (`dod_baseline_0s.log:8366-8401`). I rebuilt the timeline from frame numbers, not line order.
- **`-100` answers are not an injection effect:** `(-100,…)` answers appear in baseline ep 27 and ep 65 too. I found no code that gives them a meaning.

### Minimum evidence needed for stronger claims

- **Repeated runs:** several seeds per episode and condition, from one instrumented build, with the 0 s condition also logging DoD markers.
- **Separate the delay from its side effects:** keep context images until delayed jobs finish, and log backup counterfactuals or disable recovery.
- **Real deadline logic:** make the merger reject state older than a stated deadline.
- **For a Yo-Yo claim:** a shared slow-path service with autoscaling, bursty traffic from other clients, measured queueing and scaling delay, and deadline rejection. A sleep call can't stand in for this.
