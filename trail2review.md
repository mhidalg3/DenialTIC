# Trial 2 Review: TIC-VLA Yo-Yo and Sustained Slow-Path Attacks

Date: 2026-10-15

## Summary

The new attack is closer to the paper but still produced no break. Success went 2/4 control → 3/4 Yo-Yo → 3/4 sustained, and every condition had 1/4 episodes with a collision.

Two caveats limit how far this goes:

- **No logs:** the export has only result JSONs, so I can't confirm the attack caused any dropped updates.
- **Delay is largely cancelled:** the simulator and TIC-VLA's design together absorb most slow-path delay.

This is an evidence analysis of the existing exports. Nothing was changed and no new runs were made.

Path shorthand:

- `T2/` = `ticvla_current_results/` (this trial)
- `T1/` = `ticvla_initial_experiment_export/` (trial 1, renamed from `ticvla_experiment_export/`)

### Data used

| Condition | Result file | Settings (`T2/code/experiment_conditions.txt`) |
|---|---|---|
| Control | `T2/baseline/dod_pilot_results_20261015_040503.json` | Yo-Yo off, 5 s deadline, deterministic |
| Yo-Yo | `T2/yoyo/dod_pilot_results_20261015_033320.json` | 6 s ON / 20 s OFF, 6 burst jobs × 6 s, 5 s deadline, deterministic |
| Sustained | `T2/sustained/dod_pilot_results_20261015_043637.json` | always ON (10000 s / 0 s), 6 burst jobs × 6 s, 5 s deadline, deterministic |

- **Pairing:** all three files contain episodes 1, 27, 51 and 65. Scene, start, goal, instruction, timeout and robot match across conditions.
- **Aliases:** the `*_latest.json` files match their dated files, ignoring `timestamp`.
- **Run order:** Yo-Yo 03:33, control 04:05, sustained 04:36.

## What changed in trial 2

All from `T2/code/all_changes.diff` and `T2/code/ticvla.py`.

- **Deadline rejection:** a result older than 5 simulated seconds (150 frames) is now discarded instead of published (`ticvla.py:1083-1108`). This is the paper's "discard as stale" step (p. 4, Fig. 2).
- **Burst queueing:** during an ON window, 6 sleep jobs of 6 s each are queued ahead of the real request on the single worker thread (`all_changes.diff:315-348`).
- **Image snapshots:** the request's images are hard-linked when it is queued. This fixes trial 1's missing-image side effect (`all_changes.diff:121-151`).
- **Greedy decoding:** generation uses `do_sample=False` (`ticvla.py:1187`).
- **Delay input fixed:** the fast path now gets the true age of the cache it is using. It also gets displacement measured from that cache's source pose (`all_changes.diff:738-778, 841-852`). Trial 1 had a mismatch here.

Provenance:

- **Same base:** commit `9fa6f8b` in both trials.
- **Same benchmark and config:** `benchmark.py` and `dod_pilot.yaml` are byte-identical across `T1/code/` and `T2/code/`.
- **Behavior file:** `T2/code/nova_carter_test_ticvla.py` is byte-identical to `DenialTest/TIC-VLA/DynaNav/behavior/nova_carter_test_ticvla.py`.

**Control isn't the plain trial-1 baseline.** It runs the same new code with the attack off, so it still has the 5 s deadline, greedy decoding and the corrected delay input.

## Results

| Episode | Control | Yo-Yo (6 s on / 20 s off) | Sustained |
|---|---|---|---|
| 1 hospital | ✓ 18.47 s, 12.11 m | ✓ 23.53 s (+27.4%), 13.96 m (+15.3%) | ✓ 19.27 s (+4.3%), 12.66 m (+4.6%) |
| 27 office | ✗ 10.79 m error, physical collision | ✓ 1.49 m error, physical collision | ✓ 1.49 m error, physical collision |
| 51 outdoor | ✓ 24.57 s, 16.13 m | ✓ 22.73 s, 14.82 m | ✓ 23.73 s, 15.60 m |
| 65 warehouse | ✗ 18.17 m error, 51.16 m path | ✗ 16.49 m, 55.68 m | ✗ 26.04 m, 64.53 m |
| Success / collision | 2/4 / 1/4 | 3/4 / 1/4 | 3/4 / 1/4 |

The recomputed averages match each JSON's `average_metrics`.

### Same episodes across both trials

Format: success, final error (m), path (m), duration (s).

| Episode | T1 baseline | T1 24 s delay | T2 control | T2 Yo-Yo | T2 sustained |
|---|---|---|---|---|---|
| 1 | ✓ 1.49, 12.13, 18.5 | ✓ 1.50, 15.33, 33.1 | ✓ 1.49, 12.11, 18.47 | ✓ 1.50, 13.96, 23.53 | ✓ 1.50, 12.66, 19.27 |
| 27 | ✗ 10.73, 28.37, 60 | ✓ 1.50, 20.41, 43.23 | ✗ 10.79, 28.33, 60 | ✓ 1.49, 20.55, 48.23 | ✓ 1.49, 20.58, 49.9 |
| 51 | ✓ 3.99, 15.38, 23.4 | ✓ 3.99, 15.59, 23.73 | ✓ 3.99, 16.13, 24.57 | ✓ 3.98, 14.82, 22.73 | ✓ 3.99, 15.60, 23.73 |
| 65 | ✗ 6.26, 61.95, 110 | ✗ 8.43, 68.01, 110 | ✗ 18.17, 51.16, 110 | ✗ 16.49, 55.68, 110 | ✗ 26.04, 64.53, 110 |

## Takeaways

### 1. No break, and the clearest repeated effect goes the wrong way

Episode 27 failed in both controls (10.73 m and 10.79 m error). It succeeded in all three delayed runs: trial-1 24 s, Yo-Yo, and sustained. In the office, fresh slow-path updates appear to send the robot the wrong way, and taking them away helps.

### 2. The attack itself is unverified

None of the attack's log lines were exported: `stale_drop`, `cache_publish`, and `benign_enqueue phase=ON`. The drop counters weren't saved either. So "the attack did little" and "the attack hardly fired" can't be told apart.

### 3. Attack strength depends on the scene (INFERRED)

The burst is measured in wall seconds, but the deadline is measured in simulated frames. Using trial-1 frame rates, the expected age of a request queued during an ON window is roughly (36 s + service time) × fps / 30:

| Episode | Expected age | Under the 5 s deadline? |
|---|---|---|
| 1 | ≈ 13.1 s | no → dropped |
| 27 | ≈ 10.7 s | no → dropped |
| 51 | ≈ 4.2 s (≈ 3.2–6.2 s range) | yes, mostly → **likely barely attacked** |
| 65 | ≈ 14.4 s | no → dropped |

### 4. The Yo-Yo setting is weak by construction (INFERRED)

- **Burst only on request:** burst jobs are queued only when the victim submits a request during ON (`all_changes.diff:315-336`).
- **One victim per window:** a delayed request takes about 11–14 simulated seconds, longer than the 6 s ON window. So each 26 s cycle delays at most one victim request.
- **Episode 1:** it lasts about 18–24 s, so essentially only the first post-bootstrap update was lost.
- **Sustained:** this mode drops every post-bootstrap update in episodes 1, 27 and 65. The robot runs the whole episode on the bootstrap state plus fresh camera frames. It still succeeds in episode 1, taking just 0.8 s longer.

### 5. The simulator pauses the world while the fast path runs

`on_update` calls inference synchronously (`nova_carter_test_ticvla.py:863-880`, `1262-1265`). The robot and pedestrians stop while the fast path computes.

Trial 1 measured 2.5–11.8 simulated frames per wall second. That compresses a 36 s wall-clock attack into 3–14 s of staleness. On a real robot it would be the full 36 s.

### 6. TIC-VLA is built to tolerate stale context

The action expert runs every step on the current image. It is given the cache's true age and the displacement since that cache's source pose.

Denying updates therefore degrades the system to "first plan plus fresh vision". That is enough for "go straight and stop at X" instructions.

A denial-of-deadline attack can remove at most the slow path's added value. In these episodes that value is near zero, or negative in episode 27.

### 7. Trial 1's episode-1 harm may partly be a test-setup artifact (hypothesis)

Trial 1 fed the model a mismatched delay. With the delay corrected, even sustained denial costs episode 1 only +4% time.

The Yo-Yo run, with less denial, cost +27%. That non-monotonic result points to noise or interaction effects, and this data can't settle which.

### 8. This is still not a faithful Yo-Yo

It now models queueing and deadline rejection (paper Fig. 2, steps 2–4). Missing pieces:

- an independent ordinary-user attacker (paper p. 3, §3.1)
- real attacker requests competing for compute (these are sleeps; the code says so at `all_changes.diff:158-164`)
- autoscaling lag (paper p. 5, Fig. 3)
- a separate slow-path service

The attacker also gets free knowledge of exactly when the victim submits.

### 9. There isn't enough data to call any difference

- **Single runs:** there is one run per cell.
- **Fixed order:** conditions ran in the same order every time (Yo-Yo, then control, then sustained).
- **High noise:** episode 65's final error ranges 6.26–26.04 m across five runs. Sustained being the worst there (+7.9 m error, +26% path) is suggestive at best.
- **Greedy decoding isn't enough:** it removes VLM sampling. Pedestrians, physics, and thread timing can still vary.

## Path to a genuine break

Each step gates the next.

### 1. Prove the attack fires

Save all per-episode logs. Write these counters into the result JSON:

- publishes and stale drops
- requests queued during ON
- maximum cache age
- the distribution of the delay value the fast path receives

Without this, no trial is interpretable.

### 2. Measure the ceiling first

Run control against "bootstrap only", several repeats each. Bootstrap only means dropping every post-bootstrap result regardless of frame rate.

If bootstrap-only doesn't fail, no latency attack can fail these episodes. The next step is then to change episodes, not the attack.

The paper's collapse exists because the slow path adds accuracy (7.0 HOTA). Here that gap hasn't been shown.

### 3. Pick episodes that need the slow path

Screen the full DynaNav set for episodes where control succeeds and bootstrap-only fails. Likely candidates:

- multi-step or turning instructions
- goals not visible from the start
- long episodes
- crowded scenes where yielding matters

### 4. Fix the time base

Two options:

- **Simulated-frame queue:** express queue time in simulated frames. This is scene-invariant and controlled.
- **Real-time simulator:** run the simulator in real time so the world keeps moving while compute is late. This also exposes fast-path lateness, which the synchronous loop hides today.

### 5. Build the real threat model

- **Separate service:** host the slow-path VLM as its own service on separate hardware, with a request queue and an autoscaler that has scale-out delay.
- **Independent attacker:** run a separate client that sends real VLM requests in bursts timed against scale-in.
- **Measure it:** benign p99 latency, drop rate, and attacker request count (the paper's potency ratio).
- **Then target timing:** once that works, sweep the burst timing to find the moments where lost updates matter.

### 6. Use continuous harm metrics

Success is too coarse with four episodes. Record:

- minimum goal distance and time-to-goal
- minimum pedestrian distance and near misses
- backup events
- per-step divergence from a fresh-cache shadow or offline replay

Run 5–10 repeats per cell in randomized order and report confidence intervals.

### 7. Cheap probe for a break

Sustained denial pushes the cache age the fast path sees past 100 s. Replay recorded frames offline while sweeping that delay input to find where actions degrade.

If degradation shows up only past a certain age, long episodes under sustained attack are where to look. This is consistent with episode 65, but unproven.

## Open decision

Keeping the last plan forever is a forgiving fallback. Some deployments have a staleness watchdog, for example stopping when there is no fresh plan for T seconds. With a watchdog, denial-of-deadline becomes a direct mission failure through timeouts.

Adding a watchdog changes the victim system, so whether it's in scope is a decision for the project, not this analysis.

## Housekeeping

The trial-1 export was renamed to `ticvla_initial_experiment_export/`, so the `ticvla_experiment_export/` paths in `trail1review.md` no longer resolve. That file was left unchanged.
