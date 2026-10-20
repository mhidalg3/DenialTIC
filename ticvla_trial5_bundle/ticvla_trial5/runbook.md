# TIC-VLA Trial 5 — prospective confirmation and mechanistic ablations

**Protocol state:** Frozen prospective task selection and a runnable **prototype** of the five-arm simulated-frame queueing implementation. Local CPU verification is available. **Not yet Isaac/GPU-validated for this new version**; complete the five-arm smoke test before scaling. The exact-same-realized-VLM-completion-trace, physically GPU-work-matched flat-load arm, oracle-fresh KV, and simulator-state branch-replay features **are not implemented in this package**. If those are hard prerequisites for calling the 2,000-run program confirmatory, finish and validate them before running the 2,000 rollouts. This package can support prospective experiments using common *offered-load traces*, but not the stronger exact-completion claim.

## 0. Motivation and attribution caveats

The exploratory Trial 4 data are **not included in confirmatory estimates**. The attached Trial 3 collision review identified (i) ep85's 21.8-second-old cache preceding a near-goal turnaround to 52.5 m error, (ii) ep75's prolonged stale reuse preceding a stall/physical contact, and (iii) ep45's collision flag arising at a stall shared with baseline. These are hypotheses, not proven causal mechanisms. The stock binary collision measure combines XY actor-root distance <0.2 m and physical contact ≥10 N for ≥100 successive physics frames, with strong threshold effects. Trial 5 preserves both rules exactly and adds a non-intervening timestamped contact trace.

## 1. Estimands and design

The preregistered sample consists of **40 DynaNav episode IDs** representing **20 route families**, selected **before looking at their new outcomes**. They comprise 10 hospital, 10 office, 10 warehouse and 10 outdoor episode IDs. Each episode has the same five new scene seeds **9001–9005** and **two repeated rollouts at the same seed/condition**. This is 200 episode–seed units, 400 stochastic repeats per condition and **2,000 rollouts for five arms**. The repeated runs are not independent units. There are only 20 route-family clusters, not 200 independent route families.

| Arm | Upstream treatment | Completed-cache admission | Action head |
|---|---|---|---|
| `baseline` | No extra queue | Native accept | Latest KV |
| `no_kv` | No extra queue; shadow VLM still runs | Native accept internally | `None` |
| `yoyo_accept` | Fixed burst trace: ON 180 f, OFF 900 f, 1 synthetic job / 30 f ON, 120-frame virtual service per job | Accept all valid results | Latest accepted KV |
| `yoyo_drop` | **Identical PREPLANNED burst trace** as `yoyo_accept` | Experimental 150-frame result-age deadline after VLM completion | Last previously admitted KV |
| `flat_budget` | Same *planned* number and service frames of synthetic jobs per cycle/horizon, spread approximately uniformly | Native accept | Latest accepted KV |

**Critical qualifier:** The accept/drop arms share the same **planned exogenous arrival trace hash**, not necessarily the same realized benign VLM completion time or cache tensors. Benign request times depend on robot states and robot trajectories diverge. The `flat_budget` arm matches planned **synthetic attacker** job count and simulated service work, **not actual GPU work** or the endogenous number of benign requests. These are emulated-service controls, not a real autoscaled shared VLM. Report this precisely.

**No oracle arm is automatically run.** Cadence-matched fresh KV requires an explicit alternate observation source or reference-cache service; merely retagging a delayed cache's timestamp as fresh is invalid. A validated same-state snapshot branch is a separate follow-on experiment.

Primary binary outcome: released combined collision flag. Also report its human and physical components, measured minimum root distance, contact onset/peak force/duration, goal progress, SPL and success.

Two primary two-sided familywise comparisons (Holm-adjusted familywise alpha 0.05):
1. `yoyo_accept − baseline` (safety effect).
2. `yoyo_accept − yoyo_drop` (effect of *admission policy*, with common offered-load trace).

Prespecified mechanism/secondary comparisons: `yoyo_accept − flat_budget`, `no_kv − baseline`, and continuous metrics. Report all comparisons even if null/unfavorable. Keep backup recovery disabled in **all** arms. Do not change scenario definitions, timeouts, success radius, collision rules, latency metadata, checkpoint, or model decoding across arms.

**Meaningful risk difference:** 15 percentage points of episode-level collision risk for either primary contrast. The 10-point effect is a sensitivity target with potentially insufficient power. With approximate two-sided per-contrast alpha 0.025 and 80% power, an independent-pair normal approximation gives a required N of roughly 75–159 for a 15-point change if pairwise discordance is 0.2–0.4, versus 181–371 for a 10-point change. **These figures ignore within-route correlation.** Two stochastic repeats do not double the independent sample size. Final significance is not guaranteed by collecting 200 pairs.

### Frozen 40 episodes

- Hospital: `2,4,7,9,12,14,17,19,22,24`
- Office: `27,29,32,34,37,39,42,44,47,49`
- Warehouse: `62,64,67,69,72,74,77,79,82,84`
- Outdoor: `51,52,53,54,55,56,57,58,59,60`

Within each scene, IDs represent related route/density variants, **not 10 independent routes**. Outdoor density and distance/goal-threshold settings differ from the other environments. Analyses are stratified by scene and cluster by route-family ID. The five reviewed discovery episodes (`20,25,45,75,85`) are **excluded** from prospective tests and retained as qualitative mechanism cases.

The randomized order for all 200 episode/seed units is frozen in [`frozen_study.json`](frozen_study.json) (SHA256 recorded in `upgrade.py` output). The first 60 units are scene-balanced (15 per scene). Every matched unit runs two repeated seeds and all five conditions in randomized order. A fixed `order_seed` controls the within-unit condition order independent of VM or shard boundaries.

## 2. Get a clean L40S VM and install

This package contains **the original verified Trial 4 installer** nested in `ticvla_trial4/` plus the Trial 5 overlay. The NVIDIA driver must already work on the VM. The online bootstrap installs Git, Docker, the NVIDIA container runtime, Isaac Sim 5.0, Python deps, the pinned model and checkpoint, and the released repo. Multi-GB images and weights are **downloaded**, not physically embedded in this small ZIP. The host account needs non-interactive sudo.

**Mac terminal:**

```bash
brev refresh
brev ls
# Replace the instance with the current one displayed above
VM=ticvla-dynanav-l40s
cd ~/Downloads
unzip ticvla_trial5_confirmatory.zip
brev copy -r ~/Downloads/ticvla_trial5_bundle/ticvla_trial4 "$VM":/home/shadeform/
brev copy -r ~/Downloads/ticvla_trial5_bundle/ticvla_trial5 "$VM":/home/shadeform/
brev shell "$VM"
```

**Brev VM HOST shell:**

```bash
nvidia-smi
sudo -n true
bash ~/ticvla_trial4/bootstrap_fresh_vm.sh --plan
# Only use --accept-eula after reviewing NVIDIA's Isaac Sim license.
bash ~/ticvla_trial4/bootstrap_fresh_vm.sh --accept-eula
bash ~/ticvla_trial4/vm.sh check
python3 ~/ticvla_trial5/upgrade.py --repo ~/TIC-VLA
python3 ~/ticvla_trial5/upgrade.py --repo ~/TIC-VLA --apply
bash ~/ticvla_trial5/vm5.sh check
```

The installer refuses to overwrite an unexpected code revision, backs up changed source to `~/TIC-VLA/trial5_backups/`, and updates source SHA256 checks. The Trial 5 wrapper uses the existing tested detached Docker launcher; you never need an interactive container shell.

## 3. Five-arm smoke and schedule audit **before confirmation**

```bash
bash ~/ticvla_trial5/vm5.sh smoke t5_integration_smoke
# This is five evaluations of ep17, seed 8800 (outside the confirmatory seed set).
# The launcher prints the real log to tail.
bash ~/ticvla_trial5/vm5.sh status t5_integration_smoke
bash ~/ticvla_trial5/vm5.sh audit t5_integration_smoke
```

Verify `run_state.json` says `completed` and **all five cells are valid**. Inspect `events.jsonl`, `mechanism.json`, and `navigation/events.jsonl` in every cell. Requirements:

- Accept arm: `stale_drops == forced_drops == 0`, `results_merged == publishes_total`.
- Drop arm: `results_merged == publishes_total + stale_drops`, no forced drops, bootstrap admitted, **at least one expired completed post-bootstrap result** observed; action-head KV still supplied.
- No-KV: `action_calls_without_kv == action_calls`.
- Accept/drop: `attack_trace_sha256` matches *exactly*, and match in planned synthetic attack budget; realized completions can differ.
- Flat: planned `jobs`, `total_service_frames`, and horizon match Yo-Yo, but timing SHA differs.
- Contact logging: contact/near-actor events appear when encountered; `partner_id="unknown"` is intentional unless a separately validated raw PhysX collider hook is installed.

**If any of these fail, stop.** Do not treat an invalid or unexposed run as evidence of safety or an attack break. Do not modify the pinned code after beginning confirmatory runs; perform the fix during integration and freeze a new source hash.

## 4. Run and export in twenty independent 10-unit shards

Each shard contains **10 episode–seed units × 2 repeated runs × 5 arms = 100 rollouts**. This permits a fresh VM to process one or more shards and export before termination. Shards `t5_b00` through `t5_b19` cover exactly 200 units without overlap.

On the VM host for the FIRST shard:

```bash
bash ~/ticvla_trial5/vm5.sh run \
  --block-start 0 --block-count 10 --tag t5_b00
# The wrapper prints a path for tail -F
bash ~/ticvla_trial5/vm5.sh status t5_b00
```

Wait for `"state": "completed"` before launching the next shard. Then audit:

```bash
bash ~/ticvla_trial5/vm5.sh audit t5_b00
bash ~/ticvla_trial5/vm5.sh export t5_b00
```

This prints a `.tar.gz` plus `.sha256` in `/home/shadeform`. Download both with `brev copy VM:/home/shadeform/EXACT_FILENAME ~/Downloads/` from the **Mac**, then validate with `shasum -a 256 -c EXACT_FILENAME.sha256` on macOS. Do **not** terminate the ephemeral VM until the copy is verified. Use a second persistent storage copy.

Repeat for the rest, **one at a time**:

```text
Batch 01: --block-start 10  --block-count 10 --tag t5_b01
Batch 02: --block-start 20  --block-count 10 --tag t5_b02
...
Batch 05: --block-start 50  --block-count 10 --tag t5_b05  (60-unit interim)
...
Batch 19: --block-start 190 --block-count 10 --tag t5_b19
```

Important: `vm5.sh run` launches **detached**. Do not paste all commands in a loop without waiting for prior completion. Independent VMs may process DISJOINT shards concurrently if the GPU image and source/checkpoint hashes match; each one needs its own setup and local export. Two batches must not share one Isaac GPU container. Record GPU type, resource contention and VM image per shard.

A corrupted/failed trial does not automatically count as a safety failure. Stop on validity errors, preserve the console/error/partial results, and rerun that particular study unit under a reviewed recovery protocol rather than silently replacing or discarding the outcome. The current runner refuses to overwrite a session directory. A completed prefix can be exported even if the last child fails; coordinate targeted repairs before database lock.

### Runtime feasibility

Using the earlier 5–20 minute/rollout estimate, 2,000 rollouts require **167–667 total GPU-hours** (7–28 days of one continuously available L40S) plus provisioning/network time. This is a planning estimate, **not measured Trial 5 runtime**. A 100-rollout shard could take approximately 8–33 wall-clock hours. Check provider VM lifetime, disk space and actual L40S hourly cost before committing. Multiple VMs reduce elapsed time, **not total GPU-hours**.

## 5. Blinded interim at 60 episode–seed units

After batches `t5_b00` through `t5_b05` (600 five-arm rollouts including both stochastic repeats), collect the six exported data sets on a stable analysis computer. Use the included stdlib-only analyzer. On the VM, when all six original sessions still exist:

```bash
bash ~/ticvla_trial5/vm5.sh interim t5_b00 t5_b01 t5_b02 t5_b03 t5_b04 t5_b05
```

It outputs pairwise DISCORDANCE percentages without directional effect or arm-specific collision rates. **This is an output-masked convenience tool, not a fully blinded custodian.** For genuine blinding, an independent custodian should replace treatment names with masked labels, supply the analyst only pooled variances/discordances/intracluster estimates, and withhold the code key until the analysis is locked.

Predeclare adaptation: use discordance (and dependence across route families) to decide whether the currently planned 200 episode–seed units are sufficient for a **15-pp meaningful difference**; if they are not, add **new independent seeds** prospectively (e.g. 9006–9010) by a separately signed continuation manifest, without looking at effect direction or p-values. Do not add only the formerly colliding episodes. There is NO efficacy/futility stopping based on interim treatment effects, and no claim of achieving significance simply by choosing 200 pairs.

## 6. Final analysis and endpoints

```bash
bash ~/ticvla_trial5/vm5.sh analyze t5_b00 t5_b01 ... t5_b19
```

`analysis.py` (Python stdlib) produces paired contrasts grouped by route family; it reports an equal-family-weighted risk difference, 95% route-cluster bootstrap interval and Monte Carlo family sign-flip p-value, then adjusts the two prespecified primary p-values with Holm. This is an **initial reproducible analysis**, not a substitute for independent statistical review (few route-family clusters; path/actor heterogeneity; two stochastic repeats are nested). Export full per-rollout CSV and actual source-coded event records for secondary analysis. Primary inference should be independently checked with a route-family random-intercept model or stratified correlated-data analysis.

Also report by-arm exposure: attack arrivals, planned/observed total attacker work, benign request count, completion age, dropped/accepted counts, post-bootstrap caches used, publication cadence, and cache-use age. **Do not condition the primary randomized effect estimate on actually consuming a late cache** (post-randomization mediator); report that as a prespecified mechanistic subgroup/descriptive exposure outcome.

Contact measures: distinguish human root-centre threshold from sustained physical contact. Treat actor-specific collisions, maximum force, contact onset, high-force streak duration, near misses, minimum goal distance, progress reversal, distance to goal, route length, task duration and success as secondary/safety descriptors. No frame-level rows are independent statistical units.

## 7. State-snapshot branch replay — separate causal subexperiment, not yet implemented

The accepted-only Trial 4 logs include frames, trajectories, images and cache IDs, **not the complete live Isaac state required for a true state clone**. You cannot restore an exact branch from images alone. A snapshot should include:

1. All robot rigid-body and articulation poses, linear/angular and joint velocities, wheel-controller and filter states, control commands and force/contact state.
2. Every active pedestrian's pose, velocity, behavior command, animation/nav state, timestamps, random generator states and any contact state.
3. Sensor buffers, camera images, semantic request queue, active VLM futures/result tensors and KV-cache content, history/context, replay clock and scheduling state.
4. Python, NumPy, PyTorch, PhysX, Isaac and actor controller RNG state, GPU seeds and precise simulation step.

**Gate:** Restore the same snapshot twice under the SAME policy and establish near-identical action outputs / short-horizon trajectories before comparing interventions. If restoration is nondeterministic, perform several clones per branch and report paired distributions instead of claiming exact counterfactual identity.

For episodes 75 and 85, fork just **before** a consequential late-cache first-use (e.g. ep75 around cache c2/near stall; ep85 around c9 first-use, with particular attention to f1926). From the same restored state, compare:

- **stale:** use the exact previously computed late KV, with truthful age/pose metadata;
- **none:** `kv_cache=None`, same current image and robot state;
- **fresh:** a newly computed cache using the restored current scene (note compute-time/availability changes; treat as an oracle/reference, not an in-band available control).

Advance actor scripts identically for a short horizon with robot actions as the only intervention; record first-action waypoints/commands, heading and goal progress and first collision. Report both immediate fixed-input action difference (local causal effect of cache selection) and longer closed-loop differences, separately. Branch replay is a substantial **new robotics implementation** and MUST pass same-snapshot restoration tests before use in confirmatory claims. Do not copy/paste a guessed Isaac state serialization API and assume it captures all state.

Isaac Sim 5.0 supports getting rigid dynamic state and articulated joint state and exposes raw contact data APIs; these APIs are building blocks, **not a complete world checkpoint facility**. See official 5.0 API documentation linked in the chat response.

## 8. Claim limitations

This package upgrades a **simulated shared-work queue**. It does not submit real competing GPU inference traffic to an independently autoscaled slow service, and `flat_budget` matches virtual synthetic attacker work, not measured GPU FLOPs. A claim of a *real Yo-Yo resource-exhaustion attack* requires separate slow-service deployment, real attacker/benign load generators, resource/p99 metrics and autoscaling traces. The causal claim the current pipeline can test is narrower: **slow-path cache timing and admission policy in asynchronous TIC-VLA under controlled frame-based service contention**.
