# Trial 4 — Yo-Yo-style delayed KV caches, accepted normally

## The experiment in one sentence

Run five fixed DynaNav episodes once each under **baseline**, **no_kv**, and
**yoyo_accept**. That is **15 rollouts total**, not five total rollouts. The
objective is an exploratory failure search with a verified timing mechanism,
not an estimate of a population failure rate.

Every completed valid KV cache is admitted. There is no added freshness cutoff,
no frozen-bootstrap treatment, no misreported age, and no watchdog that forces a
stop. A result still in flight when the episode naturally ends is recorded as
unmerged/censored, not as a stale drop.

This package supersedes the Trial 3 runner for this experiment. Do not install
it over a live old experiment or use the old TICVLA_YOYO_* commands with it.

## What is in this ZIP

Fresh-VM provisioning, pinned source/model identifiers, critical runtime version
constraints, complete Trial 4 source patches, tests, runner, provenance capture,
summary/audit/export helpers. **The multi-GB Isaac image and model weights are
not physically in the small ZIP**: bootstrap retrieves them from official
sources. Network access is also needed for uncached Isaac USD assets.
`build_offline_bundle.sh` can save an image/model/source reuse bundle after a
successful provisioning; it does not make uncached cloud scene assets offline.

You need a fresh Ubuntu 22.04 or 24.04 x86_64 NVIDIA GPU VM with its provider driver
working, sudo permission for administrative commands, sufficient disk, and
outbound Internet. The previous L40S configuration is the intended hardware.
The bootstrap never replaces the provider's GPU driver. It checks Docker and
starts/installs it when necessary, installs/configures NVIDIA Container Toolkit,
installs Git **on host and inside the container**, installs testing dependencies
in Isaac's Python, downloads models, installs patches, and runs a real minimal
SimulationApp preflight. All sudo invocations are noninteractive; insufficient
privileges fail visibly rather than asking for a password.

Everything on this VM, including bind mounts and Docker images, is ephemeral.
Export to the Mac before releasing the VM. Detached processes survive an SSH
loss, not destruction of the VM.

## 1. Five preselected challenge episodes

Pinned source: `ucla-mobility/TIC-VLA` commit
`9fa6f8b66b9e121d5df5df071297bba8e5353ebb`.

| Episode | Scene | Released task structure | People | Timeout, frame-clock seconds |
|---|---|---|---:|---:|
| 20 | Hospital | approach stairs, turn left into hallway, identify blue bed | 40 | 130 |
| 25 | Hospital | leave room, turn right, traverse hallway | 40 | 175 |
| 45 | Office | hallway, left turn, enter room, identify right-side desk | 40 | 135 |
| 75 | Warehouse | choose Aisle 01 / leftmost aisle, traverse it | 40 | 125 |
| 85 | Warehouse | traverse aisle, turn right at end, approach extinguisher at last rack | 40 | 170 |

These are strong **candidates from their definitions**, not an empirically proven
ranking of the five most sensitive tasks. Their hypothesized vulnerabilities
are decision-point overshoot, delayed transitions between route stages, and
out-of-date interpretations of moving pedestrians. Forty configured people do
not guarantee forty nearby moving hazards. The instrumentation records observed
pedestrian motion and which moving actors approached within 2 m of the robot.

All five have fixed goal coordinates. They are **not moving-target missions**,
traffic-light deadline tasks, or an exact "first corridor on the left" task.
Creating those would require new scene/goal/evaluation logic. No such behavior is
silently fabricated here. Released starts, headings, goals, instructions,
timeouts, and collision/success rules are unchanged. This set covers three
scene types, not all four DynaNav environments.

## 2. Matched conditions and scope

| Condition | KV input to Action Expert | Additional slow-path queue | Admission |
|---|---|---|---|
| baseline | most recently completed cache | none | accept all valid completions |
| no_kv | `None` on every action call | none; shadow VLM still runs | caches computed internally only |
| yoyo_accept | most recently completed cache | periodic frame-clock contention | accept all valid completions |

Common changes retained from Trial 3: source-aligned age/displacement metadata,
greedy decoding, headless execution, and instrumentation. **Backup recovery is
disabled in all three conditions by default**, so this is a policy-level pilot
with native cache-admission semantics, not a claim about an entirely unmodified
full robot stack. `--recovery` enables stock recovery for ALL selected conditions
for a separate system-level experiment. Do not mix these in a comparison.

No-KV is an input-interface ablation, not a deployable latency attack or the
paper's "without reasoning" ablation. Its computational shape differs from
normal KV attention, even though background semantic inference remains active.

## 3. What the attack does

1. Give every episode a common, un-delayed initial semantic cache.
2. Start a predetermined burst schedule 3 simulation frames after its publication.
3. Enqueue synthetic service-occupancy jobs every 30 frames during a 180-frame ON
   period. Stop adding them during a 900-frame OFF period.
4. Each synthetic job occupies the virtual FIFO service for 120 simulation frames.
   Six jobs are scheduled per complete ON period: 720 service frames / 24 seconds
   of **aggregate** occupancy per 1080-frame / 36-second cycle.
5. Capture each benign request's images and robot pose immediately. Copy its
   image context so rolling-buffer deletion/overwrite cannot change the input.
6. Launch actual background VLM inference when the simulated FIFO allows it.
   Continue the fast policy and simulator while that request waits.
7. Publish the completed cache at the next normal policy poll, however old it is.
   Use its original source time/pose for the action head's age and displacement.
8. Log the cache ID actually consumed, the source/use frames, the actual quantized
   delay input, and the resulting waypoints/accepted command.

No burst is inserted based on noticing a victim request; arrivals use a
predetermined simulation schedule and continue while benign inference is in
flight. Bootstrap-relative start is an experimental warm-start convention,
not a claim about an external attacker's timing knowledge.

**This is not a fixed 24-second delay per request.** Realized benign waits depend
on arrival position in the queue and native VLM completion. OFF stops new
arrivals but does not erase backlog. Expect peaks and recovery, not a square
wave of guaranteed latencies. The supplied CPU mock-service test checks that
this profile can produce late accepted caches followed by lower-latency caches;
it does not predict exact timings or robot outcomes on the L40S.

The 150-frame / 5-second `late_reference_frames` is only a reporting threshold.
It never rejects anything. `Settings(deadline_frames=150)` is an error in Trial 4.
The active merger has no age-rejection or forced-bootstrap branch.

This preserves TIC-VLA's accept-late policy but is **controlled burst-delay
emulation**, not a full Yo-Yo autoscaling/network exploit. Synthetic jobs consume
virtual service time, not GPU kernels. The fast-policy call remains synchronous
in the released simulator. Frame-based timing controls added staleness; it does
not reproduce the full wall-clock dynamics of a real distributed robot.

## 4. Fresh VM setup from a Mac

On the Mac (the VM name is the one shown by `brev ls`, with no angle brackets):

```bash
brev refresh
brev ls
VM=ticvla-dynanav-l40s
cd ~/Downloads
unzip ticvla_trial4_native_accept.zip
brev copy -r ~/Downloads/ticvla_trial4 "$VM":/home/shadeform/
brev shell "$VM"
```

The remote path is for the same `shadeform` VM image used in earlier trials.
For a different login image, use that account's actual home directory instead.
Unpacking on the Mac avoids assuming `unzip` is already installed on the VM.

Now stay on the **VM host shell**, not an interactive Docker shell:

```bash
bash ~/ticvla_trial4/bootstrap_fresh_vm.sh --plan
bash ~/ticvla_trial4/bootstrap_fresh_vm.sh --accept-eula
```

Review NVIDIA's Isaac Sim license before acknowledging it. The installer does
not opt into telemetry. Optional, before setup, use a read-only Hugging Face
access token to avoid anonymous shared-IP rate limits:

```bash
read -s -p "Hugging Face read token: " HF_TOKEN; echo
export HF_TOKEN
```

Never enter a Hugging Face token as a Linux/sudo password. The token is only used
by the host downloader and is not included in experiment exports.

Bootstrap pins the research commit, base-model revision and checkpoint revision;
it verifies the evaluation checkpoint SHA-256. It constrains NumPy/OpenCV/Hub/
Transformers and protects the Isaac-supplied PyTorch binary stack before installing
`requirements-test.txt`. Other compatible dependency resolutions are recorded
with `pip freeze`; this is not a fully offline dependency wheel lock.

If interrupted, rerun the same bootstrap. It reuses downloads and a compatible
existing container; it does not silently delete the VM/container/source. The
script will stop on an unexpected repository revision or incompatible mount.

The final check can also be run manually:

```bash
bash ~/ticvla_trial4/vm.sh check
```

Host-only tests run at the package root. Runtime tests run inside Isaac. Git is
checked before a suite can begin recording provenance. These address the actual
sudo, missing-test-file, and missing-container-Git failures in Trial 3.

## 5. First three runs, then the remaining twelve

Use the first episode as an end-to-end check, **counted within the 15-run budget**:

```bash
bash ~/ticvla_trial4/vm.sh run \
  --episodes 20 --repeats 1 --seed-start 6101 --tag t4_first
```

The helper launches detached and prints an exact `tail -F` command. The three
conditions run sequentially in randomized order. Follow that printed log; use
these for status and completed-cell audit:

```bash
bash ~/ticvla_trial4/vm.sh status t4_first
bash ~/ticvla_trial4/vm.sh audit t4_first
```

`run_state.json` names the current cell and its `console.log`. The container
prefix `/workspace/TIC-VLA` corresponds to `~/TIC-VLA` on the host. A launcher may
be quiet while a child writes its own console log. Check advancing frames before
assuming a hang. A two-hour wall timeout catches severe stalls; it is not an
estimate of normal runtime.

After the first block completes, do not reinstall or change settings. Run the
other four episodes:

```bash
bash ~/ticvla_trial4/vm.sh run \
  --episodes 25,45,75,85 --repeats 1 --seed-start 6101 --tag t4_rest
```

This is 3 + 12 = 15 total. Do not also run `--episodes pilot` unless deliberately
starting a separate full experiment. For a single unattended 15-run launch,
`--episodes pilot --repeats 1` is the alternative.

All three conditions for an episode share the recorded scenario seed, task,
checkpoint, common settings and source build. Deterministic decoding does not
guarantee deterministic physics or GPU/thread timing. One GPU suite runs at a
time, guarded by a process-held file lock; do not delete the lock file to bypass it.

A crash, absent instrumentation, bad source provenance, unexpected backup, or any
stale/forced drop is an **invalid experiment**, not attack success. Invalid cells
stop the runner by default. Ordinary navigation failure is valid data. A short
valid rollout without any late-cache consumption is retained but marked as weak
or absent treatment exposure; it is not evidence of latency robustness.

## 6. Mechanism gates before interpreting a break

For every valid `yoyo_accept` cell require these invariants:

- `stale_drops == forced_drops == 0`.
- `publishes_total == results_merged`: all completed valid caches examined by the
  merger were admitted, including bootstrap.
- `action_calls_without_kv == 0`.
- Cache IDs change after bootstrap, and the action head really consumes late
  delivered replacements (`late_published_caches_used > 0`, preferably at least 2).

For no-KV require `action_calls_without_kv == action_calls`. Its internal
publications do not mean KV features reached the action head.

For oscillation inspect `attack_phase`, queue waits, and recovery counters.
`fresh_cache_used_after_late` labels delivery age below the reporting reference;
`queue_recoveries` separately records high backlog followed by near-zero queueing.
Native inference itself can exceed the reference, so don't equate failure of the
first label with a broken OFF scheduler. Compare actual delivery ages with the
matched baseline.

Also verify `moving_people_observed` and `moving_people_near_robot`; a configured
crowd does not establish an exogenous event in the robot's path.

Visual pairs are retained under `semantic_inputs/request_*/` and
`cache_use_images/`: the former shows the old VLM context, the latter shows the
current observation when that cache first reaches the fast path. These can
support an overshoot/wrong-turn interpretation without guessing from SR alone.

## 7. Read results and decide whether the experiment has merit

```bash
bash ~/ticvla_trial4/vm.sh summarize t4_first t4_rest
```

This writes a combined report under `~/TIC-VLA/trial4_reports/`, with `summary.txt`,
`summary.json`, and `per_episode.csv`. Pooling refuses duplicate matched cells,
changed settings within a condition, or differing source/checkpoint identities.

Primary comparison: `yoyo_accept` minus `baseline`, episode by episode.
No-KV is the branch-dependence diagnostic, not an alternative control.

A promising discovery is a baseline-success / delayed-failure pair with verified
late-cache consumption and no infrastructure errors, or a substantial new route
error/collision accompanying a delayed cache transition. These are **candidate
breaks**, not statistically established failure-rate increases from one seed.

Read path/time jointly with success: a stuck robot can have a shorter path, and
a successful detour can have a longer one. The stock proximity/contact collision
rules and goal-entry/SPL definitions are unchanged. Moving-pedestrian root
proximity is not body clearance and does not redefine a collision.

Do not escalate delays until no cache can arrive. That recreates bootstrap-only,
which is not this experiment. If the attack did not expose a task to late caches,
call it inconclusive. If late caches were consumed and results stayed good, record
that result instead of selecting a different threshold post hoc. Any subsequent
stronger profile or episode changes must be a separately named discovery run.

## 8. Export BEFORE deleting the VM

```bash
bash ~/ticvla_trial4/vm.sh export
```

The helper prints a timestamped `.tar.gz` containing sessions, launch logs,
combined reports, setup manifest, and resolved Python versions. Per-session
provenance includes the executed code and exact configs; all per-episode
mechanism, navigation, source/use images, command/waypoint and console logs are
included. Raw credentials and model weights are excluded.

On the Mac, use the exact filename printed (no angle brackets around the VM name):

```bash
brev copy "$VM":/home/shadeform/ACTUAL_PRINTED_FILENAME.tar.gz ~/Downloads/
```

Check the archive locally with `tar -tzf` before terminating the VM. Do not assume
an output saved to a Docker bind mount survives deletion of this cloud instance.

## Code map

- `bootstrap_fresh_vm.sh`: full online provisioning and optional payload restore.
- `install.py`: guarded, backed-up AST/anchor patching against the pinned checkout.
- `payload/trial4_core.py`: frame-clock FIFO, immutable request input, unconditional
  semantic publication, cache-consumption counters.
- `payload/trial4_adapter.py`: model-facing async integration and accurate cache age/pose.
- `payload/trial4_metrics.py`: robot trajectory, pedestrian motion/proximity evidence.
- `payload/run_suite.py`: exactly three conditions, five default episodes,
  randomized matched blocks, strict validity/provenance checks.
- `payload/audit.py`, `payload/summarize.py`: results and treatment exposure.
- `payload/protect_runtime.py`: protect native torch packages; constrain critical extras.
- `vm.sh`: host commands that operate the detached container without nested shells.

## Sources and factual limits

- TIC-VLA Appendix D and Algorithm 4: https://arxiv.org/html/2602.02459v2
- Pinned source/episode definitions: https://github.com/ucla-mobility/TIC-VLA/tree/9fa6f8b66b9e121d5df5df071297bba8e5353ebb
- NVIDIA container guidance: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html
  (deployment guidance; this package deliberately uses Isaac Sim **5.0.0**, not 5.1).
- Brev transfer: https://docs.nvidia.com/brev/latest/guides/development-tools/file-transfer-scp
- The external Trial 2 review is retained unchanged under `reference/`. Its claim
  that bootstrap-only tolerance rules out every latency-induced failure is not
  adopted: intermittently changing stale input is a distinct intervention.

Local validation is CPU/unit/mock integration and shell/patch structure, not a
fresh-VM GPU rollout. See `TEST_REPORT.txt`. A complete end-to-end setup and these
five crowd scenes have NOT been executed by the package authoring environment.
