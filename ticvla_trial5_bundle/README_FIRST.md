# TIC-VLA Trial 5 — Prospective protocol + runnable five-arm pilot

**Read `ticvla_trial5/runbook.md` first.**

This bundle includes the known Trial 4 online fresh-VM provisioning
package and a guarded Trial 5 overlay. The overlay adds `yoyo_drop`
and `flat_budget` conditions, fixed prospective selections, a randomized
2,000-run plan, and enhanced contact logging. It is NOT a container
image or model-weight archive.

**Scientific limitation:** common preplanned attack-arrival traces are
NOT guaranteed identical *realized victim VLM completion traces*.
Likewise `flat_budget` matches planned synthetic simulated service work,
not actual GPU workload; exact state-snapshot branch replay and an
oracle-KV arm are NOT included. The current package is a preregistered
five-arm emulation PROTOTYPE, not a completed conformant exact-trace
experiment. Run all validation gates before scaling up.

Install: first run `ticvla_trial4/bootstrap_fresh_vm.sh --accept-eula`
on the VM (read EULA), then `python3 ticvla_trial5/upgrade.py
--repo ~/TIC-VLA --apply`, then `bash ticvla_trial5/vm5.sh check`.

No real Isaac Sim/GPU integration rollout of this overlay was performed
by the package author.
