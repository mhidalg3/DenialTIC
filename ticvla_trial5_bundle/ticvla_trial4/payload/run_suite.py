#!/usr/bin/env python3
"""Run sequential, randomized, matched Trial 4 episodes inside the Isaac container."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

DYNANAV = Path(__file__).resolve().parents[1]
REPO = DYNANAV.parent
sys.path.insert(0, str(DYNANAV))
from trial4_core import Settings, atomic_json
import yaml

# Selected before observing Trial 4 outcomes. All are retained in reports.
# Avoid earlier 66/70 pairing: their initial headings differ in the released YAML.
SEMANTIC = [20, 25, 45, 75, 85]
OLD = [1, 27, 51, 65]
REQUIRED_ENV = ["TICVLA_BASE_MODEL_PATH", "TICVLA_CHECKPOINT_PATH"]


def conditions(stage, args):
    # Exactly the three requested conditions. No branch admits a deadline policy.
    base = Settings(backup_enabled=args.recovery, deterministic=not args.sample,
                    retain_visuals=not args.no_visuals)
    attack = replace(base, attack="periodic", on_frames=args.on_frames,
                     off_frames=args.off_frames,
                     arrival_interval_frames=args.arrival_frames,
                     attack_service_frames=args.service_frames,
                     warmup_frames=args.warmup_frames)
    return {"baseline": base, "no_kv": replace(base, cache_mode="none"),
            "yoyo_accept": attack}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4*1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sanitize(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize(v) for k,v in value.items()}
    return value


def validate_result(raw, status, cfg):
    errors = []
    if status != 0:
        errors.append(f"child_exit={status}")
    if raw.get("error"):
        errors.append(str(raw["error"]))
    if raw.get("duration_frames", 0) <= 0:
        errors.append("no_simulation_frames")
    if not isinstance(raw.get("navigation_error"), (int,float)) or not math.isfinite(raw["navigation_error"]):
        errors.append("no_finite_navigation_error")
    mech = raw.get("trial4", {}).get("mechanism", {})
    counts = mech.get("counts", {})
    if not mech.get("instrumentation_valid"):
        errors.append("missing_or_invalid_mechanism")
    if counts.get("action_calls", 0) <= 0:
        errors.append("no_instrumented_action_calls")
    if counts.get("commands_accepted", 0) <= 0:
        errors.append("no_instrumented_accepted_commands")
    if cfg.cache_mode == "none" and counts.get("action_calls_without_kv") != counts.get("action_calls"):
        errors.append("no_kv_invariant_failed")
    if cfg.cache_mode == "normal" and counts.get("action_calls_without_kv", -1) != 0:
        errors.append("unexpected_no_kv_action")
    if counts.get("stale_drops", -1) != 0 or counts.get("forced_drops", -1) != 0:
        errors.append("accept_late_invariant_failed")
    if counts.get("publishes_total") != counts.get("results_merged"):
        errors.append("completed_cache_not_published")
    if not cfg.backup_enabled and mech.get("backup_events", 0):
        errors.append("backup_disable_invariant_failed")
    if mech.get("settings") != asdict(cfg):
        errors.append("observed_settings_mismatch")
    return errors


def prepare_config(full):
    out = {k:v for k,v in full.items() if not k.startswith("episode_")}
    for n in SEMANTIC:
        out[f"episode_{n}"] = full[f"episode_{n}"]
    path = DYNANAV/"configs/trial4_semantic_stress.yaml"
    path.write_text(yaml.safe_dump(out, sort_keys=False))
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--stage", choices=["native"], default="native", help="native accept-late comparison only")
    p.add_argument("--only", help="optional comma-separated subset of baseline,no_kv,yoyo_accept")
    p.add_argument("--episodes", default="pilot", help="pilot (20,25,45,75,85) | comma-separated ids")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed-start", type=int, default=6101)
    p.add_argument("--order-seed", type=int, default=20261016)
    p.add_argument("--tag", help="unique session name; defaults to a UTC timestamp")
    p.add_argument("--recovery", action="store_true", help="secondary whole-system evaluation")
    p.add_argument("--sample", action="store_true", help="use stochastic VLM decoding")
    p.add_argument("--no-visuals", action="store_true", help="disable source/use JPEG evidence for ALL conditions")
    p.add_argument("--warmup-frames", type=int, default=3)
    p.add_argument("--on-frames", type=int, default=180)
    p.add_argument("--off-frames", type=int, default=900)
    p.add_argument("--arrival-frames", type=int, default=30)
    p.add_argument("--service-frames", type=int, default=120)
    p.add_argument("--wall-timeout", type=int, default=7200)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()
    if args.repeats < 1 or args.wall_timeout < 1:
        p.error("repeats and wall-timeout must be positive")
    if shutil.which("git") is None:
        raise SystemExit("Git is missing INSIDE the Isaac container. Re-run bootstrap; git is installed there as well as on the host.")
    if git("rev-parse", "HEAD") != "9fa6f8b66b9e121d5df5df071297bba8e5353ebb":
        raise SystemExit("Unexpected source commit; refusing mixed-revision experiment")
    cfg_file = DYNANAV/"configs/benchmark_full.yaml"
    if git("hash-object", str(cfg_file)) != "34b864a5ad21dbfbdbabec6fc35bd44b77fd3c25":
        raise SystemExit("Released benchmark_full.yaml was modified; retain the pinned definitions for this pilot.")
    full = yaml.safe_load(cfg_file.read_text())
    prepare_config(full)
    if args.check:
        for key in REQUIRED_ENV:
            if not os.environ.get(key) or not Path(os.environ[key]).exists():
                raise SystemExit(f"Missing path/environment variable: {key}; source .env.container inside Docker")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise SystemExit("Set CUDA_VISIBLE_DEVICES=0 for the single-L40S setup")
        for name in ("ticvla.py", "benchmark.py", "behavior/nova_carter_test_ticvla.py"):
            s = (DYNANAV/name).read_text()
            if "TICVLA_TRIAL4_INSTALLED" not in s:
                raise SystemExit(f"Trial 4 hook missing: {name}")
            compile(s, name, "exec")
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable inside Isaac Python")
        manifest_path = DYNANAV/"trial4/install_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for rel, entry in manifest["files"].items():
            if sha(REPO/rel) != entry["installed_sha256"]:
                raise SystemExit(f"Installed source changed since installation: {rel}")
        print("Paths, git, source hashes, hooks, syntax and CUDA: PASS. GPU:", torch.cuda.get_device_name(0))
        print("This preflight does not run an Isaac Sim episode.")
        return
    selected = SEMANTIC if args.episodes in {"pilot", "semantic"} else [int(x.strip()) for x in args.episodes.split(",")]
    if len(selected) != len(set(selected)) or not selected:
        raise SystemExit("Episode list is empty or contains duplicates")
    for n in selected:
        if f"episode_{n}" not in full:
            raise SystemExit(f"Unknown episode {n}")
    variants = conditions(args.stage, args)
    if args.only:
        requested = args.only.split(",")
        if len(set(requested)) != len(requested) or any(k not in variants for k in requested):
            raise SystemExit(f"--only must name distinct available conditions: {list(variants)}")
        variants = {k: variants[k] for k in requested}
    blocks = [(n, args.seed_start+r) for n in selected for r in range(args.repeats)]
    rng = random.Random(args.order_seed)
    rng.shuffle(blocks)
    plan = []
    for episode, seed in blocks:
        names = list(variants)
        rng.shuffle(names)
        plan.extend({"episode": episode, "seed": seed, "condition": name,
                     "settings": asdict(variants[name])} for name in names)
    tag = args.tag or ("t4_"+args.stage+"_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
        raise SystemExit("tag may contain only letters, numbers, underscore and hyphen")
    root = REPO/"trial4_results"/tag
    try:
        root.mkdir(parents=True, exist_ok=False)  # never mix old records silently
    except FileExistsError:
        raise SystemExit(f"Session already exists: {root}. Use a NEW tag; do not delete old data.")
    atomic_json(root/"plan.json", {"args": vars(args), "jobs": plan,
                                  "statistical_scope": "fixed episode set; paired scene seeds; exploratory unless held-out"})
    print(f"Session: {root}\nJobs: {len(plan)} (sequential, randomized within matched blocks)", flush=True)
    if args.dry_run:
        print("Dry run: plan written; no model loads or simulator runs.")
        return
    lock_file = (REPO/".trial4_suite.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another Trial 4 suite is active. Run only one suite on this L40S.")
    atomic_json(root/"run_state.json", {"state": "recording_provenance", "completed": 0, "total": len(plan)})
    provenance = root/"provenance"
    provenance.mkdir()
    code_paths = [DYNANAV/"ticvla.py", DYNANAV/"benchmark.py",
                  DYNANAV/"behavior/nova_carter_test_ticvla.py", cfg_file,
                  DYNANAV/"run_benchmark.sh", *DYNANAV.glob("trial4_*.py"),
                  *Path(__file__).parent.glob("*.py")]
    hashes = {}
    for path in code_paths:
        relative = path.relative_to(REPO)
        target = provenance/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = sha(path)
    (provenance/"git_commit.txt").write_text(git("rev-parse", "HEAD")+"\n")
    (provenance/"code_diff.patch").write_text(git("diff", "HEAD", "--", "DynaNav", "requirements-test.txt"))
    setup = REPO/"trial4_setup_manifest.json"
    if setup.exists():
        shutil.copy2(setup, provenance/setup.name)
    install_manifest = DYNANAV/"trial4/install_manifest.json"
    shutil.copy2(install_manifest, provenance/"install_manifest.json")
    model = Path(os.environ["TICVLA_CHECKPOINT_PATH"])
    atomic_json(provenance/"manifest.json", {"source_hashes": hashes,
                "checkpoint_path": str(model), "checkpoint_size": model.stat().st_size,
                "checkpoint_sha256": sha(model),
                "base_model_path": os.environ.get("TICVLA_BASE_MODEL_PATH"),
                "python": sys.version, "warning": "No tokens or full environment files exported"})
    try:
        import torch, transformers
        atomic_json(provenance/"runtime.json", {"torch": torch.__version__, "transformers": transformers.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    except Exception as exc:
        raise SystemExit(f"Dependency/GPU check failed: {exc}")
    (provenance/"pip_freeze.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"], text=True))
    results = []
    state = {"state": "running", "completed": 0, "total": len(plan)}
    atomic_json(root/"run_state.json", state)
    for index, job in enumerate(plan):
        for rel, digest in hashes.items():
            if sha(REPO/rel) != digest:
                raise SystemExit(f"Code changed mid-suite: {rel}. Refusing a mixed-build comparison.")
        name = f"{index:04d}_{job['condition']}_ep{job['episode']}_seed{job['seed']}"
        folder = root/name
        folder.mkdir()
        ep = f"episode_{job['episode']}"
        yaml_config = {k:v for k,v in full.items() if not k.startswith("episode_")}
        yaml_config.update(seed=job["seed"], navigation_method="ticvla")
        yaml_config[ep] = full[ep]
        config_path = folder/"episode.yaml"
        config_path.write_text(yaml.safe_dump(yaml_config, sort_keys=False))
        run_id = f"t4_{tag}_{name}"
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(("TICVLA_YOYO_", "TICVLA_T3_", "TICVLA_T4_")) or key in {
                "TICVLA_INJECT_QUEUE_DELAY_S", "TICVLA_DEADLINE_SIM_S", "TICVLA_ACTION_CACHE_MODE", "TICVLA_DETERMINISTIC"}:
                env.pop(key)
        env.update(TICVLA_T4_CONFIG_JSON=json.dumps(job["settings"]), TICVLA_T4_OUTPUT=str(folder),
                   TICVLA_EPISODE_NAME=ep, BENCHMARK_RUN_ID=run_id, PYTHONUNBUFFERED="1",
                   TICVLA_T4_SEED=str(job["seed"]), TICVLA_DYNANAV_ROOT=str(DYNANAV))
        raw_path = folder/"original_result.json"
        cmd = [sys.executable, str(DYNANAV/"benchmark.py"), "-c", str(config_path),
               "--navigation_method", "ticvla", "--child", "--episode_name", ep,
               "--result_json", str(raw_path), "--run_id", run_id]
        atomic_json(folder/"launch.json", {**job, "command": cmd, "run_id": run_id,
                    "seed_note": "scene config, Python, NumPy and torch seeds set; thread/physics determinism not guaranteed"})
        state.update(current_job=name, console_log=str(folder/"console.log"))
        atomic_json(root/"run_state.json", state)
        print(f"[{index+1}/{len(plan)}] {name}", flush=True)
        began = time.perf_counter()
        with (folder/"console.log").open("w") as log:
            try:
                child = subprocess.run(cmd, cwd=DYNANAV, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=args.wall_timeout)
                code = child.returncode
            except subprocess.TimeoutExpired:
                code = 124
        raw = {}
        if raw_path.exists():
            try:
                raw = json.loads(raw_path.read_text())
            except Exception as exc:
                raw = {"error": f"result_parse_error: {exc}"}
        errors = validate_result(raw, code, variants[job["condition"]])
        for required in ("events.jsonl", "mechanism.json", "navigation/events.jsonl"):
            if not (folder/required).exists():
                errors.append(f"missing_artifact:{required}")
        mech = raw.get("trial4", {}).get("mechanism", {})
        counts = mech.get("counts", {})
        nav = raw.get("trial4", {}).get("navigation", {})
        exposure = {
            "late_reference_frames": variants[job["condition"]].late_reference_frames,
            "note": "Reference age is a measurement label, NOT an admission deadline.",
            "attack_arrivals_observed": counts.get("attack_jobs_enqueued", 0) > 0,
            "late_cache_consumed": counts.get("late_published_caches_used", 0) > 0,
            "multiple_late_caches_consumed": counts.get("late_published_caches_used", 0) >= 2,
            "fresh_recovery_consumed": counts.get("fresh_cache_used_after_late", 0) > 0,
            "queue_recovery_observed": counts.get("queue_recoveries", 0) > 0,
            "moving_people_observed": nav.get("pedestrians_moved_over_0_25m", 0) > 0,
            "moving_people_near_robot": nav.get("moving_pedestrians_within_2m", 0) > 0,
        }
        wrapped = {**job, "exposure": exposure, "elapsed_wall_s": time.perf_counter()-began, "exit_code": code,
                   "valid": not errors, "validation_errors": errors, "result": sanitize(raw)}
        atomic_json(folder/"result.json", wrapped)
        results.append(wrapped)
        atomic_json(root/"results.json", results)
        state["completed"] = len(results)
        atomic_json(root/"run_state.json", state)
        print(f"  valid={not errors} navigation_success={raw.get('success')} errors={errors}", flush=True)
        if errors and not args.continue_on_error:
            state["state"] = "stopped_on_invalid_episode"
            atomic_json(root/"run_state.json", state)
            raise SystemExit(f"Inspect {folder}/console.log before proceeding.")
    state["state"] = "completed"
    state.pop("current_job", None)
    atomic_json(root/"run_state.json", state)
    print(f"Finished: {root}. All raw logs, code and mechanism records are under this directory.", flush=True)


if __name__ == "__main__":
    main()
