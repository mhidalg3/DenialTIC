"""Trial 4's CPU-testable, simulated-frame slow-path scheduler.

This is a local fault-injection/emulation harness. Synthetic queue jobs consume
SIMULATION FRAMES, not GPU work, network bandwidth, or an autoscaled service.
Only the real VLM job executes on a background thread. tick() must be called by
the simulator thread; it never waits for a frame or an unfinished future.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
import copy
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
from typing import Any, Callable

FPS = 30.0


@dataclass(frozen=True)
class Settings:
    cache_mode: str = "normal"             # normal | none; no bootstrap freeze mode
    launch_delay_frames: int = 0
    deadline_frames: int = 0                # 0=accept; >0=experimental drop-at-merge policy
    late_reference_frames: int = 150       # analysis label ONLY, never a rejection gate
    retain_visuals: bool = True
    attack: str = "none"                    # none | periodic | sustained | flat
    on_frames: int = 180
    off_frames: int = 900
    arrival_interval_frames: int = 30
    attack_service_frames: int = 120
    max_attack_jobs: int = 64
    warmup_frames: int = 3                  # after bootstrap, common to all modes
    attack_horizon_frames: int = 0         # fixed evaluation horizon for paired workload budget
    deterministic: bool = True
    backup_enabled: bool = False
    bootstrap_timeout_s: float = 120.0
    near_miss_m: float = 0.6                # root-centre proximity proxy only

    def __post_init__(self):
        if self.cache_mode not in {"normal", "none"}:
            raise ValueError("Trial 4 cache_mode must be normal or none; freezing is not a latency experiment")
        # A zero deadline preserves TIC-VLA native acceptance. A positive value
        # is an experimental victim policy and is changed ONLY in the drop arm.
        if self.attack != "none" and self.launch_delay_frames != 0:
            raise ValueError("Use either a fixed launch hold or the queue emulator, not both")
        if self.attack not in {"none", "periodic", "sustained", "flat"}:
            raise ValueError("attack must be none, periodic, sustained, or flat")
        for name in ("launch_delay_frames", "deadline_frames", "on_frames",
                     "off_frames", "warmup_frames", "late_reference_frames", "attack_horizon_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("arrival_interval_frames", "attack_service_frames", "max_attack_jobs"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.attack in {"periodic", "flat"} and (self.on_frames <= 0 or self.off_frames <= 0):
            raise ValueError("periodic attack requires positive ON and OFF durations")
        if not math.isfinite(self.bootstrap_timeout_s) or self.bootstrap_timeout_s <= 0:
            raise ValueError("bootstrap_timeout_s must be finite and positive")
        if not math.isfinite(self.near_miss_m) or self.near_miss_m <= 0:
            raise ValueError("near_miss_m must be finite and positive")

    @classmethod
    def from_env(cls):
        raw = os.environ.get("TICVLA_T4_CONFIG_JSON")
        if not raw:
            raise RuntimeError("Use Trial 4's runner: TICVLA_T4_CONFIG_JSON is required")
        return cls(**json.loads(raw))


def distribution(values):
    x = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not x:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    def pct(p):
        q = (len(x) - 1) * p
        lo, hi = math.floor(q), math.ceil(q)
        return x[lo] + (x[hi] - x[lo]) * (q - lo)
    return {"n": len(x), "mean": sum(x)/len(x), "p50": pct(.5), "p95": pct(.95), "max": x[-1]}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


class EventLog:
    """JSONL events have an episode id, monotonic timestamp and ordered sequence."""
    def __init__(self, directory, episode="unknown", run="unknown"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.episode, self.run = episode, run
        self.lock = threading.Lock()
        self.seq = 0
        self.file = (self.directory / "events.jsonl").open("a", buffering=1)

    def emit(self, event, **fields):
        with self.lock:
            self.seq += 1
            record = {"event": event, "sequence": self.seq,
                      "wall_monotonic_s": time.perf_counter(),
                      "episode": self.episode, "run": self.run, **fields}
            def convert(value):
                if hasattr(value, "tolist"):
                    return value.tolist()
                if isinstance(value, Path):
                    return str(value)
                raise TypeError(f"Non-serializable event value: {type(value).__name__}")
            self.file.write(json.dumps(record, allow_nan=False, default=convert) + "\n")


@dataclass
class Request:
    id: int
    source_step: int
    source_pose: dict | None
    source_wall: float
    input_paths: tuple[str, ...]
    input_steps: tuple[int | None, ...]
    instruction: str
    generation_config: dict
    previous_text: str
    robot_type: str
    bootstrap: bool
    snapshot_dir: str
    ready_at: int
    enqueued_on: bool
    future: Any = None
    launch_step: int | None = None
    launch_wall: float | None = None
    ready_wall: float | None = None
    first_ready_observed_step: int | None = None


class SlowPath:
    """One outstanding benign request, FIFO frame-time attacker jobs, one VLM worker."""
    def __init__(self, settings, generate, log, executor=None):
        self.cfg: Settings = settings
        self.generate: Callable = generate
        self.log: EventLog = log
        self.executor = executor or ThreadPoolExecutor(max_workers=1)
        self.queue = deque()
        self.running = None                 # (kind, payload, end_frame)
        self.ready = deque()
        self.outstanding: Request | None = None
        self.latest: tuple[Request, str, Any] | None = None
        self.origin = None
        self.next_attack = None
        self.attack_offsets = None
        self.attack_cursor = 0
        self.attack_trace_hash = None
        self.last_step = None
        self.next_id = 1
        self.closed = False
        self.fatal = None
        self.counts = Counter()
        self.samples = {key: [] for key in (
            "queue_wait_frames", "queue_wait_wall_s", "service_wall_s",
            "ready_to_merge_wall_s", "delivery_age_frames", "delivery_wall_s",
            "action_cache_age_frames", "action_cache_age_wall_s", "model_delay_input_s",
            "post_bootstrap_delivery_age_frames", "post_bootstrap_queue_wait_frames",
            "fast_call_wall_s", "publication_gap_frames")}
        self.last_publish_step = None
        self.last_action_cache_id = None
        self.used_cache_ids = set()
        self.late_published_ids = set()
        self.used_late_ids = set()
        self.last_delivery_was_late = False
        self.last_delivery_had_high_queue = False
        self.last_used_delivery_was_late = False
        self.last_phase = "OFF"
        self.log.emit("configuration", settings=asdict(settings), clock="DynaNav frame / 30")

    def _init_attack_plan(self):
        """Construct a fixed frame-based arrival plan independent of benign calls.

        Periodic and flat produce identical synthetic-job counts and service work
        over the predeclared horizon. They differ only in temporal concentration.
        """
        if self.cfg.attack not in {"periodic", "flat"} or self.cfg.attack_horizon_frames <= 0:
            return
        offsets = []
        period = self.cfg.on_frames + self.cfg.off_frames
        horizon = self.cfg.attack_horizon_frames
        for start in range(0, horizon, period):
            block = min(period, horizon - start)
            burst = list(range(0, min(self.cfg.on_frames, block), self.cfg.arrival_interval_frames))
            count = len(burst)
            if self.cfg.attack == "periodic":
                current = burst
            else:
                current = [min(block-1, (2*j+1)*block // (2*count)) for j in range(count)]
            offsets.extend(start + x for x in current)
        self.attack_offsets = offsets
        self.attack_cursor = 0
        budget = {"jobs":len(offsets),"service_frames_each":self.cfg.attack_service_frames,
                  "total_service_frames":len(offsets)*self.cfg.attack_service_frames,
                  "horizon_frames":horizon}
        self.attack_trace_hash = hashlib.sha256(json.dumps(
            {"offsets":offsets,"service_frames":self.cfg.attack_service_frames},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.attack_budget = budget
        self.log.emit("attack_plan", step=self.origin, trace_sha256=self.attack_trace_hash,
                      budget=budget, attack=self.cfg.attack)

    def phase(self, step):
        if self.cfg.attack == "none" or self.origin is None or step < self.origin:
            return "OFF"
        if self.cfg.attack == "sustained":
            return "ON"
        if self.cfg.attack == "flat":
            return "FLAT"
        return "ON" if (step-self.origin) % (self.cfg.on_frames+self.cfg.off_frames) < self.cfg.on_frames else "OFF"

    def _schedule_attack(self, step):
        if self.cfg.attack == "none" or self.origin is None:
            return
        if self.attack_offsets is not None:
            due_list = []
            while self.attack_cursor < len(self.attack_offsets):
                due = self.origin + self.attack_offsets[self.attack_cursor]
                if due > step: break
                due_list.append(due)
                self.attack_cursor += 1
        else:
            if self.next_attack is None:
                return
            due_list = []
            while self.next_attack <= step:
                due = self.next_attack
                self.next_attack += self.cfg.arrival_interval_frames
                if self.phase(due) == "ON": due_list.append(due)
        for due in due_list:
            in_system = sum(kind == "attack" for kind, _ in self.queue)
            in_system += int(self.running is not None and self.running[0] == "attack")
            if in_system >= self.cfg.max_attack_jobs:
                self.counts["attack_capacity_drops"] += 1
                self.log.emit("attack_capacity_drop", step=step, scheduled_step=due)
                continue
            self.counts["attack_jobs_enqueued"] += 1
            attack_id = self.counts["attack_jobs_enqueued"]
            self.queue.append(("attack", (attack_id, due)))
            self.counts["max_queue_depth"] = max(self.counts["max_queue_depth"], len(self.queue))
            self.log.emit("attack_enqueue", attack_id=attack_id, step=step,
                          scheduled_step=due, schedule_lateness_frames=step-due,
                          queue_depth=len(self.queue))

    def tick(self, step):
        if self.closed:
            return
        step = int(step)
        if self.last_step is not None and step < self.last_step:
            raise RuntimeError(f"Frame clock moved backwards: {self.last_step} -> {step}")
        self.last_step = step
        phase = self.phase(step)
        if phase != self.last_phase:
            self.log.emit("attack_phase", step=step, phase=phase)
            self.last_phase = phase
        self._schedule_attack(step)
        if self.running is not None:
            kind, payload, end = self.running
            if kind == "attack":
                if step < end:
                    return
                self.counts["attack_jobs_completed"] += 1
                self.log.emit("attack_complete", attack_id=payload[0], step=step)
                self.running = None
            else:
                req = payload
                if not req.future.done():
                    return
                req.first_ready_observed_step = step
                self.ready.append(req)
                self.running = None
        while self.running is None and self.queue:
            kind, payload = self.queue[0]
            if kind == "benign":
                req = payload
                if step < req.ready_at:
                    return
                self.queue.popleft()
                req.launch_step, req.launch_wall = step, time.perf_counter()
                self.counts["service_started"] += 1
                self.log.emit("service_start", request_id=req.id, step=step,
                              source_step=req.source_step, bootstrap=req.bootstrap,
                              requested_launch_hold_frames=req.ready_at-req.source_step,
                              actual_queue_wait_frames=step-req.source_step,
                              queue_wait_wall_s=req.launch_wall-req.source_wall)
                req.future = self.executor.submit(self._generate, req)
                self.running = ("benign", req, None)
                return
            self.queue.popleft()
            self.running = ("attack", payload, step+self.cfg.attack_service_frames)
            self.log.emit("attack_start", attack_id=payload[0], step=step,
                          scheduled_step=payload[1], service_frames=self.cfg.attack_service_frames)

    def _generate(self, req):
        try:
            response, cache = self.generate(req)
            req.ready_wall = time.perf_counter()
            self.log.emit("service_ready", request_id=req.id,
                          service_wall_s=req.ready_wall-req.launch_wall,
                          has_cache=cache is not None)
            return response, cache
        except Exception as exc:
            req.ready_wall = time.perf_counter()
            self.log.emit("service_error", request_id=req.id, error=repr(exc))
            raise
        finally:
            shutil.rmtree(req.snapshot_dir, ignore_errors=True)

    def offer(self, step, pose, paths, instruction, generation_config, previous_text, robot_type):
        if self.closed:
            raise RuntimeError("SlowPath is closed")
        if self.outstanding is not None:
            return None
        if not paths:
            raise ValueError("Empty VLM image context")
        step = int(step)
        start_wall = time.perf_counter()
        image_steps = tuple(int(m.group(1)) if (m := re.search(r"frame_(\d+)", Path(p).name)) else None for p in paths)
        # The released behavior captures its latest frame immediately before the call.
        # Do not silently label an older capture with a current pose.
        if image_steps[-1] is not None and image_steps[-1] != step:
            raise ValueError(f"Newest input frame {image_steps[-1]} != pose/source frame {step}; check camera provenance")
        request_id = self.next_id
        self.next_id += 1
        temp = tempfile.mkdtemp(prefix=f"ticvla_t4_{request_id}_")
        try:
            copied = []
            for i, src in enumerate(paths):
                dst = Path(temp) / f"{i:02d}_{Path(src).name}"
                shutil.copy2(src, dst)      # real copy: safe against delete AND overwrite
                copied.append(str(dst))
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        if self.cfg.retain_visuals:
            evidence = self.log.directory / "semantic_inputs" / f"request_{request_id:06d}"
            evidence.mkdir(parents=True, exist_ok=False)
            for original, copied_path in zip(paths, copied):
                shutil.copy2(copied_path, evidence / Path(copied_path).name)
        bootstrap = self.latest is None
        req = Request(request_id, step, copy.deepcopy(pose), start_wall,
                      tuple(copied), image_steps, instruction, copy.deepcopy(generation_config),
                      previous_text, robot_type, bootstrap, temp,
                      step + (0 if bootstrap else self.cfg.launch_delay_frames),
                      self.phase(step) == "ON")
        self.outstanding = req
        self.counts["requests_enqueued"] += 1
        self.counts["requests_enqueued_during_on"] += int(req.enqueued_on)
        self.queue.append(("benign", req))
        self.counts["max_queue_depth"] = max(self.counts["max_queue_depth"], len(self.queue))
        self.log.emit("benign_enqueue", request_id=req.id, step=step, source_step=step,
                      input_steps=image_steps, source_pose=req.source_pose, phase=self.phase(step), bootstrap=bootstrap,
                      ready_at_step=req.ready_at, queue_depth=len(self.queue))
        self.tick(step)
        return req

    def merge(self, step):
        """Publish every completed valid result at fast-policy cadence, whatever its age."""
        self.tick(step)
        changed = False
        while self.ready:
            req = self.ready.popleft()
            try:
                response, cache = req.future.result()
                if cache is None or len(cache) == 0:
                    raise RuntimeError("VLM produced no KV cache (not a deadline miss)")
            except Exception as exc:
                self.counts["generation_errors"] += 1
                self.fatal = repr(exc)
                self.outstanding = None
                raise RuntimeError(f"Trial 4 request {req.id} failed: {exc}") from exc
            now = time.perf_counter()
            age = int(step)-req.source_step
            if age < 0:
                raise RuntimeError("Negative semantic age")
            self.counts["results_merged"] += 1
            self.samples["queue_wait_frames"].append(req.launch_step-req.source_step)
            self.samples["queue_wait_wall_s"].append(req.launch_wall-req.source_wall)
            self.samples["service_wall_s"].append(req.ready_wall-req.launch_wall)
            self.samples["ready_to_merge_wall_s"].append(now-req.ready_wall)
            self.samples["delivery_age_frames"].append(age)
            self.samples["delivery_wall_s"].append(now-req.source_wall)
            if not req.bootstrap:
                self.samples["post_bootstrap_delivery_age_frames"].append(age)
                self.samples["post_bootstrap_queue_wait_frames"].append(req.launch_step-req.source_step)
            # Only the preregistered yoyo_drop arm enforces a freshness deadline.
            # Request scheduling/inference are unchanged; the check occurs at the
            # merge, AFTER a completed KV cache has been computed.
            if not req.bootstrap and self.cfg.deadline_frames > 0 and age > self.cfg.deadline_frames:
                self.counts["stale_drops"] += 1
                self.log.emit("cache_rejected", request_id=req.id,
                              source_step=req.source_step, merge_step=int(step),
                              age_frames=age, age_sim_s=age/FPS,
                              deadline_frames=self.cfg.deadline_frames,
                              admission="drop", response_text=response)
                self.outstanding = None
                changed = True
                continue
            high_queue = req.launch_step-req.source_step > 90
            if high_queue:
                self.last_delivery_had_high_queue = True
            elif self.last_delivery_had_high_queue and req.launch_step-req.source_step <= 3:
                self.counts["queue_recoveries"] += 1
                self.last_delivery_had_high_queue = False
            self.latest = (req, response, cache)
            self.counts["publishes_total"] += 1
            self.counts["post_bootstrap_publishes"] += int(not req.bootstrap)
            late = (not req.bootstrap and age > self.cfg.late_reference_frames)
            if late:
                self.late_published_ids.add(req.id)
                self.counts["late_publishes"] += 1
            elif self.last_delivery_was_late:
                self.counts["fresh_recoveries_after_late_delivery"] += 1
            self.last_delivery_was_late = late
            if self.last_publish_step is not None:
                self.samples["publication_gap_frames"].append(step-self.last_publish_step)
            self.last_publish_step = int(step)
            changed = True
            event = "bootstrap_publish" if req.bootstrap else "cache_publish"
            if req.bootstrap:
                self.origin = int(step)+self.cfg.warmup_frames
                self.next_attack = self.origin
                self._init_attack_plan()
            self.log.emit(event, request_id=req.id, source_step=req.source_step,
                          merge_step=int(step), age_frames=age, age_sim_s=age/FPS,
                          admission="accept", late_at_delivery=late,
                          late_reference_frames=self.cfg.late_reference_frames, response_text=response,
                          delivery_wall_s=now-req.source_wall,
                          ready_to_merge_wall_s=now-req.ready_wall,
                          first_ready_observed_step=req.first_ready_observed_step)
            self.outstanding = None
        return changed

    def bootstrap_wait(self, step):
        req = self.outstanding
        if req is None or not req.bootstrap or req.future is None:
            raise RuntimeError("Bootstrap wasn't launched immediately")
        req.future.result(timeout=self.cfg.bootstrap_timeout_s)
        self.merge(step)
        if self.latest is None:
            raise RuntimeError("Bootstrap did not publish")

    def record_action(self, step, effective_delay, kv_supplied, fast_wall_s, dx=None, dy=None):
        if self.latest is None:
            raise RuntimeError("Action requested before bootstrap")
        req = self.latest[0]
        age = int(step)-req.source_step
        wall_age = time.perf_counter()-req.source_wall
        self.counts["action_calls"] += 1
        self.counts["action_calls_without_kv"] += int(not kv_supplied)
        self.counts["action_calls_past_deadline"] += int(bool(self.cfg.deadline_frames) and age > self.cfg.deadline_frames)
        self.samples["action_cache_age_frames"].append(age)
        self.samples["action_cache_age_wall_s"].append(wall_age)
        self.samples["model_delay_input_s"].append(float(effective_delay))
        self.samples["fast_call_wall_s"].append(float(fast_wall_s))
        if kv_supplied and req.id not in self.used_cache_ids:
            self.used_cache_ids.add(req.id)
            self.counts["post_bootstrap_caches_used"] += int(not req.bootstrap)
            late_delivery = req.id in self.late_published_ids
            if late_delivery:
                self.used_late_ids.add(req.id)
            elif self.last_used_delivery_was_late:
                self.counts["fresh_cache_used_after_late"] += 1
            self.last_used_delivery_was_late = late_delivery
        self.counts["late_published_caches_used"] = len(self.used_late_ids)
        self.last_action_cache_id = req.id if kv_supplied else None
        self.log.emit("cache_use", step=int(step), cache_request_id=req.id,
                      source_step=req.source_step, age_frames=age, age_sim_s=age/FPS,
                      age_wall_s=wall_age, model_delay_input_s=float(effective_delay),
                      kv_supplied=bool(kv_supplied), cache_mode=self.cfg.cache_mode,
                      late_at_delivery=req.id in self.late_published_ids,
                      fast_call_wall_s=float(fast_wall_s), dx=dx, dy=dy)

    def summary(self):
        counts = {key: self.counts[key] for key in (
            "requests_enqueued", "requests_enqueued_during_on", "service_started",
            "results_merged", "publishes_total", "post_bootstrap_publishes", "stale_drops",
            "forced_drops", "generation_errors", "attack_jobs_enqueued", "attack_jobs_completed",
            "attack_capacity_drops", "max_queue_depth", "action_calls", "action_calls_without_kv",
            "action_calls_past_deadline", "commands_accepted", "late_publishes",
            "post_bootstrap_caches_used", "late_published_caches_used",
            "fresh_recoveries_after_late_delivery", "fresh_cache_used_after_late", "queue_recoveries")}
        counts["requests_unmerged_at_end"] = int(self.outstanding is not None)
        gap = 0 if self.last_publish_step is None or self.last_step is None else self.last_step-self.last_publish_step
        return {"settings": asdict(self.cfg), "counts": counts,
                "attack_trace_sha256": self.attack_trace_hash,
                "planned_attack_budget": getattr(self, "attack_budget", None),
                "distributions": {k: distribution(v) for k, v in self.samples.items()},
                "longest_observed_publication_gap_frames": max([gap]+self.samples["publication_gap_frames"]),
                "last_frame": self.last_step,
                "latest_cache_request_id": None if self.latest is None else self.latest[0].id,
                "latest_cache_source_step": None if self.latest is None else self.latest[0].source_step,
                "fatal_error": self.fatal,
                "instrumentation_valid": self.counts["action_calls"] > 0 and self.fatal is None}

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.outstanding and self.outstanding.future:
            self.outstanding.future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.outstanding:
            shutil.rmtree(self.outstanding.snapshot_dir, ignore_errors=True)
        self.queue.clear()
