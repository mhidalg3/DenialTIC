"""TICVLA async-method mixin for the Trial 4 simulator experiment.

Learned layers, parameter names, prompt construction and checkpoint loading stay
in the repository's ticvla.py. The installer replaces only its async plumbing.
"""
from __future__ import annotations
from dataclasses import asdict
import copy
import shutil
import os
from pathlib import Path
import time
import numpy as np
import torch
from trial4_core import EventLog, FPS, Settings, SlowPath, atomic_json

_ACTIVE_MODEL = None


def rotation_wxyz(q):
    q = np.asarray(q, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid source quaternion")
    w, x, y, z = q/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def current_summary():
    if _ACTIVE_MODEL is None:
        return {"instrumentation_valid": False, "fatal_error": "Trial 4 model not initialized"}
    return _ACTIVE_MODEL._trial4_summary()


def backup_enabled():
    return Settings.from_env().backup_enabled


class Trial4Mixin:
    def _trial4_initialize(self):
        global _ACTIVE_MODEL
        self._t4_settings = Settings.from_env()
        import random
        seed = int(os.environ.get("TICVLA_T4_SEED", "1101"))
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        root = os.environ.get("TICVLA_T4_OUTPUT")
        if not root:
            raise RuntimeError("TICVLA_T4_OUTPUT missing: launch with Trial 4 runner")
        self._t4_dir = Path(root)
        self._t4_log = EventLog(root, os.getenv("TICVLA_EPISODE_NAME", "unknown"),
                                os.getenv("BENCHMARK_RUN_ID", "unknown"))
        self._t4_saved_summary = None
        self._t4_last_visual_cache_id = None
        self._t4_backup_events = 0
        self._t4_backup_frames = set()
        self._t4_last_backup = False
        self._t4_engine = SlowPath(self._t4_settings, self._trial4_generate,
                                  self._t4_log, executor=self._executor)
        _ACTIVE_MODEL = self

    @torch.inference_mode()
    def _trial4_generate(self, req):
        # inference_mode is thread-local: enter it in the actual executor job.
        result = self.generate_and_extract_kv_cache(
            list(req.input_paths), req.instruction, req.generation_config,
            req.previous_text, req.robot_type)
        if self.device.type == "cuda":
            # Wait for this worker's preceding GPU work, not all later work.
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            event.synchronize()
        return result

    def trial4_tick(self, step, backing_up=False):
        engine = self._t4_engine
        if engine.closed:
            return
        try:
            engine.tick(int(step))
        except Exception as exc:
            engine.fatal = repr(exc)
            self._t4_log.emit("scheduler_error", step=int(step), error=repr(exc))
            self._trial4_save()
            raise
        if backing_up:
            self._t4_backup_frames.add(int(step))
        if backing_up and not self._t4_last_backup:
            self._t4_backup_events += 1
            self._t4_log.emit("backup_start", step=int(step))
        elif self._t4_last_backup and not backing_up:
            self._t4_log.emit("backup_end", step=int(step))
        self._t4_last_backup = bool(backing_up)

    def trial4_record_command(self, step, v, w, waypoints):
        self._t4_engine.counts["commands_accepted"] += 1
        self._t4_log.emit("command_accepted", step=int(step), v_cmd=float(v), w_cmd=float(w),
                          waypoints=waypoints, cache_request_id=self._t4_engine.last_action_cache_id)

    def _trial4_sync_cache(self):
        engine = self._t4_engine
        if engine.latest is not None:
            req, response, cache = engine.latest
            self._latest_kv_cache = cache
            self._last_response = response
            self._has_ever_had_kv_cache = True
            self._latest_kv_cache_source_step = req.source_step
            self._latest_kv_cache_source_pose = req.source_pose
            self._latest_kv_cache_publish_step = engine.last_publish_step
        outstanding = engine.outstanding
        self._kv_cache_future = outstanding.future if outstanding else None
        self._kv_cache_start_time = outstanding.source_wall if outstanding else None

    def _start_kv_cache_generation(self, delayed_img_paths, instruction,
                                  generation_config, current_step=None,
                                  previous_waypoints_text="", robot_type="mobile robot",
                                  current_robot_pose=None):
        if current_step is None:
            raise ValueError("Trial 4 requires an explicit simulation frame")
        req = self._t4_engine.offer(int(current_step), current_robot_pose,
                                    delayed_img_paths, instruction, generation_config,
                                    previous_waypoints_text, robot_type)
        if req is not None:
            self._kv_cache_generation_step = req.source_step
            self._kv_cache_generation_pose = copy.deepcopy(req.source_pose)
        self._trial4_sync_cache()
        return req

    def _poll_kv_cache_future(self, current_step=None):
        if current_step is None:
            raise ValueError("Trial 4 requires an explicit simulation frame")
        changed = self._t4_engine.merge(int(current_step))
        if changed:
            self._kv_cache_completion_step = int(current_step)
        self._trial4_sync_cache()

    @torch.inference_mode()
    def predict_async(self, image_paths, instruction=None, robot_state=None,
                      current_step=None, current_robot_pose=None, time_delay=0.0,
                      previous_waypoints_text="", delayed_image_paths=None,
                      robot_type="mobile robot"):
        try:
            return self._trial4_predict(image_paths, instruction, robot_state,
                                        current_step, current_robot_pose, time_delay,
                                        previous_waypoints_text, delayed_image_paths, robot_type)
        except Exception as exc:
            self._t4_engine.fatal = repr(exc)
            self._t4_log.emit("policy_error", step=current_step, error=repr(exc))
            self._trial4_save()
            raise

    def _trial4_predict(self, image_paths, instruction, robot_state, current_step,
                       current_robot_pose, time_delay, previous_waypoints_text,
                       delayed_image_paths, robot_type):
        from vision_utils import load_image
        if current_step is None or not image_paths:
            raise ValueError("Trial 4 needs current_step and current camera input")
        if self.action_expert.training:
            raise RuntimeError("ActionExpert must be in eval() mode for these experiments")
        step = int(current_step)
        self._poll_kv_cache_future(step)
        generation_config = {"max_new_tokens": 200, "do_sample": not self._t4_settings.deterministic}
        if not self._t4_settings.deterministic:
            generation_config.update(temperature=0.1, top_p=0.1, top_k=10)
        req = self._start_kv_cache_generation(
            delayed_image_paths if delayed_image_paths is not None else image_paths,
            instruction or "Move forward safely and efficiently.", generation_config,
            step, previous_waypoints_text, robot_type, current_robot_pose)
        new_step = req.source_step if req else None
        new_pose = copy.deepcopy(req.source_pose) if req else None
        if self._t4_engine.latest is None:
            # Common, un-delayed bootstrap. We do not wait for simulated frames here.
            self._t4_engine.bootstrap_wait(step)
            self._trial4_sync_cache()
        source, _, cache = self._t4_engine.latest
        age_s = (step-source.source_step)/FPS
        dx = dy = None
        if current_robot_pose is not None and source.source_pose is not None:
            old = source.source_pose
            if all(x is not None for x in (old.get("position"), old.get("quaternion"), current_robot_pose.get("position"))):
                delta = np.asarray(current_robot_pose["position"], dtype=float)-np.asarray(old["position"], dtype=float)
                body = rotation_wxyz(old["quaternion"]).T @ delta
                dx, dy = float(body[0]), float(body[1])
        if dx is None or dy is None:
            raise RuntimeError("No cache-aligned source pose: cannot use inconsistent displacement metadata")
        if self._t4_settings.retain_visuals and source.id != self._t4_last_visual_cache_id:
            evidence = self._t4_dir / "cache_use_images"
            evidence.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_paths[-1], evidence / f"cache_{source.id:06d}_used_frame_{step:06d}.jpg")
            self._t4_last_visual_cache_id = source.id
        fast_start = time.perf_counter()
        current = load_image(image_paths[-1], input_size=448, max_num=1).to(torch.bfloat16).to(self.device)
        image_embeds = self.vlm.extract_feature(current)
        image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1]).unsqueeze(0)
        if robot_state is None:
            raise ValueError("Missing robot state")
        rs = torch.as_tensor(robot_state, device=self.device, dtype=image_embeds.dtype).view(-1)
        if len(rs) == 6:
            compact = torch.stack([rs[0], rs[1], rs[3], rs[4], rs[5]])
        elif len(rs) == 5:
            compact = rs.clone()
        else:
            raise ValueError(f"Expected 5 or 6 robot state fields, got {len(rs)}")
        compact[3], compact[4] = dx, dy
        td = torch.tensor([age_s], device=self.device, dtype=image_embeds.dtype)
        state = torch.cat([compact, td]).view(1, -1, 1)
        supplied = self._t4_settings.cache_mode != "none"
        # No-KV is an interface ablation, NOT the paper's no-reasoning ablation.
        # Keep VLM workload and cache-aligned age/displacement as matched as possible.
        waypoints = self.action_expert(image_embeds, state, kv_cache=cache if supplied else None)
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            event.synchronize()
        self._t4_engine.record_action(step, float(td.item()), supplied,
                                     time.perf_counter()-fast_start, dx, dy)
        return self._last_response, waypoints, new_step, True, new_pose

    def _trial4_summary(self):
        data = self._t4_saved_summary or self._t4_engine.summary()
        data = copy.deepcopy(data)
        data["backup_events"] = self._t4_backup_events
        data["backup_frames"] = len(self._t4_backup_frames)
        data["backup_enabled"] = self._t4_settings.backup_enabled
        return data

    def _trial4_save(self):
        atomic_json(self._t4_dir / "mechanism.json", self._trial4_summary())

    def reset_episode_state(self):
        # An episode-level reset never silently waits for an unelapsed frame gate.
        self._t4_engine.close()
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._t4_engine = SlowPath(self._t4_settings, self._trial4_generate,
                                  self._t4_log, executor=self._executor)
        self._t4_saved_summary = None
        self._t4_backup_events = 0
        self._t4_backup_frames.clear()
        self._t4_last_visual_cache_id = None
        self._t4_last_backup = False
        for attr in ("_kv_cache_future", "_kv_cache_start_time", "_latest_kv_cache",
                     "_last_response", "_kv_cache_generation_step", "_kv_cache_generation_pose",
                     "_kv_cache_completion_step", "_latest_kv_cache_source_step",
                     "_latest_kv_cache_source_pose", "_latest_kv_cache_publish_step"):
            setattr(self, attr, None)
        self._has_ever_had_kv_cache = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(self.vlm, "reset_cache"):
            self.vlm.reset_cache()
        self._t4_log.emit("episode_reset")

    def cleanup(self):
        if self._t4_saved_summary is None:
            self._t4_saved_summary = self._t4_engine.summary()
            self._trial4_save()
        self._t4_engine.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
