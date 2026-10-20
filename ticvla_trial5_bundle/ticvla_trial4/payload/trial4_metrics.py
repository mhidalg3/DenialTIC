"""Extra benchmark metrics; leave released success/collision/SPL logic unchanged."""
from __future__ import annotations
import math
import os
from pathlib import Path
from trial4_core import EventLog, Settings, atomic_json


class NavTrace:
    def __init__(self):
        self.cfg = Settings.from_env()
        self.root = Path(os.environ["TICVLA_T4_OUTPUT"])
        self.log = EventLog(self.root / "navigation", os.getenv("TICVLA_EPISODE_NAME", "unknown"),
                            os.getenv("BENCHMARK_RUN_ID", "unknown"))
        self.min_goal = None
        self.min_human = None
        self.proximity_frames = set()
        self.first_goal_frame = None
        self.samples = 0
        self.people = {}
        self.last_actor_log_frame = {}

    def position(self, frame, position, goal, goal_threshold):
        self.samples += 1
        distance = math.hypot(float(position[0]-goal[0]), float(position[1]-goal[1]))
        self.min_goal = distance if self.min_goal is None else min(self.min_goal, distance)
        if self.first_goal_frame is None and distance < float(goal_threshold):
            self.first_goal_frame = int(frame)
        self.log.emit("position", benchmark_frame=int(frame), x=float(position[0]), y=float(position[1]),
                      goal_distance_m=distance)

    def human(self, frame, distance, actor_id=None, position=None):
        distance = float(distance)
        self.min_human = distance if self.min_human is None else min(self.min_human, distance)
        if distance < self.cfg.near_miss_m:
            self.proximity_frames.add(int(frame))
        if actor_id is not None and position is not None:
            xy = (float(position[0]), float(position[1]))
            person = self.people.setdefault(str(actor_id), {"first": xy, "max_displacement_m": 0.0, "min_distance_m": distance})
            person["max_displacement_m"] = max(person["max_displacement_m"], math.dist(person["first"], xy))
            person["min_distance_m"] = min(person["min_distance_m"], distance)
            if int(frame) - self.last_actor_log_frame.get(str(actor_id), -30) >= 30:
                self.log.emit("pedestrian", benchmark_frame=int(frame), actor_id=str(actor_id),
                              x=xy[0], y=xy[1], robot_root_distance_m=distance)
                self.last_actor_log_frame[str(actor_id)] = int(frame)

    def summary(self):
        return {"minimum_goal_distance_m": self.min_goal,
                "minimum_pedestrian_root_distance_m": self.min_human,
                "pedestrian_proximity_threshold_m": self.cfg.near_miss_m,
                "pedestrian_proximity_frames": len(self.proximity_frames),
                "first_goal_frame": self.first_goal_frame, "position_samples": self.samples,
                "pedestrians_observed": len(self.people),
                "pedestrians_moved_over_0_25m": sum(p["max_displacement_m"] > .25 for p in self.people.values()),
                "moving_pedestrians_within_2m": sum(p["max_displacement_m"] > .25 and p["min_distance_m"] < 2 for p in self.people.values()),
                "max_observed_pedestrian_displacement_m": max([0.0]+[p["max_displacement_m"] for p in self.people.values()]),
                "notes": "Human distance is a 2D root-centre proxy, not body clearance or a new collision definition."}


def attach(result, runner):
    from trial4_adapter import current_summary
    mechanism = current_summary()
    navigation = runner._t4_trace.summary() if hasattr(runner, "_t4_trace") else {"position_samples": 0}
    navigation["time_to_goal_sim_s"] = float(result["duration"]) if result.get("success") else None
    result["trial4"] = {"mechanism": mechanism, "navigation": navigation}
    root = Path(os.environ["TICVLA_T4_OUTPUT"])
    atomic_json(root / "mechanism.json", mechanism)
    atomic_json(root / "navigation_metrics.json", navigation)
    return result
