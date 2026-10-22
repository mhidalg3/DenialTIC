#!/usr/bin/env python3
"""Figures and scene videos for the Trial 3 evidence package.

Conditions: baseline (fresh KV caches), no_kv (cache never supplied to the action
expert), yoyo_accept (periodic queue attack; late caches admitted and consumed).
Episodes 20/25/45/75/85 with scene seed 6101, one rollout per cell, so every number
is descriptive rather than a rate estimate.

Reads the evidence package read-only. Requires numpy, matplotlib, Pillow, and
ffmpeg on PATH for the videos.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import subprocess
import textwrap
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from PIL import Image  # noqa: E402

FPS = 30
EPISODES = (20, 25, 45, 75, 85)
CONDITIONS = ("baseline", "no_kv", "yoyo_accept")
LABEL = {"baseline": "baseline", "no_kv": "no KV cache", "yoyo_accept": "yoyo (late caches)"}
COLOR = {"baseline": "#1f77b4", "no_kv": "#7f7f7f", "yoyo_accept": "#ff7f0e"}
EP_COLOR = dict(zip(EPISODES, ("#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf")))
HUMAN_COLOR = "#d62728"
PHYSICAL_COLOR = "#6a3d9a"
ATTACK_COLOR = "#bcbd22"
HUMAN_THRESHOLD_M = 0.2  # DynaNav/benchmark.py:708, robot-to-character XY root distance
NEAR_MISS_M = 0.6  # launch.json settings.near_miss_m
CONTACT_FRAMES = 100  # DynaNav/benchmark.py:132, >= 10 N for 100 consecutive frames
GOAL_RADIUS_M = 1.5
STALL_RADIUS_M = 0.5
STALLED_SPEED = 0.05  # m/s
MIN_BIN_EXPOSURE_MIN = 0.5
AGE_BINS = np.array([0, 2.5, 5, 7.5, 10, 15, 20, 25, 30])
# (episode, kind, key). ep75's yoyo robot enters the 0.5 m region it never leaves while still using
# the 22 s-old cache 2, before late cache 3 arrives; the two human encounters happen only under
# yoyo; ep85's late cache 9 is first used near the goal, after which the robot drives away.
STORYBOARD_MOMENTS = ((75, "final_region", "entry"), (75, "human", "Character_09"), (85, "human", "Character_12"),
                      (85, "cache", 9))
VIDEO_EPISODES = (75, 85)

PCUR = re.compile(r"^p_cur=\[")
HUMAN_LINE = re.compile(r"Collision detected with (\S+) at distance ([\d.]+)m")
PHYSICAL_LINE = "Contact sensor detected sustained collision"
THINK = re.compile(r"<think>(.*?)</think>", re.S)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
TRIPLE = re.compile(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)")
CONTEXT_FRAME = re.compile(r"frame_(\d+)\.jpg$")


@dataclass
class Cell:
    ep: int
    cond: str
    root: Path
    sup: Path
    res: dict
    nav: dict
    x: np.ndarray
    y: np.ndarray
    goal: np.ndarray
    peds: dict  # actor prim path -> ndarray (n, 4): frame, x, y, robot root distance
    use_step: np.ndarray
    use_cache: np.ndarray
    use_age: np.ndarray
    kv: bool
    pubs: dict  # request_id -> cache_publish / bootstrap_publish event
    enq: dict  # request_id -> benign_enqueue event
    attack_on: list  # [(first ON frame, first OFF frame)]
    humans: list  # human encounters grouped from console lines
    physical_flag: tuple | None  # stdout-order bracket of the one-time contact flag print
    stall: int | None
    tp_frames: list

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def human(self) -> bool:
        return self.res["human_collision"] > 0

    @property
    def physical(self) -> bool:
        return self.res["physical_collision"] > 0

    @property
    def streak(self) -> int:
        return int(self.res["physical_collision_count"])

    @cached_property
    def age(self) -> np.ndarray:
        """Age (s) of the KV cache the action expert was using at each frame; NaN without KV."""
        out = np.full(self.n, np.nan)
        if not self.kv:
            return out
        frames = np.arange(self.n)
        i = np.searchsorted(self.use_step, frames, side="right") - 1
        ok = i >= 0
        out[ok] = self.use_age[i[ok]] + (frames[ok] - self.use_step[i[ok]]) / FPS
        return out

    @cached_property
    def speed(self) -> np.ndarray:
        """Displacement over a centred 1 s window, i.e. m/s."""
        h, out = FPS // 2, np.full(self.n, np.nan)
        out[h : self.n - h] = np.hypot(self.x[2 * h :] - self.x[: -2 * h], self.y[2 * h :] - self.y[: -2 * h])
        return out

    @cached_property
    def progress(self) -> np.ndarray:
        """Goal-distance decrease over a centred 1 s window, i.e. m/s toward the goal."""
        h, out = FPS // 2, np.full(self.n, np.nan)
        out[h : self.n - h] = self.goal[: -2 * h] - self.goal[2 * h :]
        return out

    @cached_property
    def first_uses(self) -> list:
        """[(frame, cache_id, age_s, late_at_delivery)] for every cache the action expert started using."""
        out, seen = [], set()
        for step, cid, age in zip(self.use_step, self.use_cache, self.use_age):
            if cid not in seen:
                seen.add(cid)
                out.append((int(step), int(cid), float(age), bool(self.pubs[int(cid)]["late_at_delivery"])))
        return out

    def cache_at(self, frame: int) -> int | None:
        i = bisect.bisect_right(self.use_step, frame) - 1
        return int(self.use_cache[i]) if i >= 0 else None

    def first_use(self, cache_id: int) -> tuple[int, float]:
        i = int(np.flatnonzero(self.use_cache == cache_id)[0])
        return int(self.use_step[i]), float(self.use_age[i])

    def tp_path(self, frame: int) -> Path:
        i = max(bisect.bisect_right(self.tp_frames, frame) - 1, 0)
        return self.sup / f"tp_frame_{self.tp_frames[i]:06d}.jpg"


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def parse_console(path: Path, cmd_steps: list, n_frames: int) -> tuple[list, tuple | None]:
    """Bracket collision prints by the accepted commands printed around them.

    Each accepted command prints one `p_cur=` line, so the n-th `p_cur` line belongs to the
    n-th command_accepted event. A collision line between two `p_cur` lines happened between
    those two commands' frames. The contact flag prints once, when a streak reaches 100 frames.
    """
    lines = path.read_text(errors="replace").splitlines()
    pcur = [i for i, line in enumerate(lines) if PCUR.match(line)]
    if len(pcur) != len(cmd_steps):
        raise SystemExit(f"{path}: {len(pcur)} p_cur lines vs {len(cmd_steps)} accepted commands")

    def bracket(i: int) -> tuple[int, int]:
        k = bisect.bisect_left(pcur, i)
        return (cmd_steps[k - 1] if k else 0), (cmd_steps[k] if k < len(cmd_steps) else n_frames - 1)

    hits, physical = [], None
    for i, line in enumerate(lines):
        m = HUMAN_LINE.search(line)
        if m:
            hits.append((m.group(1), float(m.group(2)), *bracket(i)))
        elif PHYSICAL_LINE in line and physical is None:
            physical = bracket(i)

    encounters = []
    for prim in sorted({h[0] for h in hits}):
        current = None
        for _, dist, lo, hi in sorted((h for h in hits if h[0] == prim), key=lambda h: h[2]):
            if current and lo - current["hi"] <= 2 * FPS:
                current["hi"], current["min_d"] = max(current["hi"], hi), min(current["min_d"], dist)
            else:
                parts = prim.split("/")
                folder = parts[parts.index("Characters") + 1]  # bare "Character" folders: use the asset name
                actor = parts[-1] if folder == "Character" else folder
                current = {"prim": prim, "actor": actor, "lo": lo, "hi": hi, "min_d": dist}
                encounters.append(current)
    return sorted(encounters, key=lambda e: e["lo"]), physical


def stall_onset(x: np.ndarray, y: np.ndarray, success: bool) -> int | None:
    """First frame after which the robot never leaves a 0.5 m disc again (>= 5 s before the end)."""
    if success:
        return None
    onset = None
    for f in range(len(x) - 1, -1, -1):
        if np.hypot(x[f:] - x[f], y[f:] - y[f]).max() > STALL_RADIUS_M:
            break
        onset = f
    return onset if onset is not None and len(x) - onset >= 5 * FPS else None


def load_cell(evidence: Path, row: dict) -> Cell:
    root = evidence / row["folder"]
    sup = evidence / "supplemental" / "DynaNav" / "logs" / f"t4_{row['session']}_{root.name}" / "carter_ticvla_data"
    res = json.loads((root / "result.json").read_text())["result"]
    nav = json.loads((root / "navigation_metrics.json").read_text())

    pos, peds = {}, {}
    for e in read_jsonl(root / "navigation" / "events.jsonl"):
        if e["event"] == "position":
            pos[e["benchmark_frame"]] = (e["x"], e["y"], e["goal_distance_m"])
        elif e["event"] == "pedestrian":
            peds.setdefault(e["actor_id"], []).append(
                (e["benchmark_frame"], e["x"], e["y"], e["robot_root_distance_m"])
            )
    n = max(pos) + 1
    if len(pos) != n:
        raise SystemExit(f"{root}: position frames are not contiguous")
    xyg = np.array([pos[f] for f in range(n)])

    uses, cmds, pubs, enq, phases = [], [], {}, {}, []
    for e in read_jsonl(root / "events.jsonl"):
        kind = e["event"]
        if kind == "cache_use":
            uses.append((e["step"], e["cache_request_id"], e["age_sim_s"], e["kv_supplied"]))
        elif kind == "command_accepted":
            cmds.append(e["step"])
        elif kind in ("cache_publish", "bootstrap_publish"):
            pubs[e["request_id"]] = e
        elif kind == "benign_enqueue":
            enq[e["request_id"]] = e
        elif kind == "attack_phase":
            phases.append((e["step"], e["phase"]))

    attack_on, start = [], None
    for step, phase in phases:
        if phase == "ON" and start is None:
            start = step
        elif phase != "ON" and start is not None:
            attack_on.append((start, step))
            start = None
    if start is not None:
        attack_on.append((start, n))

    humans, physical_flag = parse_console(root / "console.log", cmds, n)
    return Cell(
        ep=int(row["episode"]),
        cond=row["condition"],
        root=root,
        sup=sup,
        res=res,
        nav=nav,
        x=xyg[:, 0],
        y=xyg[:, 1],
        goal=xyg[:, 2],
        peds={k: np.array(v) for k, v in peds.items()},
        use_step=np.array([u[0] for u in uses]),
        use_cache=np.array([u[1] for u in uses]),
        use_age=np.array([u[2] for u in uses], dtype=float),
        kv=any(u[3] for u in uses),
        pubs=pubs,
        enq=enq,
        attack_on=attack_on,
        humans=humans,
        physical_flag=physical_flag,
        stall=stall_onset(xyg[:, 0], xyg[:, 1], bool(res["success"])),
        tp_frames=sorted(int(p.stem.rsplit("_", 1)[1]) for p in sup.glob("tp_frame_*.jpg")),
    )


def load_cells(evidence: Path) -> dict:
    with open(evidence / "analysis_index" / "episode_inventory.csv") as f:
        rows = list(csv.DictReader(f))
    cells = {(int(r["episode"]), r["condition"]): load_cell(evidence, r) for r in rows}
    missing = {(ep, c) for ep in EPISODES for c in CONDITIONS} - cells.keys()
    if missing:
        raise SystemExit(f"missing cells: {sorted(missing)}")
    return cells


def print_summary(cells: dict) -> None:
    for (ep, cond), c in sorted(cells.items()):
        humans = ", ".join(f"{e['actor']} {e['lo']}-{e['hi']} ({e['min_d']:.2f} m)" for e in c.humans) or "-"
        flag = "-" if c.physical_flag is None else f"{c.physical_flag[0]}-{c.physical_flag[1]}"
        age = f"{np.nanmedian(c.age):5.2f}s" if c.kv else "  n/a "
        print(f"ep{ep} {cond:12s} frames={c.n:5d} median_age={age} stall={c.stall} contact_flag={flag} human={humans}")


# ----------------------------------------------------------------------------- helpers


def collision_marker(c: Cell) -> str:
    return {(True, True): "D", (True, False): "^", (False, True): "s"}.get((c.human, c.physical), "o")


def short_actor(actor: str) -> str:
    """Character_09 -> C09; asset names such as male_adult_construction_05 -> construction_05."""
    return actor.replace("Character_", "C") if actor.startswith("Character_") else actor.split("_", 2)[-1]


def condition_axis(ax, title: str) -> None:
    ax.set_xticks(range(len(CONDITIONS)), [LABEL[c] for c in CONDITIONS])
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)


def episode_handles() -> list:
    return [
        Line2D([], [], marker="o", color=EP_COLOR[ep], markeredgecolor="black", label=f"episode {ep}")
        for ep in EPISODES
    ]


def load_image(path: Path, size=(480, 270)) -> np.ndarray:
    im = Image.open(path)
    im.draft("RGB", size)
    return np.asarray(im.convert("RGB"))


def source_pose(enq: dict) -> tuple[float, float, float]:
    p = enq["source_pose"]["position"]
    w, qx, qy, qz = enq["source_pose"]["quaternion"]
    return p[0], p[1], math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def text_waypoints_in_world(response_text: str, pose: tuple[float, float, float]) -> np.ndarray:
    """VLM <answer> text waypoints (body frame: x forward, y left) placed at the pose the VLM saw.

    Context only: the executed waypoints come from the action expert, which combines the cached
    VLM state with the current camera image, displacement since the source pose, and cache age
    (DynaNav/trial4_adapter.py:197-215).
    """
    m = ANSWER.search(response_text)
    pts = [(float(a), float(b)) for a, b, _ in TRIPLE.findall(m.group(1))] if m else []
    x0, y0, yaw = pose
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([(x0, y0)] + [(x0 + a * cy - b * sy, y0 + a * sy + b * cy) for a, b in pts])


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ----------------------------------------------------------------------------- figures


def fig_condition_outcomes(cells: dict, out: Path) -> None:
    panels = (
        ("Success rate (%)", lambda c: 100.0 * bool(c.res["success"]), "{:.0f}%"),
        None,
        ("Path length (m)", lambda c: c.res["path_length"], "{:.1f}"),
        ("Navigation error at end (m)", lambda c: c.res["navigation_error"], "{:.1f}"),
        ("Closest approach to goal (m)", lambda c: c.nav["minimum_goal_distance_m"], "{:.1f}"),
        ("SPL", lambda c: c.res["spl"], "{:.2f}"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    xs = np.arange(len(CONDITIONS))
    for ax, panel in zip(axes.flat, panels):
        if panel is None:
            collision_type_bars(ax, cells)
            continue
        title, fn, fmt = panel
        vals = np.array([[fn(cells[ep, c]) for c in CONDITIONS] for ep in EPISODES])
        means = vals.mean(axis=0)
        ax.bar(xs, means, 0.62, color=[COLOR[c] for c in CONDITIONS], alpha=0.5, edgecolor="black", lw=0.6)
        for k, ep in enumerate(EPISODES):
            dx = (k - 2) * 0.07
            ax.plot(xs + dx, vals[k], color=EP_COLOR[ep], lw=1, alpha=0.45)
            ax.scatter(xs + dx, vals[k], color=EP_COLOR[ep], edgecolor="black", s=28, lw=0.5, zorder=3)
        for xi, m in zip(xs, means):
            ax.text(xi + 0.33, m, fmt.format(m), ha="left", va="center", fontsize=9, fontweight="bold")
        condition_axis(ax, title)
    fig.legend(handles=episode_handles(), loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Trial 3 outcomes by condition — bars = mean of 5 episodes (seed 6101), dots joined by lines = the same "
        "episode across conditions\nOne rollout per cell: descriptive, not rate estimates",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, out / "fig1_condition_outcomes.png")


def collision_type_bars(ax, cells: dict) -> None:
    xs, bottom = np.arange(len(CONDITIONS)), np.zeros(len(CONDITIONS))
    kinds = (
        ("human only", HUMAN_COLOR, "", lambda c: c.human and not c.physical),
        ("physical only", PHYSICAL_COLOR, "", lambda c: c.physical and not c.human),
        ("human + physical", HUMAN_COLOR, "///", lambda c: c.human and c.physical),
    )
    for name, color, hatch, test in kinds:
        eps = [[ep for ep in EPISODES if test(cells[ep, cond])] for cond in CONDITIONS]
        h = np.array([100.0 * len(e) / len(EPISODES) for e in eps])
        ax.bar(xs, h, 0.62, bottom=bottom, color=color, hatch=hatch, edgecolor=PHYSICAL_COLOR if hatch else "black",
               lw=0.6, alpha=0.85, label=name)
        for xi, b, hh, e in zip(xs, bottom, h, eps):
            if hh:
                ax.text(xi, b + hh / 2, "ep " + ", ".join(map(str, e)), ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        bottom += h
    for xi, total in zip(xs, bottom):
        ax.text(xi, total + 1.5, f"{total:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.legend(fontsize=8, loc="upper left")
    condition_axis(ax, "Collision rate (%) split by type")


def fig_latency_per_episode(cells: dict, out: Path) -> None:
    panels = (
        ("Collision events", "human encounters + contact flag (count)",
         lambda c: len(c.humans) + (c.physical_flag is not None)),
        ("Path length", "m", lambda c: c.res["path_length"]),
        ("Navigation error at end", "m", lambda c: c.res["navigation_error"]),
    )
    fig = plt.figure(figsize=(19, 7))
    gs = fig.add_gridspec(1, 8, width_ratios=[4, 0.7, 0.75, 4, 0.7, 0.75, 4, 0.7], wspace=0.06, top=0.8,
                          bottom=0.25)
    for p, (title, ylabel, fn) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 3 * p])
        side = fig.add_subplot(gs[0, 3 * p + 1], sharey=ax)
        for k, ep in enumerate(EPISODES):
            dy = (k - 2) * 0.04 if p == 0 else 0.0
            base, nokv, yoyo = (cells[ep, c] for c in CONDITIONS)
            xb, xy = np.nanmedian(base.age), np.nanmedian(yoyo.age)
            ax.plot([xb, xy], [fn(base) + dy, fn(yoyo) + dy], color=EP_COLOR[ep], lw=1.8, alpha=0.8)
            for c, xv in ((base, xb), (yoyo, xy)):
                ax.scatter(xv, fn(c) + dy, marker=collision_marker(c) if p == 0 else "o", s=80, color=COLOR[c.cond],
                           edgecolor=EP_COLOR[ep], lw=2, zorder=3)
            side.scatter((k - 2) * 0.12, fn(nokv) + dy, marker=collision_marker(nokv) if p == 0 else "o", s=80,
                         color=COLOR["no_kv"], edgecolor=EP_COLOR[ep], lw=2, zorder=3)
        mx = [np.mean([np.nanmedian(cells[ep, c].age) for ep in EPISODES]) for c in ("baseline", "yoyo_accept")]
        my = [np.mean([fn(cells[ep, c]) for ep in EPISODES]) for c in ("baseline", "yoyo_accept")]
        ax.plot(mx, my, color="black", lw=3, marker="X", ms=12, zorder=4)
        side.scatter(0, np.mean([fn(cells[ep, "no_kv"]) for ep in EPISODES]), marker="X", s=140, color="black", zorder=4)
        ax.set_xlabel("median age of the KV cache in use during the run (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3)
        side.set_title("no KV\n(KV tensor\nwithheld)", fontsize=8.5)
        side.set_xlim(-0.45, 0.45)
        side.set_xticks([])
        side.tick_params(labelleft=False)
        side.grid(axis="y", alpha=0.3)
        if p == 0:
            ax.set_yticks(range(4))
    handles = [
        Patch(color=COLOR[c], label=f"fill = {LABEL[c]}") for c in CONDITIONS
    ] + episode_handles() + [
        Line2D([], [], color="black", lw=3, marker="X", ms=10, label="mean of 5 episodes"),
        Line2D([], [], ls="", marker="o", color="0.6", label="no collision"),
        Line2D([], [], ls="", marker="^", color="0.6", label="human"),
        Line2D([], [], ls="", marker="s", color="0.6", label="physical"),
        Line2D([], [], ls="", marker="D", color="0.6", label="human + physical"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Latency vs outcome, one point per run — each coloured line joins an episode's baseline run (fresh caches) "
        "to its yoyo run (late caches)\nTrial 3 has no controlled latency levels; x is the cache age the action "
        "expert actually experienced. no_kv withholds the KV tensor (age/displacement scalars are still fed), "
        "so it sits on its own axis.",
        fontsize=11, y=0.97,
    )
    save(fig, out / "fig2_latency_vs_outcome_per_episode.png")


def binned_stats(cells: dict, cond: str) -> dict:
    nb = len(AGE_BINS) - 1
    frames = np.zeros(nb)
    speed_sum, progress_sum, stalled = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    near = np.zeros(nb)
    events = []
    for ep in EPISODES:
        c = cells[ep, cond]
        b = np.digitize(c.age, AGE_BINS) - 1
        ok = (b >= 0) & (b < nb) & ~np.isnan(c.speed)
        np.add.at(frames, b[ok], 1)
        np.add.at(speed_sum, b[ok], c.speed[ok])
        np.add.at(progress_sum, b[ok], c.progress[ok])
        np.add.at(stalled, b[ok], c.speed[ok] < STALLED_SPEED)
        for samples in c.peds.values():
            for f in samples[samples[:, 3] < NEAR_MISS_M, 0].astype(int):
                if ok[f]:
                    near[b[f]] += 1
        events += [(c.age[e["lo"]], "human", ep) for e in c.humans]
        if c.physical_flag:
            events.append((c.age[max(c.physical_flag[0] - CONTACT_FRAMES, 0)], "physical", ep))
    minutes = frames / FPS / 60
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = {
            "minutes": minutes,
            "speed": speed_sum / frames,
            "progress": progress_sum / frames,
            "stalled": stalled / frames,
            "near": near / minutes,
        }
    sparse = minutes < MIN_BIN_EXPOSURE_MIN
    for key in ("speed", "progress", "stalled", "near"):
        stats[key][sparse] = np.nan
    stats["events"] = events
    return stats


def overall_stats(cells: dict, cond: str) -> dict:
    speed = np.concatenate([cells[ep, cond].speed for ep in EPISODES])
    progress = np.concatenate([cells[ep, cond].progress for ep in EPISODES])
    ok = ~np.isnan(speed)
    near = 0
    for ep in EPISODES:
        c = cells[ep, cond]
        for samples in c.peds.values():
            near += int(sum(1 for f in samples[samples[:, 3] < NEAR_MISS_M, 0].astype(int) if not np.isnan(c.speed[f])))
    return {
        "speed": speed[ok].mean(),
        "progress": progress[ok].mean(),
        "stalled": (speed[ok] < STALLED_SPEED).mean(),
        "near": near / (ok.sum() / FPS / 60),
    }


def fig_latency_binned(cells: dict, out: Path) -> None:
    centers = (AGE_BINS[:-1] + AGE_BINS[1:]) / 2
    widths = np.diff(AGE_BINS)
    stats = {c: binned_stats(cells, c) for c in ("baseline", "yoyo_accept")}
    ref = overall_stats(cells, "no_kv")
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.18, top=0.9)
    top_left = gs[0, 0].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.08)
    a_rug = fig.add_subplot(top_left[0])
    a_near = fig.add_subplot(top_left[1], sharex=a_rug)
    a_speed, a_stall, a_exp = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    for cond, s in stats.items():
        kw = {"color": COLOR[cond], "marker": "o", "lw": 2.2, "label": LABEL[cond]}
        a_near.plot(centers, s["near"], **kw)
        a_speed.plot(centers, s["speed"], **kw)
        a_speed.plot(centers, s["progress"], color=COLOR[cond], marker="o", mfc="white", lw=1.6, ls="--",
                     label=f"{LABEL[cond]}: progress toward goal")
        a_stall.plot(centers, 100 * s["stalled"], **kw)
    for ax, key, scale in ((a_near, "near", 1), (a_speed, "speed", 1), (a_stall, "stalled", 100)):
        ax.axhline(scale * ref[key], color=COLOR["no_kv"], ls=":", lw=2, label="no KV cache (whole-run average)")
    a_speed.axhline(ref["progress"], color=COLOR["no_kv"], ls="-.", lw=1.4, label="no KV cache: progress")

    rows = {"human": 1, "physical": 0}
    placed = {"human": [], "physical": []}
    for age, kind, ep, cond in sorted((e + (cond,) for cond, s in stats.items() for e in s["events"]),
                                      key=lambda e: e[0]):
        crowded = sum(abs(age - a) < 1.5 for a in placed[kind])
        placed[kind].append(age)
        yy = rows[kind] + (0.22 if crowded % 2 else 0)
        a_rug.scatter(age, yy, marker="^" if kind == "human" else "s", s=90, color=COLOR[cond],
                      edgecolor=HUMAN_COLOR if kind == "human" else PHYSICAL_COLOR, lw=2, zorder=5)
        a_rug.text(age + 0.35, yy, f"ep{ep}", fontsize=7.5, va="center")
    a_rug.set_yticks([0, 1], ["contact-flag\nonset", "human\nencounter"], fontsize=8)
    a_rug.set_ylim(-0.45, 1.6)
    a_rug.tick_params(labelbottom=False)
    a_rug.grid(axis="x", alpha=0.3)
    a_rug.set_title("Collision onsets (fill = condition) and pedestrian near-miss exposure", fontsize=10)
    a_near.set_ylim(0, None)
    a_near.set_ylabel("1 Hz pedestrian samples < 0.6 m\nper robot-minute")
    a_speed.set_ylabel("m/s")
    a_speed.set_title("Speed (solid) and progress toward the goal (dashed)", fontsize=10)
    a_stall.set_ylabel("% of frames")
    a_stall.set_title(f"Time stalled (speed < {STALLED_SPEED} m/s)", fontsize=10)
    for k, (cond, s) in enumerate(stats.items()):
        a_exp.bar(centers + (k - 0.5) * widths * 0.4, s["minutes"], widths * 0.4, color=COLOR[cond], label=LABEL[cond])
    a_exp.axhline(MIN_BIN_EXPOSURE_MIN, color="black", ls=":", lw=1, label=f"{MIN_BIN_EXPOSURE_MIN} min cut-off")
    a_exp.set_ylabel("robot minutes")
    a_exp.set_title(f"Exposure: time spent at each cache age (bins under {MIN_BIN_EXPOSURE_MIN} min are not "
                    "plotted elsewhere)", fontsize=10)
    for ax in (a_rug, a_near, a_speed, a_stall, a_exp):
        ax.set_xticks(AGE_BINS)
        ax.set_xlim(AGE_BINS[0], AGE_BINS[-1])
    for ax in (a_near, a_speed, a_stall, a_exp):
        ax.set_xlabel("age of the KV cache the action expert was using (s)")
        ax.grid(alpha=0.3)
        if ax is a_speed:
            ax.legend(fontsize=8, loc="center", bbox_to_anchor=(0.55, 0.5))
        else:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Within-run latency view — every frame of every baseline and yoyo run, binned by the age of the cache in use\n"
        "Ages above ~5 s occur only under the yoyo attack, whose queue flooding delays cache refresh: "
        "association, not a controlled dose-response",
        fontsize=12,
    )
    save(fig, out / "fig3_latency_binned_within_run.png")


def fig_collision_types(cells: dict, out: Path) -> None:
    fig = plt.figure(figsize=(20, 11.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[1.15, 1], wspace=0.16, hspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    quad = fig.add_subplot(gs[0, 1])
    notes = fig.add_subplot(gs[1, 1])
    notes.axis("off")

    y, ticks, labels = 0.0, [], []
    for ep in EPISODES:
        for cond in CONDITIONS:
            c = cells[ep, cond]
            ticks.append(y)
            labels.append(f"ep{ep}  {LABEL[cond]}")
            dur = c.n / FPS
            ax.add_patch(Rectangle((0, y - 0.3), dur, 0.6, color=COLOR[cond], alpha=0.14, lw=0))
            for a, b in c.attack_on:
                ax.add_patch(Rectangle((a / FPS, y - 0.3), (b - a) / FPS, 0.6, color=ATTACK_COLOR, alpha=0.3, lw=0))
            near = [s[0] for arr in c.peds.values() for s in arr if s[3] < NEAR_MISS_M]
            ax.scatter(np.array(near) / FPS, np.full(len(near), y), s=5, color="0.3", zorder=2)
            if c.cond == "yoyo_accept":
                late = [f for f, _, _, is_late in c.first_uses if is_late]
                ax.scatter(np.array(late) / FPS, np.full(len(late), y + 0.4), marker="v", s=22,
                           color=COLOR["yoyo_accept"], edgecolor="black", lw=0.4, zorder=3)
            for k, e in enumerate(c.humans):
                ax.add_patch(Rectangle((e["lo"] / FPS, y - 0.32), max((e["hi"] - e["lo"]) / FPS, 0.8), 0.64,
                                       color=HUMAN_COLOR, zorder=3))
                ax.text(e["hi"] / FPS + 0.8, y + (0.16 if k % 2 == 0 else -0.16),
                        f"{short_actor(e['actor'])} {e['min_d']:.2f} m", fontsize=7, color=HUMAN_COLOR, va="center",
                        zorder=6)
            if c.physical_flag:
                lo, hi = c.physical_flag
                onset = max(lo - CONTACT_FRAMES, 0)
                ax.add_patch(Rectangle((onset / FPS, y - 0.32), (hi - onset) / FPS, 0.64, color=PHYSICAL_COLOR,
                                       zorder=3))
                ax.text(hi / FPS + 0.8, y - 0.17,
                        f"contact flag; longest streak {c.streak} fr ({c.streak / FPS:.0f} s)",
                        fontsize=7, color=PHYSICAL_COLOR, va="center", zorder=6)
            elif c.streak:
                ax.text(dur + 1.5, y - 0.17, f"sub-threshold contact, longest streak {c.streak} fr", fontsize=7,
                        color=PHYSICAL_COLOR, style="italic", va="center")
            if c.stall is not None:
                ax.scatter(c.stall / FPS, y, marker="|", s=320, color="black", lw=2.5, zorder=4)
            if c.res["success"]:
                ax.scatter(dur, y, marker="*", s=170, color="#2ca02c", edgecolor="black", zorder=4)
            y -= 1
        y -= 0.7
    ax.set_yticks(ticks, labels, fontsize=9)
    for label, cond in zip(ax.get_yticklabels(), CONDITIONS * len(EPISODES)):
        label.set_color(COLOR[cond])
    ax.set_xlim(0, 232)
    ax.set_ylim(y + 0.3, 0.8)
    ax.set_xlabel("simulation time (s)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("When and how each run collided", fontsize=12)
    ax.legend(handles=[
        Patch(color=HUMAN_COLOR, label="human encounter (< 0.2 m; drawn ≥ 0.8 s wide), label = actor, closest distance"),
        Patch(color=PHYSICAL_COLOR, label="100-frame contact that set the physical flag"),
        Patch(color=ATTACK_COLOR, alpha=0.4, label="yoyo attack ON window"),
        Line2D([], [], ls="", marker="v", color=COLOR["yoyo_accept"], markeredgecolor="black",
               label="late cache starts being used"),
        Line2D([], [], ls="", marker=".", color="0.3", label="pedestrian within 0.6 m (1 s sample)"),
        Line2D([], [], ls="", marker="|", color="black", markersize=12, mew=2.5,
               label="robot stays within 0.5 m for the rest of the run"),
        Line2D([], [], ls="", marker="*", color="#2ca02c", markeredgecolor="black", markersize=12,
               label="goal reached"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8.5, frameon=False)

    for ep in EPISODES:
        for cond in CONDITIONS:
            c = cells[ep, cond]
            xq = c.nav["minimum_pedestrian_root_distance_m"]
            quad.scatter(xq, c.streak, s=110, color=COLOR[cond], edgecolor=EP_COLOR[ep], lw=2.2,
                         marker=collision_marker(c), zorder=3)
            quad.annotate(str(ep), (xq, c.streak), xytext=(6, 4), textcoords="offset points", fontsize=8)
    quad.set_xscale("log")
    quad.set_yscale("symlog", linthresh=10)
    quad.set_xlim(0.004, 4)
    quad.set_ylim(-3, 8000)
    quad.axvspan(0.004, HUMAN_THRESHOLD_M, color=HUMAN_COLOR, alpha=0.08)
    quad.axhspan(CONTACT_FRAMES, 8000, color=PHYSICAL_COLOR, alpha=0.08)
    quad.axvline(HUMAN_THRESHOLD_M, color=HUMAN_COLOR, ls="--")
    quad.axvline(NEAR_MISS_M, color="0.4", ls=":")
    quad.axhline(CONTACT_FRAMES, color=PHYSICAL_COLOR, ls="--")
    quad.text(0.005, 6000, "human + physical", fontsize=9, va="top", fontweight="bold")
    quad.text(3.6, 6000, "physical only", fontsize=9, va="top", ha="right", color=PHYSICAL_COLOR, fontweight="bold")
    quad.text(0.005, 30, "human only", fontsize=9, color=HUMAN_COLOR, fontweight="bold")
    quad.text(3.6, 30, "no collision", fontsize=9, ha="right", fontweight="bold")
    quad.set_xlabel("closest robot-to-pedestrian XY root distance in run (m, log)")
    quad.set_ylabel("longest consecutive contact streak (frames, symlog)")
    quad.set_title("The two independent collision rules\n(fill = condition, edge = episode, shape = flags)",
                   fontsize=11)
    quad.grid(alpha=0.3, which="both")

    flagged, below = [], []
    for ep in EPISODES:
        for cond in CONDITIONS:
            c = cells[ep, cond]
            if c.physical_flag:
                state = (f"inside final 0.5 m region from {c.stall / FPS:.0f} s" if c.stall is not None
                         else "no final 0.5 m region")
                end = c.physical_flag[0] - CONTACT_FRAMES + c.streak
                flagged.append(f"  ep{ep} {LABEL[cond]}: {state}; flag at {c.physical_flag[1] / FPS:.0f} s; "
                               f"longest streak {c.streak} fr, implied end ≈ frame {end} of {c.n - 1}")
            elif c.streak:
                below.append(f"ep{ep} {LABEL[cond]} {c.streak} fr")
    wrap = lambda s: textwrap.fill(s, 70, subsequent_indent="    ")  # noqa: E731
    notes.text(0, 1, "\n".join([
        "How DynaNav labels a collision (benchmark.py)",
        wrap(f"• Human: any frame where the robot root and a character's SkelRoot are < {HUMAN_THRESHOLD_M} m "
             "apart in XY (line 708). A proximity proxy: no force, no body geometry. Red bars are stdout-order "
             "brackets between the accepted commands printed around each collision line."),
        wrap(f"• Physical: contact-sensor force ≥ 10 N on {CONTACT_FRAMES} consecutive frames "
             f"({CONTACT_FRAMES / FPS:.1f} s; lines 131-132, 1295). The flag prints once per run, so only the first "
             "sustained contact is timestamped. The streak is the longest run of contact frames, not a count "
             "of impacts, and the contact partner is not logged."),
        "• Total collision = human OR physical.",
        "",
        "Physical flags in this package",
        *[wrap(s) for s in flagged],
        wrap("  Below the 100-frame threshold: " + ", ".join(below) + "."),
        "  (implied end assumes the flagged contact is the longest streak.)",
    ]), va="top", ha="left", fontsize=9, family="monospace", transform=notes.transAxes)
    fig.suptitle("Collision types in Trial 3: human proximity vs sustained physical contact", fontsize=14)
    save(fig, out / "fig4_collision_types.png")


# ----------------------------------------------------------------------------- storyboard


def storyboard(cells: dict, ep: int, kind: str, key, out: Path) -> None:
    """One yoyo moment: what the VLM saw, what the robot saw when that cache was first used, and what followed."""
    yoyo, base, nokv = cells[ep, "yoyo_accept"], cells[ep, "baseline"], cells[ep, "no_kv"]
    encounter = None
    if kind == "cache":
        cache_id = int(key)
        frame = yoyo.first_use(cache_id)[0]
        what = f"late cache {cache_id} starts being used"
    elif kind == "human":
        encounter = next(e for e in yoyo.humans if e["actor"] == key)
        frame = encounter["lo"]
        cache_id = yoyo.cache_at(frame)
        what = f"human encounter with {key}"
    else:
        frame = yoyo.stall
        cache_id = yoyo.cache_at(frame)
        what = f"the robot enters the final 0.5 m region it never leaves (moving {yoyo.speed[frame]:.2f} m/s)"
    used_at, age_at_use = yoyo.first_use(cache_id)
    pub = yoyo.pubs[cache_id]
    pose = source_pose(yoyo.enq[cache_id])
    text_wps = text_waypoints_in_world(pub["response_text"], pose)
    think = " ".join((THINK.search(pub["response_text"]) or [None, ""])[1].split())
    context = sorted((yoyo.root / "semantic_inputs" / f"request_{cache_id:06d}").glob("*.jpg"))
    context_frames = [int(CONTEXT_FRAME.search(p.name).group(1)) for p in context]
    later = min(frame + 10 * FPS, yoyo.n - 1)
    baseline_note = []
    if encounter:
        track = base.peds[encounter["prim"]]
        window = (track[:, 0] >= encounter["lo"] - FPS) & (track[:, 0] <= encounter["hi"] + FPS)
        baseline_note = textwrap.wrap(
            f"baseline over the same frames: {encounter['actor']} closest 1 Hz sample {track[window, 3].min():.2f} m; "
            f"baseline run minimum to any person {base.nav['minimum_pedestrian_root_distance_m']:.2f} m; "
            f"baseline human flag: {'yes' if base.human else 'no'}", 58)

    fig = plt.figure(figsize=(21, 10.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[4, 1.5], wspace=0.16, left=0.03, right=0.98, top=0.9, bottom=0.14)
    left = outer[0].subgridspec(3, 4, height_ratios=[1, 1, 1.25], hspace=0.25, wspace=0.05)
    right = outer[1].subgridspec(2, 1, height_ratios=[0.8, 1.7], hspace=0.08)

    for i in range(4):
        ax = fig.add_subplot(left[0, i])
        ax.axis("off")
        if i < len(context):
            ax.imshow(load_image(context[i], (640, 360)))
            ax.set_title(f"VLM input {i + 1}: frame {context_frames[i]} (t = {context_frames[i] / FPS:.1f} s)",
                         fontsize=9.5, color="#555")
        elif i == len(context):
            ax.text(0.5, 0.5, f"this request had {len(context)} input frame(s)", ha="center", va="center",
                    fontsize=10, color="#777", transform=ax.transAxes)

    use_image = yoyo.root / "cache_use_images" / f"cache_{cache_id:06d}_used_frame_{used_at:06d}.jpg"
    shots = (
        (use_image, f"yoyo camera at first use of cache {cache_id} (fr {used_at})", COLOR["yoyo_accept"]),
        (yoyo.tp_path(frame), f"yoyo at the moment (fr {frame}, t = {frame / FPS:.1f} s)", COLOR["yoyo_accept"]),
        (base.tp_path(frame), f"baseline at the same frame {frame}", COLOR["baseline"]),
        (yoyo.tp_path(later), f"yoyo 10 s later (fr {later})", COLOR["yoyo_accept"]),
    )
    for i, (path, title, color) in enumerate(shots):
        ax = fig.add_subplot(left[1, i])
        ax.axis("off")
        ax.imshow(load_image(path, (640, 360)))
        ax.set_title(title, fontsize=9.5, color=color)

    text = fig.add_subplot(right[0])
    text.axis("off")
    text.text(0, 1, "\n".join([
        f"Cache {cache_id} (episode {ep}, yoyo)",
        f"VLM input frames: {', '.join(map(str, context_frames))}",
        f"merged at frame {pub['merge_step']}; late_at_delivery={pub['late_at_delivery']}",
        f"first used at frame {used_at} (t = {used_at / FPS:.1f} s), {age_at_use:.1f} s old",
        f"age at the moment shown: {yoyo.age[frame]:.1f} s",
        *baseline_note,
        "",
        "VLM reasoning for this cache:",
        *textwrap.wrap(think, 58),
    ]), va="top", ha="left", fontsize=9, family="monospace", transform=text.transAxes)

    mp = fig.add_subplot(right[1])
    for c in (nokv, base, yoyo):
        mp.plot(c.x, c.y, color=COLOR[c.cond], lw=1, alpha=0.25)
    lo, hi = max(min(pub["source_step"], frame) - 5 * FPS, 0), min(max(frame, used_at) + 15 * FPS, yoyo.n)
    for c in (base, yoyo):
        mp.plot(c.x[lo:hi], c.y[lo:hi], color=COLOR[c.cond], lw=2.4,
                label=f"{LABEL[c.cond]} path {lo / FPS:.0f}-{hi / FPS:.0f} s (dot = moment)")
        mp.scatter(c.x[frame], c.y[frame], s=90, color=COLOR[c.cond], edgecolor="black", zorder=5)
    mp.plot(text_wps[:, 0], text_wps[:, 1], color="#ff7f0e", ls=":", lw=2, marker=".",
            label="VLM text waypoints from the pose it saw (context only, not executed)")
    mp.scatter(*pose[:2], marker="o", s=110, facecolor="white", edgecolor="#ff7f0e", lw=2.5, zorder=6,
               label=f"pose the VLM saw (frame {pub['source_step']})")
    mp.scatter(yoyo.x[used_at], yoyo.y[used_at], marker="P", s=160, color="#ff7f0e", edgecolor="black", zorder=6,
               label=f"pose at first use (frame {used_at})")
    if encounter:
        track = yoyo.peds[encounter["prim"]]
        near = np.abs(track[:, 0] - frame) <= 10 * FPS
        mp.plot(track[near, 1], track[near, 2], color=HUMAN_COLOR, lw=2, marker=".", label=f"{encounter['actor']} ±10 s")
        mp.scatter(yoyo.x[frame], yoyo.y[frame], marker="X", s=200, color=HUMAN_COLOR, edgecolor="black", zorder=7,
                   label=f"human encounter ({encounter['min_d']:.2f} m)")
    if yoyo.physical_flag and lo <= yoyo.physical_flag[1] < hi:
        f = yoyo.physical_flag[1]
        mp.scatter(yoyo.x[f], yoyo.y[f], marker="s", s=150, color=PHYSICAL_COLOR, edgecolor="black", zorder=7,
                   label="physical contact flag")
    gx, gy = yoyo.res["goal"][:2]
    mp.add_patch(Circle((gx, gy), GOAL_RADIUS_M, fill=False, color="#2ca02c", lw=2))
    mp.scatter(gx, gy, marker="*", s=200, color="#2ca02c", edgecolor="black", zorder=6, label="goal (1.5 m radius)")
    pts = np.vstack([text_wps, [[yoyo.x[f], yoyo.y[f]] for f in range(lo, hi, 15)], [[base.x[frame], base.y[frame]]]])
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 + 2.5
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    mp.set_xlim(cx - span, cx + span)
    mp.set_ylim(cy - span, cy + span)
    mp.set_aspect("equal")
    mp.grid(alpha=0.3)
    mp.set_title("Top-down (world XY, m)", fontsize=11)
    mp.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)

    ts = fig.add_subplot(left[2, :])
    w0, w1 = max(frame - 25 * FPS, 0), min(frame + 25 * FPS, yoyo.n)
    t = np.arange(w0, w1) / FPS
    for c in (base, nokv, yoyo):
        ts.plot(t, c.goal[w0:w1], color=COLOR[c.cond], lw=2, label=f"{LABEL[c.cond]}: goal distance")
    ts.axhline(GOAL_RADIUS_M, color="#2ca02c", ls=":", lw=1)
    ts.set_ylabel("distance to goal (m), solid")
    ts.set_xlabel("simulation time (s)")
    age_ax = ts.twinx()
    for c in (base, yoyo):
        age_ax.plot(t, c.age[w0:w1], color=COLOR[c.cond], lw=1.2, ls="--", alpha=0.9, label=f"{LABEL[c.cond]}: cache age")
    age_ax.set_ylabel("age of the cache in use (s), dashed")
    for a, b in yoyo.attack_on:
        ts.axvspan(a / FPS, b / FPS, color=ATTACK_COLOR, alpha=0.2)
    for e in yoyo.humans:
        ts.axvspan(e["lo"] / FPS, max(e["hi"], e["lo"] + FPS // 2) / FPS, color=HUMAN_COLOR, alpha=0.35)
    if yoyo.physical_flag:
        ts.axvline(yoyo.physical_flag[1] / FPS, color=PHYSICAL_COLOR, lw=2)
    ts.axvline(used_at / FPS, color="black", lw=1.5, ls="-.")
    ts.axvline(frame / FPS, color="black", lw=2.5, alpha=0.35)
    if yoyo.stall is not None:
        ts.axvline(yoyo.stall / FPS, color="black", lw=2, ls=":")
    ts.set_xlim(t[0], t[-1])
    ts.grid(alpha=0.3)
    h1, l1 = ts.get_legend_handles_labels()
    h2, l2 = age_ax.get_legend_handles_labels()
    ts.legend(h1 + h2 + [Line2D([], [], color="black", lw=2.5, alpha=0.35),
                         Line2D([], [], color="black", ls="-.", lw=1.5),
                         Patch(color=ATTACK_COLOR, alpha=0.3), Patch(color=HUMAN_COLOR, alpha=0.4),
                         Line2D([], [], color=PHYSICAL_COLOR, lw=2), Line2D([], [], color="black", ls=":", lw=2)],
              l1 + l2 + ["moment shown", f"cache {cache_id} first used", "attack ON (yoyo)",
                         "yoyo human encounter", "yoyo contact flag", "yoyo enters final 0.5 m region"],
              fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=6, frameon=False)

    fig.suptitle(
        f"Episode {ep}, yoyo: {what} at t = {frame / FPS:.1f} s — the action expert is conditioned on the current "
        f"image plus a cached VLM state generated from frames up to t = {max(context_frames) / FPS:.1f} s "
        f"({yoyo.age[frame]:.1f} s old)",
        fontsize=13,
    )
    save(fig, out / f"storyboard_ep{ep}_{kind}_{key}.png")


# ----------------------------------------------------------------------------- video


def scene_video(cells: dict, ep: int, out: Path, step: int = 3) -> None:
    runs = [cells[ep, c] for c in CONDITIONS]
    yoyo = cells[ep, "yoyo_accept"]
    n = max(c.n for c in runs)
    frames = np.arange(0, n, step)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.2, wspace=0.1, left=0.035, right=0.985, top=0.92,
                          bottom=0.06)
    cams = [fig.add_subplot(gs[0, i]) for i in range(3)]
    mp = fig.add_subplot(gs[1, :2])
    side = gs[1, 2].subgridspec(2, 1, hspace=0.4)
    t_age = fig.add_subplot(side[0])
    t_goal = fig.add_subplot(side[1], sharex=t_age)

    images, status = [], []
    for ax, c in zip(cams, runs):
        ax.axis("off")
        images.append(ax.imshow(load_image(c.tp_path(0))))
        ax.set_title(LABEL[c.cond], fontsize=13, color=COLOR[c.cond], fontweight="bold")
        status.append(ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                              family="monospace", color="white",
                              bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none"}))

    for c in runs:
        mp.plot(c.x, c.y, color=COLOR[c.cond], lw=1, alpha=0.18)
    trails = [mp.plot([], [], color=COLOR[c.cond], lw=2.4, label=LABEL[c.cond])[0] for c in runs]
    robots = [mp.scatter([], [], s=130, color=COLOR[c.cond], edgecolor="black", lw=1.5, zorder=6) for c in runs]
    crowds = []  # each run's own recorded pedestrians; they diverge across runs (cause not logged)
    for c in runs:
        actors = sorted(c.peds)
        hit = {e["prim"] for e in c.humans}
        xy = np.stack([
            np.column_stack([np.interp(frames, c.peds[a][:, 0], c.peds[a][:, 1]),
                             np.interp(frames, c.peds[a][:, 0], c.peds[a][:, 2])])
            for a in actors
        ], axis=1)  # (frames, actors, 2)
        layer = mp.scatter(xy[0, :, 0], xy[0, :, 1], s=[75 if a in hit else 20 for a in actors],
                           c=[HUMAN_COLOR if a in hit else COLOR[c.cond] for a in actors],
                           edgecolors=[COLOR[c.cond] if a in hit else "black" for a in actors],
                           linewidths=[2 if a in hit else 0.3 for a in actors], alpha=0.75, zorder=4)
        crowds.append((xy, layer))
    events = []  # (frame, artist) revealed once reached
    for c in runs:
        for e in c.humans:
            events.append((e["lo"], mp.scatter(c.x[e["lo"]], c.y[e["lo"]], marker="X", s=170, color=HUMAN_COLOR,
                                               edgecolor=COLOR[c.cond], lw=2, zorder=7, visible=False)))
        if c.physical_flag:
            f = c.physical_flag[1]
            events.append((f, mp.scatter(c.x[f], c.y[f], marker="s", s=150, color=PHYSICAL_COLOR,
                                         edgecolor=COLOR[c.cond], lw=2, zorder=7, visible=False)))
    gx, gy = yoyo.res["goal"][:2]
    mp.add_patch(Circle((gx, gy), GOAL_RADIUS_M, fill=False, color="#2ca02c", lw=2))
    mp.scatter(gx, gy, marker="*", s=260, color="#2ca02c", edgecolor="black", zorder=6)
    mp.scatter(yoyo.x[0], yoyo.y[0], marker="o", s=90, facecolor="white", edgecolor="black", zorder=6)
    xs = np.concatenate([c.x for c in runs] + [[gx]])
    ys = np.concatenate([c.y for c in runs] + [[gy]])
    mp.set_xlim(xs.min() - 3, xs.max() + 3)
    mp.set_ylim(ys.min() - 3, ys.max() + 3)
    mp.set_aspect("equal", adjustable="datalim")
    mp.grid(alpha=0.3)
    mp.legend(handles=[*trails,
                       Line2D([], [], ls="", marker="o", color="0.6", markeredgecolor="black",
                              label="pedestrian (fill = the run it belongs to)"),
                       Line2D([], [], ls="", marker="o", color=HUMAN_COLOR, markeredgecolor="black", ms=9,
                              label="pedestrian in that run's human collision"),
                       Line2D([], [], ls="", marker="X", color=HUMAN_COLOR, ms=10, label="human collision (< 0.2 m)"),
                       Line2D([], [], ls="", marker="s", color=PHYSICAL_COLOR, ms=10, label="physical contact flag"),
                       Line2D([], [], ls="", marker="*", color="#2ca02c", markeredgecolor="black", ms=13,
                              label="goal")],
              loc="upper left", fontsize=8.5, framealpha=0.9)
    mp.set_title("Top-down view (world XY, m) — condition-specific recorded pedestrian trajectories "
                 "(cross-run divergence cause unknown)", fontsize=10)

    t_all = np.arange(n) / FPS
    for c in (cells[ep, "baseline"], yoyo):
        t_age.plot(t_all[: c.n], c.age, color=COLOR[c.cond], lw=1.5, label=LABEL[c.cond])
    for a, b in yoyo.attack_on:
        t_age.axvspan(a / FPS, b / FPS, color=ATTACK_COLOR, alpha=0.3)
    late = [(f, age) for f, _, age, is_late in yoyo.first_uses if is_late]
    t_age.scatter([f / FPS for f, _ in late], [a for _, a in late], marker="v", s=40, color=COLOR["yoyo_accept"],
                  edgecolor="black", zorder=4, label="late cache starts being used")
    t_age.set_ylabel("cache age (s)")
    t_age.set_title("Age of the KV cache the action expert is using (shaded = attack ON)", fontsize=10)
    t_age.legend(fontsize=8, loc="upper left")
    t_age.grid(alpha=0.3)
    for c in runs:
        t_goal.plot(t_all[: c.n], c.goal, color=COLOR[c.cond], lw=1.6)
        for e in c.humans:
            t_goal.axvline(e["lo"] / FPS, color=HUMAN_COLOR, lw=1.2, ls="--")
        if c.physical_flag:
            t_goal.axvline(c.physical_flag[1] / FPS, color=PHYSICAL_COLOR, lw=1.5, ls="--")
    t_goal.axhline(GOAL_RADIUS_M, color="#2ca02c", ls=":")
    t_goal.set_ylabel("goal distance (m)")
    t_goal.set_xlabel("simulation time (s)")
    t_goal.set_title("Distance to goal (dashed = collision flags)", fontsize=10)
    t_goal.grid(alpha=0.3)
    cursors = [t_age.axvline(0, color="black", lw=1.2), t_goal.axvline(0, color="black", lw=1.2)]
    title = fig.suptitle("", fontsize=14)

    def status_text(c: Cell, f: int) -> str:
        if f >= c.n:
            return "run ended (" + ("goal reached" if c.res["success"] else "timeout") + ")"
        lines = [f"goal {c.goal[f]:5.1f} m"]
        cache = c.cache_at(f)
        if not c.kv:
            lines.append("KV cache withheld")
        elif cache is None:
            lines.append("no cache in use yet")
        else:
            lines.append(f"cache {cache}, {c.age[f]:4.1f} s old")
        if any(a <= f < b for a, b in c.attack_on):
            lines.append("ATTACK ON: VLM queue flooded")
        recent = [(cid, age) for cf, cid, age, is_late in c.first_uses if is_late and cf <= f < cf + 2 * FPS]
        lines += [f">> LATE cache {cid} now in use ({age:.1f} s old)" for cid, age in recent]
        lines += [f"!! HUMAN < 0.2 m: {e['actor']}" for e in c.humans if e["lo"] <= f <= e["hi"] + FPS // 2]
        if c.physical_flag and f >= c.physical_flag[1]:
            lines.append(f"!! CONTACT FLAG (≥10 N for {CONTACT_FRAMES} frames)")
        if c.stall is not None and f >= c.stall:
            lines.append("inside final 0.5 m region")
        return "\n".join(lines)

    w, h = fig.canvas.get_width_height()
    out_path = out / f"scene_ep{ep}.mp4"
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r",
         str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27", "-preset", "medium",
         "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE,
    )
    instruction = textwrap.shorten(yoyo.res["instruction"], 110)
    for k, f in enumerate(frames):
        for c, image, text, trail, robot in zip(runs, images, status, trails, robots):
            g = min(f, c.n - 1)
            image.set_data(load_image(c.tp_path(g)))
            text.set_text(status_text(c, f))
            trail.set_data(c.x[: g + 1], c.y[: g + 1])
            robot.set_offsets([[c.x[g], c.y[g]]])
        for xy, layer in crowds:
            layer.set_offsets(xy[k])
        for ef, artist in events:
            artist.set_visible(f >= ef)
        for cursor in cursors:
            cursor.set_xdata([f / FPS, f / FPS])
        title.set_text(f"Episode {ep} — t = {f / FPS:5.1f} s (frame {f}), played at {step}× speed\n\"{instruction}\"")
        fig.canvas.draw()
        frame_bytes = bytes(fig.canvas.buffer_rgba())
        ffmpeg.stdin.write(frame_bytes)
        if k == len(frames) - 1:
            for _ in range(2 * FPS):
                ffmpeg.stdin.write(frame_bytes)
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise SystemExit(f"ffmpeg failed for {out_path}")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence", type=Path, default=here.parent / "trial3_evidence_20261017T174723891246Z")
    parser.add_argument("--out", type=Path, default=here)
    parser.add_argument("--skip-video", action="store_true", help="render the PNG figures only")
    args = parser.parse_args()

    cells = load_cells(args.evidence)
    print_summary(cells)
    args.out.mkdir(parents=True, exist_ok=True)
    fig_condition_outcomes(cells, args.out)
    fig_latency_per_episode(cells, args.out)
    fig_latency_binned(cells, args.out)
    fig_collision_types(cells, args.out)
    for ep, kind, key in STORYBOARD_MOMENTS:
        storyboard(cells, ep, kind, key, args.out)
    if not args.skip_video:
        for ep in VIDEO_EPISODES:
            scene_video(cells, ep, args.out)


if __name__ == "__main__":
    main()
