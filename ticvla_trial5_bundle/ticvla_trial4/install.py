#!/usr/bin/env python3
"""Install Trial 4 on an existing TIC-VLA checkout; no downloads or dependencies.

Dry run: python3 install.py --repo ~/TIC-VLA
Install: python3 install.py --repo ~/TIC-VLA --apply
The current files are backed up before any replacement. Rollback instructions
are written into the backup directory. No old results or environment files are
removed. The learned architecture is taken from the verified base commit;
local benchmark/behavior changes (e.g. headless) are retained.
"""
from __future__ import annotations
import argparse
import ast
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

BASE = "9fa6f8b66b9e121d5df5df071297bba8e5353ebb"
MARKER = "# TICVLA_TRIAL4_INSTALLED"


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True)


def method(source, cls, name):
    tree = ast.parse(source)
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    matches = [n for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"Need exactly one {cls}.{name}, got {len(matches)}")
    return matches[0]


def transform_method(source, cls, name, transform):
    node = method(source, cls, name)
    lines = source.splitlines(keepends=True)
    lo = min([node.lineno]+[x.lineno for x in node.decorator_list])-1
    hi = node.end_lineno
    old = "".join(lines[lo:hi])
    new = transform(old)
    return "".join(lines[:lo])+new+"".join(lines[hi:])


def once(text, old, new, label):
    if text.count(old) != 1:
        raise ValueError(f"{label}: expected one anchor, found {text.count(old)}. No files changed.")
    return text.replace(old, new, 1)


def patch_model(base):
    s = base
    for name in ("_start_kv_cache_generation", "_poll_kv_cache_future", "predict_async",
                 "reset_episode_state", "cleanup"):
        s = transform_method(s, "TICVLA", name, lambda _: "")
    s = once(s, "class TICVLA(nn.Module):", "from trial4_adapter import Trial4Mixin\n\nclass TICVLA(Trial4Mixin, nn.Module):", "model inheritance")
    s = transform_method(s, "TICVLA", "__init__", lambda x: x.rstrip()+"\n        self._trial4_initialize()\n")
    return MARKER+"\n"+s


def patch_behavior(source):
    def init(s):
        anchor = "        self._waiting_for_inference_after_backup = False"
        # Existing annotated assignment remains; insert before that whole line.
        if "self._enable_backup_recovery" in s:
            tree = ast.parse(textwrap.dedent(s))
            assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Attribute) and t.attr == "_enable_backup_recovery" for t in n.targets)]
            if len(assignments) != 1:
                raise ValueError("Cannot unambiguously bind existing backup switch to Trial 4 settings")
            node = assignments[0]
            lines = s.splitlines(keepends=True)
            lines[node.lineno-1:node.end_lineno] = [
                "        from trial4_adapter import backup_enabled\n",
                "        self._enable_backup_recovery = backup_enabled()\n"]
            return "".join(lines)
        return once(s, anchor,
                    "        from trial4_adapter import backup_enabled\n"
                    "        self._enable_backup_recovery = backup_enabled()\n"+anchor,
                    "backup initialization")
    s = transform_method(source, "NovaCarterTICVLA", "on_init", init)
    def update(text):
        anchor = "        self._capture_camera_image()"
        text = once(text, anchor, anchor+
                    "\n        if self._ticvla_model is not None:\n"
                    "            self._ticvla_model.trial4_tick(self._frame_count, self._is_backing_up)\n",
                    "per-frame tick")
        anchor = "if self._first_inference_complete and not self._is_backing_up and not self._waiting_for_inference_after_backup:"
        if anchor in text:
            return once(text, anchor,
                        "if self._enable_backup_recovery and self._first_inference_complete and not self._is_backing_up and not self._waiting_for_inference_after_backup:",
                        "backup gate")
        tree = ast.parse(textwrap.dedent(text))
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                attrs = {n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)}
                if {"_enable_backup_recovery", "_first_inference_complete", "_is_backing_up", "_waiting_for_inference_after_backup"} <= attrs:
                    return text
        raise ValueError("Could not find the backup-start guard; no files changed")
    s = transform_method(s, "NovaCarterTICVLA", "on_update", update)
    def command(text):
        anchor = "                self._current_action = np.array([v_cmd, w_cmd], dtype=np.float32)"
        return once(text, anchor, anchor+
                    "\n                self._ticvla_model.trial4_record_command(current_step, v_cmd, w_cmd, wps)\n",
                    "accepted command trace")
    s = transform_method(s, "NovaCarterTICVLA", "_inference_task", command)
    return MARKER+"\n"+s


def patch_benchmark(source):
    def worker(s):
        anchor = "        episode_config = self._benchmark_config[episode_name]"
        s = once(s, anchor, anchor+
                 "\n        from trial4_metrics import NavTrace, attach\n"
                 "        self._t4_trace = NavTrace()\n", "navigation metrics init")
        return once(s, "            return result", "            return attach(result, self)", "result enrichment")
    s = transform_method(source, "BenchmarkRunner", "run_single_episode_worker", worker)
    def position(text):
        anchor = "                self._robot_path.append(self._robot_position.copy())"
        return once(text, anchor, anchor+
                    "\n                if hasattr(self, '_t4_trace'):\n"
                    "                    self._t4_trace.position(self._simulation_frame_count, self._robot_position, self._goal_position, self._goal_threshold)\n",
                    "position trace")
    s = transform_method(s, "BenchmarkRunner", "_update_robot_position", position)
    def human(text):
        anchor = "                distance = np.linalg.norm(robot_2d - char_2d)"
        return once(text, anchor, anchor+
                    "\n                if hasattr(self, '_t4_trace'):\n"
                    "                    self._t4_trace.human(self._simulation_frame_count, distance, str(skelroot_prim.GetPath()), char_pos)\n",
                    "pedestrian distance")
    s = transform_method(s, "BenchmarkRunner", "_check_collisions", human)
    return MARKER+"\n"+s


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    repo = args.repo.expanduser().resolve()
    if git(repo, "rev-parse", "HEAD").strip() != BASE:
        raise SystemExit(f"This installer targets base {BASE}. It will not overwrite another revision.")
    rels = ["DynaNav/ticvla.py", "DynaNav/behavior/nova_carter_test_ticvla.py", "DynaNav/benchmark.py"]
    current = {rel: (repo/rel).read_text() for rel in rels}
    if any(MARKER in s for s in current.values()):
        raise SystemExit("Trial 4 is already installed. Use its existing runner, or restore the recorded backup first.")
    generated = {
        rels[0]: patch_model(git(repo, "show", f"HEAD:{rels[0]}")),
        rels[1]: patch_behavior(current[rels[1]]),
        rels[2]: patch_benchmark(current[rels[2]]),
    }
    payload = Path(__file__).resolve().parent/"payload"
    for path in sorted(payload.glob("*.py")):
        rel = f"DynaNav/{path.name}" if path.name.startswith("trial4_") else f"DynaNav/trial4/{path.name}"
        if (repo/rel).exists():
            raise SystemExit(f"Refusing to replace an existing helper: {repo/rel}")
        generated[rel] = path.read_text()
    for rel, source in generated.items():
        compile(source, rel, "exec")
    print("Source anchors validated; all generated Python parses.")
    print("Will replace async plumbing, add recovery switch/metrics, and install runner:")
    for rel in generated:
        print(" ", rel)
    if not args.apply:
        print("Dry run only. Re-run with --apply to install.")
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = repo/"trial4_backups"/stamp
    backup.mkdir(parents=True, exist_ok=False)
    manifest = {"base_commit": BASE, "backup": str(backup), "files": {}}
    for rel in generated:
        dst = repo/rel
        if dst.exists():
            save = backup/rel
            save.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, save)
        manifest["files"][rel] = {"existed": dst.exists(), "installed_sha256": sha(generated[rel].encode()),
                                   "previous_sha256": sha(dst.read_bytes()) if dst.exists() else None}
    (backup/"before.diff").write_text(git(repo, "diff", "HEAD", "--", *rels))
    patch = "".join("".join(difflib.unified_diff(current.get(rel, "").splitlines(True), s.splitlines(True),
                                               fromfile=rel+".before", tofile=rel)) for rel,s in generated.items())
    (backup/"trial4_changes.diff").write_text(patch)
    (backup/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for rel, source in generated.items():
        dst = repo/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        temp = dst.with_name(dst.name+".trial4tmp")
        temp.write_text(source)
        temp.replace(dst)
    # Only CPU/runtime tests belong in the container-mounted checkout.
    # test_bootstrap.py and test_install.py depend on the ZIP package root;
    # installing them under DynaNav/trial4/tests made `vm.sh check` fail.
    tests = Path(__file__).resolve().parent/"tests"
    dsttests = repo/"DynaNav/trial4/tests"
    dsttests.mkdir(parents=True, exist_ok=False)
    for filename in ("test_core.py", "test_adapter.py", "test_suite.py"):
        shutil.copy2(tests/filename, dsttests/filename)
    (repo/"DynaNav/trial4/install_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (backup/"RESTORE.txt").write_text(
        "Stop experiments before rollback. Restore these three backed-up files to the same repository paths:\n"+
        "\n".join(rels)+"\nRemove the helper paths listed with existed=false in manifest.json only after review.\n")
    print(f"Installed. Original files and complete diff: {backup}")
    print("No GPU/Isaac execution has been performed by this installer.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, StopIteration, subprocess.CalledProcessError) as e:
        raise SystemExit(f"Install aborted: {e}")
