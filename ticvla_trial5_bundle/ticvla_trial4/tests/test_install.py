"""Patch-structure tests on fixtures; actual checkout anchors checked on the VM."""
import ast
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"install.py").exists():
    sys.path.insert(0,str(ROOT))
    import install
else:
    install=None

MODEL='''import torch
import torch.nn as nn
class TICVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = 3
    def forward(self, x):
        return self.weight * x
    def generate_and_extract_kv_cache(self, *args):
        return "test", args
    def _start_kv_cache_generation(self):
        pass
    def _poll_kv_cache_future(self):
        pass
    @torch.inference_mode()
    def predict_async(self):
        pass
    def reset_episode_state(self):
        pass
    def cleanup(self):
        pass
'''
BEHAVIOR='''class NovaCarterTICVLA:
    def on_init(self):
        self._waiting_for_inference_after_backup = False
    def on_update(self):
        self._capture_camera_image()
        if self._first_inference_complete and not self._is_backing_up and not self._waiting_for_inference_after_backup:
            self._is_backing_up = True
    def _inference_task(self):
        if True:
            with self._inference_lock:
                self._current_action = np.array([v_cmd, w_cmd], dtype=np.float32)
'''
BENCHMARK='''APP_CONFIG = {"headless": True}
class BenchmarkRunner:
    async def run_single_episode_worker(self, episode_name):
        episode_config = self._benchmark_config[episode_name]
        try:
            result = {}
            return result
        except Exception:
            raise
    async def _update_robot_position(self):
        if True:
            if True:
                self._robot_path.append(self._robot_position.copy())
    async def _check_collisions(self):
        if True:
            if True:
                distance = np.linalg.norm(robot_2d - char_2d)
'''

@unittest.skipIf(install is None, "Installer tests run in the distribution folder, not the installed runtime")
class InstallTests(unittest.TestCase):
    def test_model_preserves_forward_and_removes_old_async(self):
        patched=install.patch_model(MODEL)
        compile(patched,"model","exec")
        self.assertIn("return self.weight * x",patched)
        self.assertIn("self._trial4_initialize()",patched)
        self.assertNotIn("def predict_async",patched)
        self.assertIn("class TICVLA(Trial4Mixin, nn.Module)",patched)
    def test_behavior_backup_and_tick(self):
        patched=install.patch_behavior(BEHAVIOR)
        compile(patched,"behavior","exec")
        self.assertIn("trial4_tick",patched)
        self.assertIn("self._enable_backup_recovery and self._first",patched)
    def test_existing_backup_switch(self):
        b=BEHAVIOR.replace('        self._waiting', '        self._enable_backup_recovery = True\n        self._waiting',1)
        b=b.replace('if self._first','if self._enable_backup_recovery and self._first')
        patched=install.patch_behavior(b)
        compile(patched,"behavior","exec")
        self.assertIn('self._enable_backup_recovery = backup_enabled()',patched)
    def test_benchmark_preserves_headless_and_enriches_result(self):
        patched=install.patch_benchmark(BENCHMARK)
        compile(patched,"benchmark","exec")
        self.assertIn('"headless": True',patched)
        self.assertIn('return attach(result, self)',patched)
        self.assertIn('_t4_trace.position',patched)
        self.assertIn('_t4_trace.human',patched)
    def test_bad_anchor_fails_before_write(self):
        with self.assertRaises(ValueError):
            install.patch_behavior(BEHAVIOR.replace('self._capture_camera_image()', 'something_else()'))

if __name__=='__main__':unittest.main()
