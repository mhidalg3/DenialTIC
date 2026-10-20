"""CPU integration tests with dummy learned modules; no Isaac/weights required."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from dataclasses import asdict
import numpy as np
import torch

HERE=Path(__file__).resolve()
PAYLOAD=HERE.parents[1]/"payload"
sys.path.insert(0,str(PAYLOAD if PAYLOAD.exists() else HERE.parents[2]))
from trial4_adapter import Trial4Mixin, rotation_wxyz
from trial4_core import Settings
from test_core import InlineExecutor

class Vision:
    def extract_feature(self,x): return torch.ones((1,2,4),dtype=x.dtype)

class Head(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.caches=[]; self.states=[]
    def forward(self, emb, state, kv_cache=None):
        self.caches.append(kv_cache); self.states.append(state.clone())
        return torch.zeros((1,30,2),dtype=emb.dtype)

class Fake(Trial4Mixin):
    def __init__(self):
        self.device=torch.device("cpu")
        self.vlm=Vision(); self.action_expert=Head().eval()
        self.generated=0
        self._executor=InlineExecutor()
        self._trial4_initialize()
    def generate_and_extract_kv_cache(self,*a):
        self.generated+=1
        return "test",(("key",f"value{self.generated}"),)

class AdapterTests(unittest.TestCase):
    def run_mode(self, mode):
        with tempfile.TemporaryDirectory() as t:
            env={"TICVLA_T4_CONFIG_JSON":json.dumps(asdict(Settings(cache_mode=mode))),
                 "TICVLA_T4_OUTPUT":t,"TICVLA_T4_SEED":"1"}
            with patch.dict(os.environ, env), patch.dict(sys.modules, {"vision_utils":types.SimpleNamespace(load_image=lambda *a,**k: torch.zeros((1,3,2,2))) }):
                model=Fake()
                try:
                    for step in (0,3,6):
                        p=Path(t)/f"front_frame_{step:06d}.jpg";p.write_bytes(b"mock")
                        result=model.predict_async([str(p)],"go",torch.zeros(6),step,
                            {"position":[step,0,0],"quaternion":[1,0,0,0]})
                        self.assertTrue(result[3])
                    return model.action_expert.caches, model.action_expert.states, model._trial4_summary()
                finally:
                    model.cleanup();model._t4_log.file.close()
    def test_none_really_passes_no_cache(self):
        caches,_,stats=self.run_mode("none")
        self.assertTrue(all(c is None for c in caches))
        self.assertEqual(stats["counts"]["action_calls_without_kv"],3)
    def test_normal_supplies_cache(self):
        caches,_,_=self.run_mode("normal")
        self.assertTrue(all(c is not None for c in caches))
    def test_delayed_cache_reaches_head_with_original_age_and_pose(self):
        with tempfile.TemporaryDirectory() as t:
            cfg=Settings(launch_delay_frames=360)
            env={"TICVLA_T4_CONFIG_JSON":json.dumps(asdict(cfg)),"TICVLA_T4_OUTPUT":t,"TICVLA_T4_SEED":"1"}
            with patch.dict(os.environ,env), patch.dict(sys.modules,{"vision_utils":types.SimpleNamespace(load_image=lambda *a,**k:torch.zeros((1,3,2,2)))}):
                model=Fake()
                try:
                    for step in (0,3,100,363,366):
                        p=Path(t)/f"front_frame_{step:06d}.jpg";p.write_bytes(b"mock")
                        model.predict_async([str(p)],"go",torch.zeros(6),step,
                            {"position":[step,0,0],"quaternion":[1,0,0,0]})
                    self.assertEqual(model.action_expert.caches[-1][0][1],"value2")
                    # Source frame is 3, NOT publication frame 363/366.
                    state=model.action_expert.states[-1]
                    self.assertAlmostEqual(float(state[0,3,0]),363.,delta=2.) # BF16 quantization
                    self.assertAlmostEqual(float(state[0,5,0]),363/30,delta=.1)
                    stats=model._trial4_summary()
                    self.assertEqual(stats["counts"]["stale_drops"],0)
                    self.assertGreater(stats["counts"]["late_published_caches_used"],0)
                finally:
                    model.cleanup();model._t4_log.file.close()
    def test_quaternion_rotation(self):
        R=rotation_wxyz([2**-.5,0,0,2**-.5])
        np.testing.assert_allclose(R@np.array([1,0,0]), [0,1,0],atol=1e-8)

if __name__=="__main__":unittest.main()
