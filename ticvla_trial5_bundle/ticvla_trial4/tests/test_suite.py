"""Runner invariants and measured crowd evidence; no GPU or simulator required."""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from unittest.mock import patch

HERE=Path(__file__).resolve()
PAYLOAD=HERE.parents[1]/"payload"
if not PAYLOAD.exists(): PAYLOAD=HERE.parents[1]
sys.path.insert(0,str(PAYLOAD))
spec=importlib.util.spec_from_file_location('t4_runner_tests',PAYLOAD/'run_suite.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
from trial4_core import Settings
from trial4_metrics import NavTrace

class SuiteTests(unittest.TestCase):
 def variants(self):
  return runner.conditions('native',types.SimpleNamespace(recovery=False,sample=False,no_visuals=False,
       on_frames=180,off_frames=900,arrival_frames=30,service_frames=120,warmup_frames=3))
 def valid(self,cfg):
  counts=dict(action_calls=10,action_calls_without_kv=10 if cfg.cache_mode=='none' else 0,
      commands_accepted=10,stale_drops=0,forced_drops=0,publishes_total=2,results_merged=2)
  return dict(duration_frames=20,navigation_error=2.0,trial4={"mechanism":{
      "instrumentation_valid":True,"settings":asdict(cfg),"backup_events":0,"counts":counts}})
 def test_exact_three_conditions_and_five_episodes(self):
  v=self.variants()
  self.assertEqual(set(v),{'baseline','no_kv','yoyo_accept'})
  self.assertEqual(runner.SEMANTIC,[20,25,45,75,85])
  self.assertEqual(len(v)*len(runner.SEMANTIC),15)
 def test_all_conditions_accept_late(self):
  for cfg in self.variants().values():
   self.assertEqual(cfg.deadline_frames,0)
   self.assertIn(cfg.cache_mode,('normal','none'))
   self.assertFalse(cfg.backup_enabled)
 def test_only_timing_changes_between_baseline_and_attack(self):
  v=self.variants();a=asdict(v['baseline']);b=asdict(v['yoyo_accept'])
  self.assertEqual({k for k in a if a[k]!=b[k]},{'attack'})
 def test_validator_rejects_any_stale_or_forced_drop(self):
  cfg=Settings();raw=self.valid(cfg)
  self.assertEqual(runner.validate_result(raw,0,cfg),[])
  for name in ['stale_drops','forced_drops']:
   raw=self.valid(cfg);raw['trial4']['mechanism']['counts'][name]=1
   self.assertIn('accept_late_invariant_failed',runner.validate_result(raw,0,cfg))
 def test_validator_rejects_incomplete_publication_and_no_kv_leak(self):
  cfg=Settings(cache_mode='none');raw=self.valid(cfg)
  raw['trial4']['mechanism']['counts']['publishes_total']=1
  self.assertIn('completed_cache_not_published',runner.validate_result(raw,0,cfg))
  raw=self.valid(cfg);raw['trial4']['mechanism']['counts']['action_calls_without_kv']=9
  self.assertIn('no_kv_invariant_failed',runner.validate_result(raw,0,cfg))
 def test_crowd_motion_and_near_robot_evidence(self):
  with tempfile.TemporaryDirectory() as t, patch.dict(os.environ,{
      'TICVLA_T4_CONFIG_JSON':json.dumps(asdict(Settings())), 'TICVLA_T4_OUTPUT':t}):
   trace=NavTrace()
   try:
    trace.human(0,1.5,'person1',[0,0,0]);trace.human(30,1.6,'person1',[.5,0,0])
    trace.human(30,8,'person2',[4,0,0])
    s=trace.summary()
    self.assertEqual(s['pedestrians_observed'],2)
    self.assertEqual(s['pedestrians_moved_over_0_25m'],1)
    self.assertEqual(s['moving_pedestrians_within_2m'],1)
   finally: trace.log.file.close()

class SummaryTests(unittest.TestCase):
 def summary_module(self):
  spec=importlib.util.spec_from_file_location('t4_summary_test_module',PAYLOAD/'summarize.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module
 def session(self,root,name,episode):
  folder=root/'trial4_results'/name; (folder/'provenance').mkdir(parents=True)
  (folder/'provenance/manifest.json').write_text(json.dumps({"source_hashes":{"code":"hash"},"checkpoint_sha256":"ckpt"}))
  records=[]
  for condition in ('baseline','yoyo_accept'):
   cfg=Settings(attack='periodic' if condition=='yoyo_accept' else 'none')
   records.append(dict(episode=episode,seed=6101,condition=condition,settings=asdict(cfg),
        valid=True,exposure={},result=dict(success=True,navigation_error=1.4,path_length=15.,
        duration=30.,collision=0.,spl=.9,trial4={"mechanism":{"counts":{},"distributions":{}},"navigation":{}})))
  (folder/'results.json').write_text(json.dumps(records))
  return folder
 def test_combined_summary_keeps_disjoint_sessions_and_outputs(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); a=self.session(root,'first',20);b=self.session(root,'rest',25)
   module=self.summary_module()
   with patch.object(sys,'argv',['summarize.py',str(a),str(b)]),contextlib.redirect_stdout(io.StringIO()):
    module.main()
   summaries=list((root/'trial4_reports').glob('combined_*/summary.json'))
   self.assertEqual(len(summaries),1)
   data=json.loads(summaries[0].read_text())
   self.assertEqual(data['valid_runs'],4)
   self.assertTrue((summaries[0].parent/'per_episode.csv').exists())
   self.assertTrue((summaries[0].parent/'summary.txt').exists())
   self.assertTrue(all(x['exploratory_95_percent_interval'] is None for x in data['paired_differences']))
 def test_summary_rejects_duplicate_cells(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);a=self.session(root,'first',20);b=self.session(root,'retry',20)
   module=self.summary_module()
   with patch.object(sys,'argv',['summarize.py',str(a),str(b)]),contextlib.redirect_stdout(io.StringIO()):
    with self.assertRaisesRegex(SystemExit,'Duplicate matched cell'):
     module.main()

if __name__=='__main__':unittest.main()

