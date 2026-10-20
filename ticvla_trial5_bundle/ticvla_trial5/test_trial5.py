"""CPU-only Trial 5 scheduler/admission tests (runnable inside Isaac Python)."""
from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import unittest
import sys

HERE=Path(__file__).resolve()
DYNANAV=HERE.parents[1] if HERE.parent.name=='trial5' else HERE.parent/'overlay'
sys.path.insert(0,str(DYNANAV))
from trial4_core import Settings,SlowPath,EventLog

class InlineExecutor:
    def submit(self,fn,*args):
        f=Future()
        try:f.set_result(fn(*args))
        except Exception as exc:f.set_exception(exc)
        return f
    def shutdown(self,**kwargs):pass

class Trial5CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.engines=[]
    def tearDown(self):
        for e in self.engines:
            e.close();e.log.file.close()
        self.tmp.cleanup()
    def mk(self,**kw):
        engine=SlowPath(Settings(**kw),lambda r:("text",("dummy-cache",)),
                        EventLog(self.root/str(len(self.engines))),InlineExecutor())
        self.engines.append(engine)
        return engine
    def offer(self,e,frame):
        p=self.root/f'front_frame_{frame:06d}.jpg'
        p.write_bytes(b'fake-frame')
        return e.offer(frame,{"position":[0,0,0],"quaternion":[1,0,0,0]},
                       [str(p)],'turn left',{},'','robot')
    def boot(self,e):
        self.offer(e,0)
        e.bootstrap_wait(0)
        self.assertEqual(e.latest[0].id,1)
    def test_accept_and_drop_are_identical_until_completion(self):
        aa=self.mk(deadline_frames=0,launch_delay_frames=360)
        dd=self.mk(deadline_frames=150,launch_delay_frames=360)
        for e in (aa,dd):
            self.boot(e);self.offer(e,1)
            e.tick(361);e.merge(361)
            self.assertEqual(e.counts['results_merged'],2)
        self.assertEqual(aa.latest[0].id,2)
        self.assertEqual(aa.counts['stale_drops'],0)
        self.assertEqual(dd.latest[0].id,1)
        self.assertEqual(dd.counts['stale_drops'],1)
        self.assertEqual(dd.counts['publishes_total']+dd.counts['stale_drops'],dd.counts['results_merged'])
        rows=[json.loads(x) for x in dd.log.file.name and Path(dd.log.file.name).read_text().splitlines()]
        self.assertEqual([x['event'] for x in rows].count('cache_rejected'),1)
    def test_bootstrap_is_exempt(self):
        e=self.mk(deadline_frames=1)
        self.boot(e)
        self.assertEqual(e.counts['stale_drops'],0)
    def test_same_exogenous_burst_trace_for_accept_and_drop(self):
        a=self.mk(attack='periodic',deadline_frames=0,attack_horizon_frames=4000)
        b=self.mk(attack='periodic',deadline_frames=150,attack_horizon_frames=4000)
        self.boot(a);self.boot(b)
        self.assertEqual(a.attack_trace_hash,b.attack_trace_hash)
        self.assertEqual(a.attack_offsets,b.attack_offsets)
        self.assertGreater(len(a.attack_offsets),0)
    def test_flat_and_yoyo_have_same_total_synthetic_work(self):
        for horizon in (1,50,150,180,300,899,1080,1400,4000,5100):
            b=self.mk(attack='periodic',attack_horizon_frames=horizon)
            f=self.mk(attack='flat',attack_horizon_frames=horizon)
            self.boot(b);self.boot(f)
            self.assertEqual(b.attack_budget,f.attack_budget)
            self.assertEqual(len(b.attack_offsets),len(f.attack_offsets))
            if horizon>30:
                self.assertNotEqual(b.attack_trace_hash,f.attack_trace_hash)
    def test_flat_arrival_independent_of_benign_submissions(self):
        e=self.mk(attack='flat',attack_horizon_frames=1080)
        self.boot(e)
        for frame in range(1,1080):e.tick(frame)
        self.assertEqual(e.counts['attack_jobs_enqueued'],6)
        self.assertEqual(e.counts['requests_enqueued'],1)
    def test_settings_reject_negative_deadline_and_nonpositive_counts(self):
        for kw in (dict(deadline_frames=-1),dict(attack_service_frames=0),dict(attack='z')):
            with self.assertRaises(ValueError): Settings(**kw)

if __name__=='__main__': unittest.main()
