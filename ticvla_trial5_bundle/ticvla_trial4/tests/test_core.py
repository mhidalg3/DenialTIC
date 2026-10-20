import json
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve()
PAYLOAD = HERE.parents[1]/"payload"
sys.path.insert(0, str(PAYLOAD if PAYLOAD.exists() else HERE.parents[2]))
from trial4_core import EventLog, Settings, SlowPath, distribution


class InlineExecutor:
    def submit(self, function, *args):
        f = Future()
        try:
            f.set_result(function(*args))
        except BaseException as e:
            f.set_exception(e)
        return f
    def shutdown(self, **kwargs):
        pass


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.engines = []
        self.seen = []

    def tearDown(self):
        for e in self.engines:
            e.close()
            e.log.file.close()
        self.tmp.cleanup()

    def engine(self, **kwargs):
        def generate(req):
            self.seen.append((req.id, req.source_step, req.launch_step, Path(req.input_paths[-1]).read_bytes()))
            return f"response-{req.id}", (f"cache-{req.id}",)
        e = SlowPath(Settings(**kwargs), generate, EventLog(self.path/f"log{len(self.engines)}"), InlineExecutor())
        self.engines.append(e)
        return e

    def offer(self, e, frame):
        path = self.path/f"front_frame_{frame:06d}.jpg"
        path.write_bytes(b"original")
        return e.offer(frame, {"position":[0,0,0], "quaternion":[1,0,0,0]}, [str(path)], "turn left", {}, "", "robot")

    def bootstrap(self, e):
        req = self.offer(e,0)
        e.bootstrap_wait(0)
        self.assertEqual(e.latest[0].id, req.id)
        return req

    def test_frame_delay_and_bootstrap_exemption(self):
        e = self.engine(launch_delay_frames=90)
        first = self.bootstrap(e)
        self.assertEqual(first.launch_step, 0)
        r = self.offer(e,1)
        e.tick(90)
        self.assertIsNone(r.future)
        e.tick(91)
        e.merge(91)
        self.assertEqual(r.launch_step,91)
        self.assertEqual(e.latest[0].id,2)
        self.assertEqual(e.summary()["distributions"]["queue_wait_frames"]["max"],90)

    def test_completed_future_cannot_be_overwritten(self):
        e = self.engine()
        self.bootstrap(e)
        req = self.offer(e,1)
        self.assertTrue(req.future.done())
        self.assertIsNone(self.offer(e,2))
        self.assertIs(e.outstanding, req)
        e.merge(2)
        self.assertEqual(e.latest[0].id,2)

    def test_deadline_policy_and_bootstrap_freeze_are_forbidden(self):
        with self.assertRaisesRegex(ValueError, "never rejects"):
            Settings(deadline_frames=150)
        with self.assertRaises(ValueError):
            Settings(cache_mode="bootstrap")

    def test_accept_late_has_no_rejection(self):
        e = self.engine(launch_delay_frames=360, deadline_frames=0)
        self.bootstrap(e)
        self.offer(e,1)
        e.tick(361)
        e.merge(361)
        self.assertEqual(e.latest[0].id,2)
        self.assertEqual(e.counts["stale_drops"],0)

    def test_extremely_old_completed_cache_is_still_published(self):
        e = self.engine()
        self.bootstrap(e)
        self.offer(e,1)
        e.merge(10001)
        self.assertEqual(e.latest[1], "response-2")
        self.assertEqual(e.counts["publishes_total"],e.counts["results_merged"])
        self.assertEqual(e.counts["stale_drops"],0)
        self.assertEqual(e.counts["forced_drops"],0)

    def test_attacker_arrives_without_victim_request(self):
        e = self.engine(attack="periodic", on_frames=60, off_frames=120,
                        arrival_interval_frames=30, attack_service_frames=10)
        self.bootstrap(e)
        for frame in range(1,183):
            e.tick(frame)
        self.assertEqual(e.counts["requests_enqueued"],1)
        self.assertEqual(e.counts["attack_jobs_enqueued"],2)
        self.assertEqual(e.counts["attack_jobs_completed"],2)

    def test_bounded_attack_queue(self):
        e = self.engine(attack="sustained", arrival_interval_frames=1,
                        attack_service_frames=100, max_attack_jobs=3)
        self.bootstrap(e)
        for frame in range(1,40):
            e.tick(frame)
        self.assertEqual(e.counts["attack_jobs_enqueued"],3)
        self.assertGreater(e.counts["attack_capacity_drops"],0)

    def test_snapshot_survives_delete_and_overwrite(self):
        e = self.engine(launch_delay_frames=90)
        self.bootstrap(e)
        req = self.offer(e,1)
        original = self.path/"front_frame_000001.jpg"
        original.write_bytes(b"changed")
        original.unlink()
        e.merge(91)
        self.assertEqual(self.seen[-1][-1], b"original")
        self.assertFalse(Path(req.snapshot_dir).exists())

    def test_cancel_pending_gate_leaves_no_snapshot(self):
        e = self.engine(launch_delay_frames=900)
        self.bootstrap(e)
        req = self.offer(e,1)
        e.close()
        self.assertIsNone(req.future)
        self.assertFalse(Path(req.snapshot_dir).exists())

    def test_frame_clock_is_not_wall_clock(self):
        import time
        outputs = []
        for pause in (0, .001):
            e = self.engine(launch_delay_frames=9)
            self.bootstrap(e)
            r = self.offer(e,1)
            for f in range(2,11):
                if pause: time.sleep(pause)
                e.tick(f)
            e.merge(10)
            outputs.append(r.launch_step-r.source_step)
        self.assertEqual(outputs,[9,9])

    def test_periodic_queue_admits_late_and_then_fresh(self):
        e = self.engine(attack="periodic", on_frames=90,
                        off_frames=300, arrival_interval_frames=15, attack_service_frames=30,
                        late_reference_frames=60)
        self.bootstrap(e)
        for frame in range(1,390):
            e.tick(frame)
            if frame % 3 == 0:
                e.merge(frame)
                e.record_action(frame,(frame-e.latest[0].source_step)/30,True,.01)
                if e.outstanding is None:
                    self.offer(e,frame)
        self.assertGreater(e.counts["late_publishes"],0)
        self.assertGreater(e.counts["late_published_caches_used"],0)
        self.assertGreater(e.counts["fresh_cache_used_after_late"],0)
        self.assertEqual(e.counts["publishes_total"],e.counts["results_merged"])
        self.assertEqual(e.counts["stale_drops"],0)

    def test_old_at_use_is_not_mistaken_for_late_at_delivery(self):
        e=self.engine()
        self.bootstrap(e)
        e.record_action(1000,1000/30,True,.01)
        self.assertEqual(e.counts["late_published_caches_used"],0)
        self.assertEqual(e.counts["post_bootstrap_caches_used"],0)

    def test_visual_source_evidence_survives_request_cleanup(self):
        e=self.engine()
        self.bootstrap(e)
        evidence=e.log.directory/"semantic_inputs/request_000001/00_front_frame_000000.jpg"
        self.assertEqual(evidence.read_bytes(),b"original")

    def test_background_generation_does_not_block_ticks(self):
        import threading
        import time
        started,release=threading.Event(),threading.Event()
        def generate(req):
            started.set()
            if not release.wait(2): raise RuntimeError("test release timeout")
            return "ok", ("cache",)
        e=SlowPath(Settings(),generate,EventLog(self.path/"threaded"))
        self.engines.append(e)
        try:
            req=self.offer(e,0)
            self.assertTrue(started.wait(1))
            before=time.perf_counter()
            e.tick(1);e.tick(2)
            self.assertLess(time.perf_counter()-before, .5)
            self.assertFalse(req.future.done())
        finally:
            release.set()
        req.future.result(timeout=1)
        e.merge(2)
        self.assertIsNotNone(e.latest)

    def test_invalid_config_and_image_provenance(self):
        for kw in ({"deadline_frames":-1},{"attack":"periodic","off_frames":0},
                   {"arrival_interval_frames":0},{"cache_mode":"fake"}):
            with self.assertRaises(ValueError): Settings(**kw)
        e=self.engine()
        p=self.path/"front_frame_000001.jpg"
        p.write_bytes(b"x")
        with self.assertRaises(ValueError):
            e.offer(3,{},[str(p)],"test",{},"","robot")

    def test_json_log_is_parseable_and_summary_has_counters(self):
        e=self.engine(cache_mode="none")
        self.bootstrap(e)
        e.record_action(4,4/30,False,.1,0.,0.)
        rows=[json.loads(x) for x in (e.log.directory/"events.jsonl").read_text().splitlines()]
        self.assertEqual([r["sequence"] for r in rows], list(range(1,len(rows)+1)))
        self.assertEqual(e.summary()["counts"]["action_calls_without_kv"],1)
        self.assertTrue(e.summary()["instrumentation_valid"])
        self.assertEqual(distribution([])["p95"],None)


if __name__ == "__main__": unittest.main()

class ProfileTests(unittest.TestCase):
    setUp=CoreTests.setUp
    tearDown=CoreTests.tearDown
    engine=CoreTests.engine
    offer=CoreTests.offer
    bootstrap=CoreTests.bootstrap
    def test_default_profile_with_mock_1_5s_service_delivers_stale_then_fresh(self):
        from concurrent.futures import Future
        class FrameExecutor:
            def __init__(self): self.step=0; self.jobs=[]
            def submit(self,fn,*a):
                f=Future();self.jobs.append((self.step+45,f,fn,a));return f
            def advance(self,k):
                self.step=k
                for item in list(self.jobs):
                    due,f,fn,a=item
                    if due<=k:
                        f.set_result(fn(*a));self.jobs.remove(item)
            def shutdown(self,**kw):
                for _,f,_,_ in self.jobs:f.cancel()
        e=self.engine(attack="periodic",retain_visuals=False)
        self.bootstrap(e)
        e.executor=FrameExecutor()
        for frame in range(1,1800):
            e.executor.advance(frame);e.tick(frame)
            if frame%3==0:
                e.merge(frame)
                e.record_action(frame,(frame-e.latest[0].source_step)/30,True,.001)
                if e.outstanding is None:self.offer(e,frame)
        self.assertGreaterEqual(e.counts["late_published_caches_used"],2)
        self.assertGreaterEqual(e.counts["fresh_cache_used_after_late"],1)
        self.assertGreaterEqual(e.counts["queue_recoveries"],1)
        self.assertEqual(e.counts["stale_drops"],0)
        self.assertEqual(e.counts["forced_drops"],0)
