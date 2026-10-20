#!/usr/bin/env python3
"""Summarize validated single-episode records, not duplicated 'latest' aliases.

Intervals are exploratory paired within-episode bootstrap intervals across the
fixed tested task set. They are not claims about the entire DynaNav benchmark.
"""
from __future__ import annotations
import argparse
import datetime
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics


def mean(x): return statistics.mean(x) if x else None

def frames_to_seconds(value): return float(value)/30.0 if value is not None else None


def nested(d,*keys):
    for k in keys:
        if not isinstance(d,dict): return None
        d=d.get(k)
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("session",type=Path,nargs="+")
    args=p.parse_args()
    if len(args.session)==1:
        root=args.session[0]
    else:
        root=args.session[0].parents[1]/"trial4_reports"/("combined_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        root.mkdir(parents=True,exist_ok=False)
    all_rows=[]
    seen=set()
    condition_settings={}
    source_identity=None
    for source in args.session:
        if len(args.session)>1:
            m=json.loads((source/"provenance/manifest.json").read_text())
            ident=(m["source_hashes"],m["checkpoint_sha256"])
            if source_identity is not None and ident!=source_identity:
                raise SystemExit("Source/checkpoint differs across sessions; do not pool.")
            source_identity=ident
        for record in json.loads((source/"results.json").read_text()):
            key=(record["episode"],record["seed"],record["condition"])
            if key in seen:
                raise SystemExit(f"Duplicate matched cell {key}; choose disjoint sessions rather than count a retry twice.")
            cfg=json.dumps(record["settings"],sort_keys=True)
            previous=condition_settings.setdefault(record["condition"],cfg)
            if previous!=cfg:
                raise SystemExit(f"Settings differ within condition {record['condition']}; keep these sessions separate.")
            seen.add(key);all_rows.append(record)
    valid=[r for r in all_rows if r.get("valid")]
    flat=[]
    for row in all_rows:
        d=row.get("result",{})
        m=nested(d,"trial4","mechanism") or {}
        c=m.get("counts",{})
        n=nested(d,"trial4","navigation") or {}
        max_cache_frames=nested(m,"distributions","action_cache_age_frames","max")
        counts_completed=c.get("results_merged",0)
        post_completed=max(0,counts_completed-1)
        flat.append({"episode":row["episode"],"seed":row["seed"],"condition":row["condition"],
             "valid":row["valid"],"success":d.get("success"),"navigation_error_m":d.get("navigation_error"),
             "path_length_m":d.get("path_length"),"duration_sim_s":d.get("duration"),"collision":d.get("collision"),
             "spl":d.get("spl"),"post_bootstrap_publishes":c.get("post_bootstrap_publishes"),
             "stale_drops":c.get("stale_drops"),"forced_drops":c.get("forced_drops"),
             "stale_drop_fraction_completed_post_bootstrap": c.get("stale_drops",0)/post_completed if post_completed else None,
             "on_requests":c.get("requests_enqueued_during_on"),"unmerged_at_end":c.get("requests_unmerged_at_end"),
             "max_cache_age_sim_s":max_cache_frames/30 if max_cache_frames is not None else None,
             "p95_model_delay_input_s":nested(m,"distributions","model_delay_input_s","p95"),
             "p95_queue_frames":nested(m,"distributions","queue_wait_frames","p95"),
             "backup_events":m.get("backup_events"),"min_goal_distance_m":n.get("minimum_goal_distance_m"),
             "min_human_root_distance_m":n.get("minimum_pedestrian_root_distance_m"),
             "human_proximity_frames":n.get("pedestrian_proximity_frames"),
             "post_bootstrap_caches_used":c.get("post_bootstrap_caches_used"),
             "late_publishes":c.get("late_publishes"),
             "late_published_caches_used":c.get("late_published_caches_used"),
             "fresh_cache_used_after_late":c.get("fresh_cache_used_after_late"),
             "attack_jobs_enqueued":c.get("attack_jobs_enqueued"),
             "moving_people":n.get("pedestrians_moved_over_0_25m"),
             "moving_people_near_robot":n.get("moving_pedestrians_within_2m"),
             "mean_delivery_age_sim_s":frames_to_seconds(nested(m,"distributions","post_bootstrap_delivery_age_frames","mean")),
             "max_delivery_age_sim_s":frames_to_seconds(nested(m,"distributions","post_bootstrap_delivery_age_frames","max")),
             "action_calls_without_kv":c.get("action_calls_without_kv"),
             "action_calls":c.get("action_calls")})
    if flat:
        with (root/"per_episode.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    grouped=defaultdict(list)
    for r in flat:
        if r["valid"]:grouped[r["condition"]].append(r)
    report={"invalid_runs":len(all_rows)-len(valid),"valid_runs":len(valid),"conditions":{},"paired_differences":[],
            "source_sessions":[str(x) for x in args.session],
            "scope":"Five fixed episode pairs at one seed by default. Descriptive discovery only; not statistical significance.",
            "treatment_audit":[{"episode":r["episode"],"seed":r["seed"],"condition":r["condition"],
                "valid":r.get("valid"),**r.get("exposure",{})} for r in all_rows]}
    for condition,rs in sorted(grouped.items()):
        report["conditions"][condition]={"n":len(rs),"success_rate":mean([float(r["success"]) for r in rs]),
               **{key:mean([r[key] for r in rs if r[key] is not None]) for key in (
                   "navigation_error_m","path_length_m","duration_sim_s","collision","spl","stale_drops",
                   "post_bootstrap_publishes","max_cache_age_sim_s","backup_events", "late_published_caches_used",
                   "fresh_cache_used_after_late", "mean_delivery_age_sim_s", "moving_people_near_robot")}}
    bases=["baseline"]
    baseline=next((b for b in bases if b in grouped),None)
    if baseline:
        controls={(r["episode"],r["seed"]):r for r in grouped[baseline]}
        rng=random.Random(312)
        for condition,rs in grouped.items():
            if condition==baseline:continue
            for metric in ("success","navigation_error_m","path_length_m","duration_sim_s","collision","spl"):
                by_ep=defaultdict(list)
                for r in rs:
                    b=controls.get((r["episode"],r["seed"]))
                    if b and r[metric] is not None and b[metric] is not None:
                        by_ep[r["episode"]].append(float(r[metric])-float(b[metric]))
                if not by_ep:continue
                estimate=mean([mean(x) for x in by_ep.values()])
                interval=None
                if all(len(x)>=5 for x in by_ep.values()):
                    draws=sorted(mean([mean([rng.choice(x) for _ in x]) for x in by_ep.values()]) for _ in range(2000))
                    interval=[draws[49],draws[1949]]
                report["paired_differences"].append({"condition":condition,"reference":baseline,"metric":metric,
                    "macro_mean_difference":estimate,"paired_runs":sum(map(len,by_ep.values())),
                    "fixed_tasks":len(by_ep),"exploratory_95_percent_interval":interval,
                    "interval_scope":"Within-episode resampling of paired seeds; fixed tested tasks, not population generalization. Requires >=5 pairs/task."})
    lines = ["TRIAL 4 — DELAYED CACHES ACCEPTED NORMALLY", report["scope"], ""]
    for name, values in report["conditions"].items():
        lines.append(f"{name}: n={values['n']}, success={values['success_rate']:.1%}, "
                     f"NE={values['navigation_error_m']:.3f}m, SPL={values['spl']:.3f}, "
                     f"collision={values['collision']:.1%}")
    lines += ["", "Per-episode evidence is in per_episode.csv, treatment_audit, and each episode/events.jsonl.",
              "No-KV is an interface ablation, not a latency attack. Zero treatment exposure is not a robustness result."]
    (root/"summary.txt").write_text("\n".join(lines)+"\n")
    (root/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))
    print("Wrote",root/"summary.json","and",root/"per_episode.csv")

if __name__=="__main__": main()
