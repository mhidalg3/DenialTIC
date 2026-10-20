#!/usr/bin/env python3
"""Apply the tested Trial 5 protocol overlay to a completed Trial 4 installation.

Run on Brev HOST after Trial 4's bootstrap, never while a suite is active.
Changes are backed up, source hashes refreshed, and core source is compiled.
"""
from __future__ import annotations
import argparse
import ast
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PACKAGE=Path(__file__).resolve().parent
PIN="9fa6f8b66b9e121d5df5df071297bba8e5353ebb"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def install(repo,apply=False):
    repo=Path(repo).expanduser().resolve()
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if git!=PIN:
        raise RuntimeError(f"Unexpected repository commit {git}; expected {PIN}")
    base_manifest=repo/'DynaNav/trial4/install_manifest.json'
    if not base_manifest.exists(): raise RuntimeError('Trial 4 must be installed first')
    manifest=json.loads(base_manifest.read_text())
    if (repo/'DynaNav/trial5/install_info.json').exists():
        raise RuntimeError('Trial 5 upgrade already installed; do not overwrite during research')
    lock=(repo/'.trial4_suite.lock').open('a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Experiment currently running; do not patch source')
    files={}
    for name in ['trial4_core.py','trial4_metrics.py','run_suite.py']:
        dest=f'DynaNav/{name}' if name.startswith('trial4_') else f'DynaNav/trial4/{name}'
        files[dest]=(PACKAGE/'overlay'/name).read_text()
    benchmark=repo/'DynaNav/benchmark.py'
    source=benchmark.read_text()
    anchor='''            if reading.is_valid and reading.in_contact and float(reading.value) >= self._contact_threshold:'''
    if source.count(anchor)!=1:raise RuntimeError('Unexpected contact-sensor source; expected exactly one known anchor')
    if 'self._t4_trace.contact(' in source:raise RuntimeError('Contact instrumentation already installed')
    replacement='''            if hasattr(self, "_t4_trace"):
                self._t4_trace.contact(self._simulation_frame_count, reading,
                                       self._contact_threshold, self._contact_duration_threshold_frames)
''' + anchor
    files['DynaNav/benchmark.py']=source.replace(anchor,replacement,1)
    for rel,content in files.items():
        ast.parse(content,filename=rel)
        current=repo/rel
        if not current.is_file():raise RuntimeError(f'Missing Trial 4 source: {current}')
        known=manifest['files'].get(rel)
        if known and sha(current)!=known['installed_sha256']:
            raise RuntimeError(f'{rel} differs from Trial 4 installed hash; refusing unexpected source overwrite')
    print('Trial 5 overlay source checks pass. Will update:')
    for rel in files: print('  ',rel)
    print('Preserves latest-cache acceptance in baseline and yoyo_accept; adds discard arm and flat-load control.')
    if not apply:
        print('Dry run. Rerun --apply when no experiments are active.')
        return
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=repo/'trial5_backups'/stamp
    backup.mkdir(parents=True,exist_ok=False)
    for rel,content in files.items():
        dest=repo/rel
        copy=backup/rel
        copy.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(dest,copy)
        temp=dest.with_name(dest.name+'.trial5tmp')
        temp.write_text(content)
        temp.replace(dest)
        manifest['files'][rel]['installed_sha256']=sha(dest)
    shutil.copy2(base_manifest,backup/'install_manifest.json')
    base_manifest.write_text(json.dumps(manifest,indent=2)+'\n')
    trial5=repo/'DynaNav/trial5'
    trial5.mkdir(parents=True,exist_ok=False)
    for name in ['frozen_study.json','test_trial5.py','runbook.md','analysis.py']:
        shutil.copy2(PACKAGE/name,trial5/name)
    info={'base_git_commit':PIN,'trial5_package_overlay':'v1','timestamp_utc':stamp,
          'modified_source_hashes':{rel:sha(repo/rel) for rel in files},
          'study_sha256':sha(trial5/'frozen_study.json'),
          'backup':str(backup),'pre_existing_trial4_installed':True,
          'notes':'This is a limited VM-execution upgrade. Exact per-request VLM completion times are not guaranteed equal across closed-loop treatments.'}
    (trial5/'install_info.json').write_text(json.dumps(info,indent=2)+'\n')
    print('Upgraded. Backups:',backup)
    print('Study SHA256:',info['study_sha256'])

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',default='~/TIC-VLA')
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    install(a.repo,a.apply)
