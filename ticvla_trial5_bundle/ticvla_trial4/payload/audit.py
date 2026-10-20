#!/usr/bin/env python3
"""Compact native-admission/treatment audit; works on partial or completed sessions."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('session',type=Path);a=p.parse_args()
 result=a.session/'results.json'
 if not result.exists():
  print('No completed cells yet. Read run_state.json and the current console.log.');return
 rows=json.loads(result.read_text())
 print('condition       ep valid SR  pub  used late_used recovery drops moving_near')
 for row in rows:
  data=row.get('result',{});m=data.get('trial4',{}).get('mechanism',{});c=m.get('counts',{})
  n=data.get('trial4',{}).get('navigation',{})
  print(f"{row['condition']:15} {row['episode']:2} {str(row.get('valid')):5} "
        f"{int(bool(data.get('success'))):2} {c.get('post_bootstrap_publishes',0):4} "
        f"{c.get('post_bootstrap_caches_used',0):5} {c.get('late_published_caches_used',0):9} "
        f"{c.get('fresh_cache_used_after_late',0):8} "
        f"{c.get('stale_drops',0)+c.get('forced_drops',0):5} "
        f"{n.get('moving_pedestrians_within_2m',0):11}")
  if row.get('validation_errors'): print('  INVALID:',row['validation_errors'])
  if row['condition']=='yoyo_accept':
   d=m.get('distributions',{})
   for key in ['post_bootstrap_queue_wait_frames','post_bootstrap_delivery_age_frames','model_delay_input_s']:
    z=d.get(key,{})
    print(f"  {key}: n={z.get('n')} mean={z.get('mean')} p95={z.get('p95')} max={z.get('max')}")
   print('  exposure:',json.dumps(row.get('exposure',{}),sort_keys=True))
 print('\npub/used exclude bootstrap. late_used counts distinct caches already >150 frames old at delivery, actually consumed.')
 print('The 150-frame reference is ONLY an analysis label. All valid completed caches are admitted.')
 print('A weak/unobserved treatment is not a demonstrated robustness result. Recovery is an age label; queue recovery is separately recorded.')
if __name__=='__main__':main()
