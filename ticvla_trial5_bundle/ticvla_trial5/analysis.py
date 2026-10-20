#!/usr/bin/env python3
"""Trial 5 preregistered paired analysis. Python stdlib only.

Resample *route families*, not frames or individual stochastic repeats.
Primary outcomes: stock episode-level collision flags. Uses Monte Carlo
family sign-flip tests as a transparent confirmatory approximation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, statistics, math
from collections import defaultdict,Counter
from pathlib import Path

def m(xs):return sum(xs)/len(xs) if xs else None

def read_rows(paths):
    rows=[]; seen=set(); id0=None; frozen=None
    for path in paths:
        root=Path(path)
        prov=root/'provenance/manifest.json'
        if not prov.exists():raise ValueError(f'Missing session provenance: {prov}')
        obj=json.loads(prov.read_text())
        identity=(obj['source_hashes'],obj['checkpoint_sha256'])
        if id0 is not None and id0!=identity:raise ValueError('Different source/model identities: do not pool')
        id0=identity
        plan=json.loads((root/'plan.json').read_text())
        if frozen is not None and frozen!=plan.get('frozen_manifest_sha256'):
            raise ValueError('Frozen protocol manifest mismatch')
        frozen=plan.get('frozen_manifest_sha256')
        if not frozen:raise ValueError('Session is not a frozen-manifest prospective Trial 5 session')
        for r in json.loads((root/'results.json').read_text()):
            key=(r['episode'],r['seed'],r.get('replicate',0),r['condition'])
            if key in seen:raise ValueError(f'Duplicate paired cell {key}')
            seen.add(key)
            nav=r.get('result',{})
            mech=nav.get('trial4',{}).get('mechanism',{})
            ct=mech.get('counts',{})
            row={'episode':int(r['episode']),'seed':int(r['seed']),'replicate':int(r.get('replicate',0)),
                 'family':r.get('family','unknown'), 'scene':r.get('scene','unknown'),
                 'condition':r['condition'],'valid':bool(r['valid']),
                 'collision':int(bool(nav.get('collision'))) if nav.get('collision') is not None else None,
                 'human_collision':int(bool(nav.get('human_collision'))) if nav.get('human_collision') is not None else None,
                 'physical_collision':int(bool(nav.get('physical_collision'))) if nav.get('physical_collision') is not None else None,
                 'success':int(bool(nav.get('success'))) if nav.get('success') is not None else None,
                 'navigation_error_m':nav.get('navigation_error'),
                 'path_length_m':nav.get('path_length'),
                 'late_used':ct.get('late_published_caches_used'),
                 'stale_drops':ct.get('stale_drops'),
                 'publishes':ct.get('publishes_total'),
                 'merged':ct.get('results_merged'),
                 'attack_trace_sha256':mech.get('attack_trace_sha256'),
                 'budget':mech.get('planned_attack_budget'),
                 'invalid_reasons':';'.join(r.get('validation_errors',[]))}
            rows.append(row)
    # Protocol-matching audit: the exogenous burst trace must be the same for
    # accept and drop. Flat demand must have the same planned virtual work.
    groups=defaultdict(dict)
    for r in rows:
        groups[(r['episode'],r['seed'],r['replicate'])][r['condition']]=r
    for unit, arm in groups.items():
        a,d,f=(arm.get(k) for k in ('yoyo_accept','yoyo_drop','flat_budget'))
        if a is not None and d is not None:
            if not a['attack_trace_sha256'] or a['attack_trace_sha256']!=d['attack_trace_sha256']:
                raise ValueError(f'Mismatched planned burst trace in {unit}')
            if a['budget']!=d['budget']:
                raise ValueError(f'Mismatched work budgets accept/drop in {unit}')
        if a is not None and f is not None and a['budget']!=f['budget']:
            raise ValueError(f'Flat/Yo-Yo offered synthetic work mismatch in {unit}')
    return rows, frozen

def make_pairs(rows,a,b):
    grouped=defaultdict(dict)
    for r in rows:
        grouped[(r['episode'],r['seed'],r['replicate'])][r['condition']]=r
    pairs=[];missing=0
    for key, arms in grouped.items():
        if a not in arms or b not in arms or not arms[a]['valid'] or not arms[b]['valid']:
            missing+=1
            continue
        x,y=arms[a],arms[b]
        if x['family']!=y['family']:raise ValueError('Family differs within matched pair')
        pairs.append({'episode':key[0],'seed':key[1],'replicate':key[2],'family':x['family'],
                      'scene':x['scene'],'x':x,'y':y})
    return pairs,missing

def family_averages(pairs,metric):
    per=defaultdict(list)
    for p in pairs:
        if p['x'][metric] is None or p['y'][metric] is None:continue
        per[p['family']].append(float(p['x'][metric])-float(p['y'][metric]))
    return {family:m(v) for family,v in per.items()}

def estimate(pairs,metric,nboot=5000,seed=9101):
    fa=family_averages(pairs,metric)
    if len(fa)<3:return {'estimate':m(list(fa.values())),'family_clusters':len(fa),'interval95':None,'p_two_sided':None}
    xs=list(fa.values());k=len(xs);rng=random.Random(seed)
    est=m(xs)
    draws=sorted(m([xs[rng.randrange(k)] for _ in range(k)]) for _ in range(nboot))
    ci=(draws[int(.025*nboot)],draws[min(nboot-1,int(.975*nboot))])
    # Swap treatments at the FAMILY level under the sharp no-effect null.
    greater=0
    for _ in range(nboot):
        draw=abs(m([value * (-1 if rng.getrandbits(1) else 1) for value in xs]))
        greater+=int(draw>=abs(est)-1e-12)
    p=(greater+1)/(nboot+1)
    return {'estimate':est,'family_clusters':k,'interval95':ci,'p_two_sided':p}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('sessions',nargs='+',type=Path)
    parser.add_argument('--out',type=Path,default=Path('trial5_analysis'))
    parser.add_argument('--interim-blind',action='store_true')
    p=parser.parse_args()
    rows,digest=read_rows(p.sessions)
    p.out.mkdir(parents=True,exist_ok=True)
    with (p.out/'per_rollout.csv').open('w',newline='') as f:
        fields=[k for k in rows[0] if k!='budget']
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows({k:v for k,v in r.items() if k!='budget'} for r in rows)
    contrasts=[('yoyo_accept','baseline'),('yoyo_accept','yoyo_drop'),
               ('yoyo_accept','flat_budget'),('no_kv','baseline')]
    values=[]
    for i,(a,b) in enumerate(contrasts):
        pairs,miss=make_pairs(rows,a,b)
        discord=sum(x['x']['collision']!=x['y']['collision'] for x in pairs if x['x']['collision'] is not None and x['y']['collision'] is not None)
        value={'contrast':f'{a} - {b}', 'complete_replicate_pairs':len(pairs),
               'missing_or_invalid_comparisons':miss,
               'discordance_count':discord,
               'discordance_rate':discord/len(pairs) if pairs else None}
        if not p.interim_blind:
            value['collision_risk_difference']=estimate(pairs,'collision',seed=9101+i)
            for name in ('human_collision','physical_collision','success','navigation_error_m','path_length_m'):
                value[f'{name}_difference']=estimate(pairs,name,seed=9201+i)
            value['baseline_and_attack_collision_table']={
                str(x)+str(y):sum(p_['x']['collision']==x and p_['y']['collision']==y for p_ in pairs)
                for x in (0,1) for y in (0,1)}
        values.append(value)
    if p.interim_blind:
        # Do not emit any arm-specific rate, signed difference or label-specific result.
        blinded={
            'blinded':True,'frozen_manifest_sha256':digest,
            'completed_observations':len(rows),
            'paired_complete_units':len({(r['episode'],r['seed']) for r in rows}),
            'comparison_discordance_rate_unlabeled':[x['discordance_rate'] for x in values[:2]],
            'comparison_pair_counts_unlabeled':[x['complete_replicate_pairs'] for x in values[:2]],
            'warning':'Analyst running this script can read condition names in raw data. For true masking, an independent custodian must supply codes A/B/C and hide mapping until database lock.'}
        (p.out/'blinded_interim.json').write_text(json.dumps(blinded,indent=2)+'\n')
        print(json.dumps(blinded,indent=2))
        return
    # Holm adjustment for 2 prespecified primary contrasts, familywise alpha 0.05.
    ps=[(i,v['collision_risk_difference']['p_two_sided']) for i,v in enumerate(values[:2])]
    if all(x[1] is not None for x in ps):
        ps.sort(key=lambda x:x[1]);a=ps[0][1];b=ps[1][1]
        values[ps[0][0]]['holm_adjusted_p']=min(1,2*a)
        values[ps[1][0]]['holm_adjusted_p']=min(1,max(2*a,b))
    report={'frozen_manifest_sha256':digest,'n_rollouts':len(rows),'n_valid':sum(r['valid'] for r in rows),
            'invalid':sum(not r['valid'] for r in rows),
            'n_unique_episode_seed_units':len({(r['episode'],r['seed']) for r in rows}),
            'n_route_families':len({r['family'] for r in rows}),
            'method':'equal route-family-weighted risk differences, 5000 route-family bootstrap replicates and 5000 paired family-level sign flips; 2 primary comparisons with Holm correction',
            'warnings':['These confidence intervals are exploratory if fewer than 20 route families or fewer than 200 prospective units are complete.',
                        'Original exploratory five episodes are excluded from this prospective design.',
                        'Paired external-arrival trace hashes do NOT imply identical realized VLM completion times or equal robot/actor states.',
                        'Sign-flip test assumes treatment exchangeability at the family-level; family heterogeneity and finite clusters warrant independent statistical review.'],
            'contrasts':values}
    (p.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
