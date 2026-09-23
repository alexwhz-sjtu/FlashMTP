import json,pathlib,argparse,collections
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();run=pathlib.Path(args.run)
rows=[json.loads(l) for f in sorted(run.glob('results_*.jsonl')) for l in f.read_text().splitlines()]
manifest=[json.loads(l) for l in (run/'manifest.jsonl').read_text().splitlines()]
assert len({(r['cohort'],r['sample_id']) for r in rows})==len(rows),'duplicate results'
assert {(r['cohort'],r['sample_id']) for r in rows}=={(r['cohort'],r['sample_id']) for r in manifest},'incomplete results'
valid=[r for r in rows if 'models'in r]
rng=np.random.default_rng(20260923);B=10000
def ci(x):return np.quantile(x,[.025,.975]).tolist()
def delta(r,key):return r['models']['convk2g16'][key]-r['models']['noconv'][key]
out={'completed_rows':len(rows),'evaluated_rows':len(valid),'anchors':sum(len(r['anchors']) for r in valid),'primary_macro':{},'matched_controls':{},'e2e':{}}
primary=['alpaca_greedy','gsm8k_greedy','mbpp_greedy']
strata=[[r for r in valid if r['cohort']==c] for c in primary]
boots=[rng.integers(0,len(s),(B,len(s))) for s in strata]
for key in ['loss','ce','tv_l1','accuracy','tf_prefix','free_accept']:
 ds=[np.array([delta(r,key) for r in s]) for s in strata]
 out['primary_macro'][key]={'noconv':float(np.mean([np.mean([r['models']['noconv'][key] for r in s]) for s in strata])),
  'conv':float(np.mean([np.mean([r['models']['convk2g16'][key] for r in s]) for s in strata])),
  'delta':float(np.mean([d.mean() for d in ds])),
  'stratified_paired_bootstrap95':ci(np.mean([d[b].mean(1) for d,b in zip(ds,boots)],axis=0))}
by={(r['cohort'],r['sample_id']):r for r in valid}
ids=sorted({r['sample_id'] for r in valid if r['cohort']=='train_original'})
for lhs,rhs in [('train_original','train_original_short'),('train_original_short','train_greedy')]:
 pairs=[(by[lhs,i],by[rhs,i]) for i in ids if (lhs,i)in by and (rhs,i)in by]
 bs=rng.integers(0,len(pairs),(B,len(pairs)))
 stats={}
 for key in ['accuracy','tf_prefix','free_accept']:
  d=np.array([delta(b,key)-delta(a,key) for a,b in pairs])
  stats[key]={'delta_of_conv_advantage_rhs_minus_lhs':float(d.mean()),'paired_bootstrap95':ci(d[bs].mean(1))}
 out['matched_controls'][lhs+' -> '+rhs]={'pairs':len(pairs),'metrics':stats}
for cohort in sorted({r['cohort'] for r in valid}):
 rs=[r for r in valid if r['cohort']==cohort and 'e2e'in r['models']['noconv']]
 if not rs:continue
 bs=rng.integers(0,len(rs),(B,len(rs)));vectors={}
 for tag in ['noconv','convk2g16']:
  e=[r['models'][tag]['e2e'] for r in rs]
  n=np.array([x['tokens'] for x in e]);tm=np.array([x['decode_seconds'] for x in e]);steps=np.array([x['steps'] for x in e]);accepted=np.array([sum(x['accept_lengths']) for x in e])
  vectors[tag]={'latency':1000*tm[bs].sum(1)/n[bs].sum(1),'accept':accepted[bs].sum(1)/steps[bs].sum(1),
   'point_latency':float(1000*tm.sum()/n.sum()),'point_accept':float(accepted.sum()/steps.sum())}
 no,co=vectors['noconv'],vectors['convk2g16']
 out['e2e'][cohort]={'samples':len(rs),'accept_delta':co['point_accept']-no['point_accept'],
  'accept_delta_paired_bootstrap95':ci(co['accept']-no['accept']),
  'latency_relative_change':co['point_latency']/no['point_latency']-1,
  'latency_relative_change_paired_bootstrap95':ci(co['latency']/no['latency']-1)}
(run/'extended_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
