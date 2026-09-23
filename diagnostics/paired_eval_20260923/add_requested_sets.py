import json, pathlib, random, hashlib
from datasets import load_from_disk
from transformers import AutoTokenizer

run=pathlib.Path('/data/wanghanzhen/FlashMTP_v2swa/diagnostics/paired_eval_20260923')
tok=AutoTokenizer.from_pretrained('/data/wanghanzhen/models/Qwen3-8B')
def h(text):return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
train_hashes=set()
with open('/data/wanghanzhen/training_data/generated/qwen3-8b/open_perfectblend_80k_qwen3_8b.jsonl') as f:
 for line in f:
  r=json.loads(line)
  for m in r['conversations']:
   if m['role']=='user':train_hashes.add(h(m['content']))
tasks=[];stats={};rng=random.Random(20260923)
for di,name in enumerate(['alpaca','gsm8k','mbpp']):
 ds=load_from_disk('/data/processed_dataset_cache/'+name+'_v1')
 indices=list(range(len(ds)));rng.shuffle(indices);count=0;overlap=0
 for index in indices:
  if count==64:break
  row=ds[index];prompt=row['turns'][0]
  possibilities=[prompt]+[row[k] for k in ['question','prompt','instruction'] if isinstance(row.get(k),str)]
  if any(h(x) in train_hashes for x in possibilities):overlap+=1;continue
  ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  if len(ids)>3840:continue
  tasks.append({'sample_id':name+'_'+str(index),'unit':256+di*64+count,'cohort':name+'_greedy',
    'source':name,'dataset_index':index,'prefix_ids':ids,'prompt_hash':h(prompt),'e2e':count<16})
  count+=1
 stats[name]={'cache_rows':len(ds),'selected':count,'exact_train_prompt_overlaps_excluded':overlap,
              'split':'train (used only for evaluation here)' if name=='alpaca' else 'test'}
existing=[json.loads(l) for l in (run/'manifest.jsonl').read_text().splitlines()]
assert not any(r['cohort'] in ('alpaca_greedy','gsm8k_greedy','mbpp_greedy') for r in existing)
with (run/'manifest.jsonl').open('a') as f:
 for t in tasks:f.write(json.dumps(t,ensure_ascii=False)+'\n')
cfg=json.loads((run/'config.json').read_text());cfg['primary_benchmarks']=stats;cfg['specbench_role']='supplementary only, superseded by user-selected primary datasets'
(run/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
print(json.dumps(stats,ensure_ascii=False),flush=True)
