import os, sys, json, random, hashlib, pathlib, argparse, time, gc, collections
import numpy as np

P=argparse.ArgumentParser()
P.add_argument('mode',choices=['prepare','worker','summarize'])
P.add_argument('--run',required=True)
P.add_argument('--rank',type=int,default=0)
P.add_argument('--world',type=int,default=8)
P.add_argument('--limit',type=int,default=0)
A=P.parse_args(); RUN=pathlib.Path(A.run); RUN.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
ROOT=pathlib.Path('/data/wanghanzhen/FlashMTP_v2swa')
TARGET='/data/wanghanzhen/models/Qwen3-8B'
DATA='/data/wanghanzhen/training_data/generated/qwen3-8b/open_perfectblend_80k_qwen3_8b.jsonl'
SEED=20260923
def writejson(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def normhash(x):return hashlib.sha256(' '.join(x.split()).encode()).hexdigest()

if A.mode=='prepare':
 import torch
 from transformers import AutoTokenizer
 from specforge.data.parse import GeneralParser
 from specforge.data.template import TEMPLATE_REGISTRY
 tok=AutoTokenizer.from_pretrained(TARGET)
 parser=GeneralParser(tok,TEMPLATE_REGISTRY.get('qwen'))
 rng=random.Random(SEED); pool=[]; hashes=set(); population=0
 with open(DATA) as f:
  for i,line in enumerate(f):
   r=json.loads(line);population+=1
   users=[m['content'] for m in r.get('conversations',[]) if m['role']=='user']
   for text in users: hashes.add(normhash(text))
   item=(i,r)
   if len(pool)<1024:pool.append(item)
   else:
    j=rng.randrange(i+1)
    if j<1024:pool[j]=item
 rng.shuffle(pool);tasks=[]; train_count=0; selected_hashes=set();think_prefix_count=0
 for line_id,r in pool:
  if train_count==128:break
  if r.get('status','success')!='success':continue
  conv=r['conversations'];users=[m['content'] for m in conv if m['role']=='user']
  if not users or normhash(users[0]) in selected_hashes:continue
  ids,mask=parser.parse(conv,4096,**{k:v for k,v in r.items() if k not in ('conversations','id')})
  valid=[a for a in range(1,len(ids)-7) if bool(mask[a:a+8].all())]
  if len(valid)<16 or int(mask.sum())<16:continue
  start=int(mask.nonzero()[0]);prefix=ids[:start].tolist()
  # Training rendering includes this fixed empty-thinking header in its loss mask.
  # Keep it as known context for greedy answer regeneration, matching thinking-off evaluation.
  empty_think=tok.encode('<think>\n\n</think>\n\n',add_special_tokens=False)
  if ids[start:start+len(empty_think)].tolist()==empty_think:
   prefix=ids[:start+len(empty_think)].tolist();think_prefix_count+=1
  if start>3840:continue
  base={'sample_id':f'train_{line_id}','unit':train_count,'line':line_id,'source':r.get('source'),
        'source_id':r.get('id'),'prompt_hash':normhash(users[0]),'prefix_ids':prefix,
        'e2e':train_count<32}
  tasks.append(dict(base,cohort='train_original',input_ids=ids.tolist(),loss_mask=mask.tolist()))
  tasks.append(dict(base,cohort='train_greedy'))
  selected_hashes.add(normhash(users[0]));train_count+=1
 bench=[json.loads(l) for l in open('/data/wanghanzhen/datasets/specbench/question.jsonl')]
 rng.shuffle(bench);bc=0;overlaps=0;bcats=collections.Counter()
 # Round-robin categories produces broad coverage instead of domination by 80-item groups.
 categories=collections.defaultdict(list)
 for r in bench:categories[r['category']].append(r)
 ordered=[]
 while any(categories.values()):
  for k in sorted(categories):
   if categories[k]:ordered.append(categories[k].pop())
 for r in ordered:
  if bc==128:break
  prompt=r['turns'][0];ph=normhash(prompt)
  if ph in hashes:overlaps+=1;continue
  prefix=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  if len(prefix)>3840:continue
  tasks.append({'sample_id':f"bench_{r['question_id']}",'unit':128+bc,'cohort':'benchmark_greedy',
    'source':'specbench','category':r['category'],'prompt_hash':ph,'prefix_ids':prefix,'e2e':bc<32})
  bcats[r['category']]+=1;bc+=1
 (RUN/'manifest.jsonl').write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in tasks))
 checkpoints={}
 for tag in ['noconv','convk2g16']:
  paths=list((ROOT/'cache/models').glob('*globalpos_'+tag+'_*/epoch_6_step_59730'))
  assert len(paths)==1
  checkpoints[tag]=str(paths[0])
 meta={'seed':SEED,'training_file':DATA,'training_rows':population,'train_samples':train_count,
       'benchmark_samples':bc,'benchmark_categories':dict(bcats),'benchmark_exact_prompt_overlap_excluded':overlaps,
       'checkpoints':checkpoints,'anchors_per_sample':16,'block_size':8,'max_length':4096,
       'generation_max_new_tokens':256,'loss_decay_gamma':4,'ce_weight':.1,'l1_weight':1.,
       'heldout_available':False,'train_greedy_prefix':'training parser prefix plus fixed empty-thinking header when present',
       'train_empty_thinking_header_preserved':think_prefix_count,
       'benchmark_template':'one user, enable_thinking=False, same as benchmark.py',
       'note':'train_original/train_greedy are seen prompts, not heldout; benchmark dedup only exact normalized user text'}
 writejson(RUN/'config.json',meta)
 writejson(RUN/'prefix_examples.json',[{'cohort':t['cohort'],'prefix_tail':tok.decode(t['prefix_ids'][-80:])} for t in [tasks[0],tasks[-1]]])
 print(json.dumps(meta,ensure_ascii=False),flush=True)
 sys.exit()

if A.mode=='worker':
 import torch
 import torch.nn.functional as F
 from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
 from specforge.modeling.draft.flashmtp import FlashMTPDraftModel
 from specforge.core.flashmtp import OnlineFlashMTPModel
 torch.set_grad_enabled(False);torch.manual_seed(SEED+A.rank)
 cfg=json.loads((RUN/'config.json').read_text())
 target=AutoModelForCausalLM.from_pretrained(TARGET,dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval()
 tok=AutoTokenizer.from_pretrained(TARGET)
 models={};loading={}
 for tag,p in cfg['checkpoints'].items():
  model,info=FlashMTPDraftModel.from_pretrained(p,dtype=torch.bfloat16,attn_implementation='flash_attention_2',output_loading_info=True)
  assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys']),info
  models[tag]=model.cuda().eval();loading[tag]=info
 writejson(RUN/f'loading_{A.rank}.json',loading)
 tasks=[json.loads(l) for l in (RUN/'manifest.jsonl').read_text().splitlines()]
 tasks=[t for t in tasks if t['unit']%A.world==A.rank]
 if A.limit:tasks=tasks[:A.limit]
 outpath=RUN/f'results_{A.rank}.jsonl'
 done=set()
 if outpath.exists():
  done={(r['sample_id'],r['cohort']) for l in outpath.read_text().splitlines() if (r:=json.loads(l))}
 outstream=outpath.open('a',buffering=1)
 checked=set()
 for ti,task in enumerate(tasks):
  key=(task['sample_id'],task['cohort'])
  if key in done:continue
  began=time.time();prefix=torch.tensor([task['prefix_ids']],device='cuda')
  if 'input_ids' in task:
   ids=torch.tensor([task['input_ids']],device='cuda');mask=torch.tensor(task['loss_mask'],device='cuda')
  else:
   ids=target.generate(prefix,attention_mask=torch.ones_like(prefix),do_sample=False,max_new_tokens=256,pad_token_id=tok.eos_token_id)
   mask=torch.zeros(ids.shape[1],device='cuda',dtype=torch.long);mask[prefix.shape[1]:]=1
  valid=[a for a in range(1,ids.shape[1]-7) if bool(mask[a:a+8].all())]
  if not valid:
   outstream.write(json.dumps(dict(sample_id=key[0],cohort=key[1],skipped='no complete supervised block',tokens=ids.shape[1]))+'\n');continue
  seed=int(hashlib.sha256(('|'.join(key)+str(SEED)).encode()).hexdigest()[:8],16)
  anchors=sorted(random.Random(seed).sample(valid,min(16,len(valid))))
  N=len(anchors);aa=torch.tensor(anchors,device='cuda');ii=aa[:,None]+torch.arange(7,device='cuda')[None]
  reference=target(ids,attention_mask=torch.ones_like(ids),output_hidden_states=True,use_cache=True)
  legacy=reference.past_key_values.to_legacy_cache()
  chs=torch.stack([reference.hidden_states[l+1][0,aa-1] for l in models['noconv'].target_layer_ids],dim=1).unsqueeze(1)
  tlogits=reference.logits[0,ii].float();labels=tlogits.argmax(-1);tp=tlogits.softmax(-1)
  prev=ids[0,ii];dataset_next=ids[0,ii+1]
  row={k:v for k,v in task.items() if k not in ('input_ids','loss_mask','prefix_ids')}
  row.update(anchors=anchors,tokens=ids.shape[1],response_tokens=int(mask.sum()),
   trajectory_top1_match=(dataset_next==labels).float().mean().item(),models={})
  weight=torch.exp(-torch.arange(7,device='cuda').float()/4)
  order=['noconv','convk2g16'] if task['unit']%2==0 else ['convk2g16','noconv']
  for tag in order:
   model=models[tag]
   qids=torch.full((N,8),model.mask_token_id,device='cuda',dtype=torch.long);qids[:,0]=ids[0,aa]
   qpos=aa[:,None]+torch.arange(8,device='cuda')[None]
   cpos=(aa-1)[:,None].expand(-1,12)
   h=model(position_ids=qpos,rotary_position_ids=torch.cat([cpos,qpos],-1),noise_embedding=target.model.embed_tokens(qids),target_hidden=chs,is_causal=False)[:,1:]
   latent=model.markov_head.forward_teacher_forcing(hidden_states=h,prev_token_ids=prev,output_mode='direct')
   logits=model.markov_head.project_logits(latent).float()
   ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none').reshape(N,7)
   l1=(logits.softmax(-1)-tp).abs().sum(-1)
   correct=logits.argmax(-1)==labels;pref=correct.long().cumprod(-1)
   ce_w=(ce*weight).sum()/(N*weight.sum());l1_w=(l1*weight).sum()/(N*weight.sum())
   metrics={'ce':ce_w.item(),'tv_l1':l1_w.item(),'tv_standard':l1_w.item()/2,
    'loss':(.1*ce_w+l1_w).item(),'accuracy':correct.float().mean().item(),'tf_prefix':(1+pref.sum(-1).float()).mean().item(),
    'position_accuracy':correct.float().mean(0).tolist(),'prefix_survival':pref.float().mean(0).tolist(),
    'anchor_tf_prefix':(1+pref.sum(-1)).tolist()}
   if tag not in checked:
    wrapper=OnlineFlashMTPModel(model,target.lm_head,target.model.embed_tokens,model.mask_token_id,block_size=8,final_ce_weight=.1,tv_loss_weight=1.,loss_decay_gamma=4.)
    values=wrapper._chunked_weighted_ce_and_metrics(prediction_hidden=h.unsqueeze(0),prev_token_ids=prev.unsqueeze(0),labels=labels.unsqueeze(0),weight_mask=weight[None,None].expand(1,N,7),binary_eval_mask=torch.ones((1,N,7),device='cuda',dtype=torch.bool),block_keep_mask=torch.ones((1,N),device='cuda',dtype=torch.bool),target_prediction_logits=tlogits.unsqueeze(0))
    torch.testing.assert_close(torch.stack([values[0],values[1],values[2],values[3],values[5]]),torch.tensor([metrics[k] for k in ['loss','accuracy','tf_prefix','ce','tv_l1']],device='cuda'),atol=2e-5,rtol=2e-5)
    checked.add(tag);metrics['training_metric_selfcheck']='passed'
   proposal,_=model.sample_draft_tokens(draft_hidden=h,lm_head=target.lm_head,first_prev_token_ids=ids[0,aa],temperature=0.)
   free=[];survival=[]
   for j,a in enumerate(anchors):
    cache=DynamicCache.from_legacy_cache(tuple((k[:,:,:a],v[:,:,:a]) for k,v in legacy))
    verify=torch.cat([ids[:,a:a+1],proposal[j:j+1]],-1)
    verified=target(verify,past_key_values=cache,position_ids=torch.arange(a,a+8,device='cuda')[None],use_cache=True,output_hidden_states=False)
    match=(proposal[j]==verified.logits[0,:7].argmax(-1)).long().cumprod(-1)
    free.append(1+int(match.sum()));survival.append(match)
    del cache,verified
   metrics.update(free_accept=sum(free)/N,anchor_free_accept=free,free_prefix_survival=torch.stack(survival).float().mean(0).tolist())
   row['models'][tag]=metrics
   del h,latent,logits,ce,l1,proposal
  # Do not retain full-sequence KV and vocabulary tensors during end-to-end timing.
  del reference,legacy,tlogits,tp,chs
  if 'input_ids' not in task and task['e2e']:
   for tag in order:
    model=models[tag];torch.cuda.synchronize();st=time.perf_counter()
    actual=model.spec_generate(target=target,input_ids=prefix,max_new_tokens=256,stop_token_ids=[tok.eos_token_id],temperature=0.,verify_block_size=8)
    stats=model.get_last_decode_stats();ls=stats['accept_lengths'];nt=actual.shape[1]-prefix.shape[1]
    row['models'][tag]['e2e']={'accept':sum(ls)/len(ls),'steps':len(ls),'tokens':nt,'decode_seconds':stats['decode_wall_time'],'ms_per_token':1000*stats['decode_wall_time']/nt,'accept_lengths':ls,'output_hash':hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()}
  row['elapsed_seconds']=time.time()-began
  outstream.write(json.dumps(row,ensure_ascii=False)+'\n')
  print(json.dumps({'rank':A.rank,'done':ti+1,'total':len(tasks),'sample':key,'seconds':row['elapsed_seconds'],'tf':{k:v['tf_prefix'] for k,v in row['models'].items()},'free':{k:v['free_accept'] for k,v in row['models'].items()}}),flush=True)
  gc.collect()
 print('WORKER_COMPLETE',A.rank,flush=True)
 sys.exit()

if A.mode=='summarize':
 rows=[json.loads(l) for p in sorted(RUN.glob('results_*.jsonl')) for l in p.read_text().splitlines()]
 rng=np.random.default_rng(SEED);summary={}
 for cohort in sorted({r['cohort'] for r in rows}):
  rs=[r for r in rows if r['cohort']==cohort and 'models'in r]
  if not rs:continue
  out={'samples':len(rs),'anchors':sum(len(r['anchors']) for r in rs),'trajectory_top1_match':float(np.mean([r['trajectory_top1_match'] for r in rs])),'metrics':{}}
  boot=rng.integers(0,len(rs),(10000,len(rs)))
  for key in ['loss','ce','tv_l1','accuracy','tf_prefix','free_accept']:
   no=np.array([r['models']['noconv'][key] for r in rs]);co=np.array([r['models']['convk2g16'][key] for r in rs]);d=co-no
   out['metrics'][key]={'noconv':float(no.mean()),'conv':float(co.mean()),'delta':float(d.mean()),'paired_bootstrap95':np.quantile(d[boot].mean(1),[.025,.975]).tolist()}
  out['positions']={tag:{key:np.mean([r['models'][tag][key] for r in rs],axis=0).tolist() for key in ['position_accuracy','prefix_survival','free_prefix_survival']} for tag in ['noconv','convk2g16']}
  er=[r for r in rs if 'e2e' in r['models']['noconv']]
  if er:
   out['e2e']={'samples':len(er),'models':{tag:{'mean_accept':sum(sum(r['models'][tag]['e2e']['accept_lengths']) for r in er)/sum(r['models'][tag]['e2e']['steps'] for r in er),'ms_per_token':1000*sum(r['models'][tag]['e2e']['decode_seconds'] for r in er)/sum(r['models'][tag]['e2e']['tokens'] for r in er),'tokens':sum(r['models'][tag]['e2e']['tokens'] for r in er)} for tag in ['noconv','convk2g16']},'identical_output_pairs':sum(r['models']['noconv']['e2e']['output_hash']==r['models']['convk2g16']['e2e']['output_hash'] for r in er)}
  summary[cohort]=out
 writejson(RUN/'summary.json',{'config':json.loads((RUN/'config.json').read_text()),'completed':len(rows),'skipped':[r for r in rows if 'skipped'in r],'cohorts':summary})
 print(json.dumps(summary,ensure_ascii=False,indent=2))
