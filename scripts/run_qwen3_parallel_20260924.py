import concurrent.futures
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

BASE = Path('/data/wanghanzhen')
PY = str(BASE / 'FlashMTP_v2swa/.venv/bin/python')
DATASETS = [('gsm8k',128),('math500',128),('aime25',30),('humaneval',164),('mbpp',128),('livecodebench',128),('mt-bench',80),('alpaca',128)] + [('longbench_v2_64000_32000_'+s,None) for s in ['single_document_qa','multi_document_qa','long_dialogue','structured_data','in_context_learning','code_repo']]
MODELS = {
 'v2': ('FlashMTP_v2swa','Flashmtp_v2_qwen3_8b_ep1','benchmark.py',10),
 'v23': ('FlashMTP_v2.3','Flashmtp_v2.3_qwen3_8b_ep1','benchmark_ep1_report.py',50),
}
ACTIVE = {0: (1879452,1845286,'v2','mbpp'), 1: (1881265,1881254,'v23','gsm8k')}
LOCK = threading.Lock()
INFLIGHT = {(x[2],x[3]) for x in ACTIVE.values()}
FAILED = set()

def root(model):
 repo,name,_,_ = MODELS[model]
 return BASE/repo/'benchmark_results'/(name+'_20260924')/'full'

def complete(model,dataset):
 p=root(model)/(dataset+'.json')
 if not p.exists(): return False
 try:
  d=json.loads(p.read_text())
  return bool(d.get('records')) and 'overall' in d
 except (ValueError,OSError): return False

def log(message):
 print(datetime.datetime.now(datetime.timezone.utc).isoformat(),message,flush=True)

def running(pid):
 try: return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0] != 'Z'
 except FileNotFoundError: return False

def worker(gpu):
 if gpu in ACTIVE:
  pid,parent,model,dataset=ACTIVE[gpu]
  log(f'GPU {gpu} waiting for existing {model}/{dataset} PID {pid}')
  while running(pid): time.sleep(5)
  # Parents were stopped before this scheduler started. Terminate before resuming
  # so their old sequential queues cannot start duplicate work.
  for sig in (signal.SIGTERM,signal.SIGCONT):
   try: os.kill(parent,sig)
   except ProcessLookupError: pass
  with LOCK:
   INFLIGHT.remove((model,dataset))
   if not complete(model,dataset): FAILED.add((model,dataset))
  log(f'GPU {gpu} existing {model}/{dataset} finished')
 while True:
  with LOCK:
   job=next(((m,d,n) for d,n in DATASETS for m in ('v23','v2') if (m,d) not in INFLIGHT and (m,d) not in FAILED and not complete(m,d)),None)
   if job is None: return
   model,dataset,count=job
   INFLIGHT.add((model,dataset))
  repo,name,script,long_count=MODELS[model]
  out=root(model)
  args=[PY,'evaluation/'+script,'--model-name-or-path',str(BASE/'models/Qwen3-8B'),'--draft-name-or-path',str(BASE/'FlashMTP_v2swa/cache/models/Qwen3-8B'/name),'--dataset',dataset,'--max-samples',str(count or long_count),'--max-new-tokens','512','--temperature','0','--output-json',str(out/(dataset+'.json'))]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',NO_COLOR='1')
  env.pop('CUDA_LAUNCH_BLOCKING',None)
  log(f'START GPU {gpu} {model}/{dataset}')
  with (out/(dataset+'.log')).open('w') as f:
   proc=subprocess.Popen(args,cwd=BASE/repo,env=env,stdout=f,stderr=subprocess.STDOUT)
   log(f'PID {proc.pid} GPU {gpu} {model}/{dataset}')
   rc=proc.wait()
  with LOCK:
   INFLIGHT.remove((model,dataset))
   if rc or not complete(model,dataset): FAILED.add((model,dataset))
  log(f'END GPU {gpu} {model}/{dataset} exit={rc}')

if __name__ == '__main__':
 lockfile=open(str(Path(__file__).with_suffix('.lock')),'w')
 fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for m in MODELS: (root(m)/'parallel_status').write_text('running\n')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  list(pool.map(worker,[0,1,2,3]))
 for m in MODELS:
  state='completed' if all(complete(m,d) for d,_ in DATASETS) else 'failed'
  (root(m)/'parallel_status').write_text(state+'\n')
  (root(m)/'status').write_text(state+'\n')
 log(f'FINISHED failures={sorted(FAILED)}')
