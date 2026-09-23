"""Synthetic backbone forward benchmark; no target, LM head or serial head timing."""
import argparse, copy, gc, json, statistics, time
from pathlib import Path
import torch
from transformers import Qwen3Config
from specforge.modeling.draft.flashmtp import FlashMTPDraftModel
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True); p.add_argument('--iterations',type=int,default=40); p.add_argument('--rounds',type=int,default=5); p.add_argument('--backend',default='sdpa'); p.add_argument('--shared-gpu', action='store_true', help='Record that other GPU workloads are present'); a=p.parse_args()
c=Qwen3Config.from_pretrained(a.config); c._attn_implementation=a.backend
# Head does not participate in model.forward; omit its allocation in all modes.
c.flashmtp_config.update(markov_head_type='none',markov_output_mode='additive')
models={}
for mode in ('none','full','simple_conv'):
    cc=copy.deepcopy(c); cc.flashmtp_config['backbone_conv_mode']=mode
    torch.manual_seed(123)
    models[mode]=FlashMTPDraftModel(cc).to(device='cuda',dtype=torch.bfloat16).eval()
m=models['none']; q=m.draft_query_length; ctx=m.condition_slot_count
args=dict(position_ids=torch.arange(q,device='cuda')[None],rotary_position_ids=torch.arange(ctx+q,device='cuda')[None],noise_embedding=torch.randn(1,q,c.hidden_size,device='cuda',dtype=torch.bfloat16),target_hidden=torch.randn(1,1,ctx,c.hidden_size,device='cuda',dtype=torch.bfloat16),attention_mask=torch.zeros(1,1,q,ctx+q,device='cuda',dtype=torch.bfloat16))
results={mode:[] for mode in models}
with torch.inference_mode():
    for model in models.values():
        for _ in range(15): model(**args)
    torch.cuda.synchronize()
    for r in range(a.rounds):
        order=list(models); order=order[r%3:]+order[:r%3]
        for mode in order:
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); wall=time.perf_counter(); start.record()
            for _ in range(a.iterations): models[mode](**args)
            end.record(); torch.cuda.synchronize()
            results[mode].append(dict(cuda_ms=start.elapsed_time(end)/a.iterations,wall_ms=(time.perf_counter()-wall)*1000/a.iterations))
out=dict(device=torch.cuda.get_device_name(),backend=a.backend,dtype='bfloat16',batch_size=1,query_length=q,context_slots=ctx,layers=c.num_hidden_layers,hidden_size=c.hidden_size,block_size=c.block_size,iterations=a.iterations,rounds=a.rounds,shared_gpu=a.shared_gpu,results=results)
out['median_cuda_ms']={k:statistics.median(x['cuda_ms'] for x in v) for k,v in results.items()}
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
