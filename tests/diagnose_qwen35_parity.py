"""Compare fused/unfused cached convolution and trace first greedy divergence."""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.model_loading import load_flashmtp_benchmark_models
from evaluation.benchmark import load_benchmark_dataset, select_max_samples, target_generate, flashmtp_generate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-name-or-path',required=True)
    p.add_argument('--draft-name-or-path',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--indices',default='5,6,7,8,9')
    args=p.parse_args(); args.block_size=None
    target,draft,tok,_=load_flashmtp_benchmark_models(args,torch.device('cuda'))
    data=select_max_samples(load_benchmark_dataset('gsm8k'),128)
    trace={}
    def hook(module, a, kw, output):
        pos=kw['position_ids'][0]
        vals,ids=output.logits[0].topk(2,dim=-1)
        length=ids.shape[0]
        positions=pos[-length:].tolist()
        for index,v,k in zip(positions,vals.tolist(),ids.tolist()):
            trace[index+1]={'ids':k,'logits':v,'gap':v[0]-v[1]}
    handle=target.register_forward_hook(hook,with_kwargs=True)
    rows=[]
    with torch.inference_mode():
        for idx in map(int,args.indices.split(',')):
            prompt=tok.apply_chat_template([{'role':'user','content':data[idx]['turns'][0]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok(prompt,return_tensors='pt').input_ids.cuda()
            trace.clear()
            ref=target_generate(target,ids,512,[tok.eos_token_id],0)
            ref_trace=dict(trace)
            for fused in (False,True):
                for layer in target.model.layers:
                    if layer.layer_type=='linear_attention': layer.linear_attn._flashmtp_fused_conv=fused
                trace.clear()
                spec=flashmtp_generate(draft,target,ids,512,draft.block_size,8,[tok.eos_token_id],0)
                a=ref.output_ids[0].tolist();b=spec.output_ids[0].tolist()
                first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
                equal=a==b
                row={'sample':idx,'fused_conv':fused,'equal':equal,'first_divergence_generated_index':None if equal else first-ids.shape[1], 'reference_at_divergence':ref_trace.get(first),'spec_at_divergence':trace.get(first),'reference_tokens':ref.num_output_tokens,'spec_tokens':spec.num_output_tokens}
                if not equal:
                    row['reference_excerpt']=tok.decode(a[max(ids.shape[1],first-12):first+16])
                    row['spec_excerpt']=tok.decode(b[max(ids.shape[1],first-12):first+16])
                rows.append(row)
                print(json.dumps(row,ensure_ascii=False),flush=True)
                Path(args.output).write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    handle.remove()

if __name__=='__main__': main()
