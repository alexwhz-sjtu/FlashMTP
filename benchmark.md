对于我给出的模型，你要用相应的评测脚本来测试
短文本：alpaca，gsm8k，math500，mbpp，livecodebench，humaneval，mt-bench,aime25
长文本："longbench_v2_64000_32000_single_document_qa":,
    "longbench_v2_64000_32000_multi_document_qa",
    "longbench_v2_64000_32000_long_dialogue",
    "longbench_v2_64000_32000_structured_data": "/share/dai-sys/wanghanzhen/datasets/longbench_v2/Long_Structured_Data_Understanding_32k_64k.jsonl",
    "longbench_v2_64000_32000_in_context_learning": "/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/datasets/longbench_v2/Long In-context Learning.json",
    "longbench_v2_64000_32000_code_repo": "/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/datasets/longbench_v2/Code Repository Understanding.json"
对于flashmtp类的模型：：均以/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2swa/evaluation/benchmark.py中采样方式，max-samples=50，max-new-tokens=512，不限制输入长度。

要求：最终以表格形式给我，如果有多张表，一个模型一个表，包含数据集名称，接受长度和加速比。利用机器上空闲gpu，注意不要抢占，一个任务一张卡。