# FlashMTP v2 SWA benchmark 规范

给定 FlashMTP v2 SWA 模型时，使用 `evaluation/benchmark.py` 测试。默认使用
`max-new-tokens=512`、`temperature=0`、不限制输入长度。一个任务只使用一张空闲
GPU，不得与其他任务抢占同一张卡。

## 短文本样本数


| 数据集 key         | 样本数 |
| --------------- | --- |
| `gsm8k`         | 128 |
| `math500`       | 128 |
| `aime25`        | 30  |
| `humaneval`     | 164 |
| `mbpp`          | 128 |
| `livecodebench` | 128 |
| `mt-bench`      | 80  |
| `alpaca`        | 128 |


未显式传入 `--max-samples` 时，评测代码自动采用上表数量。显式传参可覆盖默认值。
`mt-bench=80` 表示 80 个问题；每个问题包含两轮，因此日志最多记录 160 个 turn。

## 长文本

- `longbench_v2_64000_32000_single_document_qa`
- `longbench_v2_64000_32000_multi_document_qa`
- `longbench_v2_64000_32000_long_dialogue`
- `longbench_v2_64000_32000_structured_data`
- `longbench_v2_64000_32000_in_context_learning`
- `longbench_v2_64000_32000_code_repo`

长文本保持原有样本设置；以上数据集使用 32k–64k 分片，不设置
`max-input-tokens`。最终结果按模型分别列出数据集名称、平均接受长度和 token 加权
加速比。