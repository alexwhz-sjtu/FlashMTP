# FlashMTP SGLang 适配器

本目录包含面向 `sglang==0.5.6.post2` 的仓库内本地适配器。该适配器复用了
SGLang 的 DFlash 线性验证逻辑和调度器协议；草稿生成路径则使用 FlashMTP 的
`prefix_condition + local_position`，并支持 embedding CHS 与 direct serial Markov head。
FlashMTP 草稿侧不持有 KV cache、请求
token 表、注意力后端或内存分配器。

请使用已打补丁的 `mtp-sglang` 环境。启动器会主动丢弃继承的 `PYTHONPATH`，
防止旧的 uv/conda 环境将其自身的 `site-packages` 注入 SGLang 的多进程子进程。
启动前可通过以下命令确认当前实际使用的运行环境：

```bash
/share/dai-sys/wanghanzhen/envs/mtp-sglang/bin/python -c \
  'import inspect,sys,sglang; print(sys.executable); print(inspect.getfile(sglang))'
```

在仓库根目录执行以下命令，启动默认采用 spec-v2 重叠调度的服务：

```bash
/share/dai-sys/wanghanzhen/envs/mtp-sglang/bin/python \
  -m specforge.sglang_flashmtp.launch_server \
  --model-path /share/dai-sys/wanghanzhen/models/Qwen/Qwen3.5-35B-A3B \
  --speculative-algorithm FLASHMTP \
  --speculative-draft-model-path /path/to/flashmtp-checkpoint \
  --attention-backend fa3 \
  --tp-size 1
```

如需回退到 spec-v1，请添加 `--disable-overlap-schedule`。启动器会在子进程的
`PYTHONPATH` 中加入一个很小的 `sitecustomize` 引导模块；它不会修改 conda
环境，也不会改动 `site-packages` 中的 SGLang 文件。

目前支持 Qwen3 与 Qwen3.5 text（含 MoE）目标模型、贪心解码、
`prefix_condition`、`local_position=true`、embedding CHS 和 direct serial Markov head。
以下情况会被适配器拒绝：非零 temperature、语法约束
请求、返回 logprob 的请求、与草稿模型结构不兼容的目标模型，以及与 checkpoint
不一致的运行时 block size。

HTTP 基准测试脚本为 `evaluation/benchmark_sglang.py`，示例如下：

```bash
DATASET=math500
CUDA_VISIBLE_DEVICES=0 /share/dai-sys/wanghanzhen/envs/mtp-sglang/bin/python \
  evaluation/benchmark_sglang.py \
  --dataset ${DATASET} \
  --target-model /share/dai-sys/wanghanzhen/models/Qwen/Qwen3.5-35B-A3B \
  --draft-model /path/to/flashmtp-checkpoint \
  --concurrencies 1 \
  --tp-size 1 \
  --max-samples 50 \
  --attention-backends fa3 \
  --output-md log-sglang/${DATASET}.md \
  --response-json log-sglang/${DATASET}.json
```
