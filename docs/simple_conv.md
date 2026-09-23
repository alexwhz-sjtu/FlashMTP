### Attention 前后卷积 simple_conv

保留原有训练命令的其它参数，设置 `BACKBONE_CONV_MODE=simple_conv`。
可选模式为 `none`、`full`、`simple_conv`；显式模式优先于旧的
`ENABLE_BACKBONE_CONV`。未设置模式时，旧开关行为保持不变。

```bash
BACKBONE_CONV_MODE=simple_conv \
BACKBONE_CONV_KERNEL_SIZE=2 \
BACKBONE_CONV_GROUP_SIZE=16 \
bash scripts/run_training_flashmtp.sh --dt h100
```

直接调用 `scripts/train_flashmtp.py` 时，添加：
`--backbone-conv-mode simple_conv --conv-kernel-size 2 --conv-group-size 16`。
这里的 `--dt` 沿用原有环境配置选择，不根据 GPU 型号自动切换；数据、模型和输出路径
需沿用自己的训练配置。

`simple_conv` 保留原 full 模式每层 attention 输入与输出的动态因果卷积，
删除 MLP 输入与输出的卷积，也不再添加层输出卷积。最后一层同样保留 attention 卷积。
5 层时共 10 次卷积（full 为 20 次），基础核恒等初始化、动态投影零初始化。
新 checkpoint 记录 `simple_conv_layout=attention_pre_post`；旧层间 output_conv 版
simple_conv checkpoint 不能直接恢复，会明确报错，需要重新训练或显式转换权重。
模式保存到 checkpoint 的 `flashmtp_config.backbone_conv_mode`，推理自动读取。
旧 checkpoint 未记录模式时，根据 `backbone_conv_enabled` 恢复 none/full。
恢复训练要求卷积模式相同，不能将已有 full checkpoint 直接作为 simple_conv resume。

可复现的合成 backbone forward 测速（不含 target、LM head 和串行 head）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \
  scripts/benchmark_simple_conv_forward.py \
  --config /path/to/checkpoint/config.json \
  --output benchmark_results/simple_conv_forward.json
```
