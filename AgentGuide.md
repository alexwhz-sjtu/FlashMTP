# FlashMTP Agent Guide

本项目正在快速迭代 FlashMTP 的模型结构、训练目标和推理流程。AI 修改代码时必须先阅读本文件，并把模型侧、训练侧、推理侧和评测侧作为一个整体维护。

## 协同修改要求

1. 当用户提出“改变模型结构”“改变 hidden states / contextual pivot 的使用方式”“改变 noise/mask/block 构造”“改变 loss 或训练数据格式”时，不要只改训练代码。必须同步检查并按需修改：
  - `specforge/modeling/draft/flashmtp.py` 中的 `FlashMTPDraftModel.forward()` 和 `FlashMTPDraftModel.spec_generate()`。
  - `scripts/train_flashmtp.py` 以及 `specforge/core/flashmtp.py` 中和训练输入、loss、mask、position ids、target hidden 相关的逻辑。
  - `evaluation/` 下的评测脚本，确保新的 checkpoint 可以直接跑测试。
2. `spec_generate()` 是推理/测试的主入口。任何训练时引入的新输入约定，都要在这里有对应推理实现，例如：
  - `target_hidden` 的提取层、拼接方式、归一化方式。
  - `noise_embedding` / mask token / anchor token / block 内 token 的构造方式。
  - `position_ids`、attention mask、draft cache、target cache 的更新方式。
  - 验收逻辑、stop token 裁剪、输出 token 序列格式。
3. 修改 /evaluation 时，优先保证以下脚本和当前 `FlashMTPDraftModel` 对齐：
  - `evaluation/benchmark.py`：用于速度、acceptance length、吞吐等快速，全面，多样 benchmark。
  - `evaluation/eval.py`：用于单个prompt测试或者多轮问题集测试和保存 jsonl/log 结果。
  - `evaluation/utils.py`：用于数据集加载、prompt 格式化、采样和辅助函数。
4. 不要把 DFlash 的类名、checkpoint 路径或导入方式复制到 FlashMTP 测评脚本中。FlashMTP 评测应使用：
  - `FlashMTPDraftModel`
  - `specforge.modeling.draft.flashmtp`
  - 文件名和输出结果中的 `FlashMTP` 标识。
5. 如果修改后的 `spec_generate()` 仍然需要评测统计，请维护 `get_last_decode_stats()` 返回的字段：
  - `accept_lengths`
  - `target_total_time`
  - `draft_total_time`
  - `steps`
6. 完成模型/训练改动后，至少做一次轻量检查：
  - 确认 `FlashMTPDraftModel.from_pretrained(...)` 能被 evaluation 脚本导入。
  - 确认 `spec_generate()` 的参数和 evaluation 调用一致。
  - 确认 evaluation 不再引用不存在的 DFlash 模块或旧路径。


## 当前核心假设

FlashMTP 的核心方向是去掉草稿模型侧 KV cache 对完整历史的依赖，利用目标大模型最新 hidden states / contextual pivot 表示历史上下文，并结合 block 内 mask/noise token 进行并行草稿预测。除非用户明确要求，不要把实现退回到依赖完整草稿侧历史 KV 的方案。