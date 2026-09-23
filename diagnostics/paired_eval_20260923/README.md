# FlashMTP convk2g16 / noconv paired evaluation

2026-09-23，Inf_H800_2。比较用户提供的两个 epoch_6_step_59730 checkpoint，完整路径见 config.json。

## 结论

训练样本上的改善真实可复现，而且能转化为相同上下文下的实际 target 接受长度提升；但在用户指定的 Alpaca、GSM8K、MBPP 问题上，训练收益没有迁移为明确的接受长度提升，loss / L1 / accuracy 的等权汇总反而变差。真实解码的小样本计时也更慢。

证据支持这两个 checkpoint 存在训练收益迁移不足，叠加卷积计算成本的解释。没有独立同来源验证集、多个训练种子或逐算子计时，不能严格归因为过拟合，也不能将全部耗时差都归于卷积算子。

## 关键结果

差值统一为 conv − noconv，以下接受长度含 anchor，范围 1–8。

- 训练原回答，128 条：loss 0.5160 → 0.4413；L1 0.4112 → 0.3517；accuracy 77.895% → 81.459%；TF prefix 5.217 → 5.608；实际接受长度 5.147 → 5.544，差值 +0.397，样本级配对 95% CI [0.323, 0.474]。
- 原回答截至 generation prefix 后最多 256 token，128 条：实际接受长度 5.109 → 5.507，差值 +0.397。截短未减弱收益，因此原始回答长度本身不是主要解释。
- 相同训练问题重新生成 greedy 回答，128 条：实际接受长度 4.653 → 4.873，差值 +0.220。相对截短原回答，卷积优势减少 0.177 token，配对 95% CI [−0.288, −0.070]。这一对照支持收益对回答轨迹敏感，并不独立证明记忆化。
- Alpaca，63 条：实际接受长度 2.366 → 2.401，差值 +0.035，95% CI [−0.041, 0.115]。
- GSM8K，64 条：4.816 → 4.739，差值 −0.077，95% CI [−0.189, 0.040]。
- MBPP，64 条：4.129 → 4.150，差值 +0.021，95% CI [−0.070, 0.111]。
- 三个主测试集等权：loss 0.8531 → 0.8838；L1 0.6587 → 0.6776；accuracy 64.619% → 63.608%（−1.011 个百分点，95% CI [−1.617, −0.436]）；TF prefix 3.771 → 3.763；实际接受长度 3.770 → 3.764，差值 −0.007，95% CI [−0.062, 0.050]。

这也回答了“训练三个指标都变好，为什么测试不提升”：这三个改善是在训练分布上同时发生的；在新问题上，同样口径的指标并没有同时改善。无需用“teacher forcing 天生虚高”解释本次结果。

## 连续解码计时

每个主测试集 16 条，greedy、block 8、最多 256 新 token；只计 decode，不计 prefill。模型运行顺序在样本间交替，同一对在相同 GPU 上运行。

- Alpaca：平均接受长度 2.134 → 2.153；16.324 → 17.076 ms/token，耗时 +4.61%。
- GSM8K：4.165 → 4.001；8.404 → 9.361 ms/token，耗时 +11.40%。
- MBPP：3.773 → 3.756；9.195 → 9.736 ms/token，耗时 +5.88%。

上述接受长度按总步数汇总，耗时按总 token 汇总，与固定 anchor 的样本均值不同。每条只计时一次，8 个 GPU worker 同时运行，未做独立重复计时或逐算子 profile。固定 anchor 与真实解码的 anchor 分布不同，不应直接比较绝对值。

两模型的输出分别有 15/16、15/16、13/16 对完全一致；BF16 下不同验证分块可能改变临界 argmax，不能声称逐 token 全部相同。仅保留输出一致的配对，耗时增幅仍为 4.76%、11.32%、6.54%，方向未变。此前的层、token、cache 对齐检查及 BF16 数值差异诊断见 ../prefill-chs/README.md。本次没有对每个不一致输出重新逐 token 定因。

## 方法及范围

- 704 个任务全部完成（exit_status.txt 为 0），703 条可评估，11,138 个 anchor。Alpaca 一条因缺少完整监督 block 跳过。
- 三组训练控制各 128 条；主测试 Alpaca 63 / GSM8K 64 / MBPP 64。SpecBench 128 条只作补充，未计入主测试汇总。
- 原训练数据全部 80,000 行扫描，seed=20260923；确定性抽取有效且唯一的首轮问题。原回答最长 4096，最多 16 个完整监督 block anchor。
- 测试来自现有缓存 /data/processed_dataset_cache/alpaca_v1、gsm8k_v1、mbpp_v1；分别使用 Alpaca train、GSM8K test、sanitized MBPP test。采用项目现有问题格式，GSM8K 含 step-by-step / boxed answer 指令。没有使用标准答案进行任务正确率评分。
- 对训练 user 文本与测试问题做 normalized exact-match 去重；选定问题未发现精确重合。无法排除近似或语义重复。没有独立同来源验证集，所有训练控制均明确标记为已见样本。
- 训练控制使用训练 GeneralParser 的模板、系统提示与 loss mask；重新生成时保留已有空 thinking header。测试使用 benchmark 的 user-only、enable_thinking=False 模板。模板/来源/轨迹变化均可能影响泛化，当前实验未逐一隔离。
- 每个模型共享同一完整 token 序列、anchor、target hidden 和 target 标签。target Qwen3-8B，BF16，FA2。
- anchor=a 的 CHS 来自完整回答 prefill 的 hidden_states[layer+1] 在 a−1 的位置；目标层为 [0,1,5,8,12,16,19,23,27,30,34,35]。预测 a+1 到 a+7，标签取 target logits 在 a 到 a+6 的 argmax。
- TF 指标按训练函数同样公式计算并自检：CE 和 L1 按 exp(−j/4) 加权，loss=0.1×CE+L1；accuracy 为七个预测位置平均；prefix=1+连续正确 draft token 数。代码所称 TV 是 L1，即标准 TV 的两倍。
- free_accept 使用串行 draft 提议，再以同一 anchor 的前缀 KV 交给 target 实际验证，非 TF prefix 的替代命名。
- 每个 worker 的两模型加载检查无 missing/unexpected/mismatched keys；训练指标公式自检通过。源码快照含此前修复的首次 prefill 最后一层 RMSNorm 对齐逻辑。现有对齐测试未支持层/token 错位作为此差异的主要解释，但不等于证明所有实现路径绝无问题。
- 所有固定 anchor 指标先在样本内平均，再在样本间平均。95% CI 用 10,000 次样本级配对 bootstrap；三集主汇总按数据集分层、等权。CI 不包含训练种子不确定性。
- 原始 token 与生成 prefix 保存在 manifest.jsonl；重新生成的完整回答未单独落盘，仅记录样本、anchor、长度与指标。BF16 数值路径变化可能导致复现时逐 token 差异。

## 文件与复现

远端运行目录：/data/wanghanzhen/FlashMTP_v2swa/diagnostics/paired_eval_20260923。

- summary.json：各组完整指标、位置曲线、真实解码汇总。
- extended_summary.json：主测试等权汇总、配对长度/轨迹对照、解码置信区间。
- results_0.jsonl 至 results_7.jsonl：逐样本及逐 anchor 结果。
- config.json / manifest.jsonl：checkpoint 路径、参数、精确抽样清单与输入。
- paired_eval.py / add_requested_sets.py / extended_analysis.py / launch.sh：本次评估脚本；worker 按样本标识断点续跑。
- source_commit.txt / source_worktree.patch：源码快照标识；远端 source/specforge 保存运行源码。不要直接改当前工作区来替代快照。
- loading_*.json / worker_*.log / exit_status.txt：加载检查、日志和退出状态。

在远端项目下重算汇总：

```sh
.venv/bin/python diagnostics/paired_eval_20260923/paired_eval.py summarize --run diagnostics/paired_eval_20260923
.venv/bin/python diagnostics/paired_eval_20260923/extended_analysis.py diagnostics/paired_eval_20260923
```

完整重跑应复制脚本、source 快照、config 和 manifest 到新运行目录，使用 launch.sh 相同环境启动 worker，避免已有结果触发断点跳过。
