对于debug，测试和测评的脚本和文件，都存入/tests中

开始处理任务前，必须显式检查仓库内的 `.agents/skills/*/SKILL.md`（注意
`.agents` 是隐藏目录，不能依赖会忽略隐藏文件的默认搜索结果）。如果任务与某个
仓库 skill 的 description 匹配，必须先完整读取该 `SKILL.md` 及其要求的 reference，
再执行任务，并严格遵循其中的确认步骤，不得用自行假设代替用户确认。

涉及新数据集的下载、检查、prompt 提取或转换时，必须使用
`.agents/skills/prepare-spec-decoding-dataset/SKILL.md`。在创建 adapter 或执行完整
转换之前，必须先检查 5 条样本，并向用户确认以下两项：

1. 使用哪个字段或路径提取 prompt；
2. 使用 `multi` 保留所有 user 轮次，还是使用 `first` 只保留首个 user 轮次。

在用户明确回答这两项之前，不得自行选择，不得开始完整转换。
