# FlashMTP

详情见FRAMEWORK_GUIDE.md
## Ours core idea

由于隐状态是模型在**完整上下文**下计算得到的，因此它们可以看作对上下文的**浓缩表示**。在预测后续 block 的 token 时，我们只需要**最新的隐状态**即可。

我们提出 FlashMTP：利用**最新的隐状态**（有限上下文）并结合**双向注意力**（扩散式思路），使 draft 的 token 生成在 MTP/SD 中像“闪一下”一样高效

## Base structure

与 DFlash 类似。但我们使用**所有层**的 bonus 隐状态。原因在于：在生成隐状态时，各层会关注上下文的不同部分，因为不同层、不同注意力头的模式差异很大。我们沿**特征维度**把它们拼接起来（沿序列维度拼接效果不好），并作为条件使用。随后把 bonus 的干净 token 与若干 mask（噪声）拼接起来，**只做一次前向**。其中噪声 block 作为 **Q**，拼接后的序列作为 **KV**。

## v1.1 Improved condition injection

- 为提升模型表达能力与条件信息量，我们把**整条拼接序列**都作为 **Q** 输入模型。这样前缀可以在各层之间被逐步处理，每一层都能得到**不同的前缀表示**。
- 在构造前缀隐状态时，我们**把初始 embedding 也纳入其中**。
- **seq 模式**：各层对应的隐状态使用**相同的位置 id（position id）**。



## v2: Improved structure

加入最新一段kvcache

## Use UV

>  git clone the source code
>
> git clone [https://github.com/sgl-project/SpecForge.git](https://github.com/sgl-project/SpecForge.git)
>
> cd SpecForge
>
>  create a new virtual environment
>
> uv venv -p 3.11
>
> source .venv/bin/activate
>
>  install specforge
>
> uv pip install -v -e . --prerelease=allow

uv pip install datasets==4.8.3 pyarrow==23.0.1

## 📃 Citation

```plaintext
@misc{specforge2025,
  title={SpecForge: Train speculative decoding models effortlessly},
  author={Shenggui Li, Yikai Zhu, Chao Wang, Fan Yin, Shuai Shi, Yubo Wang, Yi Zhang, Yingyi Huang, Haoshuai Zheng, Yineng Zhang},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/sgl-project/specforge}},
}
```

