# FlashMTP

## Ours core idea

Since the hidden states are calculated by the model from its complete context, therefore they are the concentration of the context.
When predicting the following block tokens, we only need the latest hidden states.
We propose FlashMTP, which utilize the last hidden states (limited context) and bidirectional attention (diffusion-based) to make draft token generation like a flash in MTP/SD

## Base structure

Like DFlash. But we use all bonus hidden states of all layers. Cuz when generating the hidden states, all layer focus on different part of context as the attention patern of every head across layers differ a lot. We concat them along feature dim (seq dim fails) and use it as condition. Then we concat bonus clean token and several mask(noise) and forward only once. Noise block serves as Q, concat sequence serves as kv.  Every layer's kv is the same.

![basestructure](assets/base_structure.png)

## v1.1 Improved condition injection

* To improve model expression and condition info, we input the whole concat seq into the model as Q. Therefore, the prefix can be processed across layers and every layer can have different prefix.
* We buiding the prefix hidden states, we include the initial embedding to let the model know the begining point.
*

<img src="assets/v1.1.png" alt="v1.1 structure" width="400"/>

## v2: Improved structure

The basic version has a drawback. There are always repetative in adjacent positions or meaningless tokens like ("the", "," ...) in the last. This rises from the harlearning goal of the one-forward prediction paradigm. We'd like to utilize diffusion method. Continuous diffusion and disill it.

Besides, the later the posiion of tokens, the less info i can get from the condition. Inspired by sse, maybe we can enhance condition by concating it with every mask(noise).

## Use UV

> \# git clone the source code
>
> git clone https://github.com/sgl-project/SpecForge.git
>
> cd SpecForge
>
> \# create a new virtual environment
>
> uv venv -p 3.11
>
> source .venv/bin/activate
>
> \# install specforge
>
> uv pip install -v -e . --prerelease=allow
>
> uv pip install datasets==4.8.3 pyarrow==23.0.1

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
