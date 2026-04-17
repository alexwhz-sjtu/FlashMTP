"""Teacher K 与噪声 K 分别用与各自长度一致的 cos/sin 做 RoPE（全序列位置 0..T-1 + 噪声段 position_ids）。"""

import importlib.util
import unittest
from pathlib import Path

import torch
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

# 避免 import specforge 时拉取 yunchang 等可选依赖：直接加载 draft 模块文件
_ROOT = Path(__file__).resolve().parents[3]
_FLASHMTP_PATH = _ROOT / "specforge/modeling/draft/flashmtp.py"
_spec = importlib.util.spec_from_file_location("flashmtp_draft_only", _FLASHMTP_PATH)
assert _spec and _spec.loader
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
Qwen3FlashMTPAttention = _m.Qwen3FlashMTPAttention


class TestFlashMTPRotary(unittest.TestCase):
    def test_attention_long_context_shorter_noise_matches_rotary_len(self):
        """ctx 与噪声分别用 cos_ctx/sin_ctx 与 cos_noise/sin_noise；无维度冲突。"""
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            max_position_embeddings=16384,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
        )
        config._attn_implementation = "eager"
        attn = Qwen3FlashMTPAttention(config, 0).eval()
        rotary = Qwen3RotaryEmbedding(config).eval()

        b, ctx_len, q_len = 2, 3529, 8192
        noise = torch.randn(b, q_len, config.hidden_size)
        ctx = torch.randn(b, ctx_len, config.hidden_size)
        position_ids = torch.arange(q_len, dtype=torch.long).unsqueeze(0).expand(b, -1)
        ctx_pos = torch.arange(ctx_len, dtype=torch.long).unsqueeze(0).expand(b, -1)
        cos_ctx, sin_ctx = rotary(noise[:, :1, :], ctx_pos)
        cos_noise, sin_noise = rotary(noise, position_ids)
        position_embeddings = (cos_ctx, sin_ctx, cos_noise, sin_noise)

        out, _ = attn(
            hidden_states=noise,
            target_hidden=ctx,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )
        self.assertEqual(out.shape, (b, q_len, config.hidden_size))


if __name__ == "__main__":
    unittest.main()
