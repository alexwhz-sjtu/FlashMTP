import copy
import tempfile
import unittest
import torch
from transformers import Qwen3Config
from specforge.modeling.draft.flashmtp import FlashMTPDraftModel, FlashMTPGroupedConv, FLASHMTP_ARCHITECTURE_VERSION

class SimpleConvTest(unittest.TestCase):
    def config(self, mode='simple_conv', layers=3):
        c=Qwen3Config(vocab_size=29, hidden_size=16, intermediate_size=32,num_hidden_layers=layers,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        c._attn_implementation='eager'; c.num_target_layers=4; c.block_size=4
        c.flashmtp_config=dict(architecture_version=FLASHMTP_ARCHITECTURE_VERSION,sliding_window_size=4,chs_num_layers=2,target_layer_ids=[0,3],backbone_conv_mode=mode,conv_group_size=4)
        return c
    def test_identity_gradients_and_roundtrip(self):
        model=FlashMTPDraftModel(self.config())
        base=FlashMTPDraftModel(self.config('none'))
        base.load_state_dict({k:v for k,v in model.state_dict().items() if '.output_conv.' not in k})
        q=model.draft_query_length; c=model.condition_slot_count
        args=dict(position_ids=torch.arange(q)[None],rotary_position_ids=torch.arange(c+q)[None],noise_embedding=torch.randn(1,q,16),target_hidden=torch.randn(1,1,c,16),attention_mask=torch.zeros(1,1,q,c+q))
        y=model(**args); torch.testing.assert_close(y,base(**args),rtol=0,atol=0)
        y.mul(torch.randn_like(y)).sum().backward()
        self.assertIsNone(model.layers[-1].output_conv)
        for l in model.layers[:-1]:
            self.assertIsNone(l.attention_conv); self.assertIsNone(l.mlp_conv)
            self.assertEqual(l.output_conv.kernel_projection.out_features,8)
            self.assertGreater(l.output_conv.kernel_projection.weight.grad.abs().sum().item(),0)
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d); restored=FlashMTPDraftModel.from_pretrained(d)
            self.assertEqual(restored.backbone_conv_mode,'simple_conv')
            torch.testing.assert_close(restored(**args),y)
        model.set_config_block_size(6)
        for l in model.layers[:-1]: self.assertEqual(l.output_conv.block_size,model.draft_query_length)
    def test_boundary_and_one_layer(self):
        conv=FlashMTPGroupedConv(4,3,2,2,sides=1)
        x=torch.arange(24.).reshape(1,6,4)
        torch.testing.assert_close(conv(x),x)
        with torch.no_grad(): conv.base_kernel.zero_(); conv.base_kernel[:,1]=1
        expected=torch.zeros_like(x); expected[:,1:3]=x[:,:2]; expected[:,4:6]=x[:,3:5]
        torch.testing.assert_close(conv(x),expected)
        self.assertIsNone(FlashMTPDraftModel(self.config(layers=1)).layers[0].output_conv)
    def test_invalid_mode(self):
        with self.assertRaisesRegex(ValueError,'backbone_conv_mode'): FlashMTPDraftModel(self.config('invalid'))

if __name__=='__main__': unittest.main()
