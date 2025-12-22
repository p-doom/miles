import unittest
import torch
import torch.nn as nn
from miles.models.peft import LoRAConfig, LoRALinear, apply_lora, get_lora_state_dict, load_lora_state_dict

class TestLoRA(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.output_dim = 20
        self.base_layer = nn.Linear(self.input_dim, self.output_dim)
        self.config = LoRAConfig(lora_rank=4, lora_alpha=8, lora_dropout=0.0)

    def test_lora_linear_forward(self):
        lora_layer = LoRALinear(self.base_layer, rank=4, alpha=8, dropout=0.0)
        x = torch.randn(5, self.input_dim)
        
        # Initial forward should match base layer (since B is zero)
        out_lora = lora_layer(x)
        out_base = self.base_layer(x)
        torch.testing.assert_close(out_lora, out_base)
        
        # Modify LoRA weights
        lora_layer.lora_B.data.fill_(1.0)
        out_lora_mod = lora_layer(x)
        self.assertFalse(torch.allclose(out_lora_mod, out_base))

    def test_apply_lora(self):
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10)
        )
        # Name modules to match default target "q_proj", "v_proj" won't work here.
        # Let's use custom config
        config = LoRAConfig(target_modules=["0"], lora_rank=4)
        
        model = apply_lora(model, config)
        
        self.assertIsInstance(model[0], LoRALinear)
        self.assertIsInstance(model[1], nn.Linear)
        
        # Check gradients
        self.assertTrue(model[0].lora_A.requires_grad)
        self.assertFalse(model[0].base_layer.weight.requires_grad)
        self.assertFalse(model[1].weight.requires_grad) # Should be frozen by apply_lora

    def test_state_dict(self):
        model = nn.Sequential(
            nn.Linear(10, 10)
        )
        config = LoRAConfig(target_modules=["0"], lora_rank=4)
        model = apply_lora(model, config)
        
        state_dict = get_lora_state_dict(model)
        self.assertEqual(len(state_dict), 2) # A and B
        self.assertTrue(all("lora_" in k for k in state_dict.keys()))
        
        # Test loading
        new_state = {k: torch.ones_like(v) for k, v in state_dict.items()}
        load_lora_state_dict(model, new_state)
        
        self.assertTrue(torch.allclose(model[0].lora_A, torch.ones_like(model[0].lora_A)))

if __name__ == "__main__":
    unittest.main()
