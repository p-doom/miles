import math
from dataclasses import dataclass, field
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA."""
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # "none", "all", or "lora_only" - currently only "none" supported for simplicity

    def to_hf_peft_config(self) -> Dict[str, Any]:
        """Convert to Hugging Face PEFT config format."""
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "inference_mode": False,
            "r": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
        }


class LoRALinear(nn.Module):
    """
    LoRA linear layer that wraps a base linear layer.
    
    Args:
        base_layer: The existing Linear layer to wrap.
        rank: LoRA rank (r).
        alpha: LoRA alpha (scaling factor).
        dropout: Dropout probability for LoRA input.
    """
    def __init__(
        self, 
        base_layer: nn.Linear, 
        rank: int = 8, 
        alpha: int = 16, 
        dropout: float = 0.0
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        
        lora_out = self.dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        
        return result + lora_out * self.scaling

    def __repr__(self):
        return (
            f"LoRALinear(in_features={self.base_layer.in_features}, "
            f"out_features={self.base_layer.out_features}, "
            f"rank={self.rank}, alpha={self.alpha})"
        )


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Apply LoRA to the model by replacing target linear layers with LoRALinear.
    
    Args:
        model: The model to modify.
        config: LoRA configuration.
        
    Returns:
        The modified model.
    """
    assert config.bias == "none", "Only bias='none' is currently supported"
    target_modules = set(config.target_modules)
    
    # We need to collect replacements first to avoid modifying the dict while iterating
    modules_to_replace = []
    
    for name, module in model.named_modules():
        # Check if this module name ends with any of the target modules
        # e.g. "model.layers.0.self_attn.q_proj" ends with "q_proj"
        if any(name.endswith(target) for target in target_modules):
            if isinstance(module, nn.Linear):
                modules_to_replace.append((name, module))
                
    if not modules_to_replace:
        raise ValueError(f"No modules found matching {target_modules}")
        
    for name, module in modules_to_replace:
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ""
            child_name = name
            parent = model
            
        lora_layer = LoRALinear(
            base_layer=module,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout
        )
        
        setattr(parent, child_name, lora_layer)
        
    # Freeze all non-LoRA parameters
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
            
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return state dict with only LoRA parameters."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = False):
    """Load LoRA parameters into the model."""
    # We only load keys that exist in the state_dict and match LoRA params
    model.load_state_dict(state_dict, strict=strict)
