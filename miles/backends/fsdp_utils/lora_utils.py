"""LoRA utilities for FSDP backend using PEFT library.

This module provides functions for applying, detecting, and saving LoRA adapters
in a way that's compatible with the PEFT library and HuggingFace ecosystem.
"""

import logging
import os
import shutil
from pathlib import Path

import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError as err:
    raise ImportError("peft library required for LoRA. Install with: pip install peft") from err

logger = logging.getLogger(__name__)

LORA_READY_MARKER = ".lora_ready"
LORA_ADAPTER_NAME = "miles_lora"
LORA_SUBDIR = "tmp_lora"


def apply_lora_to_model(model: nn.Module, args) -> nn.Module:
    """Apply LoRA to model using PEFT library.

    Args:
        model: The base model to apply LoRA to.
        args: Arguments containing LoRA configuration (lora_rank, lora_alpha,
              target_modules, lora_adapter_path).

    Returns:
        Model wrapped with LoRA adapters.
    """
    if args.lora_adapter_path:
        logger.info(f"Loading LoRA adapter from {args.lora_adapter_path}")
        model = PeftModel.from_pretrained(model, args.lora_adapter_path, is_trainable=True)
        peft_config = model.peft_config["default"]
        if isinstance(peft_config.task_type, str):
            peft_config.task_type = TaskType.CAUSAL_LM
        model.print_trainable_parameters()
        return model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info(f"Applied LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    return model


def is_lora_model(module: nn.Module) -> bool:
    """Check if a module is a PEFT LoRA model.

    Args:
        module: The module to check.

    Returns:
        True if the module has PEFT LoRA applied.
    """
    unwrapped = getattr(module, "_fsdp_wrapped_module", module)
    return hasattr(unwrapped, "peft_config")


def save_lora_to_disk(module: nn.Module, save_dir: str) -> str:
    """Save LoRA adapter to disk in HuggingFace PEFT format.

    Args:
        module: The PEFT model to save.
        save_dir: Directory to save the adapter to.

    Returns:
        The save directory path.
    """
    # Gather full state dict (all-gather LoRA weights)
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_state_dict = get_model_state_dict(module, options=options)

    lora_state_dict = {name: param for name, param in full_state_dict.items() if "lora_" in name}

    if dist.get_rank() == 0:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        module.save_pretrained(str(save_path), state_dict=lora_state_dict)

        # Sync filesystem
        os.sync()

        logger.info(f"Saved LoRA adapter to {save_path}")
    return save_dir


def delete_lora_from_disk(save_dir: str) -> None:
    """Delete LoRA adapter files from disk.

    Args:
        save_dir: Directory containing the adapter to delete.
    """
    save_path = Path(save_dir)
    if save_path.exists():
        shutil.rmtree(save_path)
        logger.info(f"Deleted LoRA adapter from {save_path}")
