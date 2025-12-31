import argparse

def add_lora_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add LoRA arguments to the parser."""
    group = parser.add_argument_group(title="LoRA")
    
    group.add_argument(
        "--use-lora",
        action="store_true",
        help="Whether to use LoRA for training.",
    )
    group.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    group.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha.",
    )
    group.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout.",
    )
    group.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
        help="List of module names to apply LoRA to.",
    )
    
    return parser
