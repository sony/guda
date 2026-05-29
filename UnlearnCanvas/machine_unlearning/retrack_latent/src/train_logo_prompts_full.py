"""LOGO full fine-tuning entry point for Stable Diffusion."""

from train_prompts_full_base import main as base_main


if __name__ == "__main__":
    base_main(require_exclude_style=True)
