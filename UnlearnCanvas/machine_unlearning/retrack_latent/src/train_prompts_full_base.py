"""Full fine-tuning utilities for UnlearnCanvas Stable Diffusion training.

This module provides a shared entry point for all-class and LOGO full fine-tuning
runs that follow the original UnlearnCanvas training recipe (diffusers pipeline
with full UNet updates, EMA tracking, constant learning rate, and 256px data
preprocessing).
"""

import argparse
import glob
import os
import re
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from transformers import CLIPTextModel, CLIPTokenizer

from dataset_prompts import PromptDataset
from utils import AverageMeter, format_time


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser shared by all-class and LOGO training."""
    parser = argparse.ArgumentParser(
        description="Full fine-tuning of Stable Diffusion on UnlearnCanvas prompts"
    )

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing UnlearnCanvas images")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Path to train_prompts.jsonl")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store checkpoints")
    parser.add_argument("--exclude_style", type=str, default=None,
                        help="Style to exclude for LOGO training")

    # Model
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Pretrained Stable Diffusion model ID")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Optional cache directory for pretrained weights")

    # Optimisation
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5,
                        help="Gradient accumulation steps to reach effective batch size 80")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker count")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Constant learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="AdamW weight decay (matches official recipe)")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--num_steps", type=int, default=10000,
                        help="Number of optimisation steps (official recipe uses 20000)")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Checkpoint interval in steps")
    parser.add_argument("--keep_milestone", type=int, default=2500,
                        help="Always keep checkpoints at multiples of this step")

    # Data preprocessing
    parser.add_argument("--resolution", type=int, default=256,
                        help="Training image resolution (default 256)")
    parser.add_argument("--no_center_crop", action="store_false", dest="center_crop",
                        help="Disable center crop (use random crop instead)")
    parser.add_argument("--random_flip_prob", type=float, default=0.5,
                        help="Probability of horizontal flip (0 disables flips)")

    # Misc
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Accelerate mixed precision mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output_dir")
    parser.add_argument("--allow_tf32", action="store_true",
                        help="Enable TF32 matmul on Ampere/ADA GPUs")
    parser.add_argument("--no_ema", action="store_false", dest="use_ema",
                        help="Disable EMA tracking")
    parser.set_defaults(center_crop=True, use_ema=True)

    return parser


def build_transform(resolution: int, center_crop: bool, random_flip_prob: float) -> transforms.Compose:
    """Construct the training transform pipeline."""
    ops = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
    ]
    if random_flip_prob > 0:
        ops.append(transforms.RandomHorizontalFlip(p=random_flip_prob))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(ops)


def find_latest_checkpoint(output_dir: str) -> tuple[Optional[str], int]:
    pattern = os.path.join(output_dir, "checkpoint_step_*.pt")
    checkpoints = glob.glob(pattern)
    latest_step = 0
    latest_path: Optional[str] = None

    for ckpt in checkpoints:
        match = re.search(r"checkpoint_step_(\d+)\.pt", ckpt)
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_path = ckpt

    return latest_path, latest_step


def cleanup_checkpoints(output_dir: str, current_step: int, keep_milestone: int) -> None:
    pattern = os.path.join(output_dir, "checkpoint_step_*.pt")
    for ckpt in glob.glob(pattern):
        match = re.search(r"checkpoint_step_(\d+)\.pt", ckpt)
        if not match:
            continue
        step = int(match.group(1))
        if step == current_step:
            continue
        if keep_milestone > 0 and step % keep_milestone == 0:
            continue
        # Remove stale checkpoint
        os.remove(ckpt)


def save_checkpoint(
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ema_unet: Optional[EMAModel],
    output_dir: str,
    step: int,
) -> None:
    if not accelerator.is_main_process:
        return

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")

    unwrapped_unet = accelerator.unwrap_model(unet)
    checkpoint = {
        "step": step,
        "unet_state_dict": unwrapped_unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
    }

    if ema_unet is not None:
        checkpoint["ema_state_dict"] = ema_unet.state_dict()

    torch.save(checkpoint, checkpoint_path)
    accelerator.print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ema_unet: Optional[EMAModel],
    checkpoint_path: str,
) -> int:
    accelerator.print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.load_state_dict(checkpoint["unet_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    if ema_unet is not None and "ema_state_dict" in checkpoint:
        ema_unet.load_state_dict(checkpoint["ema_state_dict"])

    step = int(checkpoint.get("step", 0))
    accelerator.print(f"Resumed from step {step}")
    return step


def main(argv: Optional[Sequence[str]] = None, *, require_exclude_style: bool = False) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if require_exclude_style and not args.exclude_style:
        parser.error("--exclude_style is required for LOGO full fine-tuning")

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None,
    )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    accelerate_set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        mode = f"LOGO (exclude: {args.exclude_style})" if args.exclude_style else "All-class"
        print("=" * 80)
        print(f"Full SD fine-tuning | Mode: {mode}")
        print("=" * 80)
        print(f"Data root: {args.data_root}")
        print(f"Prompt file: {args.prompt_file}")
        print(f"Output dir: {args.output_dir}")
        print(f"Resolution: {args.resolution}")
        print(f"Batch size: {args.batch_size} | Grad accum: {args.gradient_accumulation_steps}")
        print(f"Learning rate: {args.lr} | Weight decay: {args.weight_decay}")
        print(f"EMA: {'enabled' if args.use_ema else 'disabled'}")
        print("=" * 80)

    # Load pretrained components
    accelerator.print("Loading pretrained components...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer", cache_dir=args.cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder", cache_dir=args.cache_dir)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", cache_dir=args.cache_dir)
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", cache_dir=args.cache_dir)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler", cache_dir=args.cache_dir)

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    ema_unet: Optional[EMAModel] = None
    if args.use_ema:
        ema_unet_model = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", cache_dir=args.cache_dir)
        ema_unet = EMAModel(ema_unet_model.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet_model.config)
        del ema_unet_model

    # Dataset & loader
    transform = build_transform(args.resolution, args.center_crop, args.random_flip_prob)
    dataset = PromptDataset(
        data_root=args.data_root,
        prompt_file=args.prompt_file,
        styles=None,
        exclude_styles=[args.exclude_style] if args.exclude_style else None,
        transform=transform,
    )

    if len(dataset) == 0:
        raise ValueError("No training samples found with the provided configuration")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if accelerator.is_main_process:
        print(f"Loaded {len(dataset)} samples")

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    if ema_unet is not None:
        ema_unet.to(device=accelerator.device, dtype=weight_dtype)

    global_step = 0
    if args.resume:
        latest_ckpt, latest_step = find_latest_checkpoint(args.output_dir)
        if latest_ckpt is not None:
            global_step = load_checkpoint(accelerator, unet, optimizer, lr_scheduler, ema_unet, latest_ckpt)
            if ema_unet is not None:
                ema_unet.to(device=accelerator.device, dtype=weight_dtype)
            if global_step >= args.num_steps:
                accelerator.print("Checkpoint step exceeds or equals num_steps; exiting.")
                return
        else:
            accelerator.print("No checkpoint found; starting from scratch")

    progress_bar = tqdm_range(global_step, args.num_steps, accelerator)
    loss_meter = AverageMeter()
    start_time = time.time()

    unet.train()

    while global_step < args.num_steps:
        for batch in dataloader:
            if global_step >= args.num_steps:
                break

            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                captions = batch["caption"]

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    text_inputs = tokenizer(
                        captions,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = text_inputs.input_ids.to(accelerator.device)
                    text_embeddings = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                          (bsz,), device=latents.device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = unet(noisy_latents, timesteps, text_embeddings).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

                if ema_unet is not None and accelerator.sync_gradients:
                    ema_unet.step(unet.parameters())

            if accelerator.sync_gradients:
                global_step += 1
                loss_meter.update(loss.item())
                progress_bar.update(1)

                if global_step % args.save_interval == 0 or global_step == args.num_steps:
                    if ema_unet is not None:
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())

                    save_checkpoint(accelerator, unet, optimizer, lr_scheduler, ema_unet, args.output_dir, global_step)

                    if ema_unet is not None:
                        ema_unet.restore(unet.parameters())

                    cleanup_checkpoints(args.output_dir, global_step, args.keep_milestone)

                if global_step % args.save_interval == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = max(1e-8, global_step / elapsed)
                    eta = (args.num_steps - global_step) / steps_per_sec
                    if accelerator.is_main_process:
                        print(
                            f"\nStep {global_step}/{args.num_steps} | "
                            f"Loss: {loss_meter.avg:.4f} | "
                            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                            f"Speed: {steps_per_sec:.2f} steps/s | "
                            f"ETA: {format_time(eta)}"
                        )
                    loss_meter.reset()

    progress_bar.close()

    if accelerator.is_main_process:
        total_time = time.time() - start_time
        print("\nTraining complete!")
        print(f"Total time: {format_time(total_time)}")

    accelerator.wait_for_everyone()


def tqdm_range(start_step: int, total_steps: int, accelerator: Accelerator):
    """Utility to create a tqdm progress bar with Accelerate awareness."""
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - fallback without tqdm
        class DummyBar:
            def update(self, *_args, **_kwargs):
                pass

            def close(self):
                pass

        return DummyBar()

    return tqdm(
        total=total_steps - start_step,
        initial=0,
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )


if __name__ == "__main__":
    main()
