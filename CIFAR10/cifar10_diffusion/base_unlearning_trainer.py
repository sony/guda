"""
Base Unlearning Trainer for Diffusion Models.

This module implements the base class for unlearning methods for diffusion models.
It provides common functionality for forgetting specific classes from unconditional diffusion models
by setting up retention/forgetting data loaders, model sampling, and CLIP-based evaluation.
"""

import os
# Set environment variables to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import warnings
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from transformers import CLIPModel, CLIPProcessor

# Suppress various warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.graph")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

from .trainer import BaseTrainer
from evaluation.data_manifests import load_train_manifest_checked


class BaseUnlearningTrainer(BaseTrainer):
    """
    Base class for unlearning methods on unconditional diffusion models.
    
    This class provides common functionality for unlearning specific classes from 
    unconditional diffusion models, including:
    - Data loader setup for retention/forgetting data splits
    - CLIP-based image classification for evaluation
    - Sample generation and monitoring during unlearning
    
    Subclasses must implement the _calculate_loss method to define the specific
    unlearning loss function.
    """
    
    def __init__(self, args):
        # Validate required unlearning arguments
        if not hasattr(args, 'unlearn_class') or args.unlearn_class is None:
            raise ValueError("unlearn_class must be specified for BaseUnlearningTrainer")
            
        # Force unconditional mode (remove conditional support)
        if hasattr(args, 'conditional') and args.conditional:
            warnings.warn("BaseUnlearningTrainer only supports unconditional models. Setting conditional=False.")
            args.conditional = False
            
        self.unlearn_class = args.unlearn_class
        
        # Batch size for paired retain/forget batches
        self.pair_batch_size = getattr(args, 'pair_batch_size', 128)
        
        super().__init__(args)
        
        # Initialize logging state used across training/validation/sample flows
        self._latest_forget_ratio = None
        self._sample_log_step = None

        # Initialize class names mapping
        self.class_names_dict = self._get_class_names()
        self.class_names = [self.class_names_dict[i] for i in range(self.num_classes)]
        
        # Setup CLIP classifier for evaluation
        self.clip_available = self._setup_clip_classifier()

    # ---------------------------------------------------------------------
    # Logging helpers
    # ---------------------------------------------------------------------

    def _can_log_to_wandb(self) -> bool:
        """Return True if the current process should emit wandb logs."""
        return getattr(self.args, "use_wandb", False) and self.accelerator.is_main_process

    def _log_metrics(self, metrics: dict, *, step: int | None = None, commit: bool | None = None) -> None:
        """Safely log a dictionary of metrics to wandb."""
        if not metrics:
            return
        if not self._can_log_to_wandb():
            return

        import wandb

        log_kwargs = {"step": step} if step is not None else {}
        if commit is not None:
            log_kwargs["commit"] = commit

        wandb.log(metrics, **log_kwargs)

    def _update_forget_ratio(self, ratio: float | None) -> None:
        """Track the most recent forget class ratio for downstream logging."""
        self._latest_forget_ratio = ratio

    def _next_sample_log_step(self, epoch: int) -> int:
        """Return the next monotonic step to use for sample logging."""
        if epoch < 0:
            self._sample_log_step = 0
            return self._sample_log_step

        previous = self._sample_log_step if self._sample_log_step is not None else 0
        proposed = max(self.global_step, previous + 1)
        self._sample_log_step = proposed
        return self._sample_log_step

    def _format_train_metrics(self, *, total_loss, learning_rate, mse_loss=None, vlb_loss=None, extra: dict | None = None) -> dict:
        """Create a standardized dictionary for training loss logs."""
        metrics = {
            "train/total_loss": total_loss.item() if hasattr(total_loss, "item") else total_loss,
            "train/learning_rate": learning_rate,
        }

        if mse_loss is not None:
            metrics["train/mse_loss"] = mse_loss.item() if hasattr(mse_loss, "item") else mse_loss

        if vlb_loss is not None:
            metrics["train/vlb_loss"] = vlb_loss.item() if hasattr(vlb_loss, "item") else vlb_loss

        if extra:
            metrics.update(extra)

        return metrics

    def _format_validation_metrics(self, *, loss, mse_loss=None, vlb_loss=None, forget_ratio=None, extra: dict | None = None) -> dict:
        """Create a standardized dictionary for validation logs."""
        metrics = {
            "val/loss": loss,
        }

        if mse_loss is not None:
            metrics["val/mse_loss"] = mse_loss

        if vlb_loss is not None:
            metrics["val/vlb_loss"] = vlb_loss

        if forget_ratio is not None:
            metrics["val/forget_class_ratio"] = forget_ratio

        if extra:
            metrics.update(extra)

        return metrics
        
    def _get_class_names(self):
        """Get class names for CIFAR-10 or CIFAR-100."""
        if self.num_classes == 10:
            return {
                0: "airplane", 1: "car", 2: "bird", 3: "cat", 4: "deer",
                5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
            }
        elif self.num_classes == 100:
            try:
                from torchvision.datasets import CIFAR100
                dataset = CIFAR100(root="./data", train=True, download=True)
                return {i: name for i, name in enumerate(dataset.classes)}
            except (ImportError, Exception) as e:
                print(f"Warning: Could not load CIFAR100 class names, falling back to numbers. Error: {e}")
                return {i: str(i) for i in range(self.num_classes)}
        else:
            return {i: str(i) for i in range(self.num_classes)}

    def _setup_clip_classifier(self):
        """Setup CLIP model for classification. Only on main process in distributed setting."""
        try:
            # Only initialize CLIP on main process to avoid download conflicts
            if hasattr(self, 'accelerator') and not self.accelerator.is_main_process:
                return False
                
            print("Loading CLIP model for classification...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
            
            # Create text embeddings for CIFAR-10 classes
            prompts = [f'a photo of a {cls}' for cls in self.class_names]
            text_inputs = self.clip_processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                self.text_embeddings = self.clip_model.get_text_features(**text_inputs)
                self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=-1, keepdim=True)
            
            print(f"CLIP prompts: {prompts}")
            return True
        except Exception as e:
            print(f"Error setting up CLIP classifier: {e}")
            return False

    @torch.no_grad()
    def _classify_images_with_clip(self, images):
        """
        Classify images using CLIP model.
        
        Args:
            images: Tensor of shape (B, C, H, W) with values in [-1, 1]
            
        Returns:
            List of predicted class IDs, or None if classification fails
        """
        if not hasattr(self, 'clip_model') or not hasattr(self, 'text_embeddings'):
            print("CLIP classifier not properly initialized")
            return None
            
        try:
            # Convert from [-1, 1] to [0, 1] and then to PIL Images
            images_pil = []
            images_normalized = (images.clamp(-1, 1) + 1) / 2
            for img in images_normalized:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                images_pil.append(img_pil)
            
            # Process images with CLIP
            image_inputs = self.clip_processor(images=images_pil, return_tensors="pt").to(self.device)
            image_features = self.clip_model.get_image_features(**image_inputs)
            image_embeddings = F.normalize(image_features, dim=1)
            
            # Validate shapes before matrix multiplication
            if image_embeddings.shape[1] != self.text_embeddings.shape[1]:
                print(f"Embedding dimension mismatch: image {image_embeddings.shape[1]} vs text {self.text_embeddings.shape[1]}")
                return None
            
            # Compute similarities with text embeddings
            # image_embeddings: (batch_size, embedding_dim)
            # text_embeddings: (num_classes, embedding_dim)
            # Result: (batch_size, num_classes)
            similarities = torch.matmul(image_embeddings, self.text_embeddings.T)
            predicted_classes = similarities.argmax(dim=1).cpu().tolist()
            
            return predicted_classes
            
        except Exception as e:
            print(f"Error in CLIP classification: {e}")
            return None

    def _generate_samples_batch(self, unwrapped_model, ddim_scheduler, batch_size, desc_prefix):
        """
        Generate a batch of samples using DDIM sampling (unconditional only).
        
        Args:
            unwrapped_model: The unwrapped diffusion model
            ddim_scheduler: DDIM scheduler configured for sampling  
            batch_size: Number of samples to generate
            desc_prefix: Description prefix for progress bar
            
        Returns:
            Tensor of generated samples with shape (batch_size, C, H, W)
        """
        # Unconditional sampling only
        samples = torch.randn((batch_size, unwrapped_model.config.in_channels, 32, 32), device=self.device)
        
        for t in tqdm(ddim_scheduler.timesteps, desc=f"{desc_prefix} (Unconditional)", disable=not self.accelerator.is_main_process):
            with torch.no_grad():
                model_output = unwrapped_model(samples, t).sample
                # Handle models trained with learn_sigma
                if model_output.shape[1] == 6:  # learn_sigma=True case
                    residual, _ = torch.chunk(model_output, 2, dim=1)
                else:  # learn_sigma=False case
                    residual = model_output
                samples = ddim_scheduler.step(residual, t, samples).prev_sample
                
        return samples[:batch_size]  # Return exactly the requested number of samples

    def setup_dataloaders(self):
        """Setup dataloaders for retention and forgetting data with equal batch sizes."""
        # Training transform with data augmentation
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
        
        # Validation transform without data augmentation (deterministic)
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
        
        if self.args.dataset == "cifar100":
            DatasetClass = datasets.CIFAR100
            self.num_classes = 100
        else:
            DatasetClass = datasets.CIFAR10
            self.num_classes = 10
            
        # Load full datasets with separate transforms
        train_ds_full = DatasetClass(self.args.data_dir, train=True, download=True, transform=train_transform)
        val_ds_full = DatasetClass(self.args.data_dir, train=False, download=True, transform=val_transform)

        manifest_path = getattr(self.args, "group_manifest", None)
        if manifest_path:
            self.train_group_manifest = load_train_manifest_checked(
                Path(manifest_path),
                dataset_name=self.args.dataset,
                split="train",
                expected_len=len(train_ds_full),
            )
            train_group_ids = np.asarray(self.train_group_manifest.group_ids, dtype=np.int64)
        else:
            self.train_group_manifest = None
            train_group_ids = np.asarray(train_ds_full.targets, dtype=np.int64)

        val_group_ids = np.asarray(val_ds_full.targets, dtype=np.int64)

        # Split into retention (D_neg_c) and forgetting (D_c) datasets
        train_forget_indices = [
            i for i, group_id in enumerate(train_group_ids) if int(group_id) == self.unlearn_class
        ]
        train_retain_indices = [
            i for i, group_id in enumerate(train_group_ids) if int(group_id) != self.unlearn_class
        ]

        val_forget_indices = [
            i for i, label in enumerate(val_group_ids) if int(label) == self.unlearn_class
        ]
        val_retain_indices = [
            i for i, label in enumerate(val_group_ids) if int(label) != self.unlearn_class
        ]
        
        # Create datasets
        self.train_forget_ds = Subset(train_ds_full, train_forget_indices)
        self.train_retain_ds = Subset(train_ds_full, train_retain_indices)
        self.val_forget_ds = Subset(val_ds_full, val_forget_indices)
        self.val_retain_ds = Subset(val_ds_full, val_retain_indices)
        
        # Store dataset sizes for loss calculations
        self.n_total = len(train_ds_full)
        self.k_forget = len(self.train_forget_ds)

        if self.accelerator.is_main_process and self.train_group_manifest is not None:
            unique_groups, counts = np.unique(train_group_ids, return_counts=True)
            summary = ", ".join(
                f"{int(group_id)}:{int(count)}"
                for group_id, count in zip(unique_groups.tolist(), counts.tolist())
            )
            print(f"Loaded train group manifest: {manifest_path}")
            print(f"Train group sizes: {summary}")
            print(
                f"Using manifest-defined forget/retain split for train data: "
                f"forget={len(train_forget_indices)}, retain={len(train_retain_indices)}"
            )
        
        # Create data loaders with equal batch sizes
        self.train_forget_loader = DataLoader(
            self.train_forget_ds, batch_size=self.pair_batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, drop_last=True
        )
        self.train_retain_loader = DataLoader(
            self.train_retain_ds, batch_size=self.pair_batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, drop_last=True
        )
        
        # Validation loaders
        self.val_loader = DataLoader(
            val_ds_full, batch_size=self.args.batch_size, shuffle=False, 
            num_workers=8, pin_memory=True
        )
        
        # Also create a dummy train_loader for BaseTrainer compatibility
        self.train_loader = self.train_retain_loader

    def prepare_with_accelerator(self):
        """Prepare models and optimizers with accelerator, handling unlearning-specific loaders."""
        # Save reference to original optimizer before accelerator wrapping
        self.original_optimizer = self.optimizer
        
        self.model, self.ema_model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.ema_model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler
        )
        
        # Also prepare unlearning-specific loaders
        self.train_retain_loader, self.train_forget_loader = self.accelerator.prepare(
            self.train_retain_loader, self.train_forget_loader
        )

    def setup_model_specifics(self):
        """Setup the unconditional model for unlearning."""
        if self.args.resume:
            model_source_path = self.args.resume
            self.model = UNet2DModel.from_pretrained(model_source_path, subfolder="unet")
        else:
            # Load the teacher model from the specified path
            model_source_path = self.args.teacher_model_path
            print(f"Loading teacher model from: {model_source_path}")
            self.model = UNet2DModel.from_pretrained(model_source_path)

        # Noise scheduler setup – prefer loading the original scheduler config when available
        scheduler_config_path = os.path.join(model_source_path, "scheduler_config.json")
        if os.path.exists(scheduler_config_path):
            self.noise_scheduler = DDPMScheduler.from_pretrained(model_source_path)

            # Keep args in sync with the loaded scheduler to avoid mismatches downstream
            self.args.timesteps = self.noise_scheduler.config.num_train_timesteps
            if hasattr(self.noise_scheduler.config, "beta_schedule") and self.noise_scheduler.config.beta_schedule is not None:
                self.args.beta_schedule = self.noise_scheduler.config.beta_schedule

            if self.accelerator.is_main_process:
                print("Loaded noise scheduler from checkpoint:")
                print(f"  timesteps: {self.noise_scheduler.config.num_train_timesteps}")
                beta_sched = getattr(self.noise_scheduler.config, "beta_schedule", self.args.beta_schedule)
                print(f"  beta_schedule: {beta_sched}")
        else:
            if self.accelerator.is_main_process:
                print("scheduler_config.json not found; falling back to args-provided scheduler settings.")
                print(f"  timesteps: {self.args.timesteps}")
                print(f"  beta_schedule: {self.args.beta_schedule}")

            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.args.timesteps,
                beta_schedule=self.args.beta_schedule,
                prediction_type="epsilon"
            )
        
        # EMA model setup
        self.ema_model = copy.deepcopy(self.model)
        if self.args.resume and os.path.exists(os.path.join(self.args.resume, "ema_model.pt")):
            self.ema_model.load_state_dict(torch.load(os.path.join(self.args.resume, "ema_model.pt"), map_location="cpu"))
        for p in self.ema_model.parameters(): 
            p.requires_grad_(False)

    def setup_common_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        total_steps = self.args.epochs * len(self.train_retain_loader)  # Use retain loader length
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=5000, num_training_steps=total_steps
        )

    def train(self):
        """Main training loop for unlearning."""
        # Perform initial validation and sample generation to establish baseline performance
        if self.accelerator.is_main_process:
            print("=" * 60)
            print("Evaluating initial performance before unlearning...")
            print("=" * 60)
        
        # Initial validation (epoch -1 to indicate pre-training)
        self._validate_epoch(-1)
        
        # Initial sample generation to see what the model produces before unlearning
        self._generate_samples(-1, use_ema=False)
        
        if self.accelerator.is_main_process:
            print("=" * 60)
            print("Starting unlearning training...")
            print("=" * 60)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            self._on_epoch_start(epoch)
            self._train_epoch(epoch)
            
            # Run validation epoch
            self._validate_epoch(epoch)
            
            # Generate samples (multi-GPU safe)
            if (epoch + 1) % self.args.sample_every == 0:
                self._generate_samples(epoch, use_ema=False)
            
            # Save checkpoints (multi-GPU safe)
            if (epoch + 1) % self.args.save_latest_every == 0:
                self._save_latest_checkpoint(epoch)
                
            if (epoch + 1) % self.args.save_every == 0:
                self._save_milestone_checkpoint(epoch)

        # Final checkpoint save
        self._save_latest_checkpoint(self.args.epochs - 1)
        if self.accelerator.is_main_process:
            print("Unlearning training completed!")

        if self.args.use_wandb:
            if self.accelerator.is_main_process:
                import wandb
                wandb.finish()

    def _calculate_loss(self, batch, is_validation=False):
        """
        Calculate the unlearning loss. Must be implemented by subclasses.
        
        Args:
            batch: Can be a single batch or tuple of (retain_batch, forget_batch)
            is_validation: Whether this is validation (not used in unlearning)
            
        Returns:
            Tuple of (total_loss, mse_loss, vlb_loss) for compatibility with BaseTrainer
        """
        raise NotImplementedError("Subclasses must implement _calculate_loss method")

    def _train_epoch(self, epoch):
        """Training epoch for unlearning methods."""
        self.model.train()
        running_loss = 0.0
        
        # Create iterators for both loaders
        retain_iter = iter(self.train_retain_loader)
        forget_iter = iter(self.train_forget_loader)
        
        # Use the smaller loader length
        num_batches = min(len(self.train_retain_loader), len(self.train_forget_loader))
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.args.epochs}", disable=not self.accelerator.is_main_process)
        
        for _ in pbar:
            try:
                retain_batch = next(retain_iter)
                forget_batch = next(forget_iter)
            except StopIteration:
                break
                
            # Calculate loss using both batches
            loss, mse_loss, vlb_loss = self._calculate_loss((retain_batch, forget_batch), is_validation=False)
            
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
            running_loss += loss.item()
            self._update_ema()

            if self.global_step % self.args.log_every == 0:
                metrics = self._format_train_metrics(
                    total_loss=loss,
                    mse_loss=mse_loss,
                    vlb_loss=vlb_loss,
                    learning_rate=self.lr_scheduler.get_last_lr()[0],
                )
                self._log_metrics(metrics, step=self.global_step)
        
        if self.accelerator.is_main_process:
            avg_train_loss = running_loss / num_batches
            print(f"[Epoch {epoch+1:03d}] train loss: {avg_train_loss:.4f}")
            self._log_metrics({"train/epoch_avg_total_loss": avg_train_loss}, step=self.global_step)

    def _validate_epoch(self, epoch):
        """
        Validation epoch for unlearning methods with Multi-GPU support.
        """
        self.model.eval()
        
        # Lists to store gathered loss tensors
        all_losses = []
        all_mse_losses = []
        all_vlb_losses = []
        
        # Handle initial evaluation vs regular epochs
        if epoch == -1:
            desc = "Initial Validation"
            print("Performing initial validation to establish baseline loss...")
        else:
            desc = "Validation"
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=desc, disable=not self.accelerator.is_main_process)
            for batch in pbar:
                # Calculate loss (using retain loss calculation for validation)
                loss, mse_loss, vlb_loss = self._calculate_loss((batch, batch), is_validation=True)
                
                # Gather losses from all processes
                gathered_loss = self.accelerator.gather(loss)
                gathered_mse = self.accelerator.gather(mse_loss)
                gathered_vlb = self.accelerator.gather(vlb_loss)
                
                # Append gathered tensors (ensure they are at least 1D)
                all_losses.append(gathered_loss.flatten())
                all_mse_losses.append(gathered_mse.flatten())
                all_vlb_losses.append(gathered_vlb.flatten())

        # Calculate averages across all batches and processes
        avg_val_loss = torch.cat(all_losses).mean().item()
        avg_val_mse = torch.cat(all_mse_losses).mean().item()
        avg_val_vlb = torch.cat(all_vlb_losses).mean().item()
        
        # Only main process logs
        if self.accelerator.is_main_process:
            if epoch == -1:
                print(f"[Initial] val_loss: {avg_val_loss:.4f}, val_mse: {avg_val_mse:.4f}")
            else:
                print(f"[Epoch {epoch+1:03d}] val_loss: {avg_val_loss:.4f}, val_mse: {avg_val_mse:.4f}")
            
            step_value = self.global_step if epoch >= 0 else 0
            val_logs = self._format_validation_metrics(
                loss=avg_val_loss,
                mse_loss=avg_val_mse,
                vlb_loss=avg_val_vlb if getattr(self.args, "learn_sigma", False) else None,
            )
            self._log_metrics(val_logs, step=step_value)
        
        # Ensure all processes are synchronized after validation
        self.accelerator.wait_for_everyone()
        
        # Set model back to training mode
        self.model.train()

    def _save_milestone_checkpoint(self, epoch: int):
        """Saves a permanent checkpoint to a directory named 'epoch_xxxx'."""
        self.accelerator.wait_for_everyone()  # Ensure all processes are synchronized before saving

        if self.accelerator.is_main_process:
            import os
            import shutil
            
            ckpt_dir = os.path.join(self.args.output_dir, f"epoch_{epoch+1:04d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(ckpt_dir)
            
            if hasattr(self, 'noise_scheduler'):
                self.noise_scheduler.save_pretrained(ckpt_dir)
            
            torch.save({
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": epoch + 1,
                "global_step": self.global_step
            }, os.path.join(ckpt_dir, "training_state.pt"))
            
            torch.save(
                self.accelerator.unwrap_model(self.ema_model).state_dict(),
                os.path.join(ckpt_dir, "ema_model.pt")
            )
            print(f"Saved milestone checkpoint to {ckpt_dir}")
        
        self.accelerator.wait_for_everyone()  # Ensure all processes wait until the main process finishes saving

    def _save_latest_checkpoint(self, epoch: int):
        """Saves a temporary checkpoint to 'latest' for resuming, overwriting previous ones."""
        self.accelerator.wait_for_everyone()  # Ensure all processes are synchronized before saving
        
        if self.accelerator.is_main_process:
            import os
            import shutil
            
            latest_ckpt_dir = os.path.join(self.args.output_dir, "latest")
            tmp_ckpt_dir = os.path.join(self.args.output_dir, "tmp_checkpoint")

            if os.path.exists(tmp_ckpt_dir):
                shutil.rmtree(tmp_ckpt_dir)
            os.makedirs(tmp_ckpt_dir, exist_ok=True)

            # Save all components to the temporary directory
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(tmp_ckpt_dir)
            
            if hasattr(self, 'noise_scheduler'):
                self.noise_scheduler.save_pretrained(tmp_ckpt_dir)
            
            torch.save({
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": epoch + 1,
                "global_step": self.global_step
            }, os.path.join(tmp_ckpt_dir, "training_state.pt"))
            
            torch.save(
                self.accelerator.unwrap_model(self.ema_model).state_dict(),
                os.path.join(tmp_ckpt_dir, "ema_model.pt")
            )

            # Atomically replace the old 'latest' with the new one
            if os.path.exists(latest_ckpt_dir):
                shutil.rmtree(latest_ckpt_dir)
            os.rename(tmp_ckpt_dir, latest_ckpt_dir)
            
            print(f"Updated 'latest' resume checkpoint at epoch {epoch+1}")
        
        self.accelerator.wait_for_everyone()  # Ensure all processes wait until the main process finishes saving

    def _generate_samples(self, epoch, use_ema=False):
        """Generate samples using DDIM sampling (100 steps) for unlearning evaluation."""
        # Wait for all processes to sync before starting the long sampling process.
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            model_to_use = self.ema_model if use_ema else self.model
            unwrapped_model = self.accelerator.unwrap_model(model_to_use)
            unwrapped_model.eval()

            # Setup DDIM scheduler for sampling (100 steps)
            ddim_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
            ddim_scheduler.set_timesteps(num_inference_steps=100)

            # Generate samples
            num_samples = 1024  # Set to 1024 as requested
            
            # Handle initial evaluation (epoch -1) vs regular epochs
            if epoch == -1:
                desc_prefix = "Initial Eval"
                epoch_label = "initial"
                print("Generating initial samples to establish baseline performance...")
            else:
                desc_prefix = f"Epoch {epoch+1}"
                epoch_label = f"epoch_{epoch+1:04d}"
            
            samples = self._generate_samples_batch(unwrapped_model, ddim_scheduler, num_samples, desc_prefix)

            # Save sample grid (show first 64 for visualization)
            samples_denorm = (samples + 1) / 2
            grid = make_grid(samples_denorm[:64], nrow=8, normalize=False, scale_each=False)  # Show first 64 samples
            sample_path = os.path.join(self.samples_dir, f"{epoch_label}_samples.png")
            transforms.ToPILImage()(grid).save(sample_path)
            print(f"Saved samples to {sample_path} (showing first 64 of {num_samples})")

            wandb_payload = {}

            # Determine wandb step for sample logging while keeping it monotonic
            step_value = self._next_sample_log_step(epoch)

            # Log sample grid to wandb if available
            if self._can_log_to_wandb():
                import wandb

                grid_array = grid.permute(1, 2, 0).cpu().numpy()
                caption = "Initial baseline samples" if epoch == -1 else f"Generated samples at epoch {epoch+1}"
                wandb_payload["samples/image_grid"] = wandb.Image(grid_array, caption=caption)

            # Classify samples with CLIP if available
            if self.clip_available:
                predictions = self._classify_images_with_clip(samples)
                if predictions is not None:
                    # Count predictions for each class
                    class_counts = {i: predictions.count(i) for i in range(self.num_classes)}
                    forget_class_ratio = class_counts.get(self.unlearn_class, 0) / len(predictions)
                    
                    status_prefix = "Initial baseline" if epoch == -1 else f"Epoch {epoch+1}"
                    print(f"{status_prefix} - Class distribution in generated samples:")
                    for class_id, count in class_counts.items():
                        print(f"  Class {class_id}: {count} ({count/len(predictions)*100:.1f}%)")
                    print(f"Forget class ({self.unlearn_class}) ratio: {forget_class_ratio:.3f}")
                    
                    self._update_forget_ratio(forget_class_ratio)
                    if self._can_log_to_wandb():
                        import wandb

                        wandb_payload["samples/forget_class_ratio"] = forget_class_ratio

            if wandb_payload:
                self._log_metrics(wandb_payload, step=step_value)

            unwrapped_model.train()

        # All processes wait here until the main process has finished sampling.
        self.accelerator.wait_for_everyone()
