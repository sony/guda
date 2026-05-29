import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from torchvision.utils import make_grid
import math
import wandb
import shutil
import random
from scipy.stats import spearmanr

from cifar10_diffusion.diffusion_utils import DiffusionHelpers, normal_kl
from evaluation.data_manifests import load_train_manifest_checked

class BaseTrainer:
    """
    Base class for training diffusion models.
    It handles the common logic for setup, training loops, validation, and saving.
    Subclasses must implement methods for model-specific setup and loss calculation.
    """
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.global_step = 0

        self.accelerator = Accelerator(mixed_precision="fp16")
        self.device = self.accelerator.device

        self.setup_directories()
        
        # These methods should be implemented by subclasses
        self.setup_dataloaders()
        self.setup_model_specifics()
        
        # These methods are common for all trainers
        self.setup_common_optimizer_and_scheduler()
        self.resume_from_checkpoint_if_needed()

        self.diffusion_helpers = DiffusionHelpers(self.noise_scheduler)
        self.prepare_with_accelerator()

        if self.accelerator.is_main_process:
            self.setup_logging()

    def setup_directories(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.samples_dir = os.path.join(self.args.output_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)

    def setup_dataloaders(self):
        raise NotImplementedError("Subclasses must implement setup_dataloaders()")

    def setup_model_specifics(self):
        raise NotImplementedError("Subclasses must implement setup_model_specifics()")

    def _calculate_loss(self, batch, t):
        raise NotImplementedError("Subclasses must implement _calculate_loss()")

    def _generate_samples(self, epoch, use_ema=False):
        raise NotImplementedError("Subclasses must implement _generate_samples()")

    def _on_epoch_start(self, epoch):
        """Hook for subclasses to perform actions at the start of each epoch."""
        pass
        
    def setup_common_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        total_steps = self.args.epochs * len(self.train_loader)
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=5000, num_training_steps=total_steps)

    def resume_from_checkpoint_if_needed(self):
        if self.args.resume:
            ckpt = torch.load(os.path.join(self.args.resume, "training_state.pt"), map_location="cpu")
            self.start_epoch = ckpt["epoch"]
            self.global_step = ckpt.get("global_step", 0)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    def prepare_with_accelerator(self):
        self.model, self.ema_model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.ema_model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler
        )

    def setup_logging(self):
        if self.args.use_wandb:
            import wandb
            wandb_kwargs = {
                "project": self.args.wandb_project,
                "config": vars(self.args)
            }
            
            # Add group if specified
            if hasattr(self.args, 'wandb_group') and self.args.wandb_group:
                wandb_kwargs["group"] = self.args.wandb_group
            
            # Add tags if specified
            if hasattr(self.args, 'wandb_tags') and self.args.wandb_tags:
                wandb_kwargs["tags"] = self.args.wandb_tags
                
            if hasattr(self.args, 'wandb_run_name') and self.args.wandb_run_name:
                # Initialize wandb first to get the random name
                wandb.init(**wandb_kwargs)
                # Then update the name to include our identifier
                random_name = wandb.run.name
                custom_name = f"{random_name}-{self.args.wandb_run_name}"
                wandb.run.name = custom_name
            else:
                wandb.init(**wandb_kwargs)
            # wandb.watch(self.accelerator.unwrap_model(self.model), log="all", log_freq=self.args.log_every)

    def _update_ema(self):
        if self.global_step > self.args.ema_start:
            with torch.no_grad():
                for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                    ema_p.data.mul_(self.args.ema_decay).add_(p.data, alpha=1 - self.args.ema_decay)

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}", disable=not self.accelerator.is_main_process)
        
        for batch in pbar:
            loss, mse_loss, vlb_loss = self._calculate_loss(batch, is_validation=False)
            
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
            running_loss += loss.item()
            self._update_ema()
            
            if self.accelerator.is_main_process and (self.global_step % self.args.log_every == 0):
                lr = self.lr_scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=loss.item(), lr=f"{lr:.1e}")
                if self.args.use_wandb:
                    logs = {"train/loss": loss.item(), "train/lr": lr}
                    if self.args.learn_sigma:
                        logs.update({"train/mse_loss": mse_loss.item(), "train/vlb_loss": vlb_loss.item()})
                    wandb.log(logs, step=self.global_step)
        
        if self.accelerator.is_main_process:
            avg_train_loss = running_loss / len(self.train_loader)
            print(f"[Epoch {epoch+1:03d}] train loss: {avg_train_loss:.4f}")
            if self.args.use_wandb:
                wandb.log({"train/avg_loss": avg_train_loss}, step=self.global_step)

    def _validate_epoch(self, epoch):
        # 1. Set model to evaluation mode
        self.model.eval()
        
        # Lists to store the gathered loss tensors from each batch
        all_losses = []
        all_mse_losses = []
        all_vlb_losses = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", disable=not self.accelerator.is_main_process)
            for batch in pbar:
                # Calculate loss on the local batch for each process.
                # Note: The loss returned here is already a mean over the batch on this process.
                loss, mse_loss, vlb_loss = self._calculate_loss(batch, is_validation=True)
                
                # 2. Gather the scalar loss values from all processes.
                # The result will be a tensor of shape (num_processes,) on each process.
                gathered_loss = self.accelerator.gather(loss)
                gathered_mse = self.accelerator.gather(mse_loss)
                gathered_vlb = self.accelerator.gather(vlb_loss)
                
                # Append the gathered tensors to our lists
                # Ensure tensors are at least 1D for concatenation
                all_losses.append(gathered_loss.flatten())
                all_mse_losses.append(gathered_mse.flatten())
                all_vlb_losses.append(gathered_vlb.flatten())

        # 3. After the loop, calculate the true average across all batches and all processes.
        # Concatenate all gathered batch losses and compute the final mean.
        avg_val_loss = torch.cat(all_losses).mean().item()
        avg_val_mse = torch.cat(all_mse_losses).mean().item()
        avg_val_vlb = torch.cat(all_vlb_losses).mean().item()
        
        # 4. Only the main process should print and log the final, correct metrics.
        if self.accelerator.is_main_process:
            print(f"[Epoch {epoch+1:03d}] val_loss: {avg_val_loss:.4f}, val_mse: {avg_val_mse:.4f}")
            
            if self.args.use_wandb:
                val_logs = {"val/loss": avg_val_loss, "val/mse_loss": avg_val_mse}
                if self.args.learn_sigma:
                    val_logs["val/vlb_loss"] = avg_val_vlb
                wandb.log(val_logs, step=self.global_step)
        
        # 5. Set model back to training mode
        self.model.train()

    def _save_milestone_checkpoint(self, epoch: int):
        """Saves a permanent checkpoint to a directory named 'epoch_xxxx'."""

        self.accelerator.wait_for_everyone()  # Ensure all processes are synchronized before saving

        if self.accelerator.is_main_process:

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

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self._on_epoch_start(epoch)
            self._train_epoch(epoch)
            
            self._validate_epoch(epoch)
            
            if (epoch + 1) % self.args.sample_every == 0:
                self._generate_samples(epoch, use_ema=False)
                # self._generate_samples(epoch, use_ema=True)
                
            if (epoch + 1) % self.args.save_latest_every == 0:
                self._save_latest_checkpoint(epoch)
                
            # Save the permanent, milestone checkpoint at lower frequency
            if (epoch + 1) % self.args.save_every == 0:
                self._save_milestone_checkpoint(epoch)
            
            self.accelerator.wait_for_everyone()

        self._save_latest_checkpoint(self.args.epochs - 1)
        if self.accelerator.is_main_process:
            print("Training complete!")

        if self.args.use_wandb:
            if self.accelerator.is_main_process:
                wandb.finish()

class UnconditionalTrainer(BaseTrainer):

    def _load_train_group_ids(self) -> np.ndarray:
        """Load train-time group ids from manifest when provided, else use raw labels."""
        manifest_path = getattr(self.args, "group_manifest", None)
        if not manifest_path:
            self.train_group_manifest = None
            return np.asarray(self.train_ds_full.targets, dtype=np.int64)

        manifest = load_train_manifest_checked(
            Path(manifest_path),
            dataset_name=self.args.dataset,
            split="train",
            expected_len=len(self.train_ds_full),
        )
        self.train_group_manifest = manifest
        return np.asarray(manifest.group_ids, dtype=np.int64)

    def _rebuild_dataloaders(self):
        """Helper function to build/rebuild dataloaders with current exclusion rules."""
        classes_to_exclude = []
        # Check if any exclusion rule is active
        if self.args.exclude_classes:
            # Check for the 'random' keyword
            if self.args.exclude_classes[0].lower() == 'random':
                # Randomly select one class to exclude from the total available classes
                classes_to_exclude = [random.choice(range(self.num_classes))]
                if self.accelerator.is_main_process:
                    print(f"This epoch, randomly excluding class: {classes_to_exclude}")
            else:
                # Convert list of string numbers to integers
                try:
                    classes_to_exclude = [int(c) for c in self.args.exclude_classes]
                except ValueError:
                    raise ValueError("exclude_classes must be a list of integers or the keyword 'random'.")

        # Create subsets based on the classes to exclude
        if classes_to_exclude:
            excluded_set = set(classes_to_exclude)
            train_indices = [
                i
                for i, group_id in enumerate(self.train_group_ids)
                if int(group_id) not in excluded_set
            ]
            train_ds = Subset(self.train_ds_full, train_indices)
            if self.train_group_manifest is None:
                val_indices = [
                    i
                    for i, label in enumerate(self.val_ds_full.targets)
                    if label not in excluded_set
                ]
                val_ds = Subset(self.val_ds_full, val_indices)
            else:
                # Validation/test labels remain the original CIFAR labels for monitoring only.
                val_ds = self.val_ds_full
        else:
            # If no exclusion, use the full datasets
            train_ds = self.train_ds_full
            val_ds = self.val_ds_full

        # Create new DataLoader instances
        self.train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    def _on_epoch_start(self, epoch):
        """
        Overridden hook to rebuild dataloaders if 'random' exclusion is active.
        This ensures a new random class is excluded for each epoch.
        """
        if self.args.exclude_classes and self.args.exclude_classes[0].lower() == 'random':
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}: Rebuilding dataloaders for random class exclusion.")
            # Rebuild dataloaders with new random exclusion
            self._rebuild_dataloaders()
            # Prepare the new dataloaders with the accelerator
            self.train_loader, self.val_loader = self.accelerator.prepare(self.train_loader, self.val_loader)

    def setup_dataloaders(self):
        """
        Initial setup for dataloaders. Loads the full dataset and then
        builds the first version of the dataloaders.
        """
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

        # Load the full datasets once and store them with separate transforms
        self.train_ds_full = DatasetClass(self.args.data_dir, train=True, download=True, transform=train_transform)
        self.val_ds_full = DatasetClass(self.args.data_dir, train=False, download=True, transform=val_transform)
        self.train_group_ids = self._load_train_group_ids()

        if self.accelerator.is_main_process and self.train_group_manifest is not None:
            unique_groups, counts = np.unique(self.train_group_ids, return_counts=True)
            summary = ", ".join(
                f"{int(group_id)}:{int(count)}"
                for group_id, count in zip(unique_groups.tolist(), counts.tolist())
            )
            print(f"Loaded train group manifest: {self.args.group_manifest}")
            print(f"Train group sizes: {summary}")
        
        # Build the initial dataloaders
        self._rebuild_dataloaders()

    def setup_model_specifics(self):
        if self.args.resume:
            self.model = UNet2DModel.from_pretrained(self.args.resume)
            self.noise_scheduler = DDPMScheduler.from_pretrained(self.args.resume)
        else:
            self.model = UNet2DModel(
                sample_size=32, in_channels=3, out_channels=6 if self.args.learn_sigma else 3,
                layers_per_block=3, block_out_channels=(128, 256, 256, 256),
                down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
                dropout=0.3,
            )
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.args.timesteps, beta_schedule=self.args.beta_schedule)
        
        self.ema_model = copy.deepcopy(self.model)
        if self.args.resume and os.path.exists(os.path.join(self.args.resume, "ema_model.pt")):
            self.ema_model.load_state_dict(torch.load(os.path.join(self.args.resume, "ema_model.pt"), map_location="cpu"))
        for p in self.ema_model.parameters(): p.requires_grad_(False)

    def _calculate_loss(self, batch, is_validation=False):
        imgs, _ = batch
        imgs = imgs.to(self.device)
        noise = torch.randn_like(imgs)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (imgs.size(0),), device=self.device).long()
        noisy_imgs = self.noise_scheduler.add_noise(imgs, noise, t)
        
        with self.accelerator.autocast():
            model_output = self.model(noisy_imgs, t).sample
            
            if self.args.learn_sigma:
                pred_eps, model_var_values = torch.chunk(model_output, 2, dim=1)
            else:
                pred_eps = model_output
            
            mse_loss = F.mse_loss(pred_eps, noise, reduction="none").mean(dim=(1, 2, 3))
            total_loss = mse_loss
            
            vlb_loss = torch.zeros_like(total_loss)
            if self.args.learn_sigma:
                frozen_out = torch.cat([pred_eps.detach(), model_var_values], dim=1)
                true_mean, _, true_log_var = self.diffusion_helpers.q_posterior_mean_variance(x_start=imgs, x_t=noisy_imgs, t=t)
                model_mean, _, model_log_var = self.diffusion_helpers.p_mean_variance(frozen_out, noisy_imgs, t)
                
                kl = normal_kl(true_mean, true_log_var, model_mean, model_log_var)
                kl = kl.mean(dim=(1, 2, 3)) / math.log(2.0)
                
                decoder_nll = -F.mse_loss(imgs, model_mean, reduction="none").mean(dim=(1, 2, 3))
                vlb_loss = torch.where((t == 0), decoder_nll, kl)
                total_loss += self.args.lambda_vlb * vlb_loss

        return total_loss.mean(), mse_loss.mean(), vlb_loss.mean()

    def _generate_samples(self, epoch, use_ema=False):
        # Wait for all processes to sync before starting the long sampling process.
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            # This method is called only on the main process, so no need for an extra check here.
            unwrapped_model = self.accelerator.unwrap_model(self.ema_model if use_ema else self.model)
            unwrapped_model.eval()  # Set model to evaluation mode

            # --- Sampling logic ---
            samples = torch.randn((64, unwrapped_model.in_channels, 32, 32), device=self.device)
            for t in tqdm(self.noise_scheduler.timesteps, desc=f"Sampling {'EMA' if use_ema else ''} (Uncond)", disable=not self.accelerator.is_main_process):
                t_batch = torch.full((64,), t, device=self.device, dtype=torch.long)
                with torch.no_grad():
                    out = unwrapped_model(samples, t_batch).sample
                
                if self.args.learn_sigma:
                    model_mean, _, model_log_var = self.diffusion_helpers.p_mean_variance(out, samples, t_batch)
                    noise = torch.randn_like(samples) if t > 0 else 0
                    samples = model_mean + torch.exp(0.5 * model_log_var) * noise
                else:
                    samples = self.noise_scheduler.step(out, t, samples).prev_sample
            # --- End of sampling logic ---

            # --- Save images and log to wandb ---
            grid = make_grid((samples.clamp(-1,1)+1)/2, nrow=8)
            arr = grid.permute(1,2,0).cpu().numpy()
            fname = f"{'ema_' if use_ema else ''}samples_epoch_{epoch+1:04d}.png"
            Image.fromarray((arr*255).astype(np.uint8)).save(os.path.join(self.samples_dir, fname))
            if self.args.use_wandb:
                import wandb
                wandb.log({f"samples/{'ema_' if use_ema else ''}grid": wandb.Image(arr)}, step=self.global_step)

            unwrapped_model.train() # Set model back to training mode

        # All processes wait here until the main process has finished sampling.
        # This prevents timeouts.
        self.accelerator.wait_for_everyone()


class ConditionalTrainer(BaseTrainer):
    def setup_dataloaders(self):
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
        
        self.train_loader = DataLoader(
            DatasetClass(self.args.data_dir, train=True, download=True, transform=train_transform),
            batch_size=self.args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
        )
        self.val_loader = DataLoader(
            DatasetClass(self.args.data_dir, train=False, download=True, transform=val_transform),
            batch_size=self.args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
        self.uncond_label = self.num_classes
        self.num_total_labels = self.num_classes+1

    def setup_model_specifics(self):
        if self.args.resume:
            self.model = UNet2DConditionModel.from_pretrained(self.args.resume)
            self.noise_scheduler = DDPMScheduler.from_pretrained(self.args.resume)
        else:
            self.model = UNet2DConditionModel(
                sample_size=32, in_channels=3, out_channels=6 if self.args.learn_sigma else 3,
                num_class_embeds=self.num_total_labels, class_embed_type="simple_projection",
                projection_class_embeddings_input_dim=self.num_total_labels, layers_per_block=3,
                block_out_channels=(128, 256, 256, 512),
                down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=512, dropout=0.3,
            )
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.args.timesteps, beta_schedule=self.args.beta_schedule)
        
        self.ema_model = copy.deepcopy(self.model)
        if self.args.resume and os.path.exists(os.path.join(self.args.resume, "ema_model.pt")):
            self.ema_model.load_state_dict(torch.load(os.path.join(self.args.resume, "ema_model.pt"), map_location="cpu"))
        for p in self.ema_model.parameters(): p.requires_grad_(False)
        
    def _calculate_loss(self, batch, is_validation=False):
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        noise = torch.randn_like(imgs)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (imgs.size(0),), device=self.device).long()
        noisy_imgs = self.noise_scheduler.add_noise(imgs, noise, t)

        # Classifier-Free Guidance: randomly drop labels
        mask = torch.rand(imgs.size(0), device=self.device) < self.args.drop_prob
        labels[mask] = self.uncond_label
        labels_onehot = F.one_hot(labels, num_classes=self.num_total_labels).to(imgs.dtype)

        with self.accelerator.autocast():
            model_output = self.model(noisy_imgs, t, class_labels=labels_onehot, encoder_hidden_states=None).sample
            # The rest of the loss calculation is identical to the unconditional case
            if self.args.learn_sigma:
                pred_eps, model_var_values = torch.chunk(model_output, 2, dim=1)
            else:
                pred_eps = model_output
            
            mse_loss = F.mse_loss(pred_eps, noise, reduction="none").mean(dim=(1, 2, 3))
            total_loss = mse_loss
            
            vlb_loss = torch.zeros_like(total_loss)
            if self.args.learn_sigma:
                frozen_out = torch.cat([pred_eps.detach(), model_var_values], dim=1)
                true_mean, _, true_log_var = self.diffusion_helpers.q_posterior_mean_variance(x_start=imgs, x_t=noisy_imgs, t=t)
                model_mean, _, model_log_var = self.diffusion_helpers.p_mean_variance(frozen_out, noisy_imgs, t)
                
                kl = normal_kl(true_mean, true_log_var, model_mean, model_log_var)
                kl = kl.mean(dim=(1, 2, 3)) / math.log(2.0)
                
                decoder_nll = -F.mse_loss(imgs, model_mean, reduction="none").mean(dim=(1, 2, 3))
                vlb_loss = torch.where((t == 0), decoder_nll, kl)
                total_loss += self.args.lambda_vlb * vlb_loss

        return total_loss.mean(), mse_loss.mean(), vlb_loss.mean()

    def _generate_samples(self, epoch, use_ema=False):
        # Wait for all processes to sync before starting.
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:

            unwrapped_model = self.accelerator.unwrap_model(self.ema_model if use_ema else self.model)
            unwrapped_model.eval() # Set model to evaluation mode

            # --- Sampling logic (batched by class) ---
            samples_per_class = 1
            all_samples = []
            num_batches = math.ceil(self.num_total_labels / (self.args.batch_size // samples_per_class))

            for i in range(num_batches):
                start_class = i * (self.args.batch_size // samples_per_class)
                end_class = min(start_class + (self.args.batch_size // samples_per_class), self.num_total_labels)
                batch_classes = list(range(start_class, end_class))

                if not batch_classes: continue
                
                batch_size = len(batch_classes) * samples_per_class
                samples = torch.randn((batch_size, unwrapped_model.config.in_channels, 32, 32), device=self.device)
                class_labels = F.one_hot(torch.tensor([c for c in batch_classes for _ in range(samples_per_class)], device=self.device), num_classes=self.num_total_labels).to(samples.dtype)

                for t in tqdm(self.noise_scheduler.timesteps, desc=f"Sampling {'EMA' if use_ema else ''} (Cond)", disable=not self.accelerator.is_main_process):
                    t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                    with torch.no_grad():
                        out = unwrapped_model(samples, t_batch, class_labels=class_labels, encoder_hidden_states=None).sample
                    
                    if self.args.learn_sigma:
                        model_mean, _, model_log_var = self.diffusion_helpers.p_mean_variance(out, samples, t_batch)
                        noise = torch.randn_like(samples) if t > 0 else 0
                        samples = model_mean + torch.exp(0.5 * model_log_var) * noise
                    else:
                        samples = self.noise_scheduler.step(out, t, samples).prev_sample
                all_samples.append(samples)
            # --- End of sampling logic ---
                
            # --- Save images and log to wandb ---
            all_samples = torch.cat(all_samples, dim=0)
            grid = make_grid((all_samples.clamp(-1,1)+1)/2, nrow=samples_per_class)
            arr = grid.permute(1,2,0).cpu().numpy()
            fname = f"{'ema_' if use_ema else ''}samples_epoch_{epoch+1:04d}.png"
            Image.fromarray((arr*255).astype(np.uint8)).save(os.path.join(self.samples_dir, fname))
            if self.args.use_wandb:
                import wandb
                wandb.log({f"samples/{'ema_' if use_ema else ''}grid": wandb.Image(arr)}, step=self.global_step)
                
            unwrapped_model.train() # Set model back to training mode

        # All processes wait here for the main process to finish.
        self.accelerator.wait_for_everyone()
