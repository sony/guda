"""
EraseDiff_noise Unlearning Trainer for Diffusion Models.

This module implements the EraseDiff_noise unlearning method for forgetting specific classes
from diffusion models by training the model to output meaningless noise for forget data
while maintaining performance on retain data using a simple scalar combination loss.
"""

import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm.auto import tqdm

from .base_unlearning_trainer import BaseUnlearningTrainer


class EraseDiffNoiseUnlearningTrainer(BaseUnlearningTrainer):
    """
    EraseDiff_noise Unlearning Trainer for Diffusion Models.
    
    This method implements a simplified version of EraseDiff that uses a scalar combination
    loss L_retain + λ L_forget to minimize. For forget data, the model is trained to output
    random noise, while for retain data, normal diffusion loss is used to maintain performance.
    
    The method does not use constrained optimization and does not support learn_sigma.
    """
    
    def __init__(self, args):
        # EraseDiff_noise specific parameters
        self.lambda_forget = getattr(args, 'lambda_forget', 1.0)  # Weight for forget loss
        
        # Force learn_sigma to False as per specification
        if hasattr(args, 'learn_sigma') and args.learn_sigma:
            import warnings
            warnings.warn("EraseDiff_noise does not support learn_sigma. Setting learn_sigma=False.")
            args.learn_sigma = False
        
        super().__init__(args)
        
        print(f"EraseDiff_noise initialized with lambda_forget={self.lambda_forget}")

    def _calculate_loss(self, batch_tuple, is_validation=False):
        """
        Calculate the EraseDiff_noise unlearning loss.
        
        Loss function:
        L(θ) = L_retain(θ; D_r) + λ * L_forget(θ; D_f)
        
        Where:
        - L_retain: Standard diffusion loss on retain data
        - L_forget: Loss encouraging model to output random noise on forget data
        
        Args:
            batch_tuple: Tuple of (retain_batch, forget_batch)
            is_validation: Whether this is validation (not used in unlearning)
            
        Returns:
            Tuple of (total_loss, retain_mse, forget_mse) for compatibility with BaseTrainer
        """
        retain_batch, forget_batch = batch_tuple
        
        # Calculate retain loss (standard diffusion loss)
        retain_loss = self._calculate_retain_loss(retain_batch)
        
        # Calculate forget loss (random noise target)
        forget_loss = self._calculate_forget_loss(forget_batch)
        
        # Combine losses with lambda weighting
        total_loss = retain_loss + self.lambda_forget * forget_loss
        
        return total_loss, retain_loss, forget_loss

    def _calculate_retain_loss(self, retain_batch):
        """
        Calculate retention loss using standard diffusion model loss.
        
        L_retain(θ; D_r) = E_{t, ε ~ N(0, I), x0 ~ D_r}[ || ε - ε_θ(x_t) ||^2 ]
        
        Args:
            retain_batch: Batch of retain data (images, labels)
            
        Returns:
            Scalar tensor representing retain loss
        """
        imgs, _ = retain_batch  # Ignore labels for unconditional model
        imgs = imgs.to(self.device)
        batch_size = imgs.size(0)
        
        # Sample random timesteps for each sample independently
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # Sample noise epsilon ~ N(0, I)
        noise = torch.randn_like(imgs)
        
        # Generate noisy images x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε
        noisy_imgs = self.noise_scheduler.add_noise(imgs, noise, t)
        
        # Forward pass to predict noise
        with self.accelerator.autocast():
            model_output = self.model(noisy_imgs, t).sample
            
            # Handle models trained with learn_sigma
            if model_output.shape[1] == 6:  # learn_sigma=True case
                predicted_noise, _ = torch.chunk(model_output, 2, dim=1)
            else:  # learn_sigma=False case
                predicted_noise = model_output
        
        # Calculate MSE loss between predicted and actual noise
        retain_loss = F.mse_loss(predicted_noise, noise, reduction="mean")
        
        return retain_loss

    def _calculate_forget_loss(self, forget_batch):
        """
        Calculate forget loss using random noise as target.
        
        L_forget(θ; D_f) = E_{t, ε ~ N(0, I), x0 ~ D_f}[ || ε_f - ε_θ(x_t) ||^2 ]
        
        Where ε_f ~ N(0, I) is independently sampled Gaussian noise.
        
        Args:
            forget_batch: Batch of forget data (images, labels)
            
        Returns:
            Scalar tensor representing forget loss
        """
        imgs, _ = forget_batch  # Ignore labels for unconditional model
        imgs = imgs.to(self.device)
        batch_size = imgs.size(0)
        
        # Sample random timesteps for each sample independently
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # Sample noise epsilon ~ N(0, I) for noising the image
        noise = torch.randn_like(imgs)
        
        # Generate noisy images x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε
        noisy_imgs = self.noise_scheduler.add_noise(imgs, noise, t)
        
        # Sample independent random noise ε_f ~ N(0, I) as target
        target_noise = torch.randn_like(imgs)
        
        # Forward pass to predict noise
        with self.accelerator.autocast():
            model_output = self.model(noisy_imgs, t).sample
            
            # Handle models trained with learn_sigma
            if model_output.shape[1] == 6:  # learn_sigma=True case
                predicted_noise, _ = torch.chunk(model_output, 2, dim=1)
            else:  # learn_sigma=False case
                predicted_noise = model_output
        
        # Calculate MSE loss between predicted noise and random target noise
        forget_loss = F.mse_loss(predicted_noise, target_noise, reduction="mean")
        
        return forget_loss

    def _train_epoch(self, epoch):
        """
        Training epoch for EraseDiff_noise method with detailed logging.
        """
        self.model.train()
        running_total_loss = 0.0
        running_retain_loss = 0.0
        running_forget_loss = 0.0
        
        # Create iterators for both loaders
        retain_iter = iter(self.train_retain_loader)
        forget_iter = iter(self.train_forget_loader)
        
        # Use the smaller loader length
        num_batches = min(len(self.train_retain_loader), len(self.train_forget_loader))
        
        from tqdm.auto import tqdm
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.args.epochs}", 
                   disable=not self.accelerator.is_main_process)
        
        for batch_idx in pbar:
            try:
                retain_batch = next(retain_iter)
                forget_batch = next(forget_iter)
            except StopIteration:
                break
                
            # Calculate EraseDiff_noise loss
            total_loss, retain_loss, forget_loss = self._calculate_loss(
                (retain_batch, forget_batch), is_validation=False
            )
            
            # Backward pass
            self.accelerator.backward(total_loss)
            
            # Gradient clipping (recommended 1.0-5.0)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            # Update global step and EMA
            self.global_step += 1
            running_total_loss += total_loss.item()
            running_retain_loss += retain_loss.item()
            running_forget_loss += forget_loss.item()
            self._update_ema()
            
            # Update progress bar
            pbar.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'Retain': f'{retain_loss.item():.4f}',
                'Forget': f'{forget_loss.item():.4f}'
            })
            
            # Detailed logging
            if self.global_step % self.args.log_every == 0 and self.accelerator.is_main_process:
                current_lr = self.lr_scheduler.get_last_lr()[0]
                metrics = self._format_train_metrics(
                    total_loss=total_loss,
                    learning_rate=current_lr,
                    extra={
                        "train/retain_loss": retain_loss.item(),
                        "train/forget_loss": forget_loss.item(),
                        "train/lambda_forget": self.lambda_forget,
                        "train/global_step": self.global_step,
                    },
                )
                self._log_metrics(metrics, step=self.global_step)
        
        # Epoch summary logging
        if self.accelerator.is_main_process:
            avg_total_loss = running_total_loss / num_batches
            avg_retain_loss = running_retain_loss / num_batches
            avg_forget_loss = running_forget_loss / num_batches
            
            print(f"[Epoch {epoch+1:03d}] "
                  f"Total: {avg_total_loss:.4f}, "
                  f"Retain: {avg_retain_loss:.4f}, "
                  f"Forget: {avg_forget_loss:.4f}")

            epoch_metrics = {
                "train/epoch_avg_total_loss": avg_total_loss,
                "train/epoch_avg_retain_loss": avg_retain_loss,
                "train/epoch_avg_forget_loss": avg_forget_loss,
            }
            self._log_metrics(epoch_metrics, step=self.global_step)
        
        # Ensure all processes are synchronized after training epoch
        self.accelerator.wait_for_everyone()

    def _validate_epoch(self, epoch):
        """
        Validation epoch for EraseDiff_noise with retain and forget loss monitoring.
        """
        self.model.eval()
        
        # Lists to store gathered loss tensors
        all_total_losses = []
        all_retain_losses = []
        all_forget_losses = []
        
        with torch.no_grad():
            # Use smaller validation loaders for faster validation
            val_retain_loader = torch.utils.data.DataLoader(
                self.val_retain_ds, batch_size=self.pair_batch_size, shuffle=False, 
                num_workers=4, pin_memory=True
            )
            val_forget_loader = torch.utils.data.DataLoader(
                self.val_forget_ds, batch_size=self.pair_batch_size, shuffle=False, 
                num_workers=4, pin_memory=True
            )
            
            # Prepare validation loaders with accelerator
            val_retain_loader, val_forget_loader = self.accelerator.prepare(
                val_retain_loader, val_forget_loader
            )
            
            # Create iterators
            retain_iter = iter(val_retain_loader)
            forget_iter = iter(val_forget_loader)
            
            # Use the smaller loader length
            num_batches = min(len(val_retain_loader), len(val_forget_loader))
            
            pbar = tqdm(range(num_batches), desc="Validation", disable=not self.accelerator.is_main_process)
            for _ in pbar:
                try:
                    retain_batch = next(retain_iter)
                    forget_batch = next(forget_iter)
                except StopIteration:
                    break
                
                # Calculate EraseDiff_noise loss on validation data
                total_loss, retain_loss, forget_loss = self._calculate_loss(
                    (retain_batch, forget_batch), is_validation=True
                )
                
                # Gather losses from all processes
                gathered_total = self.accelerator.gather(total_loss)
                gathered_retain = self.accelerator.gather(retain_loss)
                gathered_forget = self.accelerator.gather(forget_loss)
                
                # Append gathered tensors (ensure they are at least 1D)
                all_total_losses.append(gathered_total.flatten())
                all_retain_losses.append(gathered_retain.flatten())
                all_forget_losses.append(gathered_forget.flatten())

        # Calculate averages across all batches and processes
        avg_val_total = torch.cat(all_total_losses).mean().item()
        avg_val_retain = torch.cat(all_retain_losses).mean().item()
        avg_val_forget = torch.cat(all_forget_losses).mean().item()
        
        # Only main process logs
        if self.accelerator.is_main_process:
            print(f"[Epoch {epoch+1:03d}] val_total: {avg_val_total:.4f}, val_retain: {avg_val_retain:.4f}, val_forget: {avg_val_forget:.4f}")

            val_logs = self._format_validation_metrics(
                loss=avg_val_total,
                mse_loss=None,
                vlb_loss=None,
                forget_ratio=self._latest_forget_ratio,
                extra={
                    "val/retain_loss": avg_val_retain,
                    "val/forget_loss": avg_val_forget,
                },
            )
            self._log_metrics(val_logs, step=self.global_step)
        
        # Ensure all processes are synchronized after validation
        self.accelerator.wait_for_everyone()
        
        # Set model back to training mode
        self.model.train()