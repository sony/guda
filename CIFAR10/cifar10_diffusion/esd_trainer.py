"""
ESD (Erased Stable Diffusion) Unlearning Trainer for Diffusion Models.

This module implements the ESD unlearning method for forgetting specific classes
from diffusion models by training the model to output negative guided scores for forget data
while maintaining performance on retain data using a simple scalar combination loss.
"""

import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm.auto import tqdm
from diffusers import UNet2DModel, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from .base_unlearning_trainer import BaseUnlearningTrainer


class ESDUnlearningTrainer(BaseUnlearningTrainer):
    """
    ESD (Erased Stable Diffusion) Unlearning Trainer for Diffusion Models.
    
    This method implements ESD that uses negative guidance for forget data.
    For forget data, the model is trained to output negative guided scores
    (unconditional_score - w * (conditional_score - unconditional_score)),
    while for retain data, normal diffusion loss is used to maintain performance.
    
    The method requires a conditional diffusion model for negative guidance calculation.
    """
    
    def __init__(self, args):
        # ESD specific parameters
        self.lambda_forget = getattr(args, 'lambda_forget', 1.0)  # Weight for forget loss
        self.guidance_weight = getattr(args, 'guidance_weight', 1.0)  # Weight w for negative guidance
        self.conditional_model_path = getattr(args, 'conditional_model_path', None)
        self.retain_loss_type = getattr(args, 'retain_loss_type', 'standard')  # 'standard' or 'distillation'
        
        if self.conditional_model_path is None:
            raise ValueError("conditional_model_path must be specified for ESD trainer")
        
        # Force learn_sigma to False as per specification
        if hasattr(args, 'learn_sigma') and args.learn_sigma:
            import warnings
            warnings.warn("ESD does not support learn_sigma. Setting learn_sigma=False.")
            args.learn_sigma = False
        
        super().__init__(args)
        
        # Set unconditional label for conditional model (same as ConditionalTrainer)
        self.uncond_label = self.num_classes
        
        # Load conditional model for negative guidance AFTER super().__init__
        # This is called after prepare_with_accelerator() so it's safe
        self._load_conditional_model()
        
        # Load teacher model for distillation if retain_loss_type is 'distillation'
        if self.retain_loss_type == 'distillation':
            self._load_teacher_model()
        
        print(f"ESD initialized with lambda_forget={self.lambda_forget}, guidance_weight={self.guidance_weight}")
        print(f"Conditional model loaded from: {self.conditional_model_path}")
        print(f"Unconditional label set to: {self.uncond_label}")
        print(f"Retain loss type: {self.retain_loss_type}")

    def _load_conditional_model(self):
        """
        Load the conditional diffusion model for negative guidance calculation.
        """
        print(f"Loading conditional model from: {self.conditional_model_path}")
        self.conditional_model = UNet2DConditionModel.from_pretrained(self.conditional_model_path)
        self.conditional_model = self.conditional_model.to(self.device)
        self.conditional_model.eval()
        
        # Disable gradients for conditional model
        for param in self.conditional_model.parameters():
            param.requires_grad = False
        
        # Prepare conditional model with accelerator
        self.conditional_model = self.accelerator.prepare(self.conditional_model)

    def _load_teacher_model(self):
        """
        Load the teacher model for distillation loss calculation.
        This is a copy of the original model that will be kept frozen.
        """
        print(f"Loading teacher model for distillation from: {self.args.teacher_model_path}")
        self.teacher_model = UNet2DModel.from_pretrained(self.args.teacher_model_path)
        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        # Disable gradients for teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Prepare teacher model with accelerator
        self.teacher_model = self.accelerator.prepare(self.teacher_model)

    def _calculate_loss(self, batch_tuple, is_validation=False):
        """
        Calculate the ESD unlearning loss.
        
        Loss function:
        L(θ) = L_retain(θ; D_r) + λ * L_forget(θ; D_f)
        
        Where:
        - L_retain: Standard diffusion loss on retain data
        - L_forget: Loss encouraging model to output negative guided scores on forget data
        
        Args:
            batch_tuple: Tuple of (retain_batch, forget_batch)
            is_validation: Whether this is validation (not used in unlearning)
            
        Returns:
            Tuple of (total_loss, retain_mse, forget_mse) for compatibility with BaseTrainer
        """
        retain_batch, forget_batch = batch_tuple
        
        # Calculate retain loss (standard diffusion loss)
        retain_loss = self._calculate_retain_loss(retain_batch)
        
        # Calculate forget loss (negative guidance target)
        forget_loss = self._calculate_forget_loss(forget_batch)
        
        # Combine losses with lambda weighting
        total_loss = retain_loss + self.lambda_forget * forget_loss
        
        return total_loss, retain_loss, forget_loss

    def _calculate_retain_loss(self, retain_batch):
        """
        Calculate retention loss using either standard diffusion model loss or distillation loss.
        
        For standard loss:
        L_retain(θ; D_r) = E_{t, ε ~ N(0, I), x0 ~ D_r}[ || ε - ε_θ(x_t) ||^2 ]
        
        For distillation loss:
        L_retain(θ; D_r) = E_{t, ε ~ N(0, I), x0 ~ D_r}[ || ε_teacher(x_t) - ε_θ(x_t) ||^2 ]
        
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
        
        # Forward pass to predict noise with current model
        with self.accelerator.autocast():
            model_output = self.model(noisy_imgs, t).sample
            
            # Handle models trained with learn_sigma
            if model_output.shape[1] == 6:  # learn_sigma=True case
                predicted_noise, _ = torch.chunk(model_output, 2, dim=1)
            else:  # learn_sigma=False case
                predicted_noise = model_output
        
        if self.retain_loss_type == 'distillation':
            # Use distillation loss: minimize difference between student and teacher predictions
            with torch.no_grad():
                with self.accelerator.autocast():
                    teacher_output = self.teacher_model(noisy_imgs, t).sample
                    
                    # Handle models trained with learn_sigma
                    if teacher_output.shape[1] == 6:  # learn_sigma=True case
                        teacher_noise, _ = torch.chunk(teacher_output, 2, dim=1)
                    else:  # learn_sigma=False case
                        teacher_noise = teacher_output
            
            # Calculate MSE loss between student and teacher predictions
            retain_loss = F.mse_loss(predicted_noise, teacher_noise, reduction="mean")
        else:
            # Use standard diffusion loss: minimize difference between predicted and actual noise
            retain_loss = F.mse_loss(predicted_noise, noise, reduction="mean")
        
        return retain_loss

    def _calculate_forget_loss(self, forget_batch):
        """
        Calculate forget loss using negative guided score as target.
        
        L_forget(θ; D_f) = E_{t, ε ~ N(0, I), x0 ~ D_f}[ || ε_neg_guided - ε_θ(x_t) ||^2 ]
        
        Where ε_neg_guided = ε_uncond - w * (ε_cond - ε_uncond) is the negative guided score.
        
        Args:
            forget_batch: Batch of forget data (images, labels)
            
        Returns:
            Scalar tensor representing forget loss
        """
        imgs, labels = forget_batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
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
        
        # Calculate negative guided score as target
        target_noise = self._calculate_negative_guided_score(noisy_imgs, t, labels)
        
        # Forward pass to predict noise
        with self.accelerator.autocast():
            model_output = self.model(noisy_imgs, t).sample
            
            # Handle models trained with learn_sigma
            if model_output.shape[1] == 6:  # learn_sigma=True case
                predicted_noise, _ = torch.chunk(model_output, 2, dim=1)
            else:  # learn_sigma=False case
                predicted_noise = model_output
        
        # Calculate MSE loss between predicted noise and negative guided target
        forget_loss = F.mse_loss(predicted_noise, target_noise, reduction="mean")
        
        return forget_loss

    def _calculate_negative_guided_score(self, noisy_imgs, t, labels):
        """
        Calculate negative guided score: ε_uncond - w * (ε_cond - ε_uncond)
        Both unconditional and conditional scores are obtained from the conditional model.
        
        Args:
            noisy_imgs: Noisy images tensor
            t: Timesteps tensor
            labels: Class labels for conditional model
            
        Returns:
            Negative guided score tensor
        """
        with torch.no_grad():
            batch_size = labels.size(0)
            num_total_classes = self.num_classes + 1
            
            # Get unconditional score using conditional model with uncond_label
            uncond_labels = torch.full((batch_size,), self.uncond_label, 
                                     device=labels.device, dtype=labels.dtype)
            uncond_labels_onehot = torch.nn.functional.one_hot(
                uncond_labels.long(), num_classes=num_total_classes
            ).float()
            
            with self.accelerator.autocast():
                uncond_output = self.conditional_model(
                    noisy_imgs, t,
                    encoder_hidden_states=None,
                    class_labels=uncond_labels_onehot
                ).sample
                
                # Handle models trained with learn_sigma
                if uncond_output.shape[1] == 6:  # learn_sigma=True case
                    uncond_score, _ = torch.chunk(uncond_output, 2, dim=1)
                else:  # learn_sigma=False case
                    uncond_score = uncond_output
            
            # Get conditional score using conditional model with actual labels
            labels_onehot = torch.nn.functional.one_hot(
                labels.long(), num_classes=num_total_classes
            ).float()
            
            with self.accelerator.autocast():
                cond_output = self.conditional_model(
                    noisy_imgs, t, 
                    encoder_hidden_states=None, 
                    class_labels=labels_onehot
                ).sample
                
                # Handle models trained with learn_sigma
                if cond_output.shape[1] == 6:  # learn_sigma=True case
                    cond_score, _ = torch.chunk(cond_output, 2, dim=1)
                else:  # learn_sigma=False case
                    cond_score = cond_output
            
            # Calculate negative guided score: ε_uncond - w * (ε_cond - ε_uncond)
            neg_guided_score = uncond_score - self.guidance_weight * (cond_score - uncond_score)
            
        return neg_guided_score

    def _train_epoch(self, epoch):
        """
        Training epoch for ESD method with detailed logging.
        """
        self.model.train()
        self.conditional_model.eval()  # Keep conditional model in eval mode
        
        # Keep teacher model in eval mode if using distillation
        if self.retain_loss_type == 'distillation' and hasattr(self, 'teacher_model'):
            self.teacher_model.eval()
        
        running_total_loss = 0.0
        running_retain_loss = 0.0
        running_forget_loss = 0.0
        
        # Create iterators for both loaders
        retain_iter = iter(self.train_retain_loader)
        forget_iter = iter(self.train_forget_loader)
        
        # Use the smaller loader length
        num_batches = min(len(self.train_retain_loader), len(self.train_forget_loader))
        
        from tqdm import tqdm
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.args.epochs}", 
                   disable=not self.accelerator.is_main_process)
        
        for batch_idx in pbar:
            try:
                retain_batch = next(retain_iter)
                forget_batch = next(forget_iter)
            except StopIteration:
                break
                
            # Calculate ESD loss
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
                        "train/guidance_weight": self.guidance_weight,
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
        Validation epoch for ESD with retain and forget loss monitoring.
        """
        self.model.eval()
        self.conditional_model.eval()
        
        # Keep teacher model in eval mode if using distillation
        if self.retain_loss_type == 'distillation' and hasattr(self, 'teacher_model'):
            self.teacher_model.eval()
        
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
            
            # Handle initial evaluation vs regular epochs
            if epoch == -1:
                desc = "Initial ESD Validation"
                print("Performing initial ESD validation to establish baseline losses...")
            else:
                desc = "Validation"
            
            pbar = tqdm(range(num_batches), desc=desc, disable=not self.accelerator.is_main_process)
            for _ in pbar:
                try:
                    retain_batch = next(retain_iter)
                    forget_batch = next(forget_iter)
                except StopIteration:
                    break
                
                # Calculate ESD loss on validation data
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
            if epoch == -1:
                print(f"[Initial] val_total: {avg_val_total:.4f}, val_retain: {avg_val_retain:.4f}, val_forget: {avg_val_forget:.4f}")
            else:
                print(f"[Epoch {epoch+1:03d}] val_total: {avg_val_total:.4f}, val_retain: {avg_val_retain:.4f}, val_forget: {avg_val_forget:.4f}")

            step_value = self.global_step if epoch >= 0 else 0
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
            self._log_metrics(val_logs, step=step_value)
        
        # Ensure all processes are synchronized after validation
        self.accelerator.wait_for_everyone()
        
        # Set model back to training mode
        self.model.train()

    def prepare_with_accelerator(self):
        """Prepare models and optimizers with accelerator, handling ESD-specific models."""
        # Save reference to original optimizer before accelerator wrapping
        self.original_optimizer = self.optimizer
        
        # Prepare main models and data loaders
        self.model, self.ema_model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.ema_model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler
        )
        
        # Also prepare unlearning-specific loaders
        self.train_retain_loader, self.train_forget_loader = self.accelerator.prepare(
            self.train_retain_loader, self.train_forget_loader
        )
        
        # Note: conditional_model and teacher_model will be prepared separately after loading

    def setup_common_optimizer_and_scheduler(self):
        """Setup ESD-optimized optimizer and learning rate scheduler."""
        # Get optimizer parameters from args
        optimizer_type = getattr(self.args, 'optimizer', 'adam')
        adam_b1 = getattr(self.args, 'adam_b1', 0.9)
        adam_b2 = getattr(self.args, 'adam_b2', 0.999)
        adam_eps = getattr(self.args, 'adam_eps', 1e-8)
        weight_decay = getattr(self.args, 'weight_decay', 0.0)
        
        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.args.lr,
                betas=(adam_b1, adam_b2),
                eps=adam_eps,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Setup learning rate scheduler
        lr_scheduler_type = getattr(self.args, 'lr_scheduler', 'cosine')
        warmup_steps = getattr(self.args, 'warmup_steps', 500)
        
        # Calculate total training steps
        total_steps = self.args.epochs * len(self.train_retain_loader)
        
        if lr_scheduler_type.lower() == 'cosine':
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=total_steps
            )
        elif lr_scheduler_type.lower() == 'linear':
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif lr_scheduler_type.lower() == 'constant':
            # Constant LR after warmup
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=0.5  # This makes it constant after warmup
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler type: {lr_scheduler_type}")
            
        print(f"ESD Optimizer: {optimizer_type.upper()} with {lr_scheduler_type} scheduler")
        print(f"Adam parameters: b1={adam_b1}, b2={adam_b2}, eps={adam_eps}")
        print(f"Weight decay: {weight_decay}, Warmup steps: {warmup_steps}")
        print(f"Total training steps: {total_steps}")

        # Note: conditional_model will be prepared separately after loading