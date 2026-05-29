import os
import torch
from torch import autocast
import numpy as np
from PIL import Image
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
import torch.nn.functional as F

from cifar10_diffusion.diffusion_utils import DiffusionHelpers

def load_ema_model(model, ema_path):
    """Helper function to load EMA weights into a model."""
    if not os.path.exists(ema_path):
        raise FileNotFoundError(f"EMA model weights not found at {ema_path}")
    
    print(f"Loading EMA model weights from {ema_path}")
    ema_state_dict = torch.load(ema_path, map_location="cpu")
    
    if "model" in ema_state_dict:
        ema_state_dict = ema_state_dict["model"]
    
    if all(k.startswith("module.") for k in ema_state_dict.keys()):
        ema_state_dict = {k[7:]: v for k, v in ema_state_dict.items()}
    
    model.load_state_dict(ema_state_dict)
    return model

class BaseSampler:
    """
    Base class for sampling from diffusion models. Handles common logic for
    model loading, scheduler setup, and directory creation.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.args.seed)
        
        self.model = self.load_model_and_scheduler()
        self.model.eval()
        self.model = torch.compile(self.model)

    def load_model_and_scheduler(self):
        raise NotImplementedError("Subclasses must implement load_model_and_scheduler()")

    def sample(self):
        raise NotImplementedError("Subclasses must implement sample()")

    def _setup_scheduler_and_helpers(self):
        """Initializes the scheduler and diffusion helpers."""
        if self.args.scheduler == "ddim":
            scheduler = DDIMScheduler.from_pretrained(self.args.ckpt_dir)
            scheduler.set_timesteps(self.args.num_inference_steps)
            scheduler.eta = self.args.eta
        else:
            scheduler = DDPMScheduler.from_pretrained(self.args.ckpt_dir)
            scheduler.set_timesteps(self.args.num_inference_steps)
        
        diffusion_helpers = None
        if self.args.learn_sigma:
            # Only DDPM with learned sigma uses the helper class for sampling
            if not isinstance(scheduler, DDPMScheduler):
                 print("Warning: Learned sigma is only supported with DDPMScheduler for sampling. Ignoring.")
            else:
                diffusion_helpers = DiffusionHelpers(scheduler)

        return scheduler, diffusion_helpers


class UnconditionalSampler(BaseSampler):
    """Sampler for unconditional models."""
    def load_model_and_scheduler(self):
        model = UNet2DModel.from_pretrained(self.args.ckpt_dir).to(self.device)
        
        if self.args.use_ema:
            ema_path = self.args.ema_path if self.args.ema_path else os.path.join(self.args.ckpt_dir, "ema_model.pt")
            model = load_ema_model(model, ema_path)
            
        if model.config.out_channels == 6 and not self.args.learn_sigma:
            print("Model has 6 output channels, enabling --learn_sigma automatically.")
            self.args.learn_sigma = True
        
        self.scheduler, self.diffusion_helpers = self._setup_scheduler_and_helpers()
        return model

    def sample(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        total = self.args.num_images
        bs = self.args.batch_size
        img_idx = 0

        fixed_noise = None
        if getattr(self.args, "noise_file", None):
            if not os.path.exists(self.args.noise_file):
                raise FileNotFoundError(f"Noise file not found: {self.args.noise_file}")
            fixed_noise = torch.load(self.args.noise_file, map_location="cpu")
            if not isinstance(fixed_noise, torch.Tensor):
                raise TypeError("Noise file must contain a torch.Tensor")
            if fixed_noise.shape[0] < total:
                raise ValueError(
                    f"Noise file has {fixed_noise.shape[0]} samples, expected >= {total}"
                )

        for _ in tqdm(range((total + bs - 1) // bs), desc="Generating unconditional images"):
            current_bs = min(bs, total - img_idx)
            if current_bs <= 0: break

            if fixed_noise is not None:
                latents = fixed_noise[img_idx:img_idx + current_bs].to(self.device).clone()
            else:
                latents = torch.randn((current_bs, self.model.in_channels, 32, 32), device=self.device)

            for t in tqdm(self.scheduler.timesteps, leave=False):
                t_batch = torch.full((current_bs,), t, device=self.device, dtype=torch.long)
                with torch.no_grad(), autocast(device_type="cuda"):
                    output = self.model(latents, t_batch).sample

                if self.args.learn_sigma and self.diffusion_helpers:
                    model_mean, _, model_log_var = self.diffusion_helpers.p_mean_variance(output, latents, t_batch)
                    noise = torch.randn_like(latents) if t > 0 else 0
                    latents = model_mean + torch.exp(0.5 * model_log_var) * noise
                else:
                    # Standard DDPM/DDIM step
                    pred_noise = output.chunk(2, dim=1)[0] if self.args.learn_sigma else output
                    latents = self.scheduler.step(pred_noise, t, latents).prev_sample

            images = (latents.clamp(-1, 1) + 1) / 2
            for i in range(current_bs):
                pil_image = to_pil_image(images[i].cpu())
                pil_image.save(os.path.join(self.args.output_dir, f"image_{img_idx:04d}.png"))
                img_idx += 1
        
        print(f"Generated {img_idx} images in -> {self.args.output_dir}")


class ConditionalSampler(BaseSampler):
    """Sampler for class-conditional models."""
    def load_model_and_scheduler(self):
        model = UNet2DConditionModel.from_pretrained(self.args.ckpt_dir).to(self.device)
        
        if self.args.use_ema:
            ema_path = self.args.ema_path if self.args.ema_path else os.path.join(self.args.ckpt_dir, "ema_model.pt")
            model = load_ema_model(model, ema_path)
            
        if model.config.out_channels == 6 and not self.args.learn_sigma:
            print("Model has 6 output channels, enabling --learn_sigma automatically.")
            self.args.learn_sigma = True

        self.scheduler, self.diffusion_helpers = self._setup_scheduler_and_helpers()

        if self.args.dataset == "cifar100":
            self.num_classes = 100
        else:
            self.num_classes = 10

        self.uncond_label = self.num_classes
        self.num_total_labels = self.num_classes + 1

        return model

    def sample(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Track the number of images that need to be generated for each class
        remaining_per_class = {class_id: self.args.num_images_per_class for class_id in range(self.num_classes)}
        
        # Continue until images for all classes have been generated
        while any(count > 0 for count in remaining_per_class.values()):
            # Generate images for multiple classes simultaneously to maximize batch size utilization
            batch_class_ids = []
            batch_indices = []  # Track which class each image in the batch belongs to
            
            # Add images from each class that has remaining images to fill the batch
            available_batch_size = self.args.batch_size
            for class_id in range(self.num_classes):
                if remaining_per_class[class_id] > 0:
                    # Determine how many images to generate for this class
                    class_count = min(remaining_per_class[class_id], available_batch_size)
                    batch_class_ids.extend([class_id] * class_count)
                    batch_indices.extend([class_id] * class_count)
                    remaining_per_class[class_id] -= class_count
                    available_batch_size -= class_count
                    
                    if available_batch_size == 0:
                        break
            
            # Actual batch size
            cur_bs = len(batch_class_ids)
            if cur_bs == 0:
                break
                
            # Convert class IDs for each image in the batch to a tensor
            batch_class_ids = torch.tensor(batch_class_ids, device=self.device)
            
            # Initialize latent variables
            latents = torch.randn((cur_bs, self.model.in_channels, 32, 32), device=self.device)
            
            # Prepare class labels for Classifier-Free Guidance
            uncond_oh = F.one_hot(torch.full((cur_bs,), self.uncond_label, device=self.device), self.num_total_labels).to(latents.dtype)
            cond_oh = F.one_hot(batch_class_ids, self.num_total_labels).to(latents.dtype)
            
            # Diffusion model sampling steps
            for t in tqdm(self.scheduler.timesteps, leave=False):
                t_batch = torch.full((cur_bs,), t, device=self.device, dtype=torch.long)
                with torch.no_grad(), autocast(device_type="cuda"):
                    uncond_pred = self.model(latents, t_batch, class_labels=uncond_oh, encoder_hidden_states=None).sample
                    cond_pred = self.model(latents, t_batch, class_labels=cond_oh, encoder_hidden_states=None).sample
                
                # Classifier-Free Guidance
                guided_pred = uncond_pred + self.args.guidance_scale * (cond_pred - uncond_pred)
                
                if self.args.learn_sigma and self.diffusion_helpers:
                    model_mean, _, model_log_var = self.diffusion_helpers.p_mean_variance(guided_pred, latents, t_batch)
                    noise = torch.randn_like(latents) if t > 0 else 0
                    latents = model_mean + torch.exp(0.5 * model_log_var) * noise
                else:
                    pred_noise = guided_pred.chunk(2, dim=1)[0] if self.args.learn_sigma else guided_pred
                    latents = self.scheduler.step(pred_noise, t, latents).prev_sample
            
            # Save generated images
            images = (latents.clamp(-1, 1) + 1) / 2
            
            # Initialize/update counters for each class
            class_counters = {}
            
            for i, class_id in enumerate(batch_indices):
                # Get or initialize the counter for this class
                if class_id not in class_counters:
                    # Create class directory
                    class_dir = os.path.join(self.args.output_dir, f"class_{class_id}")
                    os.makedirs(class_dir, exist_ok=True)
                    
                    # Count existing images
                    existing_images = [f for f in os.listdir(class_dir) if f.startswith("image_") and f.endswith(".png")]
                    class_counters[class_id] = len(existing_images)
                
                # Save the image
                img_idx = class_counters[class_id]
                class_dir = os.path.join(self.args.output_dir, f"class_{class_id}")
                pil_image = to_pil_image(images[i].cpu())
                pil_image.save(os.path.join(class_dir, f"image_{img_idx:04d}.png"))
                class_counters[class_id] += 1
        
        # Report results
        for class_id in range(self.num_classes):
            class_dir = os.path.join(self.args.output_dir, f"class_{class_id}")
            if os.path.exists(class_dir):
                img_count = len([f for f in os.listdir(class_dir) if f.endswith(".png")])
                print(f"Class {class_id} -> generated {img_count} images in {class_dir}")