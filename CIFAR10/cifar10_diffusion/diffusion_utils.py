import torch
import torch.nn.functional as F
import numpy as np
from diffusers import DDPMScheduler

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1D numpy array for a batch of indices (timesteps),
    and broadcast to the specified shape.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Calculate the KL divergence between two Gaussian distributions.
    """
    term = (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
    return 0.5 * term.view(term.shape[0], -1).sum(dim=1)

class DiffusionHelpers:
    """
    Helper class that pre-computes coefficients needed for the diffusion process
    and assists in calculating the VLB loss.
    Ported from the logic in `improved_diffusion.gaussian_diffusion.GaussianDiffusion`.
    """
    def __init__(self, scheduler: DDPMScheduler):
        self.betas = scheduler.betas.numpy().astype(np.float64)
        self.alphas_cumprod = scheduler.alphas_cumprod.numpy().astype(np.float64)
        self.num_timesteps = len(self.betas)

        # Coefficients for calculating q(x_{t-1} | x_t, x_0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(1.0 - self.betas)
            / (1.0 - self.alphas_cumprod)
        )

        # Coefficients for predicting x_0
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Calculate the mean and variance of the true posterior distribution
        q(x_{t-1} | x_t, x_0).
        """
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model_output, x_t, t):
        """
        Calculate the mean and variance of the reverse process p(x_{t-1} | x_t)
        from the model output.
        """
        B, C = x_t.shape[:2]
        pred_eps, model_var_values = torch.chunk(model_output, 2, dim=1)

        #--- Calculate variance (LEARNED_RANGE) ---
        min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        max_log = _extract_into_tensor(np.log(self.betas), t, x_t.shape)
        # Scale model output from [-1, 1] to [0, 1]
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)

        #--- Calculate mean ---
        # 1. Predict x_0 from predicted noise (eps)
        pred_xstart = self._predict_xstart_from_eps(x_t, t, pred_eps)
        pred_xstart = torch.clamp(pred_xstart, -1., 1.)

        # 2. Use predicted x_0 to calculate the mean of the reverse process
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )
        return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def compute_vlb_terms(self, model_output, x_start, x_t, t):
        """
        Compute the VLB (Variational Lower Bound) terms for ELBO calculation.
        Returns individual VLB terms that can be summed to get the full ELBO.
        """
        B, C = x_t.shape[:2]
        
        if model_output.shape[1] == 2 * C:
            # Model predicts both epsilon and variance
            pred_eps, model_var_values = torch.chunk(model_output, 2, dim=1)
        else:
            # Model only predicts epsilon
            pred_eps = model_output
            model_var_values = None
        
        # Compute true posterior
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        
        if model_var_values is not None:
            # Use learned variance
            model_mean, _, model_log_var = self.p_mean_variance(model_output, x_t, t)
            
            # KL divergence between q(x_{t-1}|x_t,x_0) and p(x_{t-1}|x_t)
            kl = normal_kl(true_mean, true_log_var, model_mean, model_log_var)
            kl = kl.mean(dim=(1, 2, 3)) / np.log(2.0)
            
            # Decoder NLL for t=0
            decoder_nll = -self._gaussian_log_likelihood(x_start, model_mean, model_log_var)
            decoder_nll = decoder_nll.mean(dim=(1, 2, 3)) / np.log(2.0)
            
            # VLB loss: KL for t>0, decoder NLL for t=0
            vlb_terms = torch.where(t == 0, decoder_nll, kl)
        else:
            # Simple MSE loss as approximation to VLB
            mse_loss = F.mse_loss(pred_eps, torch.randn_like(x_start), reduction="none")
            vlb_terms = mse_loss.mean(dim=(1, 2, 3))
        
        return vlb_terms

    def _gaussian_log_likelihood(self, x, mean, log_var):
        """Compute Gaussian log-likelihood."""
        return -0.5 * (np.log(2 * np.pi) + log_var + (x - mean) ** 2 * torch.exp(-log_var))


def compute_elbo_score(model, noise_scheduler, imgs, labels=None, num_timesteps=None, device=None):
    """
    Compute ELBO scores for a batch of images using a diffusion model.
    
    Args:
        model: The diffusion model (UNet2DModel or UNet2DConditionModel)
        noise_scheduler: The noise scheduler
        imgs: Input images [B, C, H, W]
        labels: Optional labels for conditional models [B]
        num_timesteps: Number of timesteps to sample for ELBO estimation
        device: Device to run computation on
    
    Returns:
        elbo_scores: ELBO scores for each image [B]
    """
    if device is None:
        device = imgs.device
    
    if num_timesteps is None:
        num_timesteps = noise_scheduler.config.num_train_timesteps // 10  # Sample subset for efficiency
    
    model.eval()
    diffusion_helpers = DiffusionHelpers(noise_scheduler)
    
    elbo_scores = []
    
    with torch.no_grad():
        # Sample multiple timesteps for each image to estimate ELBO
        t_samples = torch.randint(1, noise_scheduler.config.num_train_timesteps, 
                                 (num_timesteps,), device=device)
        
        batch_elbos = torch.zeros(imgs.size(0), device=device)
        
        for t_idx in range(num_timesteps):
            t = t_samples[t_idx].unsqueeze(0).expand(imgs.size(0))
            
            # Add noise
            noise = torch.randn_like(imgs)
            noisy_imgs = noise_scheduler.add_noise(imgs, noise, t)
            
            # Forward pass
            if hasattr(model, 'class_embed') and labels is not None:
                # Conditional model
                num_classes = model.config.num_class_embeds
                labels_onehot = F.one_hot(labels, num_classes=num_classes).to(imgs.dtype)
                model_output = model(noisy_imgs, t, class_labels=labels_onehot, encoder_hidden_states=None).sample
            else:
                # Unconditional model
                model_output = model(noisy_imgs, t).sample
            
            # Compute VLB terms
            vlb_terms = diffusion_helpers.compute_vlb_terms(model_output, imgs, noisy_imgs, t)
            batch_elbos += vlb_terms
        
        # Average over timesteps
        elbo_scores = -batch_elbos / num_timesteps  # Negative because VLB is upper bound on -log p(x)
    
    return elbo_scores


def evaluate_spearman_correlation(unlearned_scores, logo_scores):
    """
    Evaluate Spearman correlation between unlearned and LOGO ΔELBO scores.
    
    Args:
        unlearned_scores: ΔELBO scores from unlearned model
        logo_scores: ΔELBO scores from LOGO model (ground truth)
    
    Returns:
        correlation: Spearman correlation coefficient
        p_value: P-value of the correlation
    """
    from scipy.stats import spearmanr
    
    # Ensure arrays are numpy
    if torch.is_tensor(unlearned_scores):
        unlearned_scores = unlearned_scores.cpu().numpy()
    if torch.is_tensor(logo_scores):
        logo_scores = logo_scores.cpu().numpy()
    
    # Compute correlation
    correlation, p_value = spearmanr(unlearned_scores, logo_scores)
    
    return correlation, p_value
