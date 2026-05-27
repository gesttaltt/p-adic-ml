import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_reconstruction_error(model, digits, p, is_vqvae=False, weighted=False, alpha=1.5):
    """
    Computes the average reconstruction cross-entropy loss per sequence.
    If weighted is True, applies an exponentially decaying weight along the sequence indices.
    digits: [B, N] tensor
    p: [B] tensor
    """
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(digits, p)
            
        B, N, C = logits.shape
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Flatten to compute Cross Entropy
        loss_flat = criterion(logits.reshape(-1, C), digits.reshape(-1))
        loss_per_token = loss_flat.reshape(B, N)
        
        if weighted:
            exponents = torch.arange(N, dtype=torch.float, device=digits.device).view(1, N)
            if alpha == 'p':
                decay_base = p.view(B, 1).float()
            else:
                decay_base = torch.tensor(alpha, device=digits.device).view(1, 1)
            
            weights = decay_base ** (-exponents) # [B, N]
            # Normalize each sample's weights to sum to N (so scale is comparable to flat mean)
            weights_norm = weights / weights.mean(dim=-1, keepdim=True)
            loss_per_sample = (loss_per_token * weights_norm).mean(dim=-1)
        else:
            loss_per_sample = loss_per_token.mean(dim=-1)
            
    return loss_per_sample

class CascadeRouter:
    def __init__(self, beta_vae, vq_vae, prior):
        self.beta_vae = beta_vae
        self.vq_vae = vq_vae
        self.prior = prior
        
    def generate_cascade(self, p, threshold_tau, device='cpu', weighted=False, alpha=1.5):
        """
        Generates p-adic numbers using a cascade of Beta-VAE and VQ-VAE.
        If Beta-VAE's self-reconstruction error is below threshold_tau, it uses the Beta-VAE (fast).
        Otherwise, it falls back to VQ-VAE + Prior (slow but precise).
        p: [B] tensor of primes
        threshold_tau: float or dict mapping prime values to threshold floats
        """
        B = p.shape[0]
        self.beta_vae.eval()
        self.vq_vae.eval()
        self.prior.eval()
        
        start_time = time.time()
        
        # 1. Fast Generation Path: sample candidate from Beta-VAE
        x_beta = self.beta_vae.sample(p, device=device) # [B, N]
        
        # 2. Anomaly Detection: compute Beta-VAE self-reconstruction error
        beta_recon_err = get_reconstruction_error(
            self.beta_vae, x_beta, p, is_vqvae=False, weighted=weighted, alpha=alpha
        ) # [B]
        
        # Determine routing masks (global threshold or dictionary of base-specific thresholds)
        if isinstance(threshold_tau, dict):
            thresh_val = torch.tensor([threshold_tau[val.item()] for val in p], device=p.device)
        else:
            thresh_val = torch.full_like(p, threshold_tau, dtype=torch.float)
            
        fast_mask = beta_recon_err < thresh_val # [B] (boolean mask)
        fallback_indices = (~fast_mask).nonzero(as_tuple=True)[0]
        num_fallback = fallback_indices.shape[0]
        
        # Final output digits initialization
        final_digits = x_beta.clone()
        routed_paths = [True] * B # True means fast path, False means fallback
        
        # 3. Fallback Generation Path (if needed)
        if num_fallback > 0:
            p_fallback = p[fallback_indices]
            
            # Sample discrete latents from VQ-VAE Prior
            with torch.no_grad():
                L = self.vq_vae.N // 2
                latent_indices = self.prior.sample(p_fallback, L=L, temperature=0.7)
                quantized = self.vq_vae.quantizer.embedding(latent_indices)
                logits_vq = self.vq_vae.decode(quantized, p_fallback)
                x_vq = torch.argmax(logits_vq, dim=-1) # [num_fallback, N]
                
            # Insert VQ-VAE generations
            final_digits[fallback_indices] = x_vq
            for idx in fallback_indices.tolist():
                routed_paths[idx] = False
                
        elapsed = time.time() - start_time
        
        return final_digits, routed_paths, beta_recon_err, elapsed
