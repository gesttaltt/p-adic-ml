import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt

def batch_padic_distance(digits, p):
    """
    Computes the pairwise p-adic distance matrix for a batch of sequences.
    digits: [B, N] tensor of digits in base p
    p: [B] tensor of primes
    Returns: [B, B] pairwise distance matrix
    """
    B, N = digits.shape
    
    # 1. Compute pairwise digit differences
    d1 = digits.unsqueeze(1) # [B, 1, N]
    d2 = digits.unsqueeze(0) # [1, B, N]
    diffs = (d1 != d2).float() # [B, B, N]
    
    # 2. Exponents: [0, 1, ..., N-1]
    exponents = torch.arange(N, dtype=torch.float, device=digits.device).view(1, 1, N)
    
    # 3. Geometric weights for each sequence: p_i ** (-exponent)
    p_matrix = p.view(B, 1, 1).float() # [B, 1, 1]
    weights = p_matrix ** (-exponents) # [B, 1, N] (broadcasts to [B, B, N])
    
    # 4. Compute maximum weight at the first differing position
    weighted_diffs = diffs * weights # [B, B, N]
    D_padic, _ = torch.max(weighted_diffs, dim=-1) # [B, B]
    
    return D_padic

def compute_metric_loss(z, digits, p):
    """
    Computes the scale-invariant normalized MSE loss between latent Euclidean distances and data-space p-adic distances.
    Only aligns pairs that share the same prime base.
    z: [B, latent_dim] latent representations
    digits: [B, N] sequences
    p: [B] prime bases
    """
    B = digits.shape[0]
    if B <= 1:
        return torch.tensor(0.0, device=z.device)
        
    # 1. Compute pairwise p-adic distances
    D_padic = batch_padic_distance(digits, p) # [B, B]
    
    # 2. Compute pairwise latent Euclidean distances
    z1 = z.unsqueeze(1) # [B, 1, latent_dim]
    z2 = z.unsqueeze(0) # [1, B, latent_dim]
    D_latent = torch.norm(z1 - z2, p=2, dim=-1) # [B, B]
    
    # 3. Mask to only align pairs with matching primes
    prime_match = (p.unsqueeze(1) == p.unsqueeze(0)).float() # [B, B]
    # Exclude diagonal (distance to self is always 0, no need to optimize)
    diag_mask = 1.0 - torch.eye(B, device=z.device)
    mask = prime_match * diag_mask
    
    num_pairs = mask.sum()
    if num_pairs == 0:
        return torch.tensor(0.0, device=z.device)
        
    # 4. Normalize matrices by their masked means to achieve scale invariance
    mean_padic = (D_padic * mask).sum() / num_pairs
    mean_latent = (D_latent * mask).sum() / num_pairs
    
    if mean_padic == 0 or mean_latent == 0:
        return torch.tensor(0.0, device=z.device)
        
    D_padic_norm = D_padic / (mean_padic + 1e-8)
    D_latent_norm = D_latent / (mean_latent + 1e-8)
    
    # 5. MSE Loss over masked elements
    loss = (D_latent_norm - D_padic_norm).pow(2)
    loss = (loss * mask).sum() / num_pairs
    
    return loss

def compute_supcon_loss(z, p, temperature=0.1):
    """
    Supervised Contrastive loss (Khosla et al. 2020) using prime as the class label.

    For each anchor i, positives are all j with same prime (j ≠ i).
    Negatives are all k with different prime.

    z   : [B, D] — projected representations (already detached from encoder)
    p   : [B]    — prime bases (class labels)
    temperature : float — scales cosine similarities before softmax
    """
    B = z.shape[0]
    if B <= 1:
        return torch.tensor(0.0, device=z.device)

    z_norm = F.normalize(z, dim=-1)               # [B, D]
    sim    = z_norm @ z_norm.T / temperature       # [B, B]

    # Mask: same prime (excluding diagonal)
    same  = (p.unsqueeze(0) == p.unsqueeze(1))     # [B, B] bool
    diag  = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = same & ~diag                        # [B, B]
    neg_mask = ~same                               # [B, B] (cross-prime negatives)

    # Need at least one positive for each anchor; skip anchors with none
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=z.device)

    # Log-sum-exp over all non-self pairs (denominator)
    sim_masked = sim.masked_fill(diag, float('-inf'))
    log_denom  = torch.logsumexp(sim_masked, dim=1)  # [B]

    # For each anchor, mean log-prob over its positives
    pos_sim = sim * pos_mask.float()                 # zero out non-positives
    n_pos   = pos_mask.float().sum(dim=1).clamp(min=1)
    log_num = (pos_sim - log_denom.unsqueeze(1)) * pos_mask.float()
    loss_per = -log_num.sum(dim=1) / n_pos           # [B]

    return loss_per[has_pos].mean()


def compute_hyperbolic_metric_loss(z_ball, digits, p, manifold):
    """
    Metric alignment loss for Poincaré-ball latents.

    Replaces the Euclidean pairwise distance with geodesic distance on the
    Poincaré ball. Otherwise identical to compute_metric_loss.

    z_ball : [B, D] — points on the Poincaré ball (output of HyperbolicBetaVAE)
    manifold : geoopt.PoincareBall instance
    """
    B = z_ball.shape[0]
    if B <= 1:
        return torch.tensor(0.0, device=z_ball.device)

    D_padic = batch_padic_distance(digits, p)  # [B, B]

    # Vectorised pairwise geodesic distance
    z1 = z_ball.unsqueeze(1)  # [B, 1, D]
    z2 = z_ball.unsqueeze(0)  # [1, B, D]
    D_hyp = manifold.dist(z1, z2)  # [B, B]

    prime_match = (p.unsqueeze(1) == p.unsqueeze(0)).float()
    diag_mask   = 1.0 - torch.eye(B, device=z_ball.device)
    mask        = prime_match * diag_mask

    num_pairs = mask.sum()
    if num_pairs == 0:
        return torch.tensor(0.0, device=z_ball.device)

    mean_padic = (D_padic * mask).sum() / num_pairs
    mean_hyp   = (D_hyp   * mask).sum() / num_pairs

    if mean_padic == 0 or mean_hyp == 0:
        return torch.tensor(0.0, device=z_ball.device)

    D_padic_norm = D_padic / (mean_padic + 1e-8)
    D_hyp_norm   = D_hyp   / (mean_hyp   + 1e-8)

    loss = ((D_hyp_norm - D_padic_norm) ** 2 * mask).sum() / num_pairs
    return loss


if __name__ == "__main__":
    # Unit tests
    print("Testing batch_padic_distance:")
    digits = torch.tensor([
        [1, 0, 1, 0], # A
        [1, 0, 1, 1], # B (differs at index 3 from A)
        [1, 1, 0, 0], # C (differs at index 1 from A)
        [1, 0, 1, 0]  # D (same as A)
    ], dtype=torch.long)
    p = torch.tensor([2, 2, 2, 2], dtype=torch.long)
    
    D = batch_padic_distance(digits, p)
    print("Distance Matrix:\n", D)
    
    # Checks:
    # d(A, B) = 2^-3 = 0.125
    # d(A, C) = 2^-1 = 0.5
    # d(A, D) = 0
    assert abs(D[0, 1].item() - 0.125) < 1e-6
    assert abs(D[0, 2].item() - 0.5) < 1e-6
    assert abs(D[0, 3].item() - 0.0) < 1e-6
    print("All metric unit tests passed!")
