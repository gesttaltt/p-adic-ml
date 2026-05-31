"""
hyperbolic_vae.py

Poincaré-ball latent-space VAE for p-adic sequence generation.

Key differences from ConditionalBetaVAE:
  - Latent space is the Poincaré ball B^d_c (curvature c, dimension latent_dim)
  - Encoder maps to a tangent vector at the origin, then projects to the ball via expmap0
  - Reparameterize: sample v ~ N(0, σ²) in the tangent space at origin, parallel-transport
    to μ_ball, push to ball via expmap — this is the "wrapped-normal" approximation
  - Decoder maps from ball back to Euclidean via logmap0, then same conv decoder
  - Loss: reconstruction + β * ||μ_tangent||² (pulls toward ball origin, replaces KL)
    No metric alignment loss: hyperbolic distance already approximates the ultrametric
    by construction — the geometry enforces it

The curvature c controls how aggressively the space is curved:
  c=1.0 : standard Poincaré ball (good starting point)
  c>1.0 : sharper hierarchy separation

Requires: pip install geoopt
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt

from models import PrimeEmbedder


class HyperbolicBetaVAE(nn.Module):
    def __init__(self, vocab_size=13, hidden_dim=64, latent_dim=32, N=32,
                 cond_dim=16, curvature=1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.N          = N
        self.cond_dim   = cond_dim

        self.manifold = geoopt.PoincareBall(c=curvature)

        # Shared with ConditionalBetaVAE: prime + digit embeddings
        self.digit_emb  = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb    = nn.Parameter(torch.zeros(1, N, hidden_dim))
        self.prime_emb  = PrimeEmbedder(cond_dim)
        self.cond_proj  = nn.Linear(cond_dim, hidden_dim)

        # Encoder (same conv structure)
        self.enc_conv1   = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.enc_conv2   = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.flatten_dim = hidden_dim * (N // 4)
        # μ lives in the TANGENT space at origin (Euclidean vector, then expmap0 → ball)
        self.fc_mu      = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar  = nn.Linear(self.flatten_dim, latent_dim)  # log σ²

        # Decoder (same conv structure; receives logmap0(z_ball) ∈ R^latent_dim)
        self.fc_dec      = nn.Linear(latent_dim, self.flatten_dim)
        self.dec_deconv1 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2,
                                               padding=1, output_padding=1)
        self.dec_deconv2 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2,
                                               padding=1, output_padding=1)
        self.dec_proj    = nn.Linear(hidden_dim, vocab_size)

    # ------------------------------------------------------------------ #
    # Encoder
    # ------------------------------------------------------------------ #
    def encode(self, digits, p):
        """Returns (mu_tangent, logvar) both in Euclidean R^latent_dim."""
        B = digits.shape[0]
        x = self.digit_emb(digits) + self.pos_emb          # [B, N, H]
        cond = self.cond_proj(self.prime_emb(p)).unsqueeze(1)
        x = x + cond

        x = x.transpose(1, 2)                              # [B, H, N]
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))                      # [B, H, N/4]

        x_flat  = x.reshape(B, -1)
        mu      = self.fc_mu(x_flat)
        logvar  = self.fc_logvar(x_flat)
        return mu, logvar

    # ------------------------------------------------------------------ #
    # Reparameterize: sample on the Poincaré ball
    # ------------------------------------------------------------------ #
    def reparameterize(self, mu_tangent, logvar):
        """
        Projects μ_tangent to the ball, then samples a point near it.

        Returns z_ball ∈ B^d_c  (a geoopt.ManifoldTensor).
        """
        mu_ball = self.manifold.expmap0(mu_tangent)  # [B, D]

        if not self.training:
            return mu_ball

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)           # sample in tangent space at origin
        v   = std * eps                        # [B, D]

        # Parallel-transport v from origin to μ_ball, then push to ball
        v_transported = self.manifold.transp0(mu_ball, v)  # [B, D]
        z_ball = self.manifold.expmap(mu_ball, v_transported)
        # Project onto ball and keep clear of the boundary (norms < 1 - ε)
        z_ball = self.manifold.projx(z_ball)
        max_r  = 1.0 / (self.manifold.c ** 0.5) * (1 - 1e-3)
        norms  = z_ball.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        z_ball = z_ball * (max_r / norms).clamp(max=1.0)
        return z_ball

    # ------------------------------------------------------------------ #
    # Decoder
    # ------------------------------------------------------------------ #
    def decode(self, z_ball, p):
        """
        Maps z_ball from Poincaré ball back to Euclidean via logmap0,
        then runs the same convolutional decoder.
        """
        B = z_ball.shape[0]

        # Project from ball to tangent space at origin (Euclidean)
        z_euclidean = self.manifold.logmap0(z_ball)        # [B, D]

        x = F.relu(self.fc_dec(z_euclidean))               # [B, flatten_dim]
        x = x.reshape(B, self.hidden_dim, self.N // 4)

        x = F.relu(self.dec_deconv1(x))                    # [B, H, N/2]
        x = F.relu(self.dec_deconv2(x))                    # [B, H, N]
        x = x.transpose(1, 2)                              # [B, N, H]

        cond   = self.cond_proj(self.prime_emb(p)).unsqueeze(1)
        x      = x + cond
        logits = self.dec_proj(x)                          # [B, N, vocab_size]

        # Mask invalid digits (same as parent class)
        mask   = (torch.arange(self.vocab_size, device=logits.device)
                  .unsqueeze(0).unsqueeze(0) >= p.unsqueeze(1).unsqueeze(2))
        logits = logits.masked_fill(mask, -1e9)
        return logits

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, digits, p):
        mu_tangent, logvar = self.encode(digits, p)
        z_ball             = self.reparameterize(mu_tangent, logvar)
        logits             = self.decode(z_ball, p)
        return logits, mu_tangent, logvar

    # ------------------------------------------------------------------ #
    # Sample
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, p, device='cpu'):
        """Draw z from the origin of the ball (prior = wrapped-normal at origin)."""
        B  = p.shape[0]
        z0 = torch.zeros(B, self.latent_dim, device=device)
        logits = self.decode(z0, p)
        return torch.argmax(logits, dim=-1)
