"""
hierarchical_vqvae.py

Two-level VQ-VAE for p-adic sequences (VQ-VAE-2 style).

Architecture
────────────
Encoder (bottom-up):
  digits → embed+cond
          → enc_stride2                     [B, H, N/2]   shared feature map
          ├─ bottom path: res → proj  →  [B, N/2, D_bot]  → bottom quantizer
          └─ top    path: stride2 → res → proj → [B, N/4, D_top] → top quantizer

Decoder (top-down):
  z_q_top  → upsample ConvTranspose  → [B, N/2, H]
  z_q_bot  → proj                    → [B, N/2, H]
  sum + cond → standard conv decoder → [B, N, vocab_size]

Loss:
  L = L_recon + vq_loss_top + vq_loss_bot

Motivation for p-adic trees:
  The first few digits determine the major branch of the tree; later digits
  refine within that branch.  The top codebook (16 codes, N/4 tokens) learns
  to capture global branch identity.  The bottom codebook (64 codes, N/2
  tokens) refines the fine-grained digit pattern conditioned on the top.

Priors (trained separately in train_hierarchical.py):
  TopPriorGRU   — autoregressive over N/4 top-codebook indices
  BotPriorGRU   — autoregressive over N/2 bottom-codebook indices,
                  conditioned on top-codebook indices via cross-attention inject
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt

from models import VectorQuantizer, ResidualBlock, PrimeEmbedder


class HyperbolicVectorQuantizer(nn.Module):
    """
    VQ codebook whose entries live on the Poincaré ball.

    Encoder outputs tangent vectors at the origin; we expmap0 them to the ball,
    find the nearest codebook entry via geodesic distance, then apply the
    straight-through estimator in ambient R^D space (standard trick).

    The commitment loss uses squared geodesic distance instead of squared
    Euclidean distance.
    """

    def __init__(self, num_embeddings, embedding_dim, curvature=1.0, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings  = num_embeddings
        self.embedding_dim   = embedding_dim
        self.commitment_cost = commitment_cost
        self.manifold        = geoopt.PoincareBall(c=curvature)

        # Initialise codebook entries on the ball (small norm → near origin)
        init = torch.randn(num_embeddings, embedding_dim) * 0.05
        self.embedding = geoopt.ManifoldParameter(
            self.manifold.projx(init), manifold=self.manifold
        )

    def forward(self, inputs_tangent):
        """
        inputs_tangent: [B, L, D]  tangent vectors at origin (encoder output)
        Returns: vq_loss (scalar), z_q (on ball, [B, L, D]), indices ([B, L])
        """
        B, L, D = inputs_tangent.shape
        scale   = 1.0 / math.sqrt(D)

        # Map encoder output to ball
        z_ball = self.manifold.expmap0(inputs_tangent * scale)   # [B, L, D]
        z_flat = z_ball.reshape(-1, D)                           # [B*L, D]

        # Pairwise geodesic distances to all codebook vectors
        # manifold.dist expects broadcastable shapes
        dists = self.manifold.dist(
            z_flat.unsqueeze(1),           # [B*L, 1,  D]
            self.embedding.unsqueeze(0)    # [1,   K,  D]
        ) ** 2                             # [B*L, K]

        indices = torch.argmin(dists, dim=1).reshape(B, L)       # [B, L]
        z_q     = self.embedding[indices]                        # [B, L, D] on ball

        # Commitment loss in geodesic distance
        vq_loss = (
            self.manifold.dist(z_q,                z_ball.detach()) ** 2
        ).mean() + self.commitment_cost * (
            self.manifold.dist(z_q.detach(),       z_ball) ** 2
        ).mean()

        # Straight-through: forward = z_q, gradient sees z_ball
        z_q_st = z_ball + (z_q - z_ball).detach()
        return vq_loss, z_q_st, indices


class HierarchicalVQVAE(nn.Module):
    def __init__(self, vocab_size=13, hidden_dim=64, N=64,
                 bot_codebook=64, top_codebook=16,
                 bot_dim=32, top_dim=32, cond_dim=16,
                 hyperbolic_top=False, top_curvature=1.0):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_dim  = hidden_dim
        self.N           = N
        self.bot_dim     = bot_dim
        self.top_dim     = top_dim
        self.cond_dim    = cond_dim
        self.L_bot       = N // 2   # bottom latent length
        self.L_top       = N // 4   # top    latent length

        # ── shared input embedding ─────────────────────────────────────────
        self.digit_emb  = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb    = nn.Parameter(torch.zeros(1, N, hidden_dim))
        self.prime_emb  = PrimeEmbedder(cond_dim)
        self.cond_proj  = nn.Linear(cond_dim, hidden_dim)

        # ── shared first downsample (N → N/2) ─────────────────────────────
        self.enc_stride2  = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.enc_res_shared = ResidualBlock(hidden_dim)

        # ── bottom encoder (N/2 features → N/2 latents) ───────────────────
        self.enc_res_bot  = ResidualBlock(hidden_dim)
        self.enc_proj_bot = nn.Conv1d(hidden_dim, bot_dim, kernel_size=1)
        self.bot_quantizer = VectorQuantizer(bot_codebook, bot_dim)

        # ── top encoder (N/2 features → N/4 latents) ──────────────────────
        self.enc_stride2_top = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.enc_res_top     = ResidualBlock(hidden_dim)
        self.enc_proj_top    = nn.Conv1d(hidden_dim, top_dim, kernel_size=1)
        self.hyperbolic_top  = hyperbolic_top
        if hyperbolic_top:
            self.top_quantizer = HyperbolicVectorQuantizer(top_codebook, top_dim,
                                                           curvature=top_curvature)
        else:
            self.top_quantizer = VectorQuantizer(top_codebook, top_dim)

        # ── decoder ───────────────────────────────────────────────────────
        # top upsample: N/4 → N/2
        self.dec_top_up   = nn.ConvTranspose1d(top_dim, hidden_dim, kernel_size=3,
                                               stride=2, padding=1, output_padding=1)
        # combine top context + bottom codes
        self.dec_bot_proj = nn.Conv1d(bot_dim, hidden_dim, kernel_size=1)
        self.dec_res1     = ResidualBlock(hidden_dim)
        # N/2 → N
        self.dec_deconv   = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3,
                                               stride=2, padding=1, output_padding=1)
        self.dec_res2     = ResidualBlock(hidden_dim)
        self.dec_proj     = nn.Linear(hidden_dim, vocab_size)

    # ── encoder ───────────────────────────────────────────────────────────

    def encode(self, digits, p):
        """
        Returns (z_q_bot, z_q_top, idx_bot, idx_top, vq_loss_bot, vq_loss_top).
        z_q_bot : [B, L_bot, bot_dim]
        z_q_top : [B, L_top, top_dim]
        """
        B = digits.shape[0]

        # shared embedding + condition
        x = self.digit_emb(digits) + self.pos_emb            # [B, N, H]
        cond = self.cond_proj(self.prime_emb(p)).unsqueeze(1) # [B, 1, H]
        x = (x + cond).transpose(1, 2)                       # [B, H, N]

        # shared first stride
        h = F.relu(self.enc_stride2(x))     # [B, H, N/2]
        h = self.enc_res_shared(h)

        # bottom branch
        h_bot = self.enc_res_bot(h)                           # [B, H, N/2]
        z_bot = self.enc_proj_bot(h_bot).transpose(1, 2)     # [B, N/2, bot_dim]
        vq_loss_bot, z_q_bot, idx_bot = self.bot_quantizer(z_bot)

        # top branch
        h_top = F.relu(self.enc_stride2_top(h))              # [B, H, N/4]
        h_top = self.enc_res_top(h_top)
        z_top = self.enc_proj_top(h_top).transpose(1, 2)     # [B, N/4, top_dim]
        vq_loss_top, z_q_top, idx_top = self.top_quantizer(z_top)

        return z_q_bot, z_q_top, idx_bot, idx_top, vq_loss_bot, vq_loss_top

    # ── decoder ───────────────────────────────────────────────────────────

    def decode(self, z_q_bot, z_q_top, p):
        """
        z_q_bot : [B, L_bot, bot_dim]
        z_q_top : [B, L_top, top_dim]  — on Poincaré ball if hyperbolic_top, else Euclidean
        """
        # if top codes live on the manifold, map back to tangent space before ConvTranspose
        if self.hyperbolic_top:
            z_q_top = self.top_quantizer.manifold.logmap0(z_q_top)

        # upsample top → N/2
        top_ctx = F.relu(self.dec_top_up(z_q_top.transpose(1, 2)))  # [B, H, N/2]

        # project bottom + add top context
        bot_h = self.dec_bot_proj(z_q_bot.transpose(1, 2))          # [B, H, N/2]
        h = F.relu(bot_h + top_ctx)
        h = self.dec_res1(h)

        # upsample to N
        h = F.relu(self.dec_deconv(h))                               # [B, H, N]
        h = self.dec_res2(h)
        h = h.transpose(1, 2)                                        # [B, N, H]

        # prime conditioning in decoder
        cond = self.cond_proj(self.prime_emb(p)).unsqueeze(1)
        h = h + cond

        logits = self.dec_proj(h)                                    # [B, N, vocab_size]

        # mask invalid digits
        mask = (torch.arange(self.vocab_size, device=logits.device)
                .unsqueeze(0).unsqueeze(0) >= p.unsqueeze(1).unsqueeze(2))
        return logits.masked_fill(mask, -1e9)

    def forward(self, digits, p):
        z_q_bot, z_q_top, idx_bot, idx_top, vq_loss_bot, vq_loss_top = self.encode(digits, p)
        logits = self.decode(z_q_bot, z_q_top, p)
        vq_loss = vq_loss_bot + vq_loss_top
        return logits, vq_loss, idx_bot, idx_top


# ── priors ────────────────────────────────────────────────────────────────────

class TopPriorGRU(nn.Module):
    """Autoregressive prior over top codebook indices (unconditional on bottom)."""

    def __init__(self, top_codebook=16, top_dim=32, cond_dim=16, hidden_size=128, num_layers=2):
        super().__init__()
        self.codebook_size = top_codebook
        self.sos_token     = top_codebook
        self.token_emb     = nn.Embedding(top_codebook + 1, top_dim)
        self.prime_emb     = PrimeEmbedder(cond_dim)
        self.gru = nn.GRU(top_dim + cond_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, top_codebook)

    def forward(self, idx_top, p):
        B, L = idx_top.shape
        sos  = torch.full((B, 1), self.sos_token, dtype=torch.long, device=idx_top.device)
        inp  = torch.cat([sos, idx_top[:, :-1]], dim=1)
        x    = self.token_emb(inp)
        c    = self.prime_emb(p).unsqueeze(1).expand(-1, L, -1)
        out, _ = self.gru(torch.cat([x, c], dim=-1))
        return self.proj(out)

    @torch.no_grad()
    def sample(self, p, L, temperature=1.0):
        B, device = p.shape[0], p.device
        tok    = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        c      = self.prime_emb(p).unsqueeze(1)
        hidden = None
        tokens = []
        for _ in range(L):
            x = self.token_emb(tok)
            out, hidden = self.gru(torch.cat([x, c], dim=-1), hidden)
            logits = self.proj(out[:, 0])
            probs  = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            tok    = torch.multinomial(probs, 1)
            tokens.append(tok)
        return torch.cat(tokens, dim=1)


class BotPriorGRU(nn.Module):
    """
    Autoregressive prior over bottom codebook indices, conditioned on top indices.

    The top context is injected by:
      1. Embedding the full top index sequence.
      2. Upsampling (repeat each top token twice) to match bottom length.
      3. Concatenating with each bottom input step.
    """

    def __init__(self, bot_codebook=64, top_codebook=16,
                 bot_dim=32, top_dim=32, cond_dim=16,
                 hidden_size=256, num_layers=2):
        super().__init__()
        self.codebook_size = bot_codebook
        self.sos_token     = bot_codebook
        self.token_emb     = nn.Embedding(bot_codebook + 1, bot_dim)
        self.top_emb       = nn.Embedding(top_codebook, top_dim)
        self.prime_emb     = PrimeEmbedder(cond_dim)
        self.gru = nn.GRU(bot_dim + top_dim + cond_dim, hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, bot_codebook)

    def _top_context(self, idx_top, L_bot):
        """Upsample top indices (L_top) → bottom length (L_bot) by repeating."""
        top_emb = self.top_emb(idx_top)                         # [B, L_top, top_dim]
        # each top token covers 2 bottom positions (L_bot = 2 * L_top)
        return top_emb.repeat_interleave(2, dim=1)[:, :L_bot]   # [B, L_bot, top_dim]

    def forward(self, idx_bot, idx_top, p):
        B, L_bot = idx_bot.shape
        sos = torch.full((B, 1), self.sos_token, dtype=torch.long, device=idx_bot.device)
        inp = torch.cat([sos, idx_bot[:, :-1]], dim=1)
        x   = self.token_emb(inp)                               # [B, L_bot, bot_dim]
        ctx = self._top_context(idx_top, L_bot)                 # [B, L_bot, top_dim]
        c   = self.prime_emb(p).unsqueeze(1).expand(-1, L_bot, -1)
        out, _ = self.gru(torch.cat([x, ctx, c], dim=-1))
        return self.proj(out)

    @torch.no_grad()
    def sample(self, idx_top, p, temperature=1.0):
        B, L_top = idx_top.shape
        L_bot  = L_top * 2
        device = idx_top.device
        ctx    = self._top_context(idx_top, L_bot)              # [B, L_bot, top_dim]
        c      = self.prime_emb(p).unsqueeze(1)
        tok    = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        hidden = None
        tokens = []
        for t in range(L_bot):
            x = self.token_emb(tok)
            inp = torch.cat([x, ctx[:, t:t+1], c], dim=-1)
            out, hidden = self.gru(inp, hidden)
            logits = self.proj(out[:, 0])
            probs  = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            tok    = torch.multinomial(probs, 1)
            tokens.append(tok)
        return torch.cat(tokens, dim=1)
