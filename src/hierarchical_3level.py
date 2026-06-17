"""
hierarchical_3level.py

Three-level VQ-VAE for p-adic sequences (N=128).

Architecture
────────────
Levels (for N=128):
  Top  : N/8 = 16 tokens, codebook 16  — global branch
  Mid  : N/4 = 32 tokens, codebook 32  — intermediate branch
  Bot  : N/2 = 64 tokens, codebook 64  — local digit patterns

Encoder (bottom-up):
  digits → embed+cond            [B, N, H]
         → enc_stride2           [B, H, N/2]   shared
         ├─ bot: res → proj      [B, N/2, D_bot] → bot VQ
         → enc_stride2_mid       [B, H, N/4]
         ├─ mid: res → proj      [B, N/4, D_mid] → mid VQ
         → enc_stride2_top       [B, H, N/8]
         └─ top: res → proj      [B, N/8, D_top] → top VQ

Decoder (top-down):
  z_q_top  → upsample N/8→N/4  [B, H, N/4]
  z_q_mid  → proj + add         [B, H, N/4]  mid context
  upsample N/4→N/2              [B, H, N/2]
  z_q_bot  → proj + add         [B, H, N/2]  bot context
  upsample N/2→N  → logits      [B, N, vocab]

Priors:
  ThreeLevelTopPriorGRU  — N/8 top tokens (unconditional)
  ThreeLevelMidPriorGRU  — N/4 mid tokens, conditioned on top
  ThreeLevelBotPriorGRU  — N/2 bot tokens, conditioned on mid+top
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import VectorQuantizer, ResidualBlock, PrimeEmbedder
from hierarchical_vqvae import HyperbolicVectorQuantizer


class ThreeLevelVQVAE(nn.Module):
    def __init__(self, vocab_size=13, hidden_dim=64, N=128,
                 bot_codebook=64, mid_codebook=32, top_codebook=16,
                 bot_dim=32, mid_dim=32, top_dim=32, cond_dim=16,
                 use_attention_decoder=False,
                 hyperbolic_top=False, top_curvature=1.0):
        super().__init__()
        assert N % 8 == 0, 'N must be divisible by 8'
        self.vocab_size    = vocab_size
        self.N             = N
        self.L_bot         = N // 2
        self.L_mid         = N // 4
        self.L_top         = N // 8
        self.bot_dim       = bot_dim
        self.mid_dim       = mid_dim
        self.top_dim       = top_dim
        self.hyperbolic_top = hyperbolic_top

        # ── input embedding ──────────────────────────────────────────────────
        self.digit_emb  = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb    = nn.Parameter(torch.zeros(1, N, hidden_dim))
        self.prime_emb  = PrimeEmbedder(cond_dim)
        self.cond_proj  = nn.Linear(cond_dim, hidden_dim)

        # ── shared first stride N → N/2 ───────────────────────────────────
        self.enc_stride2    = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.enc_res_shared = ResidualBlock(hidden_dim)

        # ── bot branch ────────────────────────────────────────────────────
        self.enc_res_bot  = ResidualBlock(hidden_dim)
        self.enc_proj_bot = nn.Conv1d(hidden_dim, bot_dim, 1)
        self.bot_quantizer = VectorQuantizer(bot_codebook, bot_dim)

        # ── mid branch: N/2 → N/4 ────────────────────────────────────────
        self.enc_stride2_mid = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.enc_res_mid     = ResidualBlock(hidden_dim)
        self.enc_proj_mid    = nn.Conv1d(hidden_dim, mid_dim, 1)
        self.mid_quantizer   = VectorQuantizer(mid_codebook, mid_dim)

        # ── top branch: N/4 → N/8 ────────────────────────────────────────
        self.enc_stride2_top = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.enc_res_top     = ResidualBlock(hidden_dim)
        self.enc_proj_top    = nn.Conv1d(hidden_dim, top_dim, 1)
        if hyperbolic_top:
            self.top_quantizer = HyperbolicVectorQuantizer(
                top_codebook, top_dim, curvature=top_curvature)
        else:
            self.top_quantizer = VectorQuantizer(top_codebook, top_dim)

        # ── decoder ───────────────────────────────────────────────────────
        self.use_attention_decoder = use_attention_decoder
        if use_attention_decoder:
            self.dec_queries = nn.Parameter(torch.zeros(1, N, hidden_dim))
            nn.init.normal_(self.dec_queries, mean=0.0, std=0.02)
            self.dec_top_proj_attn = nn.Linear(top_dim, hidden_dim)
            self.dec_mid_proj_attn = nn.Linear(mid_dim, hidden_dim)
            self.dec_bot_proj_attn = nn.Linear(bot_dim, hidden_dim)
            
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='relu',
                batch_first=True
            )
            self.trans_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
            self.dec_proj = nn.Linear(hidden_dim, vocab_size)
        else:
            # top N/8 → N/4
            self.dec_top_up   = nn.ConvTranspose1d(top_dim, hidden_dim, 3, stride=2,
                                                   padding=1, output_padding=1)
            # mid context
            self.dec_mid_proj = nn.Conv1d(mid_dim, hidden_dim, 1)
            self.dec_res_mid  = ResidualBlock(hidden_dim)
            # N/4 → N/2
            self.dec_mid_up   = nn.ConvTranspose1d(hidden_dim, hidden_dim, 3, stride=2,
                                                   padding=1, output_padding=1)
            # bot context
            self.dec_bot_proj = nn.Conv1d(bot_dim, hidden_dim, 1)
            self.dec_res_bot  = ResidualBlock(hidden_dim)
            # N/2 → N
            self.dec_bot_up   = nn.ConvTranspose1d(hidden_dim, hidden_dim, 3, stride=2,
                                                   padding=1, output_padding=1)
            self.dec_res_out  = ResidualBlock(hidden_dim)
            self.dec_proj     = nn.Linear(hidden_dim, vocab_size)

    def encode(self, digits, p):
        B = digits.shape[0]
        x = (self.digit_emb(digits) + self.pos_emb)
        x = (x + self.cond_proj(self.prime_emb(p)).unsqueeze(1)).transpose(1, 2)

        h = F.relu(self.enc_stride2(x))      # [B, H, N/2]
        h = self.enc_res_shared(h)

        # bot
        z_bot = self.enc_proj_bot(self.enc_res_bot(h)).transpose(1, 2)
        vq_bot, z_q_bot, idx_bot = self.bot_quantizer(z_bot)

        # mid
        hm = F.relu(self.enc_stride2_mid(h))
        hm = self.enc_res_mid(hm)
        z_mid = self.enc_proj_mid(hm).transpose(1, 2)
        vq_mid, z_q_mid, idx_mid = self.mid_quantizer(z_mid)

        # top
        ht = F.relu(self.enc_stride2_top(hm))
        ht = self.enc_res_top(ht)
        z_top = self.enc_proj_top(ht).transpose(1, 2)
        vq_top, z_q_top, idx_top = self.top_quantizer(z_top)

        vq_loss = vq_bot + vq_mid + vq_top
        return z_q_bot, z_q_mid, z_q_top, idx_bot, idx_mid, idx_top, vq_loss

    def decode(self, z_q_bot, z_q_mid, z_q_top, p):
        if self.hyperbolic_top:
            z_q_top = self.top_quantizer.manifold.logmap0(z_q_top)
        if self.use_attention_decoder:
            B = z_q_bot.shape[0]
            top_feats = self.dec_top_proj_attn(z_q_top)  # [B, L_top, H]
            mid_feats = self.dec_mid_proj_attn(z_q_mid)  # [B, L_mid, H]
            bot_feats = self.dec_bot_proj_attn(z_q_bot)  # [B, L_bot, H]
            memory = torch.cat([top_feats, mid_feats, bot_feats], dim=1)  # [B, L_top + L_mid + L_bot, H]
            
            cond = self.cond_proj(self.prime_emb(p)).unsqueeze(1)
            queries = self.dec_queries.expand(B, -1, -1) + cond
            
            h = self.trans_decoder(queries, memory)
            logits = self.dec_proj(h)
        else:
            # top → N/4
            h = F.relu(self.dec_top_up(z_q_top.transpose(1, 2)))
            h = F.relu(h + self.dec_mid_proj(z_q_mid.transpose(1, 2)))
            h = self.dec_res_mid(h)
            # → N/2
            h = F.relu(self.dec_mid_up(h))
            h = F.relu(h + self.dec_bot_proj(z_q_bot.transpose(1, 2)))
            h = self.dec_res_bot(h)
            # → N
            h = F.relu(self.dec_bot_up(h))
            h = self.dec_res_out(h).transpose(1, 2)
            h = h + self.cond_proj(self.prime_emb(p)).unsqueeze(1)
            logits = self.dec_proj(h)

        mask = (torch.arange(self.vocab_size, device=logits.device)
                .unsqueeze(0).unsqueeze(0) >= p.unsqueeze(1).unsqueeze(2))
        return logits.masked_fill(mask, -1e9)

    def forward(self, digits, p):
        z_q_bot, z_q_mid, z_q_top, idx_bot, idx_mid, idx_top, vq_loss = self.encode(digits, p)
        logits = self.decode(z_q_bot, z_q_mid, z_q_top, p)
        return logits, vq_loss, idx_bot, idx_mid, idx_top


# ── priors ────────────────────────────────────────────────────────────────────

class ThreeLevelTopPriorGRU(nn.Module):
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
        sos = torch.full((B, 1), self.sos_token, dtype=torch.long, device=idx_top.device)
        inp = self.token_emb(torch.cat([sos, idx_top[:, :-1]], dim=1))
        c   = self.prime_emb(p).unsqueeze(1).expand(-1, L, -1)
        out, _ = self.gru(torch.cat([inp, c], dim=-1))
        return self.proj(out)

    @torch.no_grad()
    def sample(self, p, L, temperature=1.0):
        B, device = p.shape[0], p.device
        tok = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        c = self.prime_emb(p).unsqueeze(1); hidden = None; tokens = []
        for _ in range(L):
            x = self.token_emb(tok)
            out, hidden = self.gru(torch.cat([x, c], dim=-1), hidden)
            probs = F.softmax(self.proj(out[:, 0]) / max(temperature, 1e-6), dim=-1)
            tok = torch.multinomial(probs, 1); tokens.append(tok)
        return torch.cat(tokens, dim=1)


class ThreeLevelMidPriorGRU(nn.Module):
    """Autoregressive over N/4 mid tokens, conditioned on top (N/8) via repeat×2."""
    def __init__(self, mid_codebook=32, top_codebook=16, mid_dim=32, top_dim=32,
                 cond_dim=16, hidden_size=192, num_layers=2):
        super().__init__()
        self.codebook_size = mid_codebook
        self.sos_token     = mid_codebook
        self.token_emb     = nn.Embedding(mid_codebook + 1, mid_dim)
        self.top_emb       = nn.Embedding(top_codebook, top_dim)
        self.prime_emb     = PrimeEmbedder(cond_dim)
        self.gru = nn.GRU(mid_dim + top_dim + cond_dim, hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, mid_codebook)

    def _top_ctx(self, idx_top, L_mid):
        return self.top_emb(idx_top).repeat_interleave(2, dim=1)[:, :L_mid]

    def forward(self, idx_mid, idx_top, p):
        B, L_mid = idx_mid.shape
        sos = torch.full((B, 1), self.sos_token, dtype=torch.long, device=idx_mid.device)
        x   = self.token_emb(torch.cat([sos, idx_mid[:, :-1]], dim=1))
        ctx = self._top_ctx(idx_top, L_mid)
        c   = self.prime_emb(p).unsqueeze(1).expand(-1, L_mid, -1)
        out, _ = self.gru(torch.cat([x, ctx, c], dim=-1))
        return self.proj(out)

    @torch.no_grad()
    def sample(self, idx_top, p, temperature=1.0):
        B, L_top = idx_top.shape; L_mid = L_top * 2; device = idx_top.device
        ctx    = self._top_ctx(idx_top, L_mid)
        c      = self.prime_emb(p).unsqueeze(1)
        tok    = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        hidden = None; tokens = []
        for t in range(L_mid):
            x   = self.token_emb(tok)
            inp = torch.cat([x, ctx[:, t:t+1], c], dim=-1)
            out, hidden = self.gru(inp, hidden)
            probs = F.softmax(self.proj(out[:, 0]) / max(temperature, 1e-6), dim=-1)
            tok   = torch.multinomial(probs, 1); tokens.append(tok)
        return torch.cat(tokens, dim=1)


class ThreeLevelBotPriorGRU(nn.Module):
    """Autoregressive over N/2 bot tokens, conditioned on mid+top via repeat."""
    def __init__(self, bot_codebook=64, mid_codebook=32, top_codebook=16,
                 bot_dim=32, mid_dim=32, top_dim=32,
                 cond_dim=16, hidden_size=256, num_layers=2):
        super().__init__()
        self.codebook_size = bot_codebook
        self.sos_token     = bot_codebook
        self.token_emb     = nn.Embedding(bot_codebook + 1, bot_dim)
        self.mid_emb       = nn.Embedding(mid_codebook, mid_dim)
        self.top_emb       = nn.Embedding(top_codebook, top_dim)
        self.prime_emb     = PrimeEmbedder(cond_dim)
        self.gru = nn.GRU(bot_dim + mid_dim + top_dim + cond_dim, hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, bot_codebook)

    def _ctx(self, idx_mid, idx_top, L_bot):
        mid_ctx = self.mid_emb(idx_mid).repeat_interleave(2, dim=1)[:, :L_bot]
        top_ctx = self.top_emb(idx_top).repeat_interleave(4, dim=1)[:, :L_bot]
        return mid_ctx, top_ctx

    def forward(self, idx_bot, idx_mid, idx_top, p):
        B, L_bot = idx_bot.shape
        sos = torch.full((B, 1), self.sos_token, dtype=torch.long, device=idx_bot.device)
        x   = self.token_emb(torch.cat([sos, idx_bot[:, :-1]], dim=1))
        mc, tc = self._ctx(idx_mid, idx_top, L_bot)
        c   = self.prime_emb(p).unsqueeze(1).expand(-1, L_bot, -1)
        out, _ = self.gru(torch.cat([x, mc, tc, c], dim=-1))
        return self.proj(out)

    @torch.no_grad()
    def sample(self, idx_mid, idx_top, p, temperature=1.0):
        B, L_mid = idx_mid.shape; L_bot = L_mid * 2; device = idx_mid.device
        mc, tc = self._ctx(idx_mid, idx_top, L_bot)
        c      = self.prime_emb(p).unsqueeze(1)
        tok    = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        hidden = None; tokens = []
        for t in range(L_bot):
            x   = self.token_emb(tok)
            inp = torch.cat([x, mc[:, t:t+1], tc[:, t:t+1], c], dim=-1)
            out, hidden = self.gru(inp, hidden)
            probs = F.softmax(self.proj(out[:, 0]) / max(temperature, 1e-6), dim=-1)
            tok   = torch.multinomial(probs, 1); tokens.append(tok)
        return torch.cat(tokens, dim=1)
