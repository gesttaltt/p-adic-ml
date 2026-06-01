"""
evaluate_cascade_hierarchical.py  —  Item #23

Compares flat vs hierarchical slow-path in the Cascade Gating System.

Architecture:
  Fast path  : ConditionalBetaVAE (same as original cascade)
  Slow path A: flat ConditionalVQVAE + PriorGRU       (original)
  Slow path B: HierarchicalVQVAE + TopPriorGRU + BotPriorGRU (new)

Gating: Beta-VAE self-reconstruction cross-entropy vs threshold τ.
Precision metric: hierarchical VQ-VAE reconstruction accuracy on the
  final generated sequence (regardless of which path produced it).

For each τ, reports:
  - fast-path rate (% of samples taking Beta-VAE path)
  - precision (hierarchical VQ-VAE recon accuracy on final output)
  - velocity (samples/sec)

Output: plots/cascade_hierarchical.png, plots/cascade_hierarchical.md
"""

import os, math, time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from dataset import PadicDataset
from beta_vae import ConditionalBetaVAE
from models import ConditionalVQVAE, PriorGRU
from hierarchical_vqvae import HierarchicalVQVAE, TopPriorGRU, BotPriorGRU
from anomaly_detector import get_reconstruction_error

# ── config ────────────────────────────────────────────────────────────────────
PRIMES        = [2, 3, 5, 7, 11]
N             = 64
VOCAB         = 13
NUM_SAMPLES   = 500          # total sequences to generate per threshold sweep
THRESHOLDS    = [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 10.0]
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BETA_VAE_CKPT    = './checkpoints/euclidean_n64/beta_vae_metric.pt'
FLAT_VQVAE_CKPT  = './checkpoints/vqvae.pt'
FLAT_PRIOR_CKPT  = './checkpoints/prior.pt'
HIER_DIR         = './checkpoints/hierarchical'


# ── model loading ─────────────────────────────────────────────────────────────

def load_models():
    beta_vae = ConditionalBetaVAE(vocab_size=VOCAB, hidden_dim=64, latent_dim=32, N=N)
    beta_vae.load_state_dict(torch.load(BETA_VAE_CKPT, map_location=DEVICE))
    beta_vae.to(DEVICE).eval()

    flat_vqvae = ConditionalVQVAE(vocab_size=VOCAB, hidden_dim=64,
                                   codebook_size=64, latent_dim=32, N=N)
    flat_vqvae.load_state_dict(torch.load(FLAT_VQVAE_CKPT, map_location=DEVICE))
    flat_vqvae.to(DEVICE).eval()

    flat_prior = PriorGRU(codebook_size=64, latent_dim=32, cond_dim=16,
                           hidden_size=128, num_layers=2)
    flat_prior.load_state_dict(torch.load(FLAT_PRIOR_CKPT, map_location=DEVICE))
    flat_prior.to(DEVICE).eval()

    hier_vqvae = HierarchicalVQVAE(vocab_size=VOCAB, hidden_dim=64, N=N)
    hier_vqvae.load_state_dict(
        torch.load(f'{HIER_DIR}/vqvae.pt', map_location=DEVICE))
    hier_vqvae.to(DEVICE).eval()

    top_prior = TopPriorGRU(top_codebook=16, top_dim=32)
    top_prior.load_state_dict(
        torch.load(f'{HIER_DIR}/top_prior.pt', map_location=DEVICE))
    top_prior.to(DEVICE).eval()

    bot_prior = BotPriorGRU(bot_codebook=64, top_codebook=16, bot_dim=32, top_dim=32)
    bot_prior.load_state_dict(
        torch.load(f'{HIER_DIR}/bot_prior.pt', map_location=DEVICE))
    bot_prior.to(DEVICE).eval()

    return beta_vae, flat_vqvae, flat_prior, hier_vqvae, top_prior, bot_prior


# ── generation helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def beta_vae_generate(beta_vae, p_tensor):
    return beta_vae.sample(p_tensor, device=DEVICE)


@torch.no_grad()
def flat_slow_path(vqvae, prior, p_tensor):
    L = N // 2
    idx = prior.sample(p_tensor, L=L, temperature=0.7)
    q   = vqvae.quantizer.embedding(idx)
    return torch.argmax(vqvae.decode(q, p_tensor), dim=-1)


@torch.no_grad()
def hier_slow_path(hier_vqvae, top_prior, bot_prior, p_tensor):
    L_top  = N // 4
    idx_t  = top_prior.sample(p_tensor, L=L_top, temperature=0.7)
    idx_b  = bot_prior.sample(idx_t, p_tensor, temperature=0.7)
    z_top  = hier_vqvae.top_quantizer.embedding[idx_t]
    z_bot  = hier_vqvae.bot_quantizer.embedding[idx_b]
    return torch.argmax(hier_vqvae.decode(z_bot, z_top, p_tensor), dim=-1)


@torch.no_grad()
def hier_precision(hier_vqvae, digits, p_tensor):
    """Hierarchical VQ-VAE reconstruction accuracy on a set of sequences."""
    logits, _, _, _ = hier_vqvae(digits, p_tensor)
    preds = torch.argmax(logits, dim=-1)
    return (preds == digits).float().mean().item()


# ── sweep ─────────────────────────────────────────────────────────────────────

def sweep(beta_vae, slow_fn, hier_vqvae, p_tensor):
    """
    For each threshold: generate with cascade (beta_vae fast, slow_fn fallback),
    measure precision via hier_vqvae, velocity, fast-path rate.
    """
    B = p_tensor.shape[0]
    rows = []

    # Pre-generate fast-path candidates + their self-reconstruction errors
    t0 = time.time()
    x_fast = beta_vae_generate(beta_vae, p_tensor)
    errs   = get_reconstruction_error(beta_vae, x_fast, p_tensor)
    fast_gen_time = time.time() - t0

    for tau in THRESHOLDS:
        fast_mask  = errs < tau
        n_fast     = fast_mask.sum().item()
        n_slow     = B - n_fast
        slow_idx   = (~fast_mask).nonzero(as_tuple=True)[0]

        t0 = time.time()
        final = x_fast.clone()
        if n_slow > 0:
            final[slow_idx] = slow_fn(p_tensor[slow_idx])
        elapsed = fast_gen_time + (time.time() - t0)

        prec      = hier_precision(hier_vqvae, final, p_tensor)
        fast_rate = n_fast / B
        velocity  = B / elapsed

        rows.append({
            'tau':       tau,
            'fast_rate': fast_rate,
            'precision': prec,
            'velocity':  velocity,
        })
        print(f'  τ={tau:5.2f} | fast={fast_rate*100:5.1f}% | '
              f'prec={prec*100:6.2f}% | vel={velocity:7.1f} smpl/s')
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f'Device: {DEVICE}')
    beta_vae, flat_vqvae, flat_prior, hier_vqvae, top_prior, bot_prior = load_models()

    # Build test batch
    primes_rep = []
    for p in PRIMES:
        primes_rep.extend([p] * (NUM_SAMPLES // len(PRIMES)))
    p_tensor = torch.tensor(primes_rep, dtype=torch.long, device=DEVICE)

    print(f'\n=== Flat slow path ===')
    flat_rows = sweep(
        beta_vae,
        lambda p: flat_slow_path(flat_vqvae, flat_prior, p),
        hier_vqvae, p_tensor
    )

    print(f'\n=== Hierarchical slow path ===')
    hier_rows = sweep(
        beta_vae,
        lambda p: hier_slow_path(hier_vqvae, top_prior, bot_prior, p),
        hier_vqvae, p_tensor
    )

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=130)

    flat_fr = [r['fast_rate'] for r in flat_rows]
    flat_pr = [r['precision'] for r in flat_rows]
    hier_fr = [r['fast_rate'] for r in hier_rows]
    hier_pr = [r['precision'] for r in hier_rows]

    axes[0].plot(flat_fr, flat_pr, 'b-o', ms=5, label='Flat slow path')
    axes[0].plot(hier_fr, hier_pr, 'r-o', ms=5, label='Hierarchical slow path')
    axes[0].set_xlabel('Fast-path rate'); axes[0].set_ylabel('Precision (hier VQ-VAE recon acc)')
    axes[0].set_title('Precision–Speed Trade-off'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    flat_vel = [r['velocity'] for r in flat_rows]
    hier_vel = [r['velocity'] for r in hier_rows]
    axes[1].plot(flat_fr, flat_vel, 'b-o', ms=5, label='Flat slow path')
    axes[1].plot(hier_fr, hier_vel, 'r-o', ms=5, label='Hierarchical slow path')
    axes[1].set_xlabel('Fast-path rate'); axes[1].set_ylabel('Velocity (samples/sec)')
    axes[1].set_title('Velocity–Speed Trade-off'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle('Cascade Router: Flat vs Hierarchical Slow Path', fontweight='bold')
    plt.tight_layout()
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/cascade_hierarchical.png', bbox_inches='tight')
    plt.close()
    print('\nSaved ./plots/cascade_hierarchical.png')

    # ── markdown report ───────────────────────────────────────────────────────
    lines = ['# Cascade Router: Flat vs Hierarchical Slow Path\n',
             f'Fast path: Beta-VAE (`{BETA_VAE_CKPT}`)  ',
             f'Precision metric: HierarchicalVQVAE recon accuracy  ',
             f'N={N}, {NUM_SAMPLES} samples, primes={PRIMES}\n',
             '| τ | Flat fast% | Flat prec% | Hier fast% | Hier prec% |',
             '| :---: | :---: | :---: | :---: | :---: |']
    for f, h in zip(flat_rows, hier_rows):
        lines.append(
            f'| {f["tau"]:.2f} | {f["fast_rate"]*100:.1f} | {f["precision"]*100:.2f} '
            f'| {h["fast_rate"]*100:.1f} | {h["precision"]*100:.2f} |'
        )
    content = '\n'.join(lines)
    with open('./plots/cascade_hierarchical.md', 'w') as fh:
        fh.write(content + '\n')
    print('Saved ./plots/cascade_hierarchical.md')
    print('\n' + content)


if __name__ == '__main__':
    main()
