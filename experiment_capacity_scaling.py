"""
experiment_capacity_scaling.py

Side-by-side comparison: hidden_dim=64 vs hidden_dim=256 on the Broad-19 config.

Trains both variants from scratch with identical dataset split, seed, and
hyperparameters. Evaluates on held-out p=2 and p=5 sequences.

The hypothesis: the Broad-19 → Broad-23 accuracy drop is a capacity bottleneck.
If hidden_dim=256 recovers or improves Broad-19's metrics, the hypothesis holds.
"""

import os
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import PadicDataset
from models import ConditionalVQVAE, PriorGRU
from beta_vae import ConditionalBetaVAE
from metric_alignment import compute_metric_loss
from train import train_vqvae, train_prior
from train_metric import train_beta_vae_metric

# ------------------------------------------------------------------ #
# Config — Broad-19
# ------------------------------------------------------------------ #
PRIMES     = [2, 3, 5, 7, 11, 13, 17, 19]
VOCAB_SIZE = 19
N          = 32
SPT        = 300
BATCH      = 128
LR         = 1e-3
VQ_EPOCHS  = 10
PR_EPOCHS  = 10
VAE_EPOCHS = 12
BETA       = 0.05
GAMMA      = 5.0
SEED       = 42

SAVE_BASE  = './checkpoints/capacity_scaling'

HIDDEN_DIMS = [64, 256]   # the two variants to compare


# ------------------------------------------------------------------ #
# Evaluation
# ------------------------------------------------------------------ #
def evaluate(vqvae, beta_vae, eval_loader, device):
    vqvae.eval()
    beta_vae.eval()

    p2_correct, p2_total = 0, 0
    p5_correct, p5_total = 0, 0
    p2_z, p2_digits = [], []
    p5_z, p5_digits = [], []

    with torch.no_grad():
        for batch in eval_loader:
            digits = batch['digits'].to(device)
            p      = batch['p'].to(device)

            logits_vq, _, _ = vqvae(digits, p)
            preds_vq = torch.argmax(logits_vq, dim=-1)

            _, mu, logvar = beta_vae(digits, p)
            z = beta_vae.reparameterize(mu, logvar)

            for i in range(digits.shape[0]):
                prime   = p[i].item()
                correct = (preds_vq[i] == digits[i]).sum().item()
                if prime == 2:
                    p2_correct += correct; p2_total += N
                    p2_z.append(z[i]); p2_digits.append(digits[i])
                elif prime == 5:
                    p5_correct += correct; p5_total += N
                    p5_z.append(z[i]); p5_digits.append(digits[i])

    def _metric(z_list, d_list, prime_val):
        z_t = torch.stack(z_list).to(device)
        d_t = torch.stack(d_list).to(device)
        p_t = torch.full((len(z_list),), prime_val, dtype=torch.long, device=device)
        return compute_metric_loss(z_t, d_t, p_t).item()

    return {
        'vq_acc_p2': p2_correct / p2_total if p2_total else 0.0,
        'vq_acc_p5': p5_correct / p5_total if p5_total else 0.0,
        'metric_p2': _metric(p2_z, p2_digits, 2) if len(p2_z) > 1 else 0.0,
        'metric_p5': _metric(p5_z, p5_digits, 5) if len(p5_z) > 1 else 0.0,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_variant(name, hidden_dim, dataset, train_ds, val_ds, device):
    print(f"\n{'='*60}")
    print(f"  Variant: {name}  (hidden_dim={hidden_dim})")
    print(f"{'='*60}")
    t0 = time.time()

    save_dir = os.path.join(SAVE_BASE, name)
    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
    full_loader  = DataLoader(dataset,  batch_size=BATCH, shuffle=False)

    vqvae = ConditionalVQVAE(
        vocab_size=VOCAB_SIZE, hidden_dim=hidden_dim, codebook_size=64,
        latent_dim=32, N=N, cond_dim=16,
    )
    prior = PriorGRU(codebook_size=64, latent_dim=32, cond_dim=16, hidden_size=128, num_layers=2)
    beta_vae = ConditionalBetaVAE(
        vocab_size=VOCAB_SIZE, hidden_dim=hidden_dim, latent_dim=32, N=N, cond_dim=16,
    )

    # Parameter counts
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  VQ-VAE params: {count_params(vqvae):,}")
    print(f"  Prior params:  {count_params(prior):,}")
    print(f"  BetaVAE params:{count_params(beta_vae):,}")

    vqvae    = train_vqvae(vqvae, train_loader, val_loader, VQ_EPOCHS, LR, device)
    prior    = train_prior(vqvae, prior, full_loader, PR_EPOCHS, LR, device)
    beta_vae = train_beta_vae_metric(beta_vae, train_loader, val_loader, VAE_EPOCHS, LR, BETA, GAMMA, device)

    torch.save(vqvae.state_dict(),    os.path.join(save_dir, 'vqvae.pt'))
    torch.save(prior.state_dict(),    os.path.join(save_dir, 'prior.pt'))
    torch.save(beta_vae.state_dict(), os.path.join(save_dir, 'beta_vae_metric.pt'))

    elapsed = time.time() - t0
    print(f"\n  [{name}] total training time: {elapsed:.0f}s")
    return vqvae, beta_vae


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Config : N={N}, primes={PRIMES}, vocab={VOCAB_SIZE}, SPT={SPT}")
    print(f"         VQ={VQ_EPOCHS}ep, Prior={PR_EPOCHS}ep, VAE={VAE_EPOCHS}ep")

    print("\n--- Building dataset ---")
    set_seed(SEED)
    dataset = PadicDataset(primes=PRIMES, N=N, num_samples_per_type=SPT)
    val_size   = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    eval_dataset = PadicDataset(primes=[2, 5], N=N, num_samples_per_type=200)
    eval_loader  = DataLoader(eval_dataset, batch_size=BATCH, shuffle=False)

    results = {}
    for hd in HIDDEN_DIMS:
        name = f'hd{hd}'
        set_seed(SEED)
        vqvae, beta_vae = run_variant(name, hd, dataset, train_ds, val_ds, device)
        results[name] = evaluate(vqvae.to(device), beta_vae.to(device), eval_loader, device)

    # ---- Results table ----
    col  = 18
    keys = list(results.keys())
    print("\n\n" + "="*72)
    header = f"  {'Metric':<34}"
    for k in keys:
        hd_label = k.replace('hd', 'hidden_dim=')
        header += f" {hd_label:>{col}}"
    print(header)
    print("-"*72)

    rows = [
        ('VQ-VAE Accuracy p=2 (%)',  'vq_acc_p2', True,  '{:>{}.2f}'),
        ('VQ-VAE Accuracy p=5 (%)',  'vq_acc_p5', True,  '{:>{}.2f}'),
        ('Metric Alignment p=2',     'metric_p2', False, '{:>{}.5f}'),
        ('Metric Alignment p=5',     'metric_p5', False, '{:>{}.5f}'),
    ]
    for label, metric_key, higher_better, fmt in rows:
        scale  = 100 if metric_key.startswith('vq') else 1
        vals   = {k: results[k][metric_key] * scale for k in keys}
        best   = max(vals, key=vals.__getitem__) if higher_better else min(vals, key=vals.__getitem__)
        line   = f"  {label:<34}"
        for k in keys:
            marker = ' *' if k == best else '  '
            line += f" {fmt.format(vals[k], col)}{marker}"
        print(line)

    print("="*72)
    print("  (* = best for that metric)")
    print(f"\nCheckpoints saved under {os.path.abspath(SAVE_BASE)}/")


if __name__ == '__main__':
    main()
