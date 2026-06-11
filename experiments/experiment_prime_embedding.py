"""
experiment_prime_embedding.py

Side-by-side comparison: categorical prime embedding vs continuous (MLP) embedding.

Both variants share the same dataset split, random seed, and hyperparameters.
They differ only in how the prime conditioning signal is produced:

  Categorical  — nn.Embedding(prime_vocab_size, cond_dim)
                 Each prime is an independent vocabulary entry; no shared structure.

  Continuous   — PrimeEmbedder: Linear(2→cond_dim)→ReLU→Linear(cond_dim→cond_dim)
                 Input features: [p/23, log(p)/log(23)]
                 Numerically close primes (e.g. 17 and 19) start with similar
                 representations, allowing the model to share learned structure.

Note: both variants have the same number of prime-embedding parameters (~320)
so the comparison is purely about inductive bias, not capacity.
"""
import sys, os; _r = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(_r, 'src')); os.chdir(_r)

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
# Experiment config
# ------------------------------------------------------------------ #
PRIMES     = [2, 3, 5, 7, 11]
N          = 32     # sequence length (shorter for faster CPU runs)
SPT        = 300    # samples per type per prime
BATCH      = 128
LR         = 1e-3
VQ_EPOCHS  = 10
PR_EPOCHS  = 10
VAE_EPOCHS = 12
BETA       = 0.05
GAMMA      = 5.0
SEED       = 42

SAVE_BASE  = './checkpoints/embedding_comparison'

# ------------------------------------------------------------------ #
# Categorical embedding — inlined so the main codebase stays clean
# ------------------------------------------------------------------ #
class CategoricalPrimeEmbedder(nn.Module):
    """Each prime treated as an independent vocabulary index (original approach)."""
    PRIME_VOCAB = 20  # covers primes up to 19

    def __init__(self, out_dim):
        super().__init__()
        self.emb = nn.Embedding(self.PRIME_VOCAB, out_dim)

    def forward(self, p):
        return self.emb(p)


def _patch_categorical(vqvae, prior, beta_vae, cond_dim=16):
    """Replace PrimeEmbedder with CategoricalPrimeEmbedder on all three models."""
    vqvae.prime_emb    = CategoricalPrimeEmbedder(cond_dim)
    prior.prime_emb    = CategoricalPrimeEmbedder(cond_dim)
    beta_vae.prime_emb = CategoricalPrimeEmbedder(cond_dim)


# ------------------------------------------------------------------ #
# Evaluation
# ------------------------------------------------------------------ #
def evaluate(vqvae, beta_vae, eval_loader, device):
    """VQ-VAE digit accuracy and Beta-VAE metric alignment for p=2 and p=5."""
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
                    p2_correct += correct
                    p2_total   += N
                    p2_z.append(z[i])
                    p2_digits.append(digits[i])
                elif prime == 5:
                    p5_correct += correct
                    p5_total   += N
                    p5_z.append(z[i])
                    p5_digits.append(digits[i])

    results = {
        'vq_acc_p2': p2_correct / p2_total if p2_total else 0.0,
        'vq_acc_p5': p5_correct / p5_total if p5_total else 0.0,
        'metric_p2': 0.0,
        'metric_p5': 0.0,
    }

    def _metric(z_list, digits_list, prime_val):
        z_t = torch.stack(z_list).to(device)
        d_t = torch.stack(digits_list).to(device)
        p_t = torch.full((len(z_list),), prime_val, dtype=torch.long, device=device)
        return compute_metric_loss(z_t, d_t, p_t).item()

    if len(p2_z) > 1:
        results['metric_p2'] = _metric(p2_z, p2_digits, 2)
    if len(p5_z) > 1:
        results['metric_p5'] = _metric(p5_z, p5_digits, 5)

    return results


# ------------------------------------------------------------------ #
# Training helpers
# ------------------------------------------------------------------ #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_variant(name, dataset, train_ds, val_ds, device, use_categorical):
    print(f"\n{'='*60}")
    print(f"  Variant: {name}")
    print(f"{'='*60}")
    t0 = time.time()

    save_dir = os.path.join(SAVE_BASE, name.lower().replace(' ', '_'))
    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
    full_loader  = DataLoader(dataset,  batch_size=BATCH, shuffle=False)

    vqvae    = ConditionalVQVAE(vocab_size=13, hidden_dim=64, codebook_size=64, latent_dim=32, N=N, cond_dim=16)
    prior    = PriorGRU(codebook_size=64, latent_dim=32, cond_dim=16, hidden_size=128, num_layers=2)
    beta_vae = ConditionalBetaVAE(vocab_size=13, hidden_dim=64, latent_dim=32, N=N, cond_dim=16)

    if use_categorical:
        _patch_categorical(vqvae, prior, beta_vae, cond_dim=16)

    vqvae    = train_vqvae(vqvae, train_loader, val_loader, VQ_EPOCHS, LR, device)
    prior    = train_prior(vqvae, prior, full_loader, PR_EPOCHS, LR, device)
    beta_vae = train_beta_vae_metric(beta_vae, train_loader, val_loader, VAE_EPOCHS, LR, BETA, GAMMA, device)

    torch.save(vqvae.state_dict(),    os.path.join(save_dir, 'vqvae.pt'))
    torch.save(prior.state_dict(),    os.path.join(save_dir, 'prior.pt'))
    torch.save(beta_vae.state_dict(), os.path.join(save_dir, 'beta_vae_metric.pt'))

    elapsed = time.time() - t0
    print(f"\n  [{name}] total training time: {elapsed:.0f}s")
    return vqvae, beta_vae


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Config : N={N}, primes={PRIMES}, SPT={SPT}")
    print(f"         VQ={VQ_EPOCHS}ep, Prior={PR_EPOCHS}ep, VAE={VAE_EPOCHS}ep")

    # ---- Dataset — shared between both variants ----
    print("\n--- Building dataset ---")
    set_seed(SEED)
    dataset = PadicDataset(primes=PRIMES, N=N, num_samples_per_type=SPT)
    val_size   = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Held-out test set (never used during training)
    eval_dataset = PadicDataset(primes=[2, 5], N=N, num_samples_per_type=200)
    eval_loader  = DataLoader(eval_dataset, batch_size=BATCH, shuffle=False)

    # ---- Train categorical ----
    set_seed(SEED)
    vqvae_cat, beta_cat = run_variant(
        'Categorical', dataset, train_ds, val_ds, device, use_categorical=True
    )

    # ---- Train continuous ----
    set_seed(SEED)
    vqvae_cont, beta_cont = run_variant(
        'Continuous', dataset, train_ds, val_ds, device, use_categorical=False
    )

    # ---- Evaluate ----
    print("\n\n--- Evaluating on held-out p=2, p=5 test set ---")
    r_cat  = evaluate(vqvae_cat.to(device),  beta_cat.to(device),  eval_loader, device)
    r_cont = evaluate(vqvae_cont.to(device), beta_cont.to(device), eval_loader, device)

    # ---- Results table ----
    col = 20
    print("\n" + "="*74)
    print(f"  {'Metric':<34} {'Categorical':>{col}} {'Continuous':>{col}}  Winner")
    print("-"*74)

    rows = [
        ('VQ-VAE Accuracy p=2 (%)',  'vq_acc_p2', True,  '{:>{}.2f}'),
        ('VQ-VAE Accuracy p=5 (%)',  'vq_acc_p5', True,  '{:>{}.2f}'),
        ('Metric Alignment p=2',     'metric_p2', False, '{:>{}.5f}'),
        ('Metric Alignment p=5',     'metric_p5', False, '{:>{}.5f}'),
    ]
    for label, key, higher_better, fmt in rows:
        scale = 100 if key.startswith('vq') else 1
        v_cat  = r_cat[key]  * scale
        v_cont = r_cont[key] * scale
        cont_wins = (v_cont > v_cat) if higher_better else (v_cont < v_cat)
        winner = 'CONT ✓' if cont_wins else 'CAT  ✓'
        s_cat  = fmt.format(v_cat,  col)
        s_cont = fmt.format(v_cont, col)
        print(f"  {label:<34} {s_cat} {s_cont}  {winner}")

    print("="*74)

    cont_wins_total = sum(
        1 for _, key, hb, _ in rows
        if ((r_cont[key] > r_cat[key]) if hb else (r_cont[key] < r_cat[key]))
    )
    print(f"\n  Continuous wins {cont_wins_total}/{len(rows)} metrics.")
    print(f"\nCheckpoints saved under {os.path.abspath(SAVE_BASE)}/")


if __name__ == '__main__':
    main()
