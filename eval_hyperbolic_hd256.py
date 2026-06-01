"""
eval_hyperbolic_hd256.py

Evaluates all four N=64 hyperbolic/Euclidean models per prime and prints a
comparison table with metric alignment loss and Spearman r.

Models compared:
  - Euclidean Beta-VAE  hd=64  (checkpoints/euclidean_n64/beta_vae_metric.pt)
  - Hyperbolic Poincaré hd=64  (checkpoints/hyperbolic_n64/hyperbolic_vae.pt)
  - Hyperbolic Poincaré hd=256 (checkpoints/hyperbolic_n64_hd256/hyperbolic_vae.pt)
  - Hyperbolic Lorentz  hd=256 (checkpoints/lorentz_n64_hd256/hyperbolic_vae.pt)
"""

import os, math, torch
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from dataset import PadicDataset
from beta_vae import ConditionalBetaVAE
from hyperbolic_vae import HyperbolicBetaVAE
from metric_alignment import batch_padic_distance, compute_hyperbolic_metric_loss, compute_metric_loss


PRIMES   = [2, 3, 5, 7, 11]
VOCAB    = max(PRIMES) + 2   # 13
N        = 64
SAMPLES  = 300               # per type per prime on eval set
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_euclidean(path, hidden_dim):
    m = ConditionalBetaVAE(vocab_size=VOCAB, hidden_dim=hidden_dim, latent_dim=32, N=N)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()


def load_hyperbolic(path, hidden_dim, manifold='poincare', curvature=1.0):
    m = HyperbolicBetaVAE(vocab_size=VOCAB, hidden_dim=hidden_dim, latent_dim=32, N=N,
                          manifold=manifold, curvature=curvature)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()


@torch.no_grad()
def get_mu(model, digits, p_tensor, is_hyperbolic):
    if is_hyperbolic:
        mu_tangent, _ = model.encode(digits, p_tensor)
        return model.manifold.expmap0(mu_tangent * (1.0 / math.sqrt(model.latent_dim)))
    else:
        mu, _ = model.encode(digits, p_tensor)
        return mu


def eval_per_prime(model, is_hyperbolic, dataset):
    results = {}
    for p in PRIMES:
        samples = [s for s in dataset if s['p'] == p and s['type'] != 2]
        if len(samples) < 10:
            continue
        digits  = torch.stack([s['digits'] for s in samples]).to(DEVICE)
        p_ten   = torch.full((len(samples),), p, dtype=torch.long, device=DEVICE)

        mu = get_mu(model, digits, p_ten, is_hyperbolic)

        # Alignment loss
        if is_hyperbolic:
            loss = compute_hyperbolic_metric_loss(mu, digits, p_ten, model.manifold).item()
            # Pairwise geodesic distances
            d_lat = model.manifold.dist(mu.unsqueeze(1), mu.unsqueeze(0)).cpu().numpy()
        else:
            loss = compute_metric_loss(mu, digits, p_ten).item()
            diff = mu.unsqueeze(1) - mu.unsqueeze(0)
            d_lat = diff.norm(dim=-1).cpu().numpy()

        d_pad = batch_padic_distance(digits, p_ten).cpu().numpy()

        # Upper-triangle (off-diagonal pairs)
        idx = np.triu_indices(len(samples), k=1)
        r, _ = spearmanr(d_pad[idx], d_lat[idx])

        results[p] = {'loss': loss, 'spearman': r}
    return results


def weighted_average(results_by_prime, key):
    # Weight each prime by log(p)+1 (same as training weight)
    total_w, total_v = 0.0, 0.0
    for p, v in results_by_prime.items():
        if not np.isnan(v[key]):
            w = math.log(p) + 1
            total_w += w
            total_v += w * v[key]
    return total_v / total_w if total_w > 0 else float('nan')


def print_and_save(all_results, save_path):
    primes_str = [f'$p={p}$' for p in PRIMES]
    headers = ['Prime', 'Euc hd=64 Loss', 'Euc hd=64 $r$',
               'Hyp-P hd=64 Loss', 'Hyp-P hd=64 $r$',
               'Hyp-P hd=256 Loss', 'Hyp-P hd=256 $r$',
               'Hyp-L hd=256 Loss', 'Hyp-L hd=256 $r$']

    rows = []
    for p in PRIMES:
        row = [f'$p={p}$']
        for name in ['euc64', 'hyp_p64', 'hyp_p256', 'hyp_l256']:
            r = all_results[name].get(p, {})
            row.append(f"{r.get('loss', float('nan')):.5f}")
            row.append(f"{r.get('spearman', float('nan')):.4f}")
        rows.append(row)

    # Weighted average row
    avg_row = ['**All (wtd)**']
    for name in ['euc64', 'hyp_p64', 'hyp_p256', 'hyp_l256']:
        avg_row.append(f"{weighted_average(all_results[name], 'loss'):.5f}")
        avg_row.append(f"{weighted_average(all_results[name], 'spearman'):.4f}")
    rows.append(avg_row)

    # Console table
    col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = ' | '.join(f'{{:<{w}}}' for w in col_w)
    print('\n' + fmt.format(*headers))
    print('-' * sum(col_w + [3 * (len(headers) - 1)]))
    for row in rows:
        print(fmt.format(*row))

    # Markdown file
    sep = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    lines = ['# Hyperbolic VAE hd=256 vs hd=64 — per-prime comparison (N=64, Broad-11)\n',
             '| ' + ' | '.join(headers) + ' |', sep]
    for row in rows:
        lines.append('| ' + ' | '.join(row) + ' |')
    content = '\n'.join(lines)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(content + '\n')
    print(f'\nSaved to {save_path}')
    return content


def main():
    print(f'Device: {DEVICE}')

    # Build eval dataset (no random sequences)
    ds = PadicDataset(primes=PRIMES, N=N, num_samples_per_type=SAMPLES)

    models = {
        'euc64':     (load_euclidean('./checkpoints/euclidean_n64/beta_vae_metric.pt', 64),     False),
        'hyp_p64':   (load_hyperbolic('./checkpoints/hyperbolic_n64/hyperbolic_vae.pt', 64),    True),
        'hyp_p256':  (load_hyperbolic('./checkpoints/hyperbolic_n64_hd256/hyperbolic_vae.pt', 256), True),
        'hyp_l256':  (load_hyperbolic('./checkpoints/lorentz_n64_hd256/hyperbolic_vae.pt', 256,
                                      manifold='lorentz'), True),
    }

    all_results = {}
    for name, (model, is_hyp) in models.items():
        print(f'\nEvaluating {name}...')
        all_results[name] = eval_per_prime(model, is_hyp, ds)

    print_and_save(all_results, './plots/hyperbolic_hd256_comparison.md')


if __name__ == '__main__':
    main()
